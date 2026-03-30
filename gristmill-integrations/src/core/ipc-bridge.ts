/**
 * IpcBridge — TypeScript client for the GristMill daemon Unix-socket IPC.
 *
 * Implements the same `IBridge` interface as `MockBridge` and `NativeBridge`,
 * so it is a drop-in replacement.  All processing logic lives in the Rust
 * daemon; this file only handles I/O.
 *
 * ## Frame format  (mirrors gristmill-daemon/src/ipc.rs)
 *
 *   [ 4-byte LE u32 length ][ MessagePack-encoded body ]
 *
 * ## Request envelope
 *
 *   { id: number, request: { method: string, params?: object } }
 *
 * ## Response envelope
 *
 *   { id: number, ok?: unknown, error?: string }
 *
 * ## Socket path
 *
 *   Default: ~/.gristmill/gristmill.sock
 *   Override: GRISTMILL_SOCK environment variable
 */

import * as net from "node:net";
import { encode, decode } from "@msgpack/msgpack";
import { monotonicFactory } from "ulid";
import type {
  IBridge,
  GristEventInit,
  RouteDecision,
  Memory,
  RankedMemory,
  EscalationResult,
  PipelineResult,
} from "./bridge.js";

const ulid = monotonicFactory();
const HEADER_SIZE = 4;
const MAX_FRAME_BYTES = 16 * 1024 * 1024;

// ── GristEvent construction ───────────────────────────────────────────────────
// Mirrors grist_core::parse_channel and GristEvent::new in Rust so we can
// build valid GristEvent JSON without a daemon round-trip.

// Rust ChannelType uses #[serde(tag = "type", rename_all = "snake_case")].
// Unit variants → { type: "variant_name" }
// Struct variants → { type: "variant_name", field: value, ... }
function toChannelSource(channel: string): unknown {
  switch (channel.toLowerCase()) {
    case "http":        return { type: "http" };
    case "websocket":   return { type: "web_socket" };
    case "cli":         return { type: "cli" };
    case "cron":        return { type: "cron" };
    case "webhook":     return { type: "webhook", provider: "generic" };
    case "mq":          return { type: "message_queue", topic: "default" };
    case "fs":          return { type: "file_system", path: "/" };
    case "python":      return { type: "python", callback_id: "default" };
    case "typescript":  return { type: "type_script", adapter_id: "default" };
    // Slack events arrive via the TypeScript shell — use TypeScript adapter variant
    case "slack":       return { type: "type_script", adapter_id: "slack" };
    case "internal":    return { type: "internal", subsystem: "core" };
    // Any other adapter registered via the plugin system
    default:            return { type: "type_script", adapter_id: channel };
  }
}

function buildEvent(channel: string, payload: unknown, priority = "normal"): string {
  return JSON.stringify({
    id: ulid(),
    source: toChannelSource(channel),
    timestamp_ms: Date.now(),
    payload,
    metadata: {
      // Rust Priority uses #[serde(rename_all = "lowercase")] → "normal", "high", etc.
      priority: priority.toLowerCase(),
      correlation_id: null,
      reply_channel: null,
      ttl_ms: null,
      tags: {},
    },
  });
}

// ── Pending request slot ──────────────────────────────────────────────────────

interface Pending {
  frame: Buffer;
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
}

// ── IpcBridge ─────────────────────────────────────────────────────────────────

export class IpcBridge implements IBridge {
  private socket: net.Socket | null = null;
  private rxBuffer = Buffer.alloc(0);
  /** Queue of requests waiting to be sent (server is sequential). */
  private readonly queue: Pending[] = [];
  /** The one request currently awaiting a response. */
  private inflight: Omit<Pending, "frame"> | null = null;
  private nextId = 1;

  constructor(readonly socketPath: string) {}

  // ── Connection ─────────────────────────────────────────────────────────────

  private connect(): Promise<net.Socket> {
    if (this.socket?.writable) return Promise.resolve(this.socket);

    return new Promise((resolve, reject) => {
      const sock = net.createConnection(this.socketPath);

      sock.on("connect", () => {
        this.socket = sock;
        resolve(sock);
      });

      sock.on("data", (chunk: Buffer) => {
        this.rxBuffer = Buffer.concat([this.rxBuffer, chunk]);
        this.drainFrames();
      });

      sock.on("error", (err) => {
        this.socket = null;
        this.inflight?.reject(err as Error);
        this.inflight = null;
        for (const p of this.queue) p.reject(err as Error);
        this.queue.length = 0;
        reject(err);
      });

      sock.on("close", () => {
        this.socket = null;
      });
    });
  }

  // ── Frame codec ────────────────────────────────────────────────────────────

  private drainFrames(): void {
    while (this.rxBuffer.length >= HEADER_SIZE) {
      const len = this.rxBuffer.readUInt32LE(0);
      if (len > MAX_FRAME_BYTES) {
        this.socket?.destroy(new Error(`IPC frame too large: ${len} bytes`));
        return;
      }
      if (this.rxBuffer.length < HEADER_SIZE + len) break;

      const body = this.rxBuffer.subarray(HEADER_SIZE, HEADER_SIZE + len);
      this.rxBuffer = this.rxBuffer.subarray(HEADER_SIZE + len);

      const msg = decode(body) as { id: number; ok?: unknown; error?: string };
      const handler = this.inflight;
      this.inflight = null;

      if (handler) {
        if (msg.error) {
          handler.reject(new Error(msg.error));
        } else {
          handler.resolve(msg.ok);
        }
      }

      this.flushQueue();
    }
  }

  private flushQueue(): void {
    if (this.inflight || this.queue.length === 0 || !this.socket?.writable) return;
    const next = this.queue.shift()!;
    this.inflight = { resolve: next.resolve, reject: next.reject };
    this.socket.write(next.frame);
  }

  // ── Send ───────────────────────────────────────────────────────────────────

  private async send(request: unknown): Promise<unknown> {
    const sock = await this.connect();
    const id = this.nextId++;
    const encoded = Buffer.from(encode({ id, request }));
    const header = Buffer.allocUnsafe(HEADER_SIZE);
    header.writeUInt32LE(encoded.length, 0);
    const frame = Buffer.concat([header, encoded]);

    return new Promise((resolve, reject) => {
      this.queue.push({ frame, resolve, reject });
      this.flushQueue();
    });
  }

  // ── IBridge ────────────────────────────────────────────────────────────────

  async triage(event: GristEventInit): Promise<RouteDecision> {
    const result = await this.send({
      method: "triage",
      params: { event_json: this.buildEventJson(event.channel, event.payload, event.priority) },
    });
    return result as RouteDecision;
  }

  async remember(content: string, tags: string[]): Promise<string> {
    const result = await this.send({
      method: "remember",
      params: { content, tags },
    });
    return result as string;
  }

  async recall(query: string, limit = 10): Promise<RankedMemory[]> {
    const result = await this.send({
      method: "recall",
      params: { query, limit },
    });
    return result as RankedMemory[];
  }

  async getMemory(id: string): Promise<Memory | null> {
    const result = await this.send({
      method: "get_memory",
      params: { id },
    });
    return (result ?? null) as Memory | null;
  }

  async escalate(prompt: string, maxTokens = 1024): Promise<EscalationResult> {
    const result = await this.send({
      method: "escalate",
      params: { prompt, max_tokens: maxTokens },
    });
    return result as EscalationResult;
  }

  registerPipeline(pipeline: object): void {
    // Fire-and-forget: daemon receives the pipeline but we don't await.
    this.send({
      method: "register_pipeline",
      params: { pipeline_json: JSON.stringify(pipeline) },
    }).catch((err) => {
      console.error("[IpcBridge] registerPipeline failed:", err);
    });
  }

  async runPipeline(pipelineId: string, event: GristEventInit): Promise<PipelineResult> {
    const result = await this.send({
      method: "run_pipeline",
      params: {
        pipeline_id: pipelineId,
        event_json: this.buildEventJson(event.channel, event.payload, event.priority),
      },
    });
    return result as PipelineResult;
  }

  pipelineIds(): string[] {
    // IPC is async but IBridge declares this sync — return cached value.
    // Callers that need an up-to-date list should await getPipelineIds().
    return [];
  }

  /** Async variant of pipelineIds() — use this when you need the real list. */
  async getPipelineIds(): Promise<string[]> {
    const result = await this.send({ method: "pipeline_ids" });
    return result as string[];
  }

  buildEventJson(channel: string, payload: unknown, priority?: string): string {
    return buildEvent(channel, payload, priority);
  }

  // subscribe() over the Unix socket would require a separate streaming
  // protocol.  Return an empty async iterable for now — Bell Tower bus events
  // are queued for Phase 4 (gRPC streaming or SSE).
  subscribe(_topic: string): AsyncIterable<unknown> {
    return {
      [Symbol.asyncIterator]() {
        return {
          next(): Promise<IteratorResult<unknown>> {
            return new Promise(() => {}); // never resolves
          },
        };
      },
    };
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /** Gracefully close the socket. */
  close(): void {
    this.socket?.end();
    this.socket = null;
  }

  /** Check whether the daemon is reachable. */
  async ping(): Promise<boolean> {
    try {
      const result = await this.send({ method: "health" });
      return (result as { status?: string })?.status === "ok";
    } catch {
      return false;
    }
  }

  /** Retrieve routing / budget metrics from the daemon. */
  async metrics(): Promise<unknown> {
    const result = await this.send({ method: "metrics", params: {} });
    return result;
  }
}
