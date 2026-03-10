/**
 * GristMillBridge — napi-rs wrapper around the Rust core.
 *
 * This is the ONLY file that imports from `@gristmill/core`.  All other
 * modules in gristmill-integrations import from here.
 *
 * Architecture note (CLAUDE.md): TypeScript handles I/O only; all processing
 * logic is delegated to Rust via this bridge.
 *
 * When the native module is unavailable (e.g. in unit tests), a mock bridge
 * is provided automatically.  Set `GRISTMILL_MOCK_BRIDGE=1` to force the mock.
 */

import { monotonicFactory } from "ulid";

const ulid = monotonicFactory();

// ── Shared types ──────────────────────────────────────────────────────────────

export interface RouteDecision {
  route: "LOCAL_ML" | "RULES" | "HYBRID" | "LLM_NEEDED";
  confidence: number;
  modelId?: string;
  reason?: string;
  estimatedTokens?: number;
}

export interface GristEventInit {
  /** Channel label: "http" | "websocket" | "cron" | "webhook" | "mq" | "fs" | "python" | "typescript" | "internal" | "cli" */
  channel: string;
  payload: unknown;
  priority?: "low" | "normal" | "high" | "critical";
  correlationId?: string;
  tags?: Record<string, string>;
}

export interface Memory {
  id: string;
  content: string;
  tags: string[];
  createdAtMs: number;
  lastAccessedMs: number;
  tier: "hot" | "warm" | "cold";
}

export interface RankedMemory {
  memory: Memory;
  score: number;
  sources: string[];
}

export interface EscalationResult {
  requestId: string;
  content: string;
  provider: string;
  cacheHit: boolean;
  tokensUsed: number;
  elapsedMs: number;
}

export interface PipelineResult {
  runId: string;
  pipelineId: string;
  succeeded: boolean;
  elapsedMs: number;
  output: unknown;
}

// ── Bridge interface ──────────────────────────────────────────────────────────

export interface IBridge {
  triage(event: GristEventInit): Promise<RouteDecision>;
  remember(content: string, tags: string[]): Promise<string>;
  recall(query: string, limit?: number): Promise<RankedMemory[]>;
  getMemory(id: string): Promise<Memory | null>;
  escalate(prompt: string, maxTokens?: number): Promise<EscalationResult>;
  registerPipeline(pipeline: object): void;
  runPipeline(pipelineId: string, event: GristEventInit): Promise<PipelineResult>;
  pipelineIds(): string[];
  buildEventJson(channel: string, payload: unknown): string;
  subscribe(topic: string): AsyncIterable<unknown>;
}

// ── Native bridge ─────────────────────────────────────────────────────────────

interface NativeCoreInstance {
  triage(json: string): Promise<string>;
  remember(content: string, tags: string[]): Promise<string>;
  recall(query: string, limit: number): Promise<string>;
  getMemory(id: string): Promise<string | null>;
  escalate(prompt: string, maxTokens: number): Promise<string>;
  registerPipeline(json: string): void;
  runPipeline(pipelineId: string, eventJson: string): Promise<string>;
  pipelineIds(): string[];
  buildEvent(channel: string, payloadJson: string): string;
  subscribe(topic: string): { nextJson(): Promise<string | null> };
}

/** Wraps the napi-rs `GristMillBridge` struct. */
class NativeBridge implements IBridge {
  private readonly native: NativeCoreInstance;

  constructor(configPath?: string) {
    this.native = new NativeCore(configPath ?? null);
  }

  private encodeEvent(event: GristEventInit): string {
    return this.native.buildEvent(event.channel, JSON.stringify(event.payload));
  }

  async triage(event: GristEventInit): Promise<RouteDecision> {
    const json = await this.native.triage(this.encodeEvent(event));
    return JSON.parse(json) as RouteDecision;
  }

  async remember(content: string, tags: string[]): Promise<string> {
    return this.native.remember(content, tags);
  }

  async recall(query: string, limit = 10): Promise<RankedMemory[]> {
    const json = await this.native.recall(query, limit);
    return JSON.parse(json) as RankedMemory[];
  }

  async getMemory(id: string): Promise<Memory | null> {
    const json = await this.native.getMemory(id);
    return json ? (JSON.parse(json) as Memory) : null;
  }

  async escalate(prompt: string, maxTokens = 1024): Promise<EscalationResult> {
    const json = await this.native.escalate(prompt, maxTokens);
    return JSON.parse(json) as EscalationResult;
  }

  registerPipeline(pipeline: object): void {
    this.native.registerPipeline(JSON.stringify(pipeline));
  }

  async runPipeline(
    pipelineId: string,
    event: GristEventInit
  ): Promise<PipelineResult> {
    const json = await this.native.runPipeline(
      pipelineId,
      this.encodeEvent(event)
    );
    return JSON.parse(json) as PipelineResult;
  }

  pipelineIds(): string[] {
    return this.native.pipelineIds();
  }

  buildEventJson(channel: string, payload: unknown): string {
    return this.native.buildEvent(channel, JSON.stringify(payload));
  }

  subscribe(topic: string): AsyncIterable<unknown> {
    const sub = this.native.subscribe(topic);
    return {
      [Symbol.asyncIterator]() {
        return {
          async next() {
            const json = await sub.nextJson();
            if (json === null) return { done: true, value: undefined };
            return { done: false, value: JSON.parse(json) };
          },
        };
      },
    };
  }
}

// ── Mock bridge (for tests / development) ────────────────────────────────────

/** In-memory mock that satisfies the bridge interface without Rust. */
export class MockBridge implements IBridge {
  private memories: Map<string, Memory> = new Map();
  private pipelines: Map<string, object> = new Map();
  private busEmitters: Map<string, Array<(v: unknown) => void>> = new Map();

  async triage(_event: GristEventInit): Promise<RouteDecision> {
    return { route: "LOCAL_ML", confidence: 0.9, modelId: "mock-model" };
  }

  async remember(content: string, tags: string[]): Promise<string> {
    const id = ulid();
    const now = Date.now();
    this.memories.set(id, {
      id,
      content,
      tags,
      createdAtMs: now,
      lastAccessedMs: now,
      tier: "hot",
    });
    return id;
  }

  async recall(query: string, limit = 10): Promise<RankedMemory[]> {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    return [...this.memories.values()]
      .filter((m) => {
        const text = (m.content + " " + m.tags.join(" ")).toLowerCase();
        return terms.some((t) => text.includes(t));
      })
      .slice(0, limit)
      .map((m) => ({ memory: m, score: 0.8, sources: ["keyword"] }));
  }

  async getMemory(id: string): Promise<Memory | null> {
    return this.memories.get(id) ?? null;
  }

  async escalate(prompt: string, _maxTokens = 1024): Promise<EscalationResult> {
    return {
      requestId: ulid(),
      content: `[mock] Response to: ${prompt.slice(0, 50)}`,
      provider: "mock",
      cacheHit: false,
      tokensUsed: 10,
      elapsedMs: 1,
    };
  }

  registerPipeline(pipeline: object & { id?: string }): void {
    const id = (pipeline as { id: string }).id;
    if (id) this.pipelines.set(id, pipeline);
  }

  async runPipeline(
    pipelineId: string,
    _event: GristEventInit
  ): Promise<PipelineResult> {
    return {
      runId: ulid(),
      pipelineId,
      succeeded: true,
      elapsedMs: 1,
      output: {},
    };
  }

  pipelineIds(): string[] {
    return [...this.pipelines.keys()];
  }

  buildEventJson(channel: string, payload: unknown): string {
    return JSON.stringify({ channel, payload, id: ulid(), timestamp_ms: Date.now() });
  }

  /** Emit an event on the mock bus (for testing Bell Tower). */
  emit(topic: string, payload: unknown): void {
    const listeners = this.busEmitters.get(topic) ?? [];
    for (const fn of listeners) fn(payload);
  }

  subscribe(topic: string): AsyncIterable<unknown> {
    const queue: unknown[] = [];
    const resolvers: Array<(v: IteratorResult<unknown>) => void> = [];

    const listener = (v: unknown) => {
      if (resolvers.length > 0) {
        resolvers.shift()!({ done: false, value: v });
      } else {
        queue.push(v);
      }
    };

    if (!this.busEmitters.has(topic)) this.busEmitters.set(topic, []);
    this.busEmitters.get(topic)!.push(listener);

    return {
      [Symbol.asyncIterator]() {
        return {
          next(): Promise<IteratorResult<unknown>> {
            if (queue.length > 0) {
              return Promise.resolve({ done: false, value: queue.shift()! });
            }
            return new Promise((resolve) => resolvers.push(resolve));
          },
        };
      },
    };
  }
}

// ── IpcBridge re-export ───────────────────────────────────────────────────────

export { IpcBridge } from "./ipc-bridge.js";

// ── Factory ───────────────────────────────────────────────────────────────────

// Lazy-load the native module so the package is importable without it
// (tests, build-time checks, etc.)
let NativeCore: new (configPath: string | null) => NativeCoreInstance;

function tryLoadNative(): boolean {
  if (process.env["GRISTMILL_MOCK_BRIDGE"] === "1") return false;
  try {
    // Dynamic import — resolved at runtime from the optional dep
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require("@gristmill/core");
    NativeCore = mod.GristMillBridge;
    return true;
  } catch {
    return false;
  }
}

const nativeAvailable = tryLoadNative();

/**
 * Create the appropriate bridge implementation.
 *
 * If the napi-rs native module is installed and `GRISTMILL_MOCK_BRIDGE` is
 * not set, the real Rust bridge is returned.  Otherwise a mock is returned.
 */
export function createBridge(configPath?: string): IBridge {
  if (nativeAvailable) {
    return new NativeBridge(configPath);
  }
  return new MockBridge();
}

/** Convenience alias for the default bridge singleton. */
export class GristMillBridge implements IBridge {
  private readonly impl: IBridge;

  constructor(configPath?: string) {
    this.impl = createBridge(configPath);
  }

  triage(event: GristEventInit) {
    return this.impl.triage(event);
  }
  remember(content: string, tags: string[]) {
    return this.impl.remember(content, tags);
  }
  recall(query: string, limit?: number) {
    return this.impl.recall(query, limit);
  }
  getMemory(id: string) {
    return this.impl.getMemory(id);
  }
  escalate(prompt: string, maxTokens?: number) {
    return this.impl.escalate(prompt, maxTokens);
  }
  registerPipeline(pipeline: object) {
    return this.impl.registerPipeline(pipeline);
  }
  runPipeline(pipelineId: string, event: GristEventInit) {
    return this.impl.runPipeline(pipelineId, event);
  }
  pipelineIds() {
    return this.impl.pipelineIds();
  }
  buildEventJson(channel: string, payload: unknown) {
    return this.impl.buildEventJson(channel, payload);
  }
  subscribe(topic: string) {
    return this.impl.subscribe(topic);
  }
}
