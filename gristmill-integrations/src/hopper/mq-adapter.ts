/**
 * MqAdapter — message-queue hopper adapter.
 *
 * Supports AMQP (RabbitMQ) via the optional `amqplib` package, and an
 * in-process queue for testing (when no URL is provided).
 *
 * Architecture: TypeScript handles I/O only — all routing decisions
 * delegate to the Rust core via bridge.triage().
 *
 * Usage (AMQP):
 *   const adapter = new MqAdapter({ url: "amqp://localhost", queue: "events", bridge });
 *   await adapter.start();
 *   // ...
 *   await adapter.stop();
 *
 * Usage (in-process / testing):
 *   const adapter = new MqAdapter({ queue: "events", bridge });
 *   await adapter.start();
 *   await adapter.push({ content: '{"foo":1}', fields: { routingKey: "events", exchange: "" }, properties: {} });
 */

import type { IBridge, GristEventInit } from "../core/bridge.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MqMessage {
  content: Buffer | string;
  fields: { routingKey: string; exchange: string };
  properties: Record<string, unknown>;
}

export interface MqAdapterOptions {
  /**
   * AMQP connection URL, e.g. "amqp://localhost".
   * If omitted, the adapter runs in in-process mode (useful for tests).
   */
  url?: string;
  /** Queue or topic to consume from. */
  queue: string;
  bridge: IBridge;
  /** How to map a raw MQ message to a GristEventInit. */
  transform?: (msg: MqMessage) => GristEventInit;
}

// ── Default transform ─────────────────────────────────────────────────────────

function defaultTransform(msg: MqMessage): GristEventInit {
  const raw: string =
    typeof msg.content === "string"
      ? msg.content
      : (msg.content as Buffer).toString("utf-8");

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    parsed = { raw };
  }

  return {
    channel: "mq",
    payload: parsed,
    priority: "normal",
  };
}

// ── MqAdapter ─────────────────────────────────────────────────────────────────

export class MqAdapter {
  private connection: unknown = null;
  private channel: unknown = null;
  private started = false;

  constructor(private readonly opts: MqAdapterOptions) {}

  /**
   * Start consuming messages.
   *
   * With a URL: dynamically imports `amqplib` — if not installed, throws
   * with an install hint.
   *
   * Without a URL: no-op (in-process mode — use push() to inject messages).
   */
  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;

    if (!this.opts.url) {
      // In-process mode — no AMQP connection needed
      console.info(
        `[MqAdapter] Started in in-process mode on queue "${this.opts.queue}". Use push() to inject messages.`,
      );
      return;
    }

    // Dynamically import amqplib so the package is an optional dependency
    type AmqplibConnect = (url: string) => Promise<{
      createChannel(): Promise<{
        assertQueue(q: string, opts?: object): Promise<unknown>;
        consume(q: string, cb: (msg: unknown) => void): Promise<unknown>;
        cancel(tag: string): Promise<void>;
        close(): Promise<void>;
      }>;
      close(): Promise<void>;
    }>;

    let amqplibConnect: AmqplibConnect;
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mod = (await import("amqplib" as string)) as any;
      amqplibConnect = (mod.connect ?? mod.default?.connect) as AmqplibConnect;
    } catch {
      throw new Error(
        "MqAdapter: amqplib is not installed. Run: npm install amqplib",
      );
    }

    const conn = await amqplibConnect(this.opts.url);
    this.connection = conn;

    const ch = await conn.createChannel();
    this.channel = ch;

    await ch.assertQueue(this.opts.queue, { durable: true });

    await ch.consume(this.opts.queue, (rawMsg: unknown) => {
      if (rawMsg === null) return; // queue cancelled
      void this._handleMessage(rawMsg as MqMessage);
    });

    console.info(
      `[MqAdapter] Consuming from AMQP queue "${this.opts.queue}" at ${this.opts.url}`,
    );
  }

  /** Stop consuming and close the AMQP connection. */
  async stop(): Promise<void> {
    if (!this.started) return;
    this.started = false;

    try {
      const ch = this.channel as {
        close?(): Promise<void>;
      } | null;
      await ch?.close?.();
    } catch {
      // ignore
    }

    try {
      const conn = this.connection as {
        close?(): Promise<void>;
      } | null;
      await conn?.close?.();
    } catch {
      // ignore
    }

    this.channel = null;
    this.connection = null;
    console.info(`[MqAdapter] Stopped.`);
  }

  /**
   * Push a message directly — in-process / test mode.
   * Safe to call when in AMQP mode too (skips the network path, processes inline).
   */
  async push(msg: MqMessage): Promise<void> {
    await this._handleMessage(msg);
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  private async _handleMessage(msg: MqMessage): Promise<void> {
    const transform = this.opts.transform ?? defaultTransform;
    let event: GristEventInit;
    try {
      event = transform(msg);
    } catch (err) {
      console.error("[MqAdapter] transform() threw:", err);
      return;
    }

    try {
      await this.opts.bridge.triage(event);
    } catch (err) {
      console.error("[MqAdapter] bridge.triage() threw:", err);
    }
  }
}
