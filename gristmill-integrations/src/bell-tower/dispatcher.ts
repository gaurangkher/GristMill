/**
 * NotificationDispatcher — subscribes to grist-bus topics and routes
 * notifications to configured channels (Slack, email).
 *
 * Features:
 * - Subscribes to all 5 standard bus topics
 * - Quiet-hours enforcement (LOW/NORMAL suppressed; CRITICAL always sends)
 * - Digest batching for LOW priority (flushes on interval or threshold)
 * - Per-topic notification formatters
 */

import { monotonicFactory } from "ulid";
import type { GristMillBridge } from "../core/bridge.js";
import type { SlackChannel } from "./channels/slack.js";
import type { EmailChannel } from "./channels/email.js";
import { WatchEngine } from "./watch.js";

// ── Plugin channel interface ───────────────────────────────────────────────────
//
// Typed as a minimal structural interface to avoid a circular import between
// bell-tower and plugins.  Any object with a `send(Notification): Promise<void>`
// method satisfies this contract (including `NotificationChannel` from plugins/types).

export interface PluginNotificationChannel {
  send(notification: Notification): Promise<void>;
}

const ulid = monotonicFactory();

// ── Types ─────────────────────────────────────────────────────────────────────

export type Priority = "low" | "normal" | "high" | "critical";

export interface Notification {
  id: string;
  topic: string;
  title: string;
  body: string;
  priority: Priority;
  createdAt: Date;
}

export interface QuietHoursConfig {
  /** 0–23, start of quiet period (local time) */
  startHour: number;
  /** 0–23, end of quiet period (local time) */
  endHour: number;
  /** Priorities suppressed during quiet hours (default: low, normal) */
  suppressPriorities?: Priority[];
}

export interface DigestConfig {
  /** Whether to batch LOW-priority notifications into periodic digests. */
  enabled: boolean;
  /** How often to flush accumulated digest (ms). Default: 3_600_000 (1 hour). */
  intervalMs: number;
  /** Flush immediately when this many messages accumulate. Default: 20. */
  threshold: number;
}

export interface ChannelRouting {
  /**
   * Slack channel IDs (webhook URL keys) to use per priority.
   * Use "*" to match any priority.
   */
  slack?: {
    priorities: Priority[] | "*";
    webhookUrl: string;
  }[];
  /**
   * Email recipient lists per priority.
   */
  email?: {
    priorities: Priority[] | "*";
    to: string[];
  }[];
}

export interface BellTowerConfig {
  quietHours?: QuietHoursConfig;
  digest?: DigestConfig;
  routing: ChannelRouting;
}

// ── Bus topic payload shapes (mirrors Rust BusEvent JSON) ─────────────────────

interface PipelineCompletedPayload {
  pipeline_id: string;
  run_id: string;
  duration_ms: number;
  steps_completed: number;
}

interface PipelineFailedPayload {
  pipeline_id: string;
  run_id: string;
  reason: string;
  failed_step?: string;
}

interface SieveAnomalyPayload {
  event_id: string;
  confidence: number;
  route: string;
  anomaly_type?: string;
}

interface LedgerThresholdPayload {
  tier: string;
  used_bytes: number;
  limit_bytes: number;
  pct: number;
}

interface HammerBudgetPayload {
  daily_used: number;
  daily_limit: number;
  window_start_ms: number;
}

// ── NotificationDispatcher ────────────────────────────────────────────────────

const BUS_TOPICS = [
  "pipeline.completed",
  "pipeline.failed",
  "sieve.anomaly",
  "ledger.threshold",
  "hammer.budget",
] as const;

export class NotificationDispatcher {
  private digestBuffer: Notification[] = [];
  private digestTimer?: ReturnType<typeof setInterval>;
  private running = false;
  private abortControllers: AbortController[] = [];

  /**
   * @param bridge        - The GristMill bridge for bus subscriptions.
   * @param config        - Bell Tower routing / quiet-hours / digest config.
   * @param slack         - Pre-built Slack channel, or `null` to disable.
   * @param email         - Pre-built Email channel, or `null` to disable.
   * @param watchEngine   - Shared WatchEngine instance (default: new empty one).
   * @param pluginChannels - Named channels registered by plugins.  Watches can
   *                         reference any of these ids in their `channelIds` array
   *                         alongside the built-in "slack" / "email" ids.
   */
  constructor(
    private readonly bridge: GristMillBridge,
    private readonly config: BellTowerConfig,
    private readonly slack: SlackChannel | null,
    private readonly email: EmailChannel | null,
    private readonly watchEngine: WatchEngine = new WatchEngine(),
    private pluginChannels: ReadonlyMap<string, PluginNotificationChannel> = new Map(),
  ) {}

  /**
   * Start subscribing to all bus topics.
   * Each topic subscription runs as a long-lived async loop.
   */
  start(): void {
    if (this.running) return;
    this.running = true;

    for (const topic of BUS_TOPICS) {
      const ac = new AbortController();
      this.abortControllers.push(ac);
      this._subscribeToTopic(topic, ac.signal).catch((err: unknown) => {
        if (!ac.signal.aborted) {
          console.error(`[BellTower] Subscription error on ${topic}:`, err);
        }
      });
    }

    if (this.config.digest?.enabled) {
      const intervalMs =
        this.config.digest.intervalMs ?? 3_600_000;
      this.digestTimer = setInterval(() => {
        this.flushDigest().catch(console.error);
      }, intervalMs);
    }
  }

  stop(): void {
    this.running = false;
    for (const ac of this.abortControllers) ac.abort();
    this.abortControllers = [];
    if (this.digestTimer) {
      clearInterval(this.digestTimer);
      this.digestTimer = undefined;
    }
  }

  addWatch(watch: Parameters<WatchEngine["addWatch"]>[0]): void {
    this.watchEngine.addWatch(watch);
  }

  removeWatch(id: string): void {
    this.watchEngine.removeWatch(id);
  }

  /**
   * Replace the set of plugin-registered notification channels.
   * Call this after `PluginRegistry.register()` completes so that watches
   * can route to plugin-provided channels by name.
   */
  setPluginChannels(
    channels: ReadonlyMap<string, PluginNotificationChannel>,
  ): void {
    this.pluginChannels = channels;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private async _subscribeToTopic(
    topic: string,
    signal: AbortSignal,
  ): Promise<void> {
    for await (const rawEvent of this.bridge.subscribe(topic)) {
      if (signal.aborted) break;

      const notification = this._buildNotification(topic, rawEvent);

      // Fire any matching watches
      const firedWatches = this.watchEngine.evaluate(topic, rawEvent);
      for (const watch of firedWatches) {
        await this._dispatchToWatchChannels(notification, watch.channelIds);
      }

      if (!this._shouldSend(notification)) continue;

      if (
        notification.priority === "low" &&
        this.config.digest?.enabled
      ) {
        this._bufferDigest(notification);
      } else {
        await this._dispatch(notification);
      }
    }
  }

  private _buildNotification(topic: string, payload: unknown): Notification {
    const base = {
      id: ulid(),
      topic,
      createdAt: new Date(),
    };

    switch (topic) {
      case "pipeline.completed": {
        const p = payload as PipelineCompletedPayload;
        return {
          ...base,
          title: `Pipeline ${p.pipeline_id ?? "unknown"} completed`,
          body: `Run ${p.run_id ?? "-"} completed in ${p.duration_ms ?? "?"}ms (${p.steps_completed ?? 0} steps).`,
          priority: "normal",
        };
      }
      case "pipeline.failed": {
        const p = payload as PipelineFailedPayload;
        return {
          ...base,
          title: `Pipeline ${p.pipeline_id ?? "unknown"} failed`,
          body: `Run ${p.run_id ?? "-"} failed${p.failed_step ? ` at step "${p.failed_step}"` : ""}: ${p.reason ?? "unknown error"}.`,
          priority: "high",
        };
      }
      case "sieve.anomaly": {
        const p = payload as SieveAnomalyPayload;
        const conf = typeof p.confidence === "number"
          ? p.confidence.toFixed(3)
          : "?";
        return {
          ...base,
          title: "Sieve anomaly detected",
          body: `Event ${p.event_id ?? "unknown"} routed to ${p.route ?? "?"} with confidence ${conf}${p.anomaly_type ? ` (${p.anomaly_type})` : ""}.`,
          priority: "high",
        };
      }
      case "ledger.threshold": {
        const p = payload as LedgerThresholdPayload;
        const pct = typeof p.pct === "number" ? p.pct.toFixed(1) : "?";
        const priority: Priority = (p.pct ?? 0) >= 90 ? "critical" : "high";
        return {
          ...base,
          title: `Memory tier "${p.tier ?? "?"}" at ${pct}% capacity`,
          body: `Used ${_fmtBytes(p.used_bytes)} of ${_fmtBytes(p.limit_bytes)}.`,
          priority,
        };
      }
      case "hammer.budget": {
        const p = payload as HammerBudgetPayload;
        const used = p.daily_used ?? 0;
        const limit = p.daily_limit ?? 1;
        const pct = ((used / limit) * 100).toFixed(1);
        const priority: Priority = used / limit >= 0.9 ? "critical" : "normal";
        return {
          ...base,
          title: `LLM budget: ${pct}% used today`,
          body: `${used.toLocaleString()} of ${limit.toLocaleString()} tokens used today.`,
          priority,
        };
      }
      default: {
        return {
          ...base,
          title: `Bus event: ${topic}`,
          body: JSON.stringify(payload),
          priority: "low",
        };
      }
    }
  }

  private _shouldSend(n: Notification): boolean {
    const qh = this.config.quietHours;
    if (!qh) return true;

    const hour = new Date().getHours();
    const inQuiet =
      qh.startHour <= qh.endHour
        ? hour >= qh.startHour && hour < qh.endHour
        : hour >= qh.startHour || hour < qh.endHour; // spans midnight

    if (!inQuiet) return true;

    // During quiet hours, CRITICAL always sends; others check suppress list
    if (n.priority === "critical") return true;
    const suppress = qh.suppressPriorities ?? ["low", "normal"];
    return !suppress.includes(n.priority);
  }

  private _bufferDigest(n: Notification): void {
    this.digestBuffer.push(n);
    const threshold = this.config.digest?.threshold ?? 20;
    if (this.digestBuffer.length >= threshold) {
      this.flushDigest().catch(console.error);
    }
  }

  async flushDigest(): Promise<void> {
    if (this.digestBuffer.length === 0) return;

    const batch = this.digestBuffer.splice(0, this.digestBuffer.length);
    const title = `GristMill digest: ${batch.length} low-priority notifications`;
    const body = batch
      .map((n) => `• [${n.topic}] ${n.title}: ${n.body}`)
      .join("\n");

    const digest: Notification = {
      id: ulid(),
      topic: "digest",
      title,
      body,
      priority: "low",
      createdAt: new Date(),
    };

    await this._dispatch(digest);
  }

  private async _dispatch(n: Notification): Promise<void> {
    const routing = this.config.routing;
    const sends: Promise<void>[] = [];

    if (this.slack && routing.slack) {
      for (const rule of routing.slack) {
        if (
          rule.priorities === "*" ||
          rule.priorities.includes(n.priority)
        ) {
          const ch = new (this.slack.constructor as new (
            url: string,
          ) => SlackChannel)(rule.webhookUrl);
          sends.push(ch.send(n));
        }
      }
    }

    if (this.email && routing.email) {
      for (const rule of routing.email) {
        if (
          rule.priorities === "*" ||
          rule.priorities.includes(n.priority)
        ) {
          sends.push(this.email.send(n, rule.to));
        }
      }
    }

    await Promise.allSettled(sends);
  }

  private async _dispatchToWatchChannels(
    n: Notification,
    channelIds: string[],
  ): Promise<void> {
    const sends: Promise<void>[] = [];

    for (const id of channelIds) {
      if (id === "slack") {
        if (this.slack) sends.push(this.slack.send(n));
      } else if (id === "email") {
        if (this.email) {
          // Fall back to all configured email recipients
          const allTo = (this.config.routing.email ?? []).flatMap((r) => r.to);
          if (allTo.length > 0) sends.push(this.email.send(n, allTo));
        }
      } else {
        // Look up plugin-registered channels by id
        const pluginChannel = this.pluginChannels.get(id);
        if (pluginChannel) {
          sends.push(pluginChannel.send(n));
        } else {
          console.warn(
            `[BellTower] Watch references unknown channel id "${id}" — ` +
              `ensure a plugin registers it before watches are evaluated.`,
          );
        }
      }
    }

    await Promise.allSettled(sends);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _fmtBytes(bytes: number | undefined): string {
  if (bytes === undefined) return "?";
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
}
