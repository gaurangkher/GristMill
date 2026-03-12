/**
 * Plugin SDK types for GristMill integrations.
 *
 * A plugin implements `GristMillPlugin` and uses `PluginContext` to register
 * custom adapters, notification channels, and pipeline step types.
 */

import type { IBridge, GristEventInit, RouteDecision } from "../core/bridge.js";
import type { Notification } from "../bell-tower/dispatcher.js";

// ── Adapter handler ───────────────────────────────────────────────────────────

/**
 * A function that accepts an external event and returns a normalised
 * `GristEventInit` that will be submitted to the Rust core via the bridge.
 */
export type AdapterHandler = (
  rawEvent: unknown,
) => GristEventInit | Promise<GristEventInit>;

// ── Notification channel ──────────────────────────────────────────────────────

/** Custom notification channel contributed by a plugin. */
export interface NotificationChannel {
  send(notification: Notification): Promise<void>;
}

// ── Step executor ─────────────────────────────────────────────────────────────

/** Context provided to a pipeline step executor contributed by a plugin. */
export interface StepContext {
  bridge: IBridge;
  stepId: string;
  pipelineId: string;
  input: unknown;
}

/** Result returned by a custom step executor. */
export interface StepResult {
  output: unknown;
  metadata?: Record<string, unknown>;
}

/** Custom pipeline step type contributed by a plugin. */
export type StepExecutor = (
  ctx: StepContext,
) => StepResult | Promise<StepResult>;

// ── PluginContext ─────────────────────────────────────────────────────────────

export interface PluginContext {
  /** The GristMill bridge — gives plugins access to triage, memory, escalate, etc. */
  bridge: IBridge;

  /**
   * Register a named event adapter.
   * Hopper will call this handler when events arrive on the named channel.
   */
  registerAdapter(name: string, handler: AdapterHandler): void;

  /**
   * Register a named notification channel.
   * Bell Tower will route to this channel when its id is listed in a Watch's `channelIds`.
   */
  registerChannel(name: string, channel: NotificationChannel): void;

  /**
   * Register a custom pipeline step type.
   * Millwright will invoke this executor when running a step of that type.
   */
  registerStepType(name: string, executor: StepExecutor): void;

  /** Subscribe to a bus topic within the plugin's lifetime. */
  subscribe(topic: string): AsyncIterable<unknown>;

  /** Emit a log message attributed to this plugin. */
  log(level: "debug" | "info" | "warn" | "error", message: string): void;
}

// ── GristMillPlugin ───────────────────────────────────────────────────────────

export interface GristMillPlugin {
  /** Unique plugin identifier, e.g. "my-company/slack-enricher". */
  name: string;
  /** Semver version string, e.g. "1.0.0". */
  version: string;
  /**
   * Called once during system startup.  The plugin should use `ctx` to
   * register its adapters, channels, and step types.
   */
  register(ctx: PluginContext): void | Promise<void>;
  /**
   * Optional teardown hook called on graceful shutdown.
   * Use to close connections, flush buffers, etc.
   */
  unregister?(): void | Promise<void>;
}

// Re-export shared types that plugins commonly need
export type { GristEventInit, RouteDecision };
