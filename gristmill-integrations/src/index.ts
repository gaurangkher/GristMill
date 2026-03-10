/**
 * gristmill-integrations — TypeScript shell for the GristMill AI orchestration system.
 *
 * Entry points:
 *   - GristMillBridge  — napi-rs wrapper around the Rust core
 *   - HttpHopper / WebSocketHopper — inbound event adapters
 *   - NotificationDispatcher / WatchEngine — outbound notifications
 *   - createDashboardServer / startDashboard — Fastify dashboard API
 *   - PluginRegistry — dynamic plugin loader
 */

// Core bridge
export {
  GristMillBridge,
  MockBridge,
  IpcBridge,
  createBridge,
} from "./core/bridge.js";
export type {
  RouteDecision,
  GristEventInit,
  Memory,
  RankedMemory,
  EscalationResult,
  IBridge,
} from "./core/bridge.js";

// Hopper (inbound adapters)
export { HttpHopper, WebSocketHopper } from "./hopper/index.js";
export type { HopperConfig } from "./hopper/index.js";

// Bell Tower (outbound notifications)
export {
  NotificationDispatcher,
  WatchEngine,
  createWatch,
  SlackChannel,
  EmailChannel,
} from "./bell-tower/index.js";
export type {
  Notification,
  Priority,
  BellTowerConfig,
  QuietHoursConfig,
  DigestConfig,
  ChannelRouting,
  Watch,
  EmailConfig,
} from "./bell-tower/index.js";

// Dashboard
export {
  createDashboardServer,
  startDashboard,
} from "./dashboard/server.js";
export type { DashboardConfig } from "./dashboard/server.js";

// Plugins
export { PluginRegistry } from "./plugins/registry.js";
export type {
  GristMillPlugin,
  PluginContext,
  AdapterHandler,
  NotificationChannel,
  StepExecutor,
  StepContext,
  StepResult,
} from "./plugins/types.js";
