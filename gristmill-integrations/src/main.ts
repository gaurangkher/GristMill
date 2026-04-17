/**
 * Dev / daemon entrypoint — starts the GristMill dashboard server.
 *
 * Bridge selection (in priority order):
 *   1. GRISTMILL_MOCK_BRIDGE=1  → in-memory mock (no daemon required)
 *   2. default                  → IpcBridge connecting to the daemon socket
 *
 * Plugin discovery:
 *   Plugins are loaded from GRISTMILL_PLUGINS_DIR, or ~/.gristmill/plugins/.
 *   Each .js / .mjs file in that directory must export a default GristMillPlugin.
 *
 * Watch persistence:
 *   Watches are loaded from ~/.gristmill/watches.json at startup and saved
 *   back after every create / update / delete via the dashboard API.
 *   Override the path with GRISTMILL_WATCHES_FILE.
 *
 * Daemon socket path:
 *   GRISTMILL_SOCK env var, or ~/.gristmill/gristmill.sock
 *
 * Examples:
 *   GRISTMILL_MOCK_BRIDGE=1 pnpm dev          # mock, no daemon
 *   pnpm dev                                   # IPC → daemon must be running
 *   GRISTMILL_SOCK=/tmp/gm.sock pnpm dev       # custom socket path
 */

import * as path from "node:path";
import * as os from "node:os";
import { MockBridge, IpcBridge } from "./core/bridge.js";
import type { IBridge } from "./core/bridge.js";
import { PluginRegistry } from "./plugins/registry.js";
import { WatchEngine } from "./bell-tower/watch.js";
import { startDashboard } from "./dashboard/server.js";
import { HttpHopper, SlackHopper, WebhookHopper, CronHopper, FsHopper } from "./hopper/index.js";
import { loadConfig } from "./core/config.js";

// ── Bridge ─────────────────────────────────────────────────────────────────────

function resolveBridge(): IBridge {
  if (process.env["GRISTMILL_MOCK_BRIDGE"] === "1") {
    console.log("[bridge] using MockBridge (GRISTMILL_MOCK_BRIDGE=1)");
    return new MockBridge();
  }

  const sockPath =
    process.env["GRISTMILL_SOCK"] ??
    path.join(os.homedir(), ".gristmill", "gristmill.sock");

  console.log(`[bridge] using IpcBridge → ${sockPath}`);
  return new IpcBridge(sockPath);
}

const bridge = resolveBridge();

// ── Config ─────────────────────────────────────────────────────────────────────
// Reads ~/.gristmill/config.yaml with env var overrides.

const config = loadConfig();

// ── Plugin registry ────────────────────────────────────────────────────────────

const pluginsDir = config.pluginsDir;

const registry = new PluginRegistry();
await registry.load(pluginsDir);
await registry.register(bridge);

if (registry.list().length > 0) {
  console.log(
    `[plugins] Loaded ${registry.list().length} plugin(s): ${registry.list().join(", ")}`,
  );
  if (registry.adapters.size > 0) {
    console.log(`[plugins] Adapters registered: ${[...registry.adapters.keys()].join(", ")}`);
  }
  if (registry.channels.size > 0) {
    console.log(`[plugins] Channels registered: ${[...registry.channels.keys()].join(", ")}`);
  }
  if (registry.stepTypes.size > 0) {
    console.log(`[plugins] Step types registered: ${[...registry.stepTypes.keys()].join(", ")}`);
  }
} else {
  console.log(`[plugins] No plugins found in ${pluginsDir}`);
}

// ── Watch engine ───────────────────────────────────────────────────────────────

const watchPersistPath =
  process.env["GRISTMILL_WATCHES_FILE"] ??
  path.join(os.homedir(), ".gristmill", "watches.json");

const watchEngine = new WatchEngine();
await watchEngine.loadFromFile(watchPersistPath);

// ── HttpHopper — inbound HTTP event intake on a dedicated port ─────────────────

const httpHopper = new HttpHopper(bridge, {
  port: config.hoppers.http.port,
  host: config.hoppers.http.host,
  pluginAdapters: registry.adapters,
});
await httpHopper.start();

// ── WebhookHopper — HMAC-verified inbound webhooks ─────────────────────────────

const webhookHopper = new WebhookHopper(bridge, {
  port: config.hoppers.webhook.port,
  host: config.hoppers.webhook.host,
  channels: config.hoppers.webhook.channels,
});
if (Object.keys(config.hoppers.webhook.channels).length > 0) {
  await webhookHopper.start();
  console.log(
    `[WebhookHopper] Channels: ${Object.keys(config.hoppers.webhook.channels).join(", ")}`,
  );
} else {
  console.log("[WebhookHopper] No channels configured — skipping start");
}

// ── CronHopper — scheduled event emission ─────────────────────────────────────

const cronHopper = new CronHopper(bridge, {
  jobs: config.hoppers.cron.map((j) => ({
    id: j.id,
    intervalMs: j.intervalMs,
    fireImmediately: j.fireImmediately,
    event: { channel: j.channel, payload: j.payload },
  })),
});
cronHopper.start();

// ── FsHopper — filesystem change events ───────────────────────────────────────

const fsHopper = new FsHopper(bridge, {
  watches: config.hoppers.fs.map((w) => ({
    path: w.path,
    channel: w.channel,
    recursive: w.recursive,
    debounceMs: w.debounceMs,
  })),
});
if (config.hoppers.fs.length > 0) {
  fsHopper.start();
} else {
  console.log("[FsHopper] No watch paths configured — skipping start");
}

// ── Slack Socket Mode hopper (optional) ────────────────────────────────────────
//
// Activated when app_token + bot_token are set (config.yaml or env vars).
// No public URL required — uses a persistent WebSocket to Slack's servers.

let slackHopper: SlackHopper | null = null;

if (config.slack.appToken && config.slack.botToken) {
  slackHopper = new SlackHopper(bridge, {
    appToken: config.slack.appToken,
    botToken: config.slack.botToken,
    replyMode: config.slack.replyMode,
    secondBrain: config.slack.secondBrain ?? undefined,
  });
  await slackHopper.start();
} else {
  console.log(
    "[SlackHopper] integrations.slack.app_token / bot_token not set — Slack Socket Mode disabled",
  );
}

// ── Graceful shutdown ──────────────────────────────────────────────────────────

async function shutdown(): Promise<void> {
  console.log("[main] Shutting down…");
  await httpHopper.stop().catch(() => {});
  await webhookHopper.stop().catch(() => {});
  cronHopper.stop();
  fsHopper.stop();
  await slackHopper?.stop().catch(() => {});
  await watchEngine.saveToFile(watchPersistPath).catch(() => {});
  await registry.unregisterAll().catch(() => {});
  process.exit(0);
}

process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());

// ── Dashboard ──────────────────────────────────────────────────────────────────

const port = Number(process.env["PORT"] ?? 3000);
const host = process.env["HOST"] ?? "127.0.0.1";

const address = await startDashboard(bridge, {
  port,
  host,
  watchEngine,
  watchPersistPath,
  pluginRegistry: registry,
  slackSigningSecret: config.slack.signingSecret || undefined,
});

console.log(`GristMill dashboard listening at ${address}`);
console.log(`  /api/watches       — Watch CRUD`);
console.log(`  /api/plugins       — Plugin status`);
console.log(`  /webhooks/slack    — Slack Events API inbound (HTTP, optional)`);
console.log(`  /api/ecosystem     — Phase 4 portability & sharing`);
console.log(`  [HttpHopper]       — http://${config.hoppers.http.host}:${config.hoppers.http.port}/events`);
console.log(
  slackHopper
    ? `  [SlackHopper]      — Socket Mode active`
    : `  [SlackHopper]      — disabled (set SLACK_APP_TOKEN + SLACK_BOT_TOKEN to enable)`,
);
