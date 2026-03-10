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

// ── Plugin registry ────────────────────────────────────────────────────────────

const pluginsDir =
  process.env["GRISTMILL_PLUGINS_DIR"] ??
  path.join(os.homedir(), ".gristmill", "plugins");

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

// ── Graceful shutdown ──────────────────────────────────────────────────────────

async function shutdown(): Promise<void> {
  console.log("[main] Shutting down…");
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
});

console.log(`GristMill dashboard listening at ${address}`);
console.log(`  /api/watches  — Watch CRUD`);
console.log(`  /api/plugins  — Plugin status`);
