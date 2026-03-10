/**
 * PluginRegistry — discovers and registers GristMill plugins.
 *
 * Plugins are loaded from a configured directory via dynamic `import()`.
 * Each module must export a default object that satisfies `GristMillPlugin`.
 *
 * This is a stub implementation: dynamic loading works but channel/adapter
 * wiring to Hopper and Bell Tower requires the owning subsystems to call
 * `registry.adapters`, `registry.channels`, etc. after `load()`.
 */

import { readdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import type {
  GristMillPlugin,
  PluginContext,
  AdapterHandler,
  NotificationChannel,
  StepExecutor,
} from "./types.js";
import type { GristMillBridge } from "../core/bridge.js";

// ── Registry ──────────────────────────────────────────────────────────────────

export class PluginRegistry {
  private plugins: Map<string, GristMillPlugin> = new Map();
  private _adapters: Map<string, AdapterHandler> = new Map();
  private _channels: Map<string, NotificationChannel> = new Map();
  private _stepTypes: Map<string, StepExecutor> = new Map();

  // ── Accessors ──────────────────────────────────────────────────────────────

  get adapters(): ReadonlyMap<string, AdapterHandler> {
    return this._adapters;
  }

  get channels(): ReadonlyMap<string, NotificationChannel> {
    return this._channels;
  }

  get stepTypes(): ReadonlyMap<string, StepExecutor> {
    return this._stepTypes;
  }

  list(): string[] {
    return [...this.plugins.keys()];
  }

  // ── Loading ────────────────────────────────────────────────────────────────

  /**
   * Scan `pluginsDir` for `.js` / `.mjs` files and attempt to load each as a
   * plugin.  Failures are logged and skipped — a bad plugin does not abort
   * startup.
   */
  async load(pluginsDir: string): Promise<void> {
    let entries: string[];
    try {
      entries = await readdir(pluginsDir);
    } catch {
      // Directory doesn't exist — fine, no plugins installed
      return;
    }

    const jsFiles = entries.filter(
      (f) => f.endsWith(".js") || f.endsWith(".mjs"),
    );

    for (const file of jsFiles) {
      const filePath = resolve(join(pluginsDir, file));
      try {
        const mod = (await import(pathToFileURL(filePath).href)) as {
          default?: GristMillPlugin;
        };
        const plugin = mod.default;
        if (!_isPlugin(plugin)) {
          console.warn(
            `[PluginRegistry] ${file}: default export is not a GristMillPlugin — skipping`,
          );
          continue;
        }
        this.plugins.set(plugin.name, plugin);
        console.info(
          `[PluginRegistry] Loaded plugin "${plugin.name}" v${plugin.version}`,
        );
      } catch (err) {
        console.error(`[PluginRegistry] Failed to load ${file}:`, err);
      }
    }
  }

  // ── Registration ───────────────────────────────────────────────────────────

  /**
   * Call each plugin's `register()` method with a `PluginContext` bound to
   * `bridge`.  Collects adapters, channels, and step-types.
   */
  async register(bridge: GristMillBridge): Promise<void> {
    for (const plugin of this.plugins.values()) {
      const ctx = this._makeContext(bridge, plugin.name);
      try {
        await plugin.register(ctx);
      } catch (err) {
        console.error(
          `[PluginRegistry] Plugin "${plugin.name}" register() threw:`,
          err,
        );
      }
    }
  }

  async unregisterAll(): Promise<void> {
    for (const plugin of this.plugins.values()) {
      try {
        await plugin.unregister?.();
      } catch (err) {
        console.error(
          `[PluginRegistry] Plugin "${plugin.name}" unregister() threw:`,
          err,
        );
      }
    }
    this.plugins.clear();
    this._adapters.clear();
    this._channels.clear();
    this._stepTypes.clear();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _makeContext(bridge: GristMillBridge, pluginName: string): PluginContext {
    const registry = this;
    return {
      bridge,

      registerAdapter(name: string, handler: AdapterHandler): void {
        if (registry._adapters.has(name)) {
          console.warn(
            `[PluginRegistry] Plugin "${pluginName}" overwrites adapter "${name}"`,
          );
        }
        registry._adapters.set(name, handler);
      },

      registerChannel(name: string, channel: NotificationChannel): void {
        if (registry._channels.has(name)) {
          console.warn(
            `[PluginRegistry] Plugin "${pluginName}" overwrites channel "${name}"`,
          );
        }
        registry._channels.set(name, channel);
      },

      registerStepType(name: string, executor: StepExecutor): void {
        if (registry._stepTypes.has(name)) {
          console.warn(
            `[PluginRegistry] Plugin "${pluginName}" overwrites step type "${name}"`,
          );
        }
        registry._stepTypes.set(name, executor);
      },

      subscribe(topic: string): AsyncIterable<unknown> {
        return bridge.subscribe(topic);
      },

      log(
        level: "debug" | "info" | "warn" | "error",
        message: string,
      ): void {
        const prefix = `[Plugin:${pluginName}]`;
        console[level](`${prefix} ${message}`);
      },
    };
  }
}

// ── Type guard ────────────────────────────────────────────────────────────────

function _isPlugin(val: unknown): val is GristMillPlugin {
  return (
    typeof val === "object" &&
    val !== null &&
    typeof (val as GristMillPlugin).name === "string" &&
    typeof (val as GristMillPlugin).version === "string" &&
    typeof (val as GristMillPlugin).register === "function"
  );
}
