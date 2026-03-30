/**
 * PluginLoader — lifecycle wrapper around PluginRegistry.
 *
 * Handles load → register → teardown cleanly so callers don't need to
 * orchestrate the two-step PluginRegistry.load() + PluginRegistry.register().
 *
 * Usage:
 *   const loader = new PluginLoader({ bridge });
 *   await loader.start();            // scans pluginsDir, calls register()
 *   // ... application runs ...
 *   await loader.stop();             // calls unregister() on all plugins
 */

import { homedir } from "node:os";
import { join } from "node:path";

import { PluginRegistry } from "./registry.js";
import type { IBridge } from "../core/bridge.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface PluginLoaderOptions {
  /**
   * Directory to scan for .js/.mjs plugin files.
   * Defaults to ~/.gristmill/plugins
   */
  pluginsDir?: string;
  bridge: IBridge;
}

// ── PluginLoader ──────────────────────────────────────────────────────────────

export class PluginLoader {
  private registry: PluginRegistry;
  private readonly pluginsDir: string;

  constructor(private readonly opts: PluginLoaderOptions) {
    this.registry = new PluginRegistry();
    this.pluginsDir =
      opts.pluginsDir ?? join(homedir(), ".gristmill", "plugins");
  }

  /**
   * Load plugins from disk and call register() on each.
   * Failures on individual plugins are logged and skipped.
   */
  async start(): Promise<void> {
    await this.registry.load(this.pluginsDir);
    await this.registry.register(this.opts.bridge);
  }

  /**
   * Call unregister() on all loaded plugins and clear the registry.
   */
  async stop(): Promise<void> {
    await this.registry.unregisterAll();
  }

  /** Names of currently loaded plugins. */
  get loadedPlugins(): string[] {
    return this.registry.list();
  }

  /** Adapter handlers registered by loaded plugins. */
  get adapters() {
    return this.registry.adapters;
  }

  /** Notification channels registered by loaded plugins. */
  get channels() {
    return this.registry.channels;
  }

  /** Pipeline step types registered by loaded plugins. */
  get stepTypes() {
    return this.registry.stepTypes;
  }
}
