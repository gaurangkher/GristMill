/**
 * Plugin SDK — convenience helpers for GristMill plugin authors.
 *
 * Import from here instead of "types.ts" to get the factory and builder
 * in addition to all raw types.
 *
 * @example
 * import { createPlugin } from "@gristmill/integrations/plugins/sdk";
 *
 * export default createPlugin({
 *   name: "my-company/my-plugin",
 *   version: "1.0.0",
 *   register(ctx) {
 *     ctx.registerAdapter("my-channel", async (raw) => ({ ... }));
 *   },
 * });
 */

import type { GristMillPlugin, PluginContext } from "./types.js";
export * from "./types.js";

// ── createPlugin factory ──────────────────────────────────────────────────────

/**
 * Factory helper — avoids boilerplate when writing simple plugins.
 *
 * @example
 * export default createPlugin({
 *   name: "my-company/my-plugin",
 *   version: "1.0.0",
 *   register(ctx) {
 *     ctx.registerAdapter("my-channel", async (raw) => ({ ... }));
 *   },
 * });
 */
export function createPlugin(def: GristMillPlugin): GristMillPlugin {
  return def;
}

// ── PluginBuilder ─────────────────────────────────────────────────────────────

/**
 * Fluent builder for creating plugins.
 *
 * @example
 * export default new PluginBuilder()
 *   .name("my-company/my-plugin")
 *   .version("1.0.0")
 *   .onRegister((ctx) => {
 *     ctx.registerAdapter("my-channel", async (raw) => ({ ... }));
 *   })
 *   .build();
 */
export class PluginBuilder {
  private def: Partial<
    GristMillPlugin & {
      _register: (ctx: PluginContext) => void | Promise<void>;
    }
  > = {};

  name(n: string): this {
    this.def.name = n;
    return this;
  }

  version(v: string): this {
    this.def.version = v;
    return this;
  }

  onRegister(fn: (ctx: PluginContext) => void | Promise<void>): this {
    this.def._register = fn;
    return this;
  }

  onUnregister(fn: () => void | Promise<void>): this {
    this.def.unregister = fn;
    return this;
  }

  build(): GristMillPlugin {
    if (!this.def.name)
      throw new Error("PluginBuilder: name() is required");
    if (!this.def.version)
      throw new Error("PluginBuilder: version() is required");
    if (!this.def._register)
      throw new Error("PluginBuilder: onRegister() is required");

    return {
      name: this.def.name,
      version: this.def.version,
      register: this.def._register,
      unregister: this.def.unregister,
    };
  }
}
