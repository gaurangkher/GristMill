/**
 * Plugin status routes — read-only visibility into loaded plugins.
 *
 * Endpoints:
 *   GET /api/plugins          — list loaded plugin names + their contributions
 *   GET /api/plugins/:name    — detail for a single plugin (adapters, channels, etc.)
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";
import type { PluginRegistry } from "../../plugins/registry.js";

// ── Plugin options ─────────────────────────────────────────────────────────────

export interface PluginsRoutesOptions extends FastifyPluginOptions {
  /** The shared PluginRegistry instance. */
  registry: PluginRegistry;
}

// ── Routes ────────────────────────────────────────────────────────────────────

export async function pluginsRoutes(
  app: FastifyInstance,
  opts: PluginsRoutesOptions,
): Promise<void> {
  const { registry } = opts;

  // ── GET /api/plugins ──────────────────────────────────────────────────────

  /**
   * Returns a summary of all loaded plugins and the names of their
   * registered adapters, notification channels, and step types.
   *
   * Example response:
   * {
   *   "count": 2,
   *   "plugins": ["my-company/slack-enricher", "my-company/pagerduty"],
   *   "adapters": ["pagerduty", "github-webhook"],
   *   "channels": ["pagerduty-incident"],
   *   "stepTypes": ["pagerduty-ack"]
   * }
   */
  app.get("/", async (_req, reply) => {
    const plugins = registry.list();
    return reply.send({
      count: plugins.length,
      plugins,
      adapters: [...registry.adapters.keys()],
      channels: [...registry.channels.keys()],
      stepTypes: [...registry.stepTypes.keys()],
    });
  });

  // ── GET /api/plugins/:name ────────────────────────────────────────────────

  /**
   * Returns the adapters, channels, and step types contributed by the named
   * plugin.  Because the registry stores contributions in shared maps (not
   * per-plugin), we can only tell the caller which names exist globally.
   *
   * Returns 404 if no plugin with that name is loaded.
   */
  app.get<{ Params: { "*": string } }>("/*", async (req, reply) => {
    const allPlugins = registry.list();
    const name = req.params["*"];

    if (!allPlugins.includes(name)) {
      return reply.status(404).send({ error: `Plugin "${name}" is not loaded` });
    }

    // The registry tracks all contributions in shared maps; per-plugin
    // breakdown is not available without extra bookkeeping.  Return global
    // state as a convenience until per-plugin tracking is added.
    return reply.send({
      name,
      adapters: [...registry.adapters.keys()],
      channels: [...registry.channels.keys()],
      stepTypes: [...registry.stepTypes.keys()],
    });
  });
}
