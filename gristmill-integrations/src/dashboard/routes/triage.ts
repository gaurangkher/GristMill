/**
 * Triage route — expose the Sieve classifier over HTTP.
 *
 * POST /api/triage  { text: string, channel?: string }
 *   → RouteDecision { route, confidence, modelId?, reason? }
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";
import type { IBridge } from "../../core/bridge.js";

interface PluginOpts extends FastifyPluginOptions {
  bridge: IBridge;
}

interface TriageBody {
  text: string;
  channel?: string;
}

export async function triageRoutes(
  app: FastifyInstance,
  opts: PluginOpts,
): Promise<void> {
  const { bridge } = opts;

  app.post<{ Body: TriageBody }>(
    "/",
    {
      schema: {
        body: {
          type: "object",
          properties: {
            text:    { type: "string" },
            channel: { type: "string" },
          },
          required: ["text"],
        },
      },
    },
    async (req, reply) => {
      const { text, channel = "cli" } = req.body;
      const decision = await bridge.triage({ channel, payload: { text } });
      return reply.send(decision);
    },
  );
}
