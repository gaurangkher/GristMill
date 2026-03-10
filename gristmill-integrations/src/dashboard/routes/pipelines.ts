/**
 * Pipeline routes — register, list, and run pipelines.
 *
 * POST /api/pipelines              → bridge.registerPipeline(body)  (body must have "id")
 * GET  /api/pipelines              → bridge.pipelineIds()
 * POST /api/pipelines/:id/run      → bridge.runPipeline(id, body)
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";
import type { IBridge } from "../../core/bridge.js";

interface PluginOpts extends FastifyPluginOptions {
  bridge: IBridge;
}

export async function pipelinesRoutes(
  app: FastifyInstance,
  opts: PluginOpts,
): Promise<void> {
  const { bridge } = opts;

  app.post<{ Body: Record<string, unknown> }>(
    "/",
    {
      schema: {
        body: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
      },
    },
    async (req, reply) => {
      bridge.registerPipeline(req.body);
      return reply.code(201).send({ registered: true, id: req.body["id"] });
    },
  );

  app.get("/", async (_req, reply) => {
    const ids = bridge.pipelineIds();
    return reply.send({ pipelines: ids });
  });

  app.post<{ Params: { id: string }; Body: unknown }>(
    "/:id/run",
    {
      schema: {
        params: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
      },
    },
    async (req, reply) => {
      const { id } = req.params;
      const body = req.body ?? {};

      // Treat body as the event payload; wrap into a GristEventInit
      const eventInit = {
        channel: "http" as const,
        payload: body,
      };

      const result = await bridge.runPipeline(id, eventInit);
      return reply.send({ pipelineId: id, result });
    },
  );
}
