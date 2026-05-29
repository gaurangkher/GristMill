/**
 * Memory routes — remember and recall via grist-ledger.
 *
 * POST /api/memory/remember  { content: string, tags?: string[] }
 * POST /api/memory/recall    { query: string, limit?: number }
 * GET  /api/memory/:id
 */

import type { FastifyInstance, FastifyPluginOptions } from "fastify";
import type { IBridge } from "../../core/bridge.js";

interface PluginOpts extends FastifyPluginOptions {
  bridge: IBridge;
}

interface RememberBody {
  content: string;
  tags?: string[];
}

interface RecallBody {
  query: string;
  limit?: number;
}

export async function memoryRoutes(
  app: FastifyInstance,
  opts: PluginOpts,
): Promise<void> {
  const { bridge } = opts;

  app.post<{ Body: RememberBody }>(
    "/remember",
    {
      schema: {
        body: {
          type: "object",
          properties: {
            content: { type: "string" },
            tags: { type: "array", items: { type: "string" } },
          },
          required: ["content"],
        },
      },
    },
    async (req, reply) => {
      const { content, tags = [] } = req.body;
      const id = await bridge.remember(content, tags);
      return reply.code(201).send({ id });
    },
  );

  app.post<{ Body: RecallBody }>(
    "/recall",
    {
      schema: {
        body: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "number" },
          },
          required: ["query"],
        },
      },
    },
    async (req, reply) => {
      const { query, limit = 10 } = req.body;
      const results = await bridge.recall(query, limit);
      return reply.send(results);
    },
  );

  app.get<{ Params: { id: string } }>(
    "/:id",
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
      const memory = await bridge.getMemory(req.params.id);
      if (!memory) {
        return reply.code(404).send({ error: "Memory not found" });
      }
      return reply.send(memory);
    },
  );
}
