import Fastify, { type FastifyInstance } from "fastify";
import { beforeEach, describe, expect, it } from "vitest";
import { watchesRoutes } from "../src/dashboard/routes/watches.js";
import { WatchEngine, createWatch } from "../src/bell-tower/watch.js";

async function buildApp(watchEngine: WatchEngine): Promise<FastifyInstance> {
  const app = Fastify();
  await app.register(watchesRoutes, { watchEngine });
  await app.ready();
  return app;
}

describe("PATCH /api/watches/:id validation", () => {
  let engine: WatchEngine;
  let app: FastifyInstance;
  let watchId: string;

  beforeEach(async () => {
    engine = new WatchEngine();
    const watch = createWatch({
      name: "original",
      topic: "topic.a",
      condition: "x > 0",
      channelIds: ["slack"],
    });
    engine.addWatch(watch);
    watchId = watch.id;
    app = await buildApp(engine);
  });

  it("accepts a valid partial update", async () => {
    const res = await app.inject({
      method: "PATCH",
      url: `/${watchId}`,
      payload: { name: "renamed" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().name).toBe("renamed");
  });

  it("rejects a non-string condition instead of corrupting the watch", async () => {
    const res = await app.inject({
      method: "PATCH",
      url: `/${watchId}`,
      payload: { condition: 12345 },
    });
    expect(res.statusCode).toBe(400);

    // The stored watch must be untouched — evaluate() must keep working.
    expect(() => engine.evaluate("topic.a", { x: 1 })).not.toThrow();
    expect(engine.getWatch(watchId)?.condition).toBe("x > 0");
  });

  it("rejects a non-array channelIds", async () => {
    const res = await app.inject({
      method: "PATCH",
      url: `/${watchId}`,
      payload: { channelIds: "slack" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("rejects a non-boolean enabled", async () => {
    const res = await app.inject({
      method: "PATCH",
      url: `/${watchId}`,
      payload: { enabled: "yes" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("returns 404 for an unknown watch id", async () => {
    const res = await app.inject({
      method: "PATCH",
      url: "/does-not-exist",
      payload: { name: "x" },
    });
    expect(res.statusCode).toBe(404);
  });
});
