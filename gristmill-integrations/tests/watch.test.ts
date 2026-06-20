import { describe, expect, it } from "vitest";
import { WatchEngine, createWatch } from "../src/bell-tower/watch.js";

describe("WatchEngine.evaluate", () => {
  it("fires a watch whose simple condition matches the payload", () => {
    const engine = new WatchEngine();
    const watch = createWatch({
      name: "low confidence",
      topic: "sieve.anomaly",
      condition: "confidence < 0.5",
      channelIds: ["slack"],
    });
    engine.addWatch(watch);

    const fired = engine.evaluate("sieve.anomaly", { confidence: 0.2 });
    expect(fired.map((w) => w.id)).toEqual([watch.id]);
  });

  it("does not fire when the condition does not match", () => {
    const engine = new WatchEngine();
    const watch = createWatch({
      name: "low confidence",
      topic: "sieve.anomaly",
      condition: "confidence < 0.5",
      channelIds: ["slack"],
    });
    engine.addWatch(watch);

    const fired = engine.evaluate("sieve.anomaly", { confidence: 0.9 });
    expect(fired).toEqual([]);
  });

  it("respects per-watch cooldown", () => {
    const engine = new WatchEngine();
    const watch = createWatch({
      name: "cooldown test",
      topic: "topic.a",
      condition: "x > 0",
      channelIds: ["slack"],
      cooldownMs: 60_000,
    });
    engine.addWatch(watch);

    const first = engine.evaluate("topic.a", { x: 1 });
    const second = engine.evaluate("topic.a", { x: 1 });
    expect(first).toHaveLength(1);
    expect(second).toHaveLength(0);
  });

  it("a malformed watch (non-string condition) does not throw and does not block other watches", () => {
    const engine = new WatchEngine();

    // Simulates a Watch object whose `condition` bypassed validation (e.g.
    // smuggled in via an unvalidated PATCH, or loaded from a corrupted
    // persisted file) — condition.split() would throw a TypeError if not
    // guarded, which used to propagate out of evaluate() and kill the
    // caller's entire bus-subscription loop.
    const malformed = createWatch({
      name: "malformed",
      topic: "topic.a",
      condition: "x > 0",
      channelIds: ["slack"],
    });
    // @ts-expect-error intentionally violating the Watch type for the test
    malformed.condition = 12345;
    engine.addWatch(malformed);

    const healthy = createWatch({
      name: "healthy",
      topic: "topic.a",
      condition: "x > 0",
      channelIds: ["slack"],
    });
    engine.addWatch(healthy);

    let fired: ReturnType<typeof engine.evaluate> = [];
    expect(() => {
      fired = engine.evaluate("topic.a", { x: 1 });
    }).not.toThrow();

    expect(fired.map((w) => w.id)).toEqual([healthy.id]);
  });

  it("an invalid regex in a 'matches' condition is treated as non-matching, not thrown", () => {
    const engine = new WatchEngine();
    const watch = createWatch({
      name: "bad regex",
      topic: "topic.a",
      condition: "pipeline_id matches (",
      channelIds: ["slack"],
    });
    engine.addWatch(watch);

    expect(() => engine.evaluate("topic.a", { pipeline_id: "prod-1" })).not.toThrow();
    expect(engine.evaluate("topic.a", { pipeline_id: "prod-1" })).toEqual([]);
  });

  it("disabled watches never fire", () => {
    const engine = new WatchEngine();
    const watch = createWatch({
      name: "disabled",
      topic: "topic.a",
      condition: "x > 0",
      channelIds: ["slack"],
      enabled: false,
    });
    engine.addWatch(watch);

    expect(engine.evaluate("topic.a", { x: 1 })).toEqual([]);
  });
});

describe("WatchEngine.updateWatch", () => {
  it("merges a partial patch and preserves the id", () => {
    const engine = new WatchEngine();
    const watch = createWatch({
      name: "original",
      topic: "topic.a",
      condition: "x > 0",
      channelIds: ["slack"],
    });
    engine.addWatch(watch);

    const updated = engine.updateWatch(watch.id, { name: "renamed" });
    expect(updated?.id).toBe(watch.id);
    expect(updated?.name).toBe("renamed");
    expect(updated?.condition).toBe("x > 0");
  });

  it("returns null for an unknown id", () => {
    const engine = new WatchEngine();
    expect(engine.updateWatch("does-not-exist", { name: "x" })).toBeNull();
  });
});
