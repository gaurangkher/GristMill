/**
 * Unit tests for SecondBrainHandler (slack-brain.ts).
 *
 * Uses MockBridge (no native Rust core required) and a minimal WebClient stub.
 * Tests cover:
 *   - captureNote() stores content and returns a memoryId
 *   - saveUrl() stores URL content and tags it as "bookmark"
 *   - queryNotes() timing instrumentation (recallMs / escalateMs logged)
 *   - queryNotes() inline path (score ≥ threshold, no LLM)
 *   - queryNotes() LLM escalation path (score < threshold)
 *   - queryNotes() empty-results path
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { SecondBrainHandler } from "../src/hopper/slack-brain.js";
import { MockBridge } from "../src/core/bridge.js";
import type { WebClient } from "@slack/web-api";

// ── Minimal WebClient stub ────────────────────────────────────────────────────

function makeWebStub(): WebClient {
  return {
    conversations: {
      replies: vi.fn().mockResolvedValue({ messages: [] }),
    },
    chat: {
      postMessage: vi.fn().mockResolvedValue({ ok: true }),
    },
  } as unknown as WebClient;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("SecondBrainHandler.captureNote()", () => {
  it("returns a non-empty memoryId and source='capture'", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    const result = await handler.captureNote("meeting notes about project alpha");

    expect(typeof result.memoryId).toBe("string");
    expect(result.memoryId.length).toBeGreaterThan(0);
    expect(result.source).toBe("capture");
  });

  it("stores content that is retrievable via recall", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    await handler.captureNote("standup agenda rocket launch");

    const results = await bridge.recall("rocket launch", 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.memory.content).toContain("rocket launch");
  });

  it("forwards slackTs in the returned result", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    const result = await handler.captureNote("tagged note", "1234567890.000001");
    expect(result.slackTs).toBe("1234567890.000001");
  });

  it("tags stored memory with 'second_brain' and 'capture'", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    const { memoryId } = await handler.captureNote("some content");
    const mem = await bridge.getMemory(memoryId);

    expect(mem).not.toBeNull();
    expect(mem!.tags).toContain("second_brain");
    expect(mem!.tags).toContain("capture");
  });
});

describe("SecondBrainHandler.saveUrl()", () => {
  it("returns source='bookmark'", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    // URL fetch will fail in unit tests (no network); saveUrl handles the error
    // gracefully and still stores a note.
    const result = await handler.saveUrl("https://example.com/article", "ts.001");

    expect(result.source).toBe("bookmark");
    expect(typeof result.memoryId).toBe("string");
    expect(result.memoryId.length).toBeGreaterThan(0);
  });

  it("tags stored memory with 'second_brain', 'bookmark', and 'url'", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    const { memoryId } = await handler.saveUrl("https://example.com/page");
    const mem = await bridge.getMemory(memoryId);

    expect(mem).not.toBeNull();
    expect(mem!.tags).toContain("second_brain");
    expect(mem!.tags).toContain("bookmark");
    expect(mem!.tags).toContain("url");
  });
});

describe("SecondBrainHandler.queryNotes()", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  it("returns empty reply when no notes match", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub());

    const result = await handler.queryNotes("xyzzy no match");

    expect(result.sourceCount).toBe(0);
    expect(result.usedLlm).toBe(false);
    expect(result.replyText).toContain("No notes found");
  });

  it("uses inline path (no LLM) when top score ≥ confidenceThreshold", async () => {
    const bridge = new MockBridge();
    // Default mock bridge returns score 0.8; default confidenceThreshold is 0.85.
    // Set a low threshold so the inline path is taken.
    const handler = new SecondBrainHandler(bridge, makeWebStub(), {
      confidenceThreshold: 0.5,
    });

    await handler.captureNote("project alpha weekly standup");
    const result = await handler.queryNotes("project alpha");

    expect(result.usedLlm).toBe(false);
    expect(result.sourceCount).toBeGreaterThan(0);
    expect(result.replyText).toContain("Found");
  });

  it("escalates to LLM when top score < confidenceThreshold", async () => {
    const bridge = new MockBridge();
    // Set threshold above mock score (0.8) to force LLM escalation.
    const handler = new SecondBrainHandler(bridge, makeWebStub(), {
      confidenceThreshold: 0.99,
    });

    await handler.captureNote("budget review q2 planning");
    const result = await handler.queryNotes("budget review");

    expect(result.usedLlm).toBe(true);
    expect(result.sourceCount).toBeGreaterThan(0);
    // Mock escalate returns "[mock] Response to: …"
    expect(result.replyText).toContain("GristMill:");
  });

  it("logs recallMs timing after recall completes", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub(), {
      confidenceThreshold: 0.5,
    });

    const logSpy = vi.spyOn(console, "log");
    await handler.captureNote("timing test note");
    await handler.queryNotes("timing test");

    const calls = logSpy.mock.calls.flat().join(" ");
    expect(calls).toMatch(/recall completed in \d+ms/);
  });

  it("logs escalateMs timing after LLM escalation", async () => {
    const bridge = new MockBridge();
    // Force escalation path.
    const handler = new SecondBrainHandler(bridge, makeWebStub(), {
      confidenceThreshold: 0.99,
    });

    const logSpy = vi.spyOn(console, "log");
    await handler.captureNote("escalation timing note");
    await handler.queryNotes("escalation timing");

    const calls = logSpy.mock.calls.flat().join(" ");
    expect(calls).toMatch(/escalation completed in \d+ms/);
  });

  it("logs cacheHit field in escalation log line", async () => {
    const bridge = new MockBridge();
    const handler = new SecondBrainHandler(bridge, makeWebStub(), {
      confidenceThreshold: 0.99,
    });

    const logSpy = vi.spyOn(console, "log");
    await handler.captureNote("cache hit log test");
    await handler.queryNotes("cache hit log");

    const calls = logSpy.mock.calls.flat().join(" ");
    expect(calls).toMatch(/cacheHit=/);
  });
});
