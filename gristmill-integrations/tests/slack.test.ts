/**
 * Unit tests for SlackHopper (slack.ts).
 *
 * Tests the !save / !ask command prefix handling, the deduplication logic
 * that prevents @mention channel messages from being processed twice, and
 * URL-vs-plain-text detection for the !save command.
 *
 * Uses MockBridge for memory round-trip tests (no native Rust core required).
 */

import { describe, it, expect } from "vitest";
import { MockBridge } from "../src/core/bridge.js";

// ── !save / !ask prefix detection (pure logic tests) ─────────────────────────

/**
 * Strip a leading @-mention from a Slack message text, mirroring the logic
 * in SlackHopper._handleEvent().
 */
function stripMention(text: string): string {
  return text.trim().replace(/^<@[A-Z0-9]+>\s*/, "");
}

describe("!save / !ask command prefix detection", () => {
  it("recognises !save <text> after mention strip", () => {
    const raw = "<@UBOT123> !save my important note";
    const text = stripMention(raw);
    expect(text.startsWith("!save ")).toBe(true);
    expect(text.slice(6).trim()).toBe("my important note");
  });

  it("recognises !save <url> after mention strip", () => {
    const raw = "!save https://example.com/article";
    const text = stripMention(raw);
    expect(text.startsWith("!save ")).toBe(true);
    expect(text.slice(6).trim()).toBe("https://example.com/article");
  });

  it("recognises !ask after mention strip", () => {
    const raw = "<@UBOT123> !ask what did I save about project alpha?";
    const text = stripMention(raw);
    expect(text.startsWith("!ask ")).toBe(true);
    expect(text.slice(5).trim()).toBe("what did I save about project alpha?");
  });

  it("does NOT match /save (old slash-command prefix)", () => {
    const text = "/save some note";
    expect(text.startsWith("!save ")).toBe(false);
  });

  it("does NOT match /ask (old slash-command prefix)", () => {
    const text = "/ask my query";
    expect(text.startsWith("!ask ")).toBe(false);
  });

  it("plain DM without !save or !ask does not trigger second brain", () => {
    const text = "hello how are you doing today";
    expect(text.startsWith("!save ")).toBe(false);
    expect(text.startsWith("!ask ")).toBe(false);
  });
});

// ── Channel message dedup logic ───────────────────────────────────────────────

describe("channel message deduplication logic", () => {
  function isMention(text: string): boolean {
    return text.trim().startsWith("<@");
  }

  it("identifies @mention channel messages correctly", () => {
    expect(isMention("<@UBOT123> !save some note")).toBe(true);
    expect(isMention("<@UBOT123> hello bot")).toBe(true);
  });

  it("does not treat !save without mention as a duplicate", () => {
    expect(isMention("!save a standalone note")).toBe(false);
  });

  it("does not treat !ask without mention as a duplicate", () => {
    expect(isMention("!ask a standalone query")).toBe(false);
  });

  it("does not treat plain channel messages as mentions", () => {
    expect(isMention("just a regular channel message")).toBe(false);
    expect(isMention("  hello world")).toBe(false);
  });
});

// ── URL detection helper ──────────────────────────────────────────────────────

describe("URL vs plain-text detection for !save", () => {
  function looksLikeUrl(s: string): boolean {
    try {
      const u = new URL(s);
      return u.protocol === "http:" || u.protocol === "https:";
    } catch {
      return false;
    }
  }

  it("returns true for https URLs", () => {
    expect(looksLikeUrl("https://example.com/article")).toBe(true);
    expect(looksLikeUrl("https://blog.mozilla.org/path?q=1")).toBe(true);
  });

  it("returns true for http URLs", () => {
    expect(looksLikeUrl("http://internal.corp/dashboard")).toBe(true);
  });

  it("returns false for plain text", () => {
    expect(looksLikeUrl("project alpha meeting notes")).toBe(false);
    expect(looksLikeUrl("not a url at all")).toBe(false);
  });

  it("returns false for partial URLs (missing protocol)", () => {
    expect(looksLikeUrl("example.com/page")).toBe(false);
    expect(looksLikeUrl("www.google.com")).toBe(false);
  });

  it("returns false for ftp or other non-http protocols", () => {
    expect(looksLikeUrl("ftp://files.example.com/data")).toBe(false);
  });
});

// ── MockBridge memory round-trip ──────────────────────────────────────────────

describe("MockBridge memory round-trip (used by SlackHopper in tests)", () => {
  it("remember + recall works for second-brain notes", async () => {
    const bridge = new MockBridge();
    await bridge.remember("project phoenix kickoff agenda", [
      "second_brain",
      "capture",
    ]);

    const results = await bridge.recall("phoenix kickoff", 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.memory.content).toContain("phoenix kickoff");
  });

  it("remember returns a unique id on each call", async () => {
    const bridge = new MockBridge();
    const id1 = await bridge.remember("note one", []);
    const id2 = await bridge.remember("note two", []);
    expect(id1).not.toBe(id2);
  });
});
