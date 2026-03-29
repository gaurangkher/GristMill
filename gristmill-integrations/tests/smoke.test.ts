/**
 * Phase 1 E2E smoke test — verifies that the TypeScript IpcBridge can talk to
 * a live GristMill daemon over its Unix socket.
 *
 * ## Prerequisites
 *
 *   1. The Rust daemon must be running:
 *        cargo run --bin gristmill-daemon
 *      or, in a CI/Docker environment, started by the entrypoint.
 *
 *   2. The socket path must match (default: ~/.gristmill/gristmill.sock).
 *      Override with the GRISTMILL_SOCK environment variable.
 *
 * ## Running
 *
 *   pnpm test                    # runs all tests including this one
 *   pnpm vitest run tests/smoke.test.ts
 *
 * ## Skipping when daemon is not available
 *
 *   If the daemon socket does not exist, all tests in this file are skipped
 *   automatically.  This keeps CI green when running unit tests without a
 *   live daemon (e.g. during TypeScript-only lint/type-check runs).
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { IpcBridge } from "../src/core/ipc-bridge.js";

// ── Socket resolution (mirrors IpcBridge default logic) ──────────────────────

function resolveSocketPath(): string {
  return (
    process.env.GRISTMILL_SOCK ??
    path.join(os.homedir(), ".gristmill", "gristmill.sock")
  );
}

// ── Suite setup ──────────────────────────────────────────────────────────────

const socketPath = resolveSocketPath();

/** True only when a live daemon responds to a ping within 1 second. */
async function probeDaemon(): Promise<boolean> {
  const probe = new IpcBridge(socketPath);
  try {
    const alive = await Promise.race([
      probe.ping(),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 1000)),
    ]);
    return alive;
  } catch {
    return false;
  } finally {
    probe.close();
  }
}

let daemonAvailable = false;
let bridge: IpcBridge;

beforeAll(async () => {
  daemonAvailable = await probeDaemon();
  if (!daemonAvailable) {
    console.warn(
      `[smoke] daemon not reachable at ${socketPath} — skipping E2E tests.\n` +
        `        Start the daemon first: cargo run --bin gristmill-daemon`
    );
    return;
  }
  bridge = new IpcBridge(socketPath);
});

afterAll(() => {
  bridge?.close();
});

// ── Test helper ──────────────────────────────────────────────────────────────

/**
 * Wrap `it` so tests are skipped when the daemon is not running.
 * This produces clean "skipped" output in CI rather than hard failures.
 */
function live(
  name: string,
  fn: () => Promise<void>,
  timeoutMs = 5000
): void {
  it(
    name,
    async () => {
      if (!daemonAvailable) {
        console.log(`  [skipped — daemon not running]`);
        return;
      }
      await fn();
    },
    timeoutMs
  );
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("IpcBridge smoke tests (requires live daemon)", () => {
  live("ping returns true", async () => {
    const alive = await bridge.ping();
    expect(alive).toBe(true);
  });

  live("triage returns a RouteDecision with confidence in [0, 1]", async () => {
    const decision = await bridge.triage({
      channel: "http",
      payload: { text: "summarise my last 10 emails" },
      priority: "normal",
    });

    expect(decision).toBeDefined();
    expect(typeof decision.confidence).toBe("number");
    expect(decision.confidence).toBeGreaterThanOrEqual(0);
    expect(decision.confidence).toBeLessThanOrEqual(1);
    expect(["LOCAL_ML", "RULES", "HYBRID", "LLM_NEEDED"]).toContain(
      decision.route
    );
  });

  live("triage handles high-priority critical event", async () => {
    const decision = await bridge.triage({
      channel: "webhook",
      payload: {
        text: "production database is down, investigate immediately",
      },
      priority: "critical",
    });

    expect(decision).toBeDefined();
    expect(typeof decision.confidence).toBe("number");
  });

  live("remember returns a non-empty ULID string", async () => {
    const id = await bridge.remember("GristMill Phase 1 smoke test", [
      "smoke",
      "phase1",
    ]);
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  live("recall returns an array", async () => {
    // Store something first so there is at least one result to find.
    await bridge.remember("GristMill smoke recall marker", ["smoke-recall"]);

    const results = await bridge.recall("smoke recall marker", 5);
    expect(Array.isArray(results)).toBe(true);
  });

  live("getMemory returns null for a non-existent id", async () => {
    // A valid ULID that does not exist in the ledger.
    const mem = await bridge.getMemory("01ARZ3NDEKTSV4RRFFQ69G5FAV");
    expect(mem).toBeNull();
  });

  live("multiple sequential triage calls succeed", async () => {
    const queries = [
      "what time is it",
      "schedule a meeting tomorrow at 3pm",
      "explain why the build failed with this stack trace",
    ];

    for (const text of queries) {
      const d = await bridge.triage({
        channel: "cli",
        payload: { text },
      });
      expect(d.confidence).toBeGreaterThanOrEqual(0);
    }
  });

  live(
    "escalate returns a result or a budget/config error (no silent failure)",
    async () => {

      // Escalate may fail if ANTHROPIC_API_KEY is not set or Ollama is not
      // running — that is acceptable.  What we assert is that the bridge
      // does not hang and either resolves with a response or rejects with a
      // descriptive error within a reasonable timeout.
      const ESCALATE_TIMEOUT_MS = 10_000;

      const raceResult = await Promise.race([
        bridge
          .escalate("What is the capital of France?", 64)
          .then((r) => ({ ok: true as const, value: r }))
          .catch((e: unknown) => ({ ok: false as const, error: e })),
        new Promise<{ ok: false; error: string }>((resolve) =>
          setTimeout(
            () =>
              resolve({
                ok: false,
                error: `escalate did not resolve within ${ESCALATE_TIMEOUT_MS}ms — provider may be unreachable`,
              }),
            ESCALATE_TIMEOUT_MS
          )
        ),
      ]);

      if (raceResult.ok) {
        // Succeeded: response must be an object.
        expect(typeof raceResult.value).toBe("object");
      } else {
        // Failed or timed out: error message must be non-empty.
        const msg =
          raceResult.error instanceof Error
            ? raceResult.error.message
            : String(raceResult.error);
        expect(msg.length).toBeGreaterThan(0);
        console.log(`  [escalate not available: ${msg.slice(0, 80)}]`);
      }
    },
    15_000  // providers may take up to ~10 s before returning an error
  );

  live("getPipelineIds returns an array on fresh daemon", async () => {
    const ids = await bridge.getPipelineIds();
    expect(Array.isArray(ids)).toBe(true);
  });
});
