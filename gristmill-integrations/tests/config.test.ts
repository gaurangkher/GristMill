/**
 * Unit tests for config.ts — the TypeScript config loader.
 *
 * Tests use a temporary YAML file written to a tempdir and override the
 * GRISTMILL_CONFIG env var to point the loader at that file.
 *
 * loadConfig() reads GRISTMILL_CONFIG on every invocation (no caching), so
 * each test can point the env var at a freshly written YAML file and call
 * loadConfig() to get a new result without needing module cache-busting.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { loadConfig } from "../src/core/config.js";

// ── Test fixture helpers ──────────────────────────────────────────────────────

let tmpDir: string;
let savedEnv: Record<string, string | undefined>;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "gristmill-config-test-"));
  // Snapshot env vars we might mutate.
  savedEnv = {
    GRISTMILL_CONFIG: process.env["GRISTMILL_CONFIG"],
    SLACK_APP_TOKEN: process.env["SLACK_APP_TOKEN"],
    SLACK_BOT_TOKEN: process.env["SLACK_BOT_TOKEN"],
    SLACK_SIGNING_SECRET: process.env["SLACK_SIGNING_SECRET"],
    SLACK_REPLY_MODE: process.env["SLACK_REPLY_MODE"],
  };
  // Clear token env vars so they don't bleed across tests.
  delete process.env["SLACK_APP_TOKEN"];
  delete process.env["SLACK_BOT_TOKEN"];
  delete process.env["SLACK_SIGNING_SECRET"];
  delete process.env["SLACK_REPLY_MODE"];
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
  for (const [k, v] of Object.entries(savedEnv)) {
    if (v === undefined) {
      delete process.env[k];
    } else {
      process.env[k] = v;
    }
  }
});

/**
 * Write `yaml` to a temp file, point GRISTMILL_CONFIG at it, and call the
 * loader.  loadConfig() reads the env var fresh on each call.
 */
function loadWithYaml(yaml: string) {
  const cfgPath = path.join(tmpDir, "config.yaml");
  fs.writeFileSync(cfgPath, yaml, "utf-8");
  process.env["GRISTMILL_CONFIG"] = cfgPath;
  return loadConfig();
}

// ── second_brain field ────────────────────────────────────────────────────────

describe("loadConfig() — second_brain field", () => {
  it("secondBrain is null when second_brain section is absent", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    app_token: "xapp-test"
    bot_token: "xoxb-test"
`);
    expect(cfg.slack.secondBrain).toBeNull();
  });

  it("secondBrain is null when second_brain.enabled is false", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    second_brain:
      enabled: false
`);
    expect(cfg.slack.secondBrain).toBeNull();
  });

  it("secondBrain is non-null when second_brain.enabled is true", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    second_brain:
      enabled: true
`);
    expect(cfg.slack.secondBrain).not.toBeNull();
  });

  it("secondBrain uses default values when only enabled: true is set", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    second_brain:
      enabled: true
`);
    const sb = cfg.slack.secondBrain!;
    expect(sb.confidenceThreshold).toBe(0.85);
    expect(sb.maxRecallLimit).toBe(5);
    expect(sb.snippetLength).toBe(400);
  });

  it("secondBrain reads overridden values from YAML", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    second_brain:
      enabled: true
      confidence_threshold: 0.70
      max_recall_limit: 10
      snippet_length: 600
`);
    const sb = cfg.slack.secondBrain!;
    expect(sb.confidenceThreshold).toBeCloseTo(0.70);
    expect(sb.maxRecallLimit).toBe(10);
    expect(sb.snippetLength).toBe(600);
  });
});

// ── SlackConfig base fields ───────────────────────────────────────────────────

describe("loadConfig() — SlackConfig base fields", () => {
  it("reads app_token, bot_token, signing_secret, reply_mode from YAML", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    app_token: "xapp-yaml"
    bot_token: "xoxb-yaml"
    signing_secret: "sig-yaml"
    reply_mode: "off"
`);
    expect(cfg.slack.appToken).toBe("xapp-yaml");
    expect(cfg.slack.botToken).toBe("xoxb-yaml");
    expect(cfg.slack.signingSecret).toBe("sig-yaml");
    expect(cfg.slack.replyMode).toBe("off");
  });

  it("defaults replyMode to 'thread' when not specified", () => {
    const cfg = loadWithYaml(`
integrations:
  slack:
    app_token: "xapp-test"
`);
    expect(cfg.slack.replyMode).toBe("thread");
  });

  it("env var SLACK_APP_TOKEN overrides YAML value", () => {
    process.env["SLACK_APP_TOKEN"] = "xapp-from-env";
    const cfg = loadWithYaml(`
integrations:
  slack:
    app_token: "xapp-from-yaml"
`);
    expect(cfg.slack.appToken).toBe("xapp-from-env");
  });

  it("falls back to defaults when config file does not exist", () => {
    process.env["GRISTMILL_CONFIG"] = path.join(tmpDir, "nonexistent.yaml");
    const cfg = loadConfig();
    expect(cfg.slack.replyMode).toBe("thread");
    expect(cfg.slack.secondBrain).toBeNull();
  });
});

// ── hoppers section ───────────────────────────────────────────────────────────

describe("loadConfig() — hoppers section", () => {
  it("reads http hopper port from YAML", () => {
    const cfg = loadWithYaml(`
integrations:
  hoppers:
    http:
      port: 4567
`);
    expect(cfg.hoppers.http.port).toBe(4567);
  });

  it("defaults http hopper port to 3001", () => {
    const cfg = loadWithYaml(`
integrations: {}
`);
    expect(cfg.hoppers.http.port).toBe(3001);
  });
});
