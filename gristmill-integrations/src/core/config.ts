/**
 * Minimal GristMill config reader for the TypeScript shell.
 *
 * Reads `~/.gristmill/config.yaml` (or the path set by GRISTMILL_CONFIG)
 * and surfaces just the fields the TypeScript layer cares about.
 *
 * Priority (highest → lowest):
 *   1. Environment variables
 *   2. config.yaml values
 *   3. Built-in defaults
 *
 * This mirrors the behaviour of `grist-config`'s `apply_env()` on the Rust side.
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { parse as parseYaml } from "yaml";

// ── Types (matches grist-config IntegrationsConfig shape) ─────────────────────

interface SlackYamlConfig {
  app_token?: string;
  bot_token?: string;
  signing_secret?: string;
  reply_mode?: string;
}

interface IntegrationsYamlConfig {
  slack?: SlackYamlConfig;
  plugins_dir?: string;
  dashboard?: { port?: number; enabled?: boolean };
}

interface GristMillYaml {
  integrations?: IntegrationsYamlConfig;
}

// ── Resolved config shape ─────────────────────────────────────────────────────

export interface SlackConfig {
  /** App-level token (xapp-…). Set via integrations.slack.app_token or SLACK_APP_TOKEN. */
  appToken: string;
  /** Bot OAuth token (xoxb-…). Set via integrations.slack.bot_token or SLACK_BOT_TOKEN. */
  botToken: string;
  /** Signing secret for HTTP Events API. Set via integrations.slack.signing_secret or SLACK_SIGNING_SECRET. */
  signingSecret: string;
  /** "thread" | "off". Set via integrations.slack.reply_mode or SLACK_REPLY_MODE. Default: "thread". */
  replyMode: "thread" | "off";
}

export interface GristMillTsConfig {
  slack: SlackConfig;
  pluginsDir: string;
}

// ── Loader ────────────────────────────────────────────────────────────────────

/**
 * Load and merge config.yaml + environment variables.
 * Always succeeds — missing file or parse errors fall back to defaults/env vars.
 */
export function loadConfig(): GristMillTsConfig {
  const configPath =
    process.env["GRISTMILL_CONFIG"] ??
    join(homedir(), ".gristmill", "config.yaml");

  let yaml: GristMillYaml = {};
  if (existsSync(configPath)) {
    try {
      yaml = parseYaml(readFileSync(configPath, "utf-8")) as GristMillYaml ?? {};
    } catch (err) {
      console.warn(`[config] Failed to parse ${configPath}: ${err}`);
    }
  }

  const s = yaml.integrations?.slack ?? {};

  // Priority: env var > config.yaml > default
  const slack: SlackConfig = {
    appToken:      process.env["SLACK_APP_TOKEN"]      ?? s.app_token      ?? "",
    botToken:      process.env["SLACK_BOT_TOKEN"]      ?? s.bot_token      ?? "",
    signingSecret: process.env["SLACK_SIGNING_SECRET"] ?? s.signing_secret ?? "",
    replyMode:     _replyMode(process.env["SLACK_REPLY_MODE"] ?? s.reply_mode),
  };

  const pluginsDir =
    process.env["GRISTMILL_PLUGINS_DIR"] ??
    yaml.integrations?.plugins_dir ??
    join(homedir(), ".gristmill", "plugins");

  return { slack, pluginsDir };
}

function _replyMode(raw?: string): "thread" | "off" {
  return raw === "off" ? "off" : "thread";
}
