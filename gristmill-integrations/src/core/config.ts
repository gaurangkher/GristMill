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

interface SecondBrainYamlConfig {
  enabled?: boolean;
  confidence_threshold?: number;
  max_recall_limit?: number;
  snippet_length?: number;
}

interface SlackYamlConfig {
  app_token?: string;
  bot_token?: string;
  signing_secret?: string;
  reply_mode?: string;
  second_brain?: SecondBrainYamlConfig;
}

interface CronJobYaml {
  id: string;
  interval_ms: number;
  channel?: string;
  payload?: unknown;
  fire_immediately?: boolean;
}

interface FsWatchYaml {
  path: string;
  channel?: string;
  recursive?: boolean;
  debounce_ms?: number;
}

interface WebhookChannelYaml {
  secret: string;
  header?: string;
}

interface HoppersYamlConfig {
  http?: { port?: number; host?: string };
  webhook?: { port?: number; host?: string; channels?: Record<string, WebhookChannelYaml> };
  cron?: { jobs?: CronJobYaml[] };
  fs?: { watches?: FsWatchYaml[] };
}

interface IntegrationsYamlConfig {
  slack?: SlackYamlConfig;
  plugins_dir?: string;
  dashboard?: { port?: number; enabled?: boolean };
  hoppers?: HoppersYamlConfig;
}

interface GristMillYaml {
  integrations?: IntegrationsYamlConfig;
}

// ── Resolved config shape ─────────────────────────────────────────────────────

export interface SecondBrainSettings {
  /** Minimum recall score before falling back to LLM. Default: 0.85 */
  confidenceThreshold: number;
  /** Max warm-tier notes retrieved per /ask query. Default: 5 */
  maxRecallLimit: number;
  /** Max chars per note snippet in Slack reply. Default: 400 */
  snippetLength: number;
}

export interface SlackConfig {
  /** App-level token (xapp-…). Set via integrations.slack.app_token or SLACK_APP_TOKEN. */
  appToken: string;
  /** Bot OAuth token (xoxb-…). Set via integrations.slack.bot_token or SLACK_BOT_TOKEN. */
  botToken: string;
  /** Signing secret for HTTP Events API. Set via integrations.slack.signing_secret or SLACK_SIGNING_SECRET. */
  signingSecret: string;
  /** "thread" | "off". Set via integrations.slack.reply_mode or SLACK_REPLY_MODE. Default: "thread". */
  replyMode: "thread" | "off";
  /** null = disabled. Set via integrations.slack.second_brain.enabled: true */
  secondBrain: SecondBrainSettings | null;
}

export interface HttpHopperSettings {
  port: number;
  host: string;
}

export interface WebhookHopperSettings {
  port: number;
  host: string;
  channels: Record<string, { secret: string; headerName: string }>;
}

export interface CronJobConfig {
  id: string;
  intervalMs: number;
  channel: string;
  payload: unknown;
  fireImmediately: boolean;
}

export interface FsWatchConfig {
  path: string;
  channel: string;
  recursive: boolean;
  debounceMs: number;
}

export interface HoppersConfig {
  http: HttpHopperSettings;
  webhook: WebhookHopperSettings;
  cron: CronJobConfig[];
  fs: FsWatchConfig[];
}

export interface GristMillTsConfig {
  slack: SlackConfig;
  pluginsDir: string;
  hoppers: HoppersConfig;
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
  const sb = s.second_brain;
  const secondBrainEnabled = sb?.enabled ?? false;

  const slack: SlackConfig = {
    appToken:      process.env["SLACK_APP_TOKEN"]      ?? s.app_token      ?? "",
    botToken:      process.env["SLACK_BOT_TOKEN"]      ?? s.bot_token      ?? "",
    signingSecret: process.env["SLACK_SIGNING_SECRET"] ?? s.signing_secret ?? "",
    replyMode:     _replyMode(process.env["SLACK_REPLY_MODE"] ?? s.reply_mode),
    secondBrain: secondBrainEnabled
      ? {
          confidenceThreshold: sb?.confidence_threshold ?? 0.85,
          maxRecallLimit:      sb?.max_recall_limit      ?? 5,
          snippetLength:       sb?.snippet_length        ?? 400,
        }
      : null,
  };

  const pluginsDir =
    process.env["GRISTMILL_PLUGINS_DIR"] ??
    yaml.integrations?.plugins_dir ??
    join(homedir(), ".gristmill", "plugins");

  const h = yaml.integrations?.hoppers ?? {};

  const hoppers: HoppersConfig = {
    http: {
      port: Number(process.env["GRISTMILL_HTTP_HOPPER_PORT"] ?? h.http?.port ?? 3001),
      host: process.env["GRISTMILL_HTTP_HOPPER_HOST"] ?? h.http?.host ?? "0.0.0.0",
    },
    webhook: {
      port: Number(process.env["GRISTMILL_WEBHOOK_PORT"] ?? h.webhook?.port ?? 3002),
      host: process.env["GRISTMILL_WEBHOOK_HOST"] ?? h.webhook?.host ?? "0.0.0.0",
      channels: Object.fromEntries(
        Object.entries(h.webhook?.channels ?? {}).map(([ch, cfg]) => [
          ch,
          {
            secret: cfg.secret,
            headerName: cfg.header ?? "x-hub-signature-256",
          },
        ]),
      ),
    },
    cron: (h.cron?.jobs ?? []).map((j) => ({
      id: j.id,
      intervalMs: j.interval_ms,
      channel: j.channel ?? "cron",
      payload: j.payload ?? {},
      fireImmediately: j.fire_immediately ?? false,
    })),
    fs: (h.fs?.watches ?? []).map((w) => ({
      path: w.path,
      channel: w.channel ?? "fs",
      recursive: w.recursive ?? false,
      debounceMs: w.debounce_ms ?? 300,
    })),
  };

  return { slack, pluginsDir, hoppers };
}

function _replyMode(raw?: string): "thread" | "off" {
  return raw === "off" ? "off" : "thread";
}
