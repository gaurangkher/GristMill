/**
 * SlackHopper — receives Slack events via Socket Mode and submits them to the
 * Rust core as GristEvents.
 *
 * Why Socket Mode?
 *   Unlike the HTTP Events API, Socket Mode maintains a persistent WebSocket
 *   from GristMill to Slack — no public URL, no ngrok, works behind firewalls.
 *   This aligns with GristMill's local-first philosophy.
 *
 * Required env vars:
 *   SLACK_APP_TOKEN   — App-level token (xapp-…) with `connections:write` scope
 *   SLACK_BOT_TOKEN   — Bot token (xoxb-…) for sending replies
 *
 * Optional env vars:
 *   SLACK_REPLY_MODE  — "off" | "thread" (default: "thread")
 *                       off    → reply in channel
 *                       thread → reply in the same thread as the trigger message
 *
 * Event mapping:
 *   Slack event type     → GristEvent channel / tags
 *   app_mention          → channel="slack", tags.slack_event_type="app_mention"
 *   message (DM only)    → channel="slack", tags.slack_conv_type="direct"
 *   reaction_added       → channel="slack", tags.slack_event_type="reaction_added"
 *   member_joined_channel→ channel="slack", tags.slack_event_type="member_joined_channel"
 *   (all others)         → channel="slack", tags.slack_event_type=<raw type>
 *
 * Note: channel `message` events are intentionally ignored — an @mention in a
 * channel fires both `app_mention` AND `message`; handling both would produce
 * duplicate replies.  Use `app_mention` for channel interactions.
 *
 * Reply behaviour:
 *   After triage, if the RouteDecision carries a `response` string in its
 *   metadata, SlackHopper posts it back to the originating channel/thread
 *   via WebClient.chat.postMessage.
 */

import { SocketModeClient } from "@slack/socket-mode";
import { WebClient, type ChatPostMessageArguments } from "@slack/web-api";
import type { IBridge, GristEventInit, RouteDecision, EscalationResult } from "../core/bridge.js";

// ── Config ─────────────────────────────────────────────────────────────────────

export interface SlackHopperConfig {
  /** App-level token (xapp-…). Required for Socket Mode. */
  appToken: string;
  /** Bot OAuth token (xoxb-…). Required for posting replies. */
  botToken: string;
  /**
   * How to post replies.
   * - "thread"  (default) — reply in the thread of the triggering message
   * - "off"               — do not auto-reply
   */
  replyMode?: "thread" | "off";
}

// ── SlackHopper ───────────────────────────────────────────────────────────────

export class SlackHopper {
  private client: SocketModeClient;
  private web: WebClient;
  private replyMode: "thread" | "off";

  constructor(
    private readonly bridge: IBridge,
    private readonly config: SlackHopperConfig,
  ) {
    this.client = new SocketModeClient({ appToken: config.appToken });
    this.web = new WebClient(config.botToken);
    this.replyMode = config.replyMode ?? "thread";
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async start(): Promise<void> {
    this._registerHandlers();
    await this.client.start();
    console.log("[SlackHopper] Connected via Socket Mode");
  }

  async stop(): Promise<void> {
    await this.client.disconnect();
    console.log("[SlackHopper] Disconnected");
  }

  // ── Event handlers ─────────────────────────────────────────────────────────

  private _registerHandlers(): void {
    // app_mention: someone @-mentions the bot in a channel
    this.client.on("app_mention", async ({ event, ack }) => {
      await ack();
      await this._handleEvent(event as SlackMessageEvent);
    });

    // message: DMs only.
    //
    // Channel messages where the bot is @mentioned fire BOTH an `app_mention`
    // event (handled above) AND a `message` event.  Processing both would post
    // two identical replies for every mention.  The `message` handler is
    // therefore restricted to direct messages (channel_type === "im"), which
    // never produce an `app_mention` event.
    this.client.on("message", async ({ event, ack }) => {
      await ack();
      const msg = event as SlackMessageEvent;
      // Ignore bot messages to prevent reply loops.
      if (msg.bot_id || msg.subtype === "bot_message") return;
      // Only process DMs — channel @mentions are handled by app_mention above.
      if (msg.channel_type !== "im") return;
      await this._handleEvent(msg);
    });

    // reaction_added / reaction_removed
    this.client.on("reaction_added", async ({ event, ack }) => {
      await ack();
      await this._handleGenericEvent(event, "reaction_added");
    });

    this.client.on("reaction_removed", async ({ event, ack }) => {
      await ack();
      await this._handleGenericEvent(event, "reaction_removed");
    });

    // member_joined_channel
    this.client.on("member_joined_channel", async ({ event, ack }) => {
      await ack();
      await this._handleGenericEvent(event, "member_joined_channel");
    });
  }

  // ── Core processing ────────────────────────────────────────────────────────

  private async _handleEvent(event: SlackMessageEvent): Promise<void> {
    const convType = _convType(event.channel_type);
    const gristEvent: GristEventInit = {
      channel: "slack",
      payload: event,
      priority: event.channel_type === "im" ? "high" : "normal",
      tags: {
        slack_event_type: String(event.type ?? "message"),
        slack_conv_type: convType,
        slack_channel: String(event.channel ?? ""),
        slack_user: String(event.user ?? ""),
        slack_thread_ts: String(event.thread_ts ?? event.ts ?? ""),
        slack_team: String(event.team ?? ""),
      },
    };

    try {
      const decision: RouteDecision = await this.bridge.triage(gristEvent);
      console.log(
        `[SlackHopper] event from @${event.user ?? "unknown"} → route=${decision.route} confidence=${decision.confidence.toFixed(2)}${decision.reason ? ` reason="${decision.reason}"` : ""}`,
      );

      // If the sieve says LLM is needed (or hybrid), escalate and reply with
      // the model's response.  LOCAL_ML / RULES only reply if reason is set.
      if (decision.route === "LLM_NEEDED" || decision.route === "HYBRID") {
        await this._escalateAndReply(event, decision);
      } else {
        await this._maybeReply(event, decision);
      }
    } catch (err) {
      console.error("[SlackHopper] triage error:", err);
    }
  }

  private async _handleGenericEvent(event: unknown, type: string): Promise<void> {
    const gristEvent: GristEventInit = {
      channel: "slack",
      payload: event,
      priority: "low",
      tags: { slack_event_type: type },
    };
    try {
      await this.bridge.triage(gristEvent);
    } catch (err) {
      console.error(`[SlackHopper] triage error (${type}):`, err);
    }
  }

  // ── Reply ──────────────────────────────────────────────────────────────────

  /**
   * Called when the sieve routes to LLM_NEEDED or HYBRID.
   * Calls bridge.escalate() with the message text and posts the response.
   */
  private async _escalateAndReply(
    event: SlackMessageEvent,
    decision: RouteDecision,
  ): Promise<void> {
    if (this.replyMode === "off") return;
    if (!event.channel) return;

    const text = String(event.text ?? "").trim();
    if (!text) return;

    // Build a prompt that gives the LLM context about the Slack source
    const prompt = `You are GristMill, an AI assistant. A user sent the following message via Slack:\n\n"${text}"\n\nRespond helpfully and concisely.`;

    let result: EscalationResult;
    try {
      result = await this.bridge.escalate(prompt, 1024);
    } catch (err) {
      console.error("[SlackHopper] escalation failed:", err);
      await this._postMessage(event, `_GristMill:_ Sorry, escalation failed (${decision.route}).`);
      return;
    }

    console.log(
      `[SlackHopper] escalated → provider=${result.provider} tokens=${result.tokensUsed} cacheHit=${result.cacheHit}`,
    );

    await this._postMessage(event, result.content);
  }

  /**
   * For LOCAL_ML / RULES routes: post decision.reason if present.
   */
  private async _maybeReply(
    event: SlackMessageEvent,
    decision: RouteDecision,
  ): Promise<void> {
    if (this.replyMode === "off") return;
    if (!event.channel) return;

    const responseText = decision.reason;
    if (!responseText?.trim()) return;

    await this._postMessage(event, `_GristMill:_ ${responseText}`);
  }

  /**
   * Post a message back to the Slack channel/thread the event came from.
   */
  private async _postMessage(event: SlackMessageEvent, text: string): Promise<void> {
    if (!event.channel) return;

    const args: ChatPostMessageArguments = {
      channel: event.channel,
      text,
    };

    if (this.replyMode === "thread") {
      args.thread_ts = event.thread_ts ?? event.ts ?? undefined;
    }

    try {
      await this.web.chat.postMessage(args);
    } catch (err) {
      console.error("[SlackHopper] Failed to post message:", err);
    }
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

interface SlackMessageEvent {
  type?: string;
  channel?: string;
  channel_type?: string;
  user?: string;
  text?: string;
  ts?: string;
  thread_ts?: string;
  team?: string;
  bot_id?: string;
  subtype?: string;
  [key: string]: unknown;
}

function _convType(channelType?: string): string {
  switch (channelType) {
    case "im":
      return "direct";
    case "mpim":
      return "group";
    default:
      return "channel";
  }
}
