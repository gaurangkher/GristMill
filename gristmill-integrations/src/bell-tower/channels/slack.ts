/**
 * SlackChannel — sends notifications to Slack via Incoming Webhooks.
 *
 * Uses Block Kit for structured messages.  Retries once on HTTP 429
 * (rate limit) with a 1-second backoff.
 */

import type { Notification } from "../dispatcher.js";

export class SlackChannel {
  constructor(private readonly webhookUrl: string) {}

  async send(notification: Notification): Promise<void> {
    const body = JSON.stringify({
      blocks: [
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: this._format(notification),
          },
        },
        {
          type: "context",
          elements: [
            {
              type: "mrkdwn",
              text: `*Topic:* ${notification.topic} | *Priority:* ${notification.priority} | <!date^${Math.floor(notification.createdAt.getTime() / 1000)}^{time_secs}|${notification.createdAt.toISOString()}>`,
            },
          ],
        },
      ],
    });

    await this._post(body);
  }

  private _format(n: Notification): string {
    const icon = this._icon(n.priority);
    return `${icon} *${n.title}*\n${n.body}`;
  }

  private _icon(priority: Notification["priority"]): string {
    switch (priority) {
      case "critical":
        return ":rotating_light:";
      case "high":
        return ":warning:";
      case "low":
        return ":information_source:";
      default:
        return ":bell:";
    }
  }

  private async _post(body: string, retry = true): Promise<void> {
    const res = await fetch(this.webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });

    if (res.status === 429 && retry) {
      await new Promise((r) => setTimeout(r, 1000));
      return this._post(body, false);
    }

    if (!res.ok) {
      throw new Error(`Slack webhook failed: ${res.status} ${await res.text()}`);
    }
  }
}
