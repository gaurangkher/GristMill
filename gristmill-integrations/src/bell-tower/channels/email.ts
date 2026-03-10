/**
 * EmailChannel — sends notifications via SMTP using nodemailer.
 */

import nodemailer from "nodemailer";
import type Mail from "nodemailer/lib/mailer/index.js";
import type { Notification } from "../dispatcher.js";

export interface EmailConfig {
  smtpHost: string;
  smtpPort: number;
  username: string;
  password: string;
  from?: string;
  secure?: boolean;
}

export class EmailChannel {
  private readonly transporter: Mail;
  private readonly from: string;

  constructor(private readonly config: EmailConfig) {
    this.transporter = nodemailer.createTransport({
      host: config.smtpHost,
      port: config.smtpPort,
      secure: config.secure ?? config.smtpPort === 465,
      auth: {
        user: config.username,
        pass: config.password,
      },
    });
    this.from = config.from ?? `GristMill <${config.username}>`;
  }

  async send(notification: Notification, to: string[]): Promise<void> {
    if (to.length === 0) return;

    await this.transporter.sendMail({
      from: this.from,
      to: to.join(", "),
      subject: `[GristMill] ${notification.title}`,
      text: this._plainText(notification),
      html: this._html(notification),
    });
  }

  private _plainText(n: Notification): string {
    return [
      `${n.title}`,
      ``,
      n.body,
      ``,
      `---`,
      `Topic: ${n.topic}`,
      `Priority: ${n.priority}`,
      `Time: ${n.createdAt.toISOString()}`,
      `ID: ${n.id}`,
    ].join("\n");
  }

  private _html(n: Notification): string {
    const priorityColor: Record<string, string> = {
      critical: "#cc0000",
      high: "#ff6600",
      normal: "#0066cc",
      low: "#666666",
    };
    const color = priorityColor[n.priority] ?? "#333333";

    return `
<!DOCTYPE html>
<html>
<body style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
  <div style="border-left: 4px solid ${color}; padding-left: 12px; margin-bottom: 16px;">
    <h2 style="margin: 0; color: ${color};">${escapeHtml(n.title)}</h2>
    <p style="margin: 8px 0;">${escapeHtml(n.body)}</p>
  </div>
  <table style="font-size: 12px; color: #666;">
    <tr><td>Topic:</td><td>${escapeHtml(n.topic)}</td></tr>
    <tr><td>Priority:</td><td>${escapeHtml(n.priority)}</td></tr>
    <tr><td>Time:</td><td>${n.createdAt.toISOString()}</td></tr>
  </table>
</body>
</html>`.trim();
  }
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
