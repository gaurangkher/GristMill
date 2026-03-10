export { NotificationDispatcher } from "./dispatcher.js";
export type {
  Notification,
  Priority,
  BellTowerConfig,
  QuietHoursConfig,
  DigestConfig,
  ChannelRouting,
  PluginNotificationChannel,
} from "./dispatcher.js";

export { WatchEngine, createWatch } from "./watch.js";
export type { Watch } from "./watch.js";

export { SlackChannel } from "./channels/slack.js";

export { EmailChannel } from "./channels/email.js";
export type { EmailConfig } from "./channels/email.js";
