/**
 * webhook-enricher-plugin — built-in plugin that registers a
 * "webhook-enriched" adapter.
 *
 * Enriches incoming webhook payloads with a `receivedAt` ISO timestamp
 * before forwarding to the Rust core for triage.
 */

import { createPlugin } from "../sdk.js";

export default createPlugin({
  name: "gristmill/webhook-enricher",
  version: "1.0.0",
  register(ctx) {
    ctx.registerAdapter("webhook-enriched", async (raw) => {
      const payload =
        typeof raw === "object" && raw !== null ? raw : { data: raw };
      return {
        channel: "webhook",
        payload: {
          ...(payload as object),
          receivedAt: new Date().toISOString(),
        },
      };
    });
  },
});
