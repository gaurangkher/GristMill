/**
 * logger-plugin — built-in reference plugin that logs every
 * `pipeline.completed` and `pipeline.failed` bus event to stdout.
 *
 * This plugin is a good reference implementation showing how to use
 * `ctx.subscribe()` and `ctx.log()` in a real plugin.
 */

import { createPlugin } from "../sdk.js";

export default createPlugin({
  name: "gristmill/logger",
  version: "1.0.0",
  register(ctx) {
    // Subscribe to pipeline.completed and log each event
    (async () => {
      for await (const event of ctx.subscribe("pipeline.completed")) {
        ctx.log("info", `pipeline completed: ${JSON.stringify(event)}`);
      }
    })();

    // Subscribe to pipeline.failed and log each event
    (async () => {
      for await (const event of ctx.subscribe("pipeline.failed")) {
        ctx.log("warn", `pipeline failed: ${JSON.stringify(event)}`);
      }
    })();
  },
});
