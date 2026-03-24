/**
 * FsHopper — watches filesystem paths for changes and submits GristEvents.
 *
 * Uses Node's built-in `fs.watch` (no external deps).  Rapid bursts of
 * change events for the same path are debounced (default 300 ms).
 *
 * Architecture: TypeScript handles I/O only.  All routing decisions come
 * from `bridge.triage()`.
 *
 * Usage:
 *   const fs = new FsHopper(bridge);
 *   fs.watch("/data/inbox", { channel: "fs-inbox", recursive: true });
 *   fs.start();
 *
 * Config via config.yaml (integrations.hoppers.fs):
 *   watches:
 *     - path: /data/inbox
 *       channel: fs-inbox
 *       recursive: true
 *     - path: /data/reports
 *       channel: fs-reports
 *
 * GristEvent shape:
 *   channel:  configured per-watch (default "fs")
 *   payload:  { event_type, path, filename }
 *   tags:     { source: "fs", fs_event: <rename|change>, fs_path: <path> }
 */

import { watch, type FSWatcher } from "node:fs";
import { join } from "node:path";
import type { IBridge, GristEventInit } from "../core/bridge.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface FsWatchTarget {
  /** Absolute path to watch (file or directory). */
  path: string;
  /** GristEvent channel name (default: "fs"). */
  channel?: string;
  /** Watch sub-directories recursively (default: false). */
  recursive?: boolean;
  /** Debounce window in ms — suppresses repeated events for the same path (default: 300). */
  debounceMs?: number;
}

export interface FsHopperConfig {
  watches?: FsWatchTarget[];
}

// ── FsHopper ──────────────────────────────────────────────────────────────────

/**
 * Filesystem change watcher that emits GristEvents on file activity.
 *
 * Each watch target runs its own `fs.watch` listener with independent debounce.
 * Watcher errors (e.g. path disappears) are logged and the watcher is closed
 * gracefully.
 */
export class FsHopper {
  private targets: FsWatchTarget[] = [];
  private watchers: FSWatcher[] = [];
  private running = false;

  constructor(
    private readonly bridge: IBridge,
    config: FsHopperConfig = {},
  ) {
    for (const target of config.watches ?? []) {
      this.watch(target.path, target);
    }
  }

  /** Add a path to watch.  If already running, starts watching immediately. */
  watch(path: string, opts: Omit<FsWatchTarget, "path"> = {}): void {
    const target: FsWatchTarget = { path, ...opts };
    this.targets.push(target);
    if (this.running) this._startWatcher(target);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    for (const target of this.targets) {
      this._startWatcher(target);
    }
    console.log(`[FsHopper] Watching ${this.targets.length} path(s)`);
  }

  stop(): void {
    for (const w of this.watchers) {
      try { w.close(); } catch { /* ignore */ }
    }
    this.watchers = [];
    this.running = false;
    console.log("[FsHopper] Stopped");
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  private _startWatcher(target: FsWatchTarget): void {
    const channel = target.channel ?? "fs";
    const debounceMs = target.debounceMs ?? 300;
    const debounceTimers = new Map<string, ReturnType<typeof setTimeout>>();

    let watcher: FSWatcher;
    try {
      watcher = watch(
        target.path,
        { recursive: target.recursive ?? false },
        (eventType, filename) => {
          const resolvedFilename = filename
            ? join(target.path, filename)
            : target.path;

          // Debounce per resolved path
          const existing = debounceTimers.get(resolvedFilename);
          if (existing) clearTimeout(existing);

          debounceTimers.set(
            resolvedFilename,
            setTimeout(() => {
              debounceTimers.delete(resolvedFilename);
              const event: GristEventInit = {
                channel,
                payload: {
                  event_type: eventType,
                  path: resolvedFilename,
                  filename: filename ?? null,
                },
                priority: "normal",
                tags: {
                  source: "fs",
                  fs_event: eventType,
                  fs_path: target.path,
                },
              };
              void this.bridge.triage(event).catch((err: unknown) => {
                const msg = err instanceof Error ? err.message : String(err);
                console.error(`[FsHopper] triage error for ${resolvedFilename}: ${msg}`);
              });
            }, debounceMs),
          );
        },
      );

      watcher.on("error", (err: Error) => {
        console.error(`[FsHopper] Watcher error on ${target.path}: ${err.message}`);
        watcher.close();
      });

      this.watchers.push(watcher);
      console.log(`[FsHopper] Watching: ${target.path} (channel=${channel})`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[FsHopper] Failed to watch ${target.path}: ${msg}`);
    }
  }
}
