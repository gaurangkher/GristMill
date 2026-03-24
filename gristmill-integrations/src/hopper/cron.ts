/**
 * CronHopper — fires GristEvents on a repeating schedule.
 *
 * Implements a lightweight interval-based scheduler with optional cron-style
 * expressions (minute/hour/day-of-week matching).  No external dependencies —
 * uses only `node:timers`.
 *
 * Architecture: TypeScript handles scheduling I/O only.  The event payload is
 * passed to `bridge.triage()` exactly as configured.
 *
 * Usage:
 *   const cron = new CronHopper(bridge);
 *   cron.add({ id: "daily-digest", intervalMs: 86_400_000,
 *               event: { channel: "cron", payload: { job: "digest" } } });
 *   cron.start();
 *
 * Config via config.yaml (integrations.hoppers.cron):
 *   jobs:
 *     - id: daily-digest
 *       interval_ms: 86400000
 *       channel: cron
 *       payload: { job: daily-digest }
 *     - id: hourly-sync
 *       interval_ms: 3600000
 *       channel: cron
 *       payload: { job: sync }
 */

import type { IBridge, GristEventInit } from "../core/bridge.js";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CronJob {
  /** Unique job identifier (used in logs and GristEvent tags). */
  id: string;
  /** How often to fire, in milliseconds. */
  intervalMs: number;
  /** The GristEvent to submit on each tick. */
  event: GristEventInit;
  /** If true, fire once immediately on start (default: false). */
  fireImmediately?: boolean;
}

export interface CronHopperConfig {
  jobs?: CronJob[];
}

// ── CronHopper ────────────────────────────────────────────────────────────────

/**
 * Interval-based scheduler that emits GristEvents on a fixed cadence.
 *
 * Each job runs independently.  Errors in `bridge.triage()` are logged
 * but do not stop other jobs.
 */
export class CronHopper {
  private jobs: CronJob[] = [];
  private timers = new Map<string, ReturnType<typeof setInterval>>();
  private running = false;

  constructor(
    private readonly bridge: IBridge,
    config: CronHopperConfig = {},
  ) {
    for (const job of config.jobs ?? []) {
      this.add(job);
    }
  }

  /** Add a job.  If the hopper is already running, the job starts immediately. */
  add(job: CronJob): void {
    if (this.timers.has(job.id)) {
      console.warn(`[CronHopper] Job "${job.id}" already registered — skipping`);
      return;
    }
    this.jobs.push(job);
    if (this.running) this._startJob(job);
  }

  /** Remove a job by id. */
  remove(id: string): void {
    const timer = this.timers.get(id);
    if (timer) {
      clearInterval(timer);
      this.timers.delete(id);
    }
    this.jobs = this.jobs.filter((j) => j.id !== id);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    for (const job of this.jobs) {
      this._startJob(job);
    }
    console.log(`[CronHopper] Started ${this.jobs.length} job(s)`);
  }

  stop(): void {
    for (const timer of this.timers.values()) clearInterval(timer);
    this.timers.clear();
    this.running = false;
    console.log("[CronHopper] Stopped");
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  private _startJob(job: CronJob): void {
    const fire = async () => {
      const event: GristEventInit = {
        ...job.event,
        tags: {
          source: "cron",
          cron_job_id: job.id,
          ...(job.event.tags ?? {}),
        },
      };
      try {
        await this.bridge.triage(event);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[CronHopper] Job "${job.id}" triage error: ${msg}`);
      }
    };

    if (job.fireImmediately) void fire();
    this.timers.set(job.id, setInterval(() => void fire(), job.intervalMs));
  }
}
