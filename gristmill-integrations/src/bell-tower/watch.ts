/**
 * WatchEngine — evaluates user-defined watches against bus events.
 *
 * A `Watch` specifies a bus topic and a condition expression evaluated
 * against the event payload.  Supported condition syntax:
 *
 * Simple:
 *   field comparator value
 *
 * Examples:
 *   "confidence < 0.5"
 *   "daily_used > 400000"
 *   "tier == hot"
 *   "route contains ANOMALY"
 *   "anomaly_type startsWith mem"
 *   "pipeline_id matches ^prod-"
 *
 * Compound (&&  binds tighter than ||):
 *   "confidence < 0.5 && route != LOCAL_ML"
 *   "pct >= 90 || tier == cold"
 *   "pct >= 80 && tier == warm || pct >= 90 && tier == hot"
 *
 * Comparators:
 *   Numeric:  `<`  `<=`  `>`  `>=`
 *   Equality: `==`  `!=`
 *   String:   `contains`  `startsWith`  `endsWith`  `matches` (regex)
 *
 * Values: number literals, bare strings, or regex patterns (for `matches`).
 */

import { readFile, writeFile } from "node:fs/promises";
import { monotonicFactory } from "ulid";

const ulid = monotonicFactory();

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Watch {
  id: string;
  name: string;
  /** Bus topic to match, e.g. `"sieve.anomaly"`. */
  topic: string;
  /**
   * Condition expression.
   * Simple:   `"confidence < 0.5"`
   * Compound: `"confidence < 0.5 && route != LOCAL_ML"`
   */
  condition: string;
  /** Channel ids to notify when the condition fires. */
  channelIds: string[];
  /** Minimum ms between alerts on the same watch (default: 60 000). */
  cooldownMs: number;
  enabled: boolean;
}

// ── WatchEngine ───────────────────────────────────────────────────────────────

export class WatchEngine {
  private watches: Map<string, Watch> = new Map();
  private lastFired: Map<string, number> = new Map();

  // ── CRUD ──────────────────────────────────────────────────────────────────

  addWatch(watch: Watch): void {
    this.watches.set(watch.id, watch);
  }

  removeWatch(id: string): void {
    this.watches.delete(id);
    this.lastFired.delete(id);
  }

  getWatch(id: string): Watch | undefined {
    return this.watches.get(id);
  }

  listWatches(): Watch[] {
    return [...this.watches.values()];
  }

  /**
   * Partially update a watch.  Returns the updated watch, or `null` if the
   * watch was not found.
   */
  updateWatch(id: string, patch: Partial<Omit<Watch, "id">>): Watch | null {
    const existing = this.watches.get(id);
    if (!existing) return null;
    const updated: Watch = { ...existing, ...patch, id };
    this.watches.set(id, updated);
    return updated;
  }

  // ── Evaluation ────────────────────────────────────────────────────────────

  /**
   * Evaluate all watches whose topic matches `topic` and whose condition
   * evaluates to `true` against `payload`.
   *
   * Respects per-watch cooldown to suppress repeated alerts.
   * Returns the subset of watches that should fire.
   */
  evaluate(topic: string, payload: unknown): Watch[] {
    const now = Date.now();
    const fired: Watch[] = [];

    for (const watch of this.watches.values()) {
      if (!watch.enabled) continue;
      if (watch.topic !== topic) continue;

      const last = this.lastFired.get(watch.id) ?? 0;
      if (now - last < watch.cooldownMs) continue;

      if (this._evaluateCompound(watch.condition, payload)) {
        this.lastFired.set(watch.id, now);
        fired.push(watch);
      }
    }

    return fired;
  }

  // ── Persistence ───────────────────────────────────────────────────────────

  /**
   * Save all watches to a JSON file.  Creates or overwrites the file.
   */
  async saveToFile(filePath: string): Promise<void> {
    const data = JSON.stringify([...this.watches.values()], null, 2);
    await writeFile(filePath, data, "utf-8");
  }

  /**
   * Load watches from a JSON file written by `saveToFile`.
   * If the file does not exist, this is a no-op.
   * Invalid entries are skipped with a warning.
   */
  async loadFromFile(filePath: string): Promise<void> {
    let raw: string;
    try {
      raw = await readFile(filePath, "utf-8");
    } catch {
      // File doesn't exist yet — fine, no watches to load
      return;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      console.warn(`[WatchEngine] Could not parse watch file at ${filePath}`);
      return;
    }

    if (!Array.isArray(parsed)) {
      console.warn(`[WatchEngine] Watch file does not contain an array: ${filePath}`);
      return;
    }

    let loaded = 0;
    for (const entry of parsed) {
      if (_isWatch(entry)) {
        this.watches.set(entry.id, entry);
        loaded++;
      } else {
        console.warn("[WatchEngine] Skipping invalid watch entry:", entry);
      }
    }

    if (loaded > 0) {
      console.info(`[WatchEngine] Loaded ${loaded} watches from ${filePath}`);
    }
  }

  // ── Compound condition evaluator ──────────────────────────────────────────

  /**
   * Evaluates a compound condition expression.
   *
   * Precedence (low → high):  OR (`||`)  <  AND (`&&`)  <  simple expression.
   *
   * "a && b || c && d"  →  (a AND b) OR (c AND d)
   */
  private _evaluateCompound(condition: string, payload: unknown): boolean {
    // Split on " || " first to get OR-clauses
    const orClauses = condition.split(/\s*\|\|\s*/);
    return orClauses.some((clause) => {
      // Split each OR-clause on " && " to get AND-terms
      const andTerms = clause.split(/\s*&&\s*/);
      return andTerms.every((term) => this._evaluateSimple(term.trim(), payload));
    });
  }

  /**
   * Evaluates a single `field comparator value` expression.
   *
   * Supported comparators:
   *   <  <=  >  >=  ==  !=  contains  startsWith  endsWith  matches
   */
  private _evaluateSimple(condition: string, payload: unknown): boolean {
    const match = condition
      .trim()
      .match(
        /^([a-zA-Z_][a-zA-Z0-9_.]*)\s*(<=|>=|==|!=|<|>|contains|startsWith|endsWith|matches)\s*(.+)$/,
      );
    if (!match) return false;

    const [, fieldPath, op, rawValue] = match as [string, string, string, string];
    const actual = this._getField(payload, fieldPath);
    if (actual === undefined) return false;

    const expected = this._coerce(rawValue.trim(), actual);

    switch (op) {
      case "<":
        return (actual as number) < (expected as number);
      case "<=":
        return (actual as number) <= (expected as number);
      case ">":
        return (actual as number) > (expected as number);
      case ">=":
        return (actual as number) >= (expected as number);
      case "==":
        return String(actual) === String(expected);
      case "!=":
        return String(actual) !== String(expected);
      case "contains":
        return String(actual).toLowerCase().includes(String(expected).toLowerCase());
      case "startsWith":
        return String(actual).toLowerCase().startsWith(String(expected).toLowerCase());
      case "endsWith":
        return String(actual).toLowerCase().endsWith(String(expected).toLowerCase());
      case "matches": {
        try {
          return new RegExp(String(expected)).test(String(actual));
        } catch {
          console.warn(`[WatchEngine] Invalid regex in condition: "${String(expected)}"`);
          return false;
        }
      }
      default:
        return false;
    }
  }

  /** Deep-get a dot-separated field path from an object. */
  private _getField(obj: unknown, path: string): unknown {
    const parts = path.split(".");
    let cur: unknown = obj;
    for (const part of parts) {
      if (cur == null || typeof cur !== "object") return undefined;
      cur = (cur as Record<string, unknown>)[part];
    }
    return cur;
  }

  /** Coerce `rawValue` string to the same type as `actual`. */
  private _coerce(raw: string, actual: unknown): unknown {
    if (typeof actual === "number") {
      const n = Number(raw);
      return isNaN(n) ? raw : n;
    }
    if (typeof actual === "boolean") {
      return raw === "true";
    }
    return raw;
  }
}

// ── Type guard ────────────────────────────────────────────────────────────────

function _isWatch(val: unknown): val is Watch {
  return (
    typeof val === "object" &&
    val !== null &&
    typeof (val as Watch).id === "string" &&
    typeof (val as Watch).name === "string" &&
    typeof (val as Watch).topic === "string" &&
    typeof (val as Watch).condition === "string" &&
    Array.isArray((val as Watch).channelIds) &&
    typeof (val as Watch).cooldownMs === "number" &&
    typeof (val as Watch).enabled === "boolean"
  );
}

// ── Factory ───────────────────────────────────────────────────────────────────

/** Create a new Watch with sensible defaults. */
export function createWatch(
  partial: Omit<Watch, "id" | "enabled" | "cooldownMs"> &
    Partial<Pick<Watch, "id" | "enabled" | "cooldownMs">>,
): Watch {
  return {
    id: partial.id ?? ulid(),
    name: partial.name,
    topic: partial.topic,
    condition: partial.condition,
    channelIds: partial.channelIds,
    cooldownMs: partial.cooldownMs ?? 60_000,
    enabled: partial.enabled ?? true,
  };
}
