#!/usr/bin/env node
/**
 * GristMill interactive CLI — natural language interface to the dashboard API.
 *
 * Usage:
 *   pnpm cli                      # connects to http://127.0.0.1:3000
 *   pnpm cli --port 4000
 *   pnpm cli --host 0.0.0.0 --port 3001
 *
 * Commands (all also accept natural language variations):
 *   remember <text> [tags: t1, t2]    Store a memory
 *   recall   <query> [limit N]        Search memories
 *   get      <ulid>                   Fetch memory by ID
 *   triage   <text>                   Get routing decision for text
 *   health                            Server health check
 *   budget                            LLM token budget
 *   pipelines                         List registered pipelines
 *   run      <pipeline-id> [json]     Run a pipeline
 *   help                              Show this help
 *   exit                              Quit
 *
 * Unrecognised input is treated as a recall query.
 */

import * as rlPromises from "node:readline/promises";

// ── ANSI colours ──────────────────────────────────────────────────────────────

const ESC = "\x1b";
const c = {
  reset:   `${ESC}[0m`,
  bold:    `${ESC}[1m`,
  dim:     `${ESC}[2m`,
  green:   `${ESC}[32m`,
  yellow:  `${ESC}[33m`,
  cyan:    `${ESC}[36m`,
  red:     `${ESC}[31m`,
  magenta: `${ESC}[35m`,
};

const ok   = (s: string) => `${c.green}${s}${c.reset}`;
const warn = (s: string) => `${c.yellow}${s}${c.reset}`;
const fail = (s: string) => `${c.red}${s}${c.reset}`;
const head = (s: string) => `${c.cyan}${c.bold}${s}${c.reset}`;
const dim  = (s: string) => `${c.dim}${s}${c.reset}`;
const hi   = (s: string) => `${c.magenta}${s}${c.reset}`;

// ── CLI args ──────────────────────────────────────────────────────────────────

function parseArgs(): { port: number; host: string } {
  const args = process.argv.slice(2);
  let port = 3000;
  let host = "127.0.0.1";
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--port" && args[i + 1]) port = Number(args[++i]);
    if (args[i] === "--host" && args[i + 1]) host = args[++i]!;
  }
  return { port, host };
}

// ── HTTP helper ───────────────────────────────────────────────────────────────

let BASE = "";

async function api<T>(
  method: "GET" | "POST",
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Command types ─────────────────────────────────────────────────────────────

type Cmd =
  | { type: "remember";  content: string; tags: string[] }
  | { type: "recall";    query: string;   limit: number }
  | { type: "get";       id: string }
  | { type: "triage";    text: string;    channel: string }
  | { type: "health" }
  | { type: "budget" }
  | { type: "pipelines" }
  | { type: "register";  pipeline: Record<string, unknown> }
  | { type: "run";       pipelineId: string; payload: unknown }
  | { type: "help" }
  | { type: "exit" };

// ── Natural language parser ───────────────────────────────────────────────────

function parse(raw: string): Cmd | null {
  const s = raw.trim();
  if (!s) return null;

  if (/^(exit|quit|bye|q)$/i.test(s))   return { type: "exit" };
  if (/^(help|\?)$/i.test(s))           return { type: "help" };
  if (/^(health|status|ping)$/i.test(s)) return { type: "health" };
  if (/^budget$/i.test(s))              return { type: "budget" };

  if (/^(pipelines?|list pipelines?|show pipelines?)$/i.test(s))
    return { type: "pipelines" };

  // remember <content> [tags: t1, t2]
  const rem = s.match(/^(?:remember|store|save)\s+(.+?)(?:\s+tags?:\s*(.+))?$/i);
  if (rem) {
    const content = rem[1]!.replace(/^["']|["']$/g, "").trim();
    const tags = rem[2]
      ? rem[2].split(/[,\s]+/).map((t) => t.trim()).filter(Boolean)
      : [];
    return { type: "remember", content, tags };
  }

  // recall / search for / find / what do you know about
  const rec = s.match(
    /^(?:recall|search(?:\s+for)?|find|what\s+(?:do\s+you\s+know|is\s+known)\s+about)\s+(.+?)(?:\s+limit\s+(\d+))?$/i,
  );
  if (rec) {
    const query = rec[1]!.trim().replace(/[?.!,;]+$/, "");
    return { type: "recall", query, limit: rec[2] ? Number(rec[2]) : 5 };
  }

  // get <ulid-26-chars>
  const getM = s.match(/^(?:get|fetch|show)\s+([A-Z0-9]{26})\b/i);
  if (getM) return { type: "get", id: getM[1]!.toUpperCase() };

  // triage / route / classify <text>
  const tri = s.match(/^(?:triage|route|classify|analyse|analyze)\s+(.+)$/i);
  if (tri) return { type: "triage", text: tri[1]!.trim().replace(/[?.!,;]+$/, ""), channel: "cli" };

  // register <id>  OR  register <json with "id" field>
  const reg = s.match(/^register\s+(.+)$/i);
  if (reg) {
    const arg = reg[1]!.trim();
    let pipeline: Record<string, unknown>;
    try {
      pipeline = JSON.parse(arg) as Record<string, unknown>;
      if (typeof pipeline["id"] !== "string") throw new Error("missing id");
    } catch {
      // Treat bare word as pipeline id → minimal stub
      pipeline = { id: arg };
    }
    return { type: "register", pipeline };
  }

  // run <pipeline-id> [json-or-text]
  const run = s.match(/^run\s+(\S+)(?:\s+(.+))?$/i);
  if (run) {
    let payload: unknown = {};
    if (run[2]) {
      try { payload = JSON.parse(run[2]); }
      catch { payload = { text: run[2] }; }
    }
    return { type: "run", pipelineId: run[1]!, payload };
  }

  // Fallback — treat as a recall query
  return { type: "recall", query: s.replace(/[?.!,;]+$/, ""), limit: 5 };
}

// ── Executors ─────────────────────────────────────────────────────────────────

async function execute(cmd: Cmd): Promise<void> {
  switch (cmd.type) {

    case "help": {
      console.log(`
${head("GristMill CLI — commands")}

  ${ok("remember")}  <text> [tags: t1, t2]   Store a memory
  ${ok("recall")}    <query> [limit N]        Search memories
  ${ok("get")}       <ulid>                   Fetch memory by ID
  ${ok("triage")}    <text>                   Get Sieve routing decision
  ${ok("health")}                             Server health
  ${ok("budget")}                             LLM token budget snapshot
  ${ok("pipelines")}                          List registered pipelines
  ${ok("register")}  <id>  OR  <json>         Register a pipeline
  ${ok("run")}       <pipeline-id> [json]     Run a pipeline

  ${ok("help")}  ${ok("exit")}

  ${dim("Unrecognised input is treated as a recall query.")}
  ${dim("Natural variations are understood, e.g. \"search for VPN\", \"store ...\"")}
`);
      break;
    }

    case "remember": {
      const res = await api<{ id: string }>("POST", "/api/memory/remember", {
        content: cmd.content,
        tags: cmd.tags,
      });
      console.log(`  ${ok("✓ Stored")} ${dim(res.id)}`);
      if (cmd.tags.length) console.log(dim(`    tags: ${cmd.tags.join(", ")}`));
      break;
    }

    case "recall": {
      interface RecallResult {
        results: Array<{
          memory: { id: string; content: string; tags: string[]; tier: string };
          score: number;
          sources: string[];
        }>;
      }
      const res = await api<RecallResult>("POST", "/api/memory/recall", {
        query: cmd.query,
        limit: cmd.limit,
      });
      if (res.results.length === 0) {
        console.log(warn(`  No results for "${cmd.query}".`));
      } else {
        console.log(head(`\n  ${res.results.length} result(s) for "${cmd.query}":`));
        for (const r of res.results) {
          const pct = `${(r.score * 100).toFixed(0)}%`;
          console.log(`  ${hi(pct.padStart(4))}  ${r.memory.content}`);
          const meta = [
            r.memory.tier,
            r.memory.tags.length ? r.memory.tags.join(", ") : null,
            r.memory.id,
          ].filter(Boolean).join("  ·  ");
          console.log(dim(`          ${meta}`));
        }
        console.log();
      }
      break;
    }

    case "get": {
      interface MemoryRes {
        id: string; content: string; tags: string[];
        tier: string; createdAtMs: number;
      }
      const res = await api<MemoryRes | null>("GET", `/api/memory/${cmd.id}`);
      if (!res) {
        console.log(warn("  Memory not found."));
      } else {
        console.log(`\n  ${head("Memory")}  ${res.content}`);
        console.log(dim(`  tier: ${res.tier}  ·  tags: ${res.tags.join(", ") || "—"}  ·  ${new Date(res.createdAtMs).toLocaleString()}`));
        console.log();
      }
      break;
    }

    case "triage": {
      interface RouteDecision {
        route: string; confidence: number; modelId?: string; reason?: string;
      }
      const res = await api<RouteDecision>("POST", "/api/triage", {
        channel: cmd.channel,
        text: cmd.text,
      });
      const filled  = Math.round(res.confidence * 20);
      const bar     = ok("█".repeat(filled)) + dim("░".repeat(20 - filled));
      const routeColour = res.route === "LLM_NEEDED" ? warn(res.route) : ok(res.route);
      console.log(`\n  ${head("Route")}       ${routeColour}`);
      console.log(`  ${head("Confidence")}  ${bar}  ${(res.confidence * 100).toFixed(1)}%`);
      if (res.modelId) console.log(dim(`  model:  ${res.modelId}`));
      if (res.reason)  console.log(dim(`  reason: ${res.reason}`));
      console.log();
      break;
    }

    case "health": {
      interface HealthRes { status: string; uptime: number; timestamp: string }
      const res = await api<HealthRes>("GET", "/api/metrics/health");
      console.log(`  ${ok("✓")} ${res.status}  ${dim(`uptime ${res.uptime.toFixed(1)}s  ·  ${res.timestamp}`)}`);
      break;
    }

    case "budget": {
      interface BudgetRes {
        daily_used: number; daily_limit: number; pct_used: number; cached_at: string;
      }
      const res = await api<BudgetRes | null>("GET", "/api/metrics/budget");
      if (!res) {
        console.log(warn("  No budget data yet — no LLM calls have been made."));
      } else {
        const filled = Math.round(res.pct_used / 5);
        const bar    = ok("█".repeat(filled)) + dim("░".repeat(20 - filled));
        console.log(`\n  ${bar}  ${res.pct_used.toFixed(1)}%`);
        console.log(dim(`  ${res.daily_used.toLocaleString()} / ${res.daily_limit.toLocaleString()} tokens  ·  updated ${res.cached_at}`));
        console.log();
      }
      break;
    }

    case "pipelines": {
      interface PipelinesRes { pipelines: string[] }
      const res = await api<PipelinesRes>("GET", "/api/pipelines");
      if (res.pipelines.length === 0) {
        console.log(warn("  No pipelines registered."));
      } else {
        console.log(head("\n  Registered pipelines:"));
        for (const id of res.pipelines) console.log(`    • ${id}`);
        console.log();
      }
      break;
    }

    case "register": {
      interface RegisterRes { registered: boolean; id: string }
      const res = await api<RegisterRes>("POST", "/api/pipelines", cmd.pipeline);
      console.log(`  ${ok("✓ Registered")} ${head(res.id)}`);
      break;
    }

    case "run": {
      interface RunRes {
        pipelineId: string;
        result: { succeeded: boolean; elapsedMs: number; output: unknown };
      }
      const res = await api<RunRes>(
        "POST",
        `/api/pipelines/${cmd.pipelineId}/run`,
        cmd.payload,
      );
      const icon = res.result.succeeded ? ok("✓") : fail("✗");
      console.log(`  ${icon} ${head(res.pipelineId)}  ${dim(`${res.result.elapsedMs}ms`)}`);
      if (res.result.output && Object.keys(res.result.output as object).length) {
        console.log(dim("  output: " + JSON.stringify(res.result.output, null, 2).replace(/\n/g, "\n  ")));
      }
      break;
    }

    case "exit":
      break; // handled in main loop
  }
}

// ── Main REPL loop ────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { port, host } = parseArgs();
  BASE = `http://${host}:${port}`;

  // Verify server is reachable before entering the loop
  try {
    await api("GET", "/api/metrics/health");
  } catch {
    console.error(fail(`\n✗ Cannot reach GristMill at ${BASE}`));
    console.error(dim("  Start the server:  GRISTMILL_MOCK_BRIDGE=1 pnpm dev\n"));
    process.exit(1);
  }

  console.log(`
${head("GristMill")} ${dim(`→ ${BASE}`)}
${dim("Type")} ${ok("help")} ${dim("for commands, or just type naturally.")}
`);

  const iface = rlPromises.createInterface({
    input:  process.stdin,
    output: process.stdout,
  });

  // eslint-disable-next-line no-constant-condition
  while (true) {
    let line: string;
    try {
      line = await iface.question(`${c.cyan}gristmill${c.reset}${c.dim}>${c.reset} `);
    } catch {
      break; // Ctrl-D
    }

    const cmd = parse(line);
    if (!cmd) continue;
    if (cmd.type === "exit") break;

    try {
      await execute(cmd);
    } catch (e) {
      console.error(fail(`  Error: ${e instanceof Error ? e.message : String(e)}`));
    }
  }

  iface.close();
  console.log(dim("\nBye.\n"));
}

main().catch((e) => {
  console.error(fail(String(e)));
  process.exit(1);
});
