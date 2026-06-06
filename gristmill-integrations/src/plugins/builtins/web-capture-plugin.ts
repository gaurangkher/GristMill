/**
 * web-capture-plugin — Personal Knowledge Capture from the Web.
 *
 * Registers:
 *   adapter    "web-capture"       — normalises URL payloads into GristEvents
 *   step type  "fetch-page"        — fetches a URL and strips HTML to plain text
 *   step type  "summarize-local"   — summarises text via a local Ollama model
 *   step type  "triage-tags"       — maps triage route decision to content tags
 *   step type  "store-article"     — stores summary + metadata into the Ledger
 */

import { createPlugin } from "../sdk.js";
import type { StepContext, StepResult } from "../types.js";

// ── Domain tag map ────────────────────────────────────────────────────────────

const DOMAIN_TAGS: Record<string, string> = {
  "arxiv.org": "research",
  "github.com": "code",
  "gitlab.com": "code",
  "stackoverflow.com": "code",
  "news.ycombinator.com": "hackernews",
  "reddit.com": "reddit",
  "medium.com": "article",
  "substack.com": "newsletter",
  "twitter.com": "social",
  "x.com": "social",
  "youtube.com": "video",
  "wikipedia.org": "wiki",
};

// ── Helper: extract domain from URL ──────────────────────────────────────────

function extractDomain(urlStr: string): string {
  try {
    return new URL(urlStr).hostname.replace(/^www\./, "");
  } catch {
    return "unknown";
  }
}

// ── Helper: strip HTML to plain text ─────────────────────────────────────────

function stripHtml(html: string): string {
  // Remove entire script/style/nav/footer/header blocks
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
    .replace(/<footer[\s\S]*?<\/footer>/gi, " ")
    .replace(/<header[\s\S]*?<\/header>/gi, " ");

  // Remove remaining HTML tags
  text = text.replace(/<[^>]+>/g, " ");

  // Decode common HTML entities
  text = text
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&nbsp;/gi, " ");

  // Collapse whitespace and trim
  return text.replace(/\s+/g, " ").trim();
}

// ── Step: fetch-page ──────────────────────────────────────────────────────────

async function fetchPage(ctx: StepContext): Promise<StepResult> {
  const input = ctx.input as { url: string; title?: string; note?: string };
  const { url, title, note } = input;
  const domain = extractDomain(url);
  const fetchedAt = new Date().toISOString();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10_000);

    let response: Response;
    try {
      response = await fetch(url, {
        signal: controller.signal,
        headers: {
          "User-Agent": "GristMill-WebCapture/1.0",
        },
      });
    } finally {
      clearTimeout(timeoutId);
    }

    if (!response.ok) {
      const error = `HTTP ${response.status} ${response.statusText}`;
      return {
        output: {
          url,
          title: title ?? url,
          text: note ?? "",
          wordCount: 0,
          domain,
          fetchedAt,
          error,
        },
      };
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("text/html") && !contentType.includes("text/plain")) {
      return {
        output: {
          url,
          title: title ?? url,
          text: note ?? "",
          wordCount: 0,
          domain,
          fetchedAt,
          error: `Unsupported content type: ${contentType}`,
        },
      };
    }

    const html = await response.text();
    let text = stripHtml(html);

    // Truncate to 8000 chars (Phi-3 context window limit)
    if (text.length > 8000) {
      text = text.slice(0, 8000);
    }

    // Try to extract <title> tag if no title was provided
    let resolvedTitle = title;
    if (!resolvedTitle) {
      const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
      if (titleMatch && titleMatch[1]) {
        resolvedTitle = titleMatch[1].trim();
      } else {
        resolvedTitle = url;
      }
    }

    const wordCount = text.split(/\s+/).filter(Boolean).length;

    return {
      output: {
        url,
        title: resolvedTitle,
        text,
        wordCount,
        domain,
        fetchedAt,
      },
    };
  } catch (err: unknown) {
    const message =
      err instanceof Error ? err.message : "Unknown fetch error";
    return {
      output: {
        url,
        title: title ?? url,
        text: note ?? "",
        wordCount: 0,
        domain,
        fetchedAt,
        error: message,
      },
    };
  }
}

// ── Step: summarize-local ─────────────────────────────────────────────────────

async function summarizeLocal(ctx: StepContext): Promise<StepResult> {
  const input = ctx.input as {
    url: string;
    title?: string;
    text: string;
    wordCount: number;
    domain: string;
    fetchedAt: string;
    error?: string;
  };
  const { url, title, text, wordCount } = input;

  // Skip summarization if text is too short
  if (!text || wordCount < 20) {
    return {
      output: {
        summary: title ?? url,
        skipped: true,
      },
    };
  }

  const ollamaHost =
    process.env["OLLAMA_HOST"] ?? "http://localhost:11434";
  const model = process.env["WEB_CAPTURE_MODEL"] ?? "phi3:mini";

  const prompt =
    `Summarise the following article in 3-5 sentences. Focus on key facts, findings, and takeaways. Be concise.\n\nTitle: ${title ?? url}\n\nContent:\n${text}\n\nSummary:`;

  const startMs = Date.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30_000);

    let response: Response;
    try {
      response = await fetch(`${ollamaHost}/api/generate`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt,
          stream: false,
          options: { temperature: 0.1, num_predict: 300 },
        }),
      });
    } finally {
      clearTimeout(timeoutId);
    }

    if (!response.ok) {
      throw new Error(`Ollama responded with HTTP ${response.status}`);
    }

    const data = (await response.json()) as { response?: string };
    const summary = (data.response ?? "").trim() || text.slice(0, 500);
    const elapsedMs = Date.now() - startMs;

    return {
      output: {
        summary,
        model,
        elapsedMs,
      },
    };
  } catch {
    // Fallback: use first 500 chars of text
    const summary = text.slice(0, 500);
    const elapsedMs = Date.now() - startMs;
    return {
      output: {
        summary,
        model,
        elapsedMs,
        fallback: true,
      },
    };
  }
}

// ── Step: triage-tags ─────────────────────────────────────────────────────────

async function triageTags(ctx: StepContext): Promise<StepResult> {
  const input = ctx.input as {
    url: string;
    title?: string;
    text: string;
    domain: string;
    summary: string;
  };
  const { domain, summary } = input;

  try {
    const decision = await ctx.bridge.triage({
      channel: "webhook",
      payload: { text: summary },
    });

    // Map route to content tags
    const routeTagMap: Record<string, string[]> = {
      LOCAL_ML: ["local", "simple"],
      RULES: ["structured", "rules"],
      HYBRID: ["technical", "mixed"],
      LLM_NEEDED: ["complex", "deep-dive"],
    };
    const routeTags = routeTagMap[decision.route] ?? ["web"];

    // Add domain-based tag
    const domainTag = DOMAIN_TAGS[domain] ?? "web";
    const tags = [...routeTags, domainTag];

    return {
      output: {
        tags,
        route: decision.route,
        confidence: decision.confidence,
      },
    };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "triage error";
    return {
      output: {
        tags: ["web"],
        route: "LOCAL_ML",
        confidence: 0,
        error: message,
      },
    };
  }
}

// ── Step: store-article ───────────────────────────────────────────────────────

async function storeArticle(ctx: StepContext): Promise<StepResult> {
  const input = ctx.input as {
    url: string;
    title?: string;
    domain: string;
    summary: string;
    tags: string[];
    route: string;
    confidence: number;
    wordCount: number;
    fallback?: boolean;
  };

  const {
    url,
    title,
    domain,
    summary,
    tags,
    wordCount,
    fallback,
  } = input;

  try {
    const content = [
      title ?? url,
      `URL: ${url}`,
      `Domain: ${domain}`,
      `Words: ${wordCount}`,
      "",
      summary,
    ].join("\n");

    const allTags: string[] = ["web-capture", domain, ...tags];
    if (fallback === true) {
      allTags.push("fallback-summary");
    }

    const memoryId = await ctx.bridge.remember(content, allTags);

    return {
      output: {
        memoryId,
        url,
        title: title ?? url,
        tags: allTags,
        summary,
      },
    };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "store error";
    return {
      output: {
        error: message,
        url,
        title: title ?? url,
      },
    };
  }
}

// ── Plugin export ─────────────────────────────────────────────────────────────

export default createPlugin({
  name: "gristmill/web-capture",
  version: "1.0.0",
  register(ctx) {
    // Adapter: normalises URL payloads into GristEvents
    ctx.registerAdapter("web-capture", (rawEvent) => {
      const raw = rawEvent as {
        url?: unknown;
        title?: unknown;
        note?: unknown;
        source?: unknown;
      };

      if (!raw.url || typeof raw.url !== "string") {
        throw new Error(
          "web-capture adapter: 'url' field is required and must be a string"
        );
      }

      // Validate URL
      try {
        new URL(raw.url);
      } catch {
        throw new Error(
          `web-capture adapter: invalid URL '${raw.url}'`
        );
      }

      const domain = extractDomain(raw.url);

      return {
        channel: "webhook",
        priority: "normal",
        payload: {
          url: raw.url,
          title: typeof raw.title === "string" ? raw.title : undefined,
          note: typeof raw.note === "string" ? raw.note : undefined,
          source: typeof raw.source === "string" ? raw.source : undefined,
        },
        tags: {
          plugin: "web-capture",
          domain,
        },
      };
    });

    // Step type: fetch-page
    ctx.registerStepType("fetch-page", (stepCtx: StepContext) =>
      fetchPage(stepCtx)
    );

    // Step type: summarize-local
    ctx.registerStepType("summarize-local", (stepCtx: StepContext) =>
      summarizeLocal(stepCtx)
    );

    // Step type: triage-tags
    ctx.registerStepType("triage-tags", (stepCtx: StepContext) =>
      triageTags(stepCtx)
    );

    // Step type: store-article
    ctx.registerStepType("store-article", (stepCtx: StepContext) =>
      storeArticle(stepCtx)
    );

    ctx.log("info", "web-capture plugin registered (fetch-page, summarize-local, triage-tags, store-article)");
  },
});
