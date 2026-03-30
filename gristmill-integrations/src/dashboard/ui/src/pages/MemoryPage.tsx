import { useState } from "react";
import { api, type RankedMemoryItem } from "../api.js";

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius)",
      padding: 20,
    }}>
      <div style={{
        color: "var(--text-muted)",
        fontSize: 12,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        marginBottom: 12,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

export default function MemoryPage() {
  // Search state
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RankedMemoryItem[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Remember form state
  const [rememberContent, setRememberContent] = useState("");
  const [rememberTags, setRememberTags] = useState("");
  const [remembering, setRemembering] = useState(false);
  const [rememberSuccess, setRememberSuccess] = useState<string | null>(null);
  const [rememberError, setRememberError] = useState<string | null>(null);

  async function handleSearch() {
    if (!query.trim()) return;
    setSearching(true);
    setSearchError(null);
    setResults(null);
    try {
      const data = await api.memoryRecall(query.trim());
      setResults(data);
    } catch (e: unknown) {
      setSearchError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

  async function handleRemember() {
    if (!rememberContent.trim()) return;
    setRemembering(true);
    setRememberSuccess(null);
    setRememberError(null);
    try {
      const tags = rememberTags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const res = await api.memoryRemember(rememberContent.trim(), tags);
      setRememberSuccess(`Stored with id: ${res.id}`);
      setRememberContent("");
      setRememberTags("");
    } catch (e: unknown) {
      setRememberError(e instanceof Error ? e.message : String(e));
    } finally {
      setRemembering(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 20, gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))" }}>
      <Card title="Search Memory">
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void handleSearch(); }}
            placeholder="Search query…"
            style={{
              flex: 1,
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              color: "var(--text)",
              padding: "6px 10px",
              fontSize: 13,
            }}
          />
          <button
            onClick={() => void handleSearch()}
            disabled={searching || !query.trim()}
            style={{
              background: "var(--accent)",
              border: "none",
              borderRadius: "var(--radius)",
              color: "#fff",
              padding: "6px 14px",
              fontWeight: 600,
              cursor: searching || !query.trim() ? "not-allowed" : "pointer",
              opacity: searching || !query.trim() ? 0.6 : 1,
            }}
          >
            {searching ? "Searching…" : "Search"}
          </button>
        </div>

        {searchError && (
          <p style={{ color: "var(--red)", fontSize: 13, margin: "0 0 8px" }}>
            Error: {searchError}
          </p>
        )}

        {results !== null && results.length === 0 && (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>No results found.</p>
        )}

        {results !== null && results.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {results.map((r) => (
              <div
                key={r.memory.id}
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  padding: 12,
                  background: "var(--bg)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{
                    background: "var(--accent-dim)",
                    color: "var(--accent)",
                    borderRadius: 4,
                    padding: "1px 7px",
                    fontSize: 11,
                    fontWeight: 700,
                  }}>
                    {(r.score * 100).toFixed(0)}%
                  </span>
                  <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                    {r.memory.tier}
                  </span>
                </div>
                <p style={{ margin: "0 0 6px", fontSize: 13, lineHeight: 1.5 }}>
                  {r.memory.content.length > 150
                    ? r.memory.content.slice(0, 150) + "…"
                    : r.memory.content}
                </p>
                {r.memory.tags.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 6 }}>
                    {r.memory.tags.map((tag) => (
                      <span
                        key={tag}
                        style={{
                          background: "var(--border)",
                          borderRadius: 4,
                          padding: "1px 6px",
                          fontSize: 11,
                          color: "var(--text-muted)",
                        }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
                  id: {r.memory.id}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title="Remember">
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <textarea
            value={rememberContent}
            onChange={(e) => setRememberContent(e.target.value)}
            placeholder="Content to remember…"
            rows={4}
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              color: "var(--text)",
              padding: "8px 10px",
              fontSize: 13,
              resize: "vertical",
              fontFamily: "inherit",
            }}
          />
          <input
            value={rememberTags}
            onChange={(e) => setRememberTags(e.target.value)}
            placeholder="Tags (comma-separated)"
            style={{
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              color: "var(--text)",
              padding: "6px 10px",
              fontSize: 13,
            }}
          />
          <button
            onClick={() => void handleRemember()}
            disabled={remembering || !rememberContent.trim()}
            style={{
              background: "var(--accent)",
              border: "none",
              borderRadius: "var(--radius)",
              color: "#fff",
              padding: "7px 14px",
              fontWeight: 600,
              cursor: remembering || !rememberContent.trim() ? "not-allowed" : "pointer",
              opacity: remembering || !rememberContent.trim() ? 0.6 : 1,
            }}
          >
            {remembering ? "Storing…" : "Remember"}
          </button>

          {rememberSuccess && (
            <p style={{ color: "var(--green, #22c55e)", fontSize: 13, margin: 0 }}>
              {rememberSuccess}
            </p>
          )}
          {rememberError && (
            <p style={{ color: "var(--red)", fontSize: 13, margin: 0 }}>
              Error: {rememberError}
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}
