import { useEffect, useState } from "react";
import { api, type WatchItem } from "../api.js";

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

const inputStyle: React.CSSProperties = {
  background: "var(--bg)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  color: "var(--text)",
  padding: "6px 10px",
  fontSize: 13,
  width: "100%",
  boxSizing: "border-box",
};

const labelStyle: React.CSSProperties = {
  display: "block",
  color: "var(--text-muted)",
  fontSize: 12,
  marginBottom: 4,
};

export default function WatchesPage() {
  const [watches, setWatches] = useState<WatchItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create form state
  const [name, setName] = useState("");
  const [topic, setTopic] = useState("");
  const [condition, setCondition] = useState("");
  const [channelIds, setChannelIds] = useState("");
  const [cooldownMs, setCooldownMs] = useState("0");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  async function loadWatches() {
    try {
      const data = await api.watchesList();
      setWatches(data);
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadWatches();
  }, []);

  async function handleCreate() {
    if (!name.trim() || !topic.trim() || !condition.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      const ids = channelIds
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const cooldown = parseInt(cooldownMs, 10) || 0;
      await api.watchCreate({
        name: name.trim(),
        topic: topic.trim(),
        condition: condition.trim(),
        channelIds: ids,
        enabled: true,
        cooldownMs: cooldown,
      });
      setName("");
      setTopic("");
      setCondition("");
      setChannelIds("");
      setCooldownMs("0");
      await loadWatches();
    } catch (e: unknown) {
      setCreateError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  }

  async function handleToggle(w: WatchItem) {
    try {
      await fetch(`/api/watches/${w.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !w.enabled }),
      });
      await loadWatches();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleDelete(id: string) {
    try {
      await api.watchDelete(id);
      await loadWatches();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Card title="Watch Rules">
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
          <button
            onClick={() => void loadWatches()}
            style={{
              background: "transparent",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              color: "var(--text-muted)",
              padding: "4px 10px",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            Refresh
          </button>
        </div>

        {error && (
          <p style={{ color: "var(--red)", fontSize: 13, marginBottom: 12 }}>
            Error: {error}
          </p>
        )}

        {loading ? (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</p>
        ) : watches.length === 0 ? (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>No watches configured yet.</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid var(--border)" }}>
                  {["Name", "Topic", "Condition", "Channels", "Enabled", "Actions"].map((h) => (
                    <th
                      key={h}
                      style={{
                        textAlign: "left",
                        padding: "6px 10px",
                        color: "var(--text-muted)",
                        fontWeight: 600,
                        fontSize: 11,
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {watches.map((w) => (
                  <tr
                    key={w.id}
                    style={{ borderBottom: "1px solid var(--border)" }}
                  >
                    <td style={{ padding: "8px 10px", fontWeight: 500 }}>{w.name}</td>
                    <td style={{ padding: "8px 10px", color: "var(--text-muted)" }}>
                      <code style={{ fontSize: 12 }}>{w.topic}</code>
                    </td>
                    <td style={{ padding: "8px 10px", color: "var(--text-muted)" }}>
                      <code style={{ fontSize: 12 }}>{w.condition}</code>
                    </td>
                    <td style={{ padding: "8px 10px" }}>
                      {w.channelIds.length > 0
                        ? w.channelIds.join(", ")
                        : <span style={{ color: "var(--text-muted)" }}>—</span>}
                    </td>
                    <td style={{ padding: "8px 10px" }}>
                      <button
                        onClick={() => void handleToggle(w)}
                        style={{
                          background: w.enabled ? "var(--accent)" : "var(--border)",
                          border: "none",
                          borderRadius: 12,
                          width: 36,
                          height: 20,
                          cursor: "pointer",
                          position: "relative",
                          transition: "background 0.2s",
                        }}
                        title={w.enabled ? "Disable" : "Enable"}
                        aria-label={w.enabled ? "Disable watch" : "Enable watch"}
                      >
                        <span style={{
                          position: "absolute",
                          top: 3,
                          left: w.enabled ? 18 : 3,
                          width: 14,
                          height: 14,
                          background: "#fff",
                          borderRadius: "50%",
                          transition: "left 0.2s",
                        }} />
                      </button>
                    </td>
                    <td style={{ padding: "8px 10px" }}>
                      <button
                        onClick={() => void handleDelete(w.id)}
                        style={{
                          background: "transparent",
                          border: "1px solid var(--border)",
                          borderRadius: "var(--radius)",
                          color: "var(--red, #ef4444)",
                          padding: "3px 8px",
                          fontSize: 12,
                          cursor: "pointer",
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Create Watch">
        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
          <div>
            <label style={labelStyle}>Name *</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. pipeline-failure-alert"
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Topic *</label>
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. pipeline.failed"
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Condition *</label>
            <input
              value={condition}
              onChange={(e) => setCondition(e.target.value)}
              placeholder="e.g. payload.reason != null"
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Channel IDs (comma-separated)</label>
            <input
              value={channelIds}
              onChange={(e) => setChannelIds(e.target.value)}
              placeholder="e.g. slack, email"
              style={inputStyle}
            />
          </div>
          <div>
            <label style={labelStyle}>Cooldown (ms)</label>
            <input
              type="number"
              value={cooldownMs}
              onChange={(e) => setCooldownMs(e.target.value)}
              min={0}
              style={inputStyle}
            />
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <button
            onClick={() => void handleCreate()}
            disabled={creating || !name.trim() || !topic.trim() || !condition.trim()}
            style={{
              background: "var(--accent)",
              border: "none",
              borderRadius: "var(--radius)",
              color: "#fff",
              padding: "7px 18px",
              fontWeight: 600,
              cursor: creating || !name.trim() || !topic.trim() || !condition.trim()
                ? "not-allowed"
                : "pointer",
              opacity: creating || !name.trim() || !topic.trim() || !condition.trim() ? 0.6 : 1,
            }}
          >
            {creating ? "Creating…" : "Create Watch"}
          </button>
        </div>

        {createError && (
          <p style={{ color: "var(--red)", fontSize: 13, marginTop: 10 }}>
            Error: {createError}
          </p>
        )}
      </Card>
    </div>
  );
}
