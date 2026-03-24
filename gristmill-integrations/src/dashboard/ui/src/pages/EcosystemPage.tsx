import { useEffect, useState, useCallback } from "react";
import { api, type EcosystemStatus } from "../api.js";

const KNOWN_DOMAINS = ["code", "writing", "reasoning", "qa", "creative", "other"];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 20 }}>
      <div style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 16 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}

export default function EcosystemPage() {
  const [status, setStatus] = useState<EcosystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, string>>({});

  const load = useCallback(() => {
    api.ecosystemStatus()
      .then(setStatus)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => { load(); }, [load]);

  const domainAction = async (
    domain: string,
    fn: () => Promise<unknown>,
    key: string,
  ) => {
    const busyKey = `${key}-${domain}`;
    setBusy(busyKey);
    setFeedback((f) => ({ ...f, [busyKey]: "" }));
    try {
      const res = await fn();
      setFeedback((f) => ({
        ...f,
        [busyKey]: `✓ ${JSON.stringify(res).slice(0, 120)}`,
      }));
    } catch (e: unknown) {
      setFeedback((f) => ({
        ...f,
        [busyKey]: `✗ ${e instanceof Error ? e.message : String(e)}`,
      }));
    } finally {
      setBusy(null);
    }
  };

  if (error) return <p style={{ color: "var(--red)" }}>Failed to load ecosystem status: {error}</p>;
  if (!status) return <p style={{ color: "var(--text-muted)" }}>Loading…</p>;

  const { community, federated } = status;
  const budget = federated.privacy_budget;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Community repo */}
      <Section title="Community Adapter Repository">
        <Row label="Enabled" value={
          community.enabled
            ? <span className="badge badge-green">Enabled</span>
            : <span className="badge badge-gray">Disabled</span>
        } />
        <Row label="Endpoint" value={<code style={{ fontSize: 12, color: "var(--text-muted)" }}>{community.endpoint}</code>} />
        {!community.enabled && (
          <p style={{ marginTop: 12, color: "var(--text-muted)", fontSize: 13 }}>
            Set <code>community.enabled: true</code> in <code>~/.gristmill/config.yaml</code> to enable adapter sharing.
          </p>
        )}
      </Section>

      {/* Federated learning */}
      <Section title="Federated Learning">
        <Row label="Enabled" value={
          federated.enabled
            ? <span className="badge badge-green">Enabled</span>
            : <span className="badge badge-gray">Disabled</span>
        } />
        <Row label="ε used" value={`${budget.epsilon_used.toFixed(3)} / ${budget.epsilon_budget.toFixed(1)}`} />
        <Row label="ε remaining" value={
          budget.exhausted
            ? <span className="badge badge-red">Exhausted</span>
            : <strong style={{ color: "var(--green)" }}>{budget.remaining.toFixed(3)}</strong>
        } />
        <Row label="Cycles contributed" value={budget.cycles_contributed} />

        {/* ε budget bar */}
        <div style={{ marginTop: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12, color: "var(--text-muted)" }}>
            <span>Privacy budget</span>
            <span>{((budget.epsilon_used / budget.epsilon_budget) * 100).toFixed(1)}% used</span>
          </div>
          <div style={{ height: 8, background: "var(--surface2)", borderRadius: 4, overflow: "hidden" }}>
            <div style={{
              width: `${Math.min(100, (budget.epsilon_used / budget.epsilon_budget) * 100)}%`,
              height: "100%",
              background: budget.exhausted ? "var(--red)" : budget.remaining < 2 ? "var(--yellow)" : "var(--accent)",
              transition: "width 0.4s",
            }} />
          </div>
        </div>
      </Section>

      {/* Per-domain actions */}
      <Section title="Domain Actions">
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 16 }}>
          Export, push to community, or bootstrap each domain adapter.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {KNOWN_DOMAINS.map((domain) => (
            <div key={domain} style={{
              display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
              padding: "10px 0", borderBottom: "1px solid var(--border)",
            }}>
              <span style={{ fontWeight: 600, width: 80 }}>{domain}</span>
              <button
                disabled={busy !== null}
                onClick={() => void domainAction(domain, () => api.ecosystemExport(domain), "export")}
              >
                Export
              </button>
              <button
                disabled={busy !== null || !community.enabled}
                title={community.enabled ? "" : "Enable community repo first"}
                onClick={() => void domainAction(domain, () => api.ecosystemPush(domain), "push")}
              >
                Push
              </button>
              <button
                disabled={busy !== null || !community.enabled}
                title={community.enabled ? "" : "Enable community repo first"}
                onClick={() => void domainAction(domain, () => api.ecosystemBootstrap(domain), "bootstrap")}
              >
                Bootstrap
              </button>
              {["export", "push", "bootstrap"].map((k) => {
                const key = `${k}-${domain}`;
                return feedback[key] ? (
                  <span key={k} style={{
                    fontSize: 12,
                    color: feedback[key].startsWith("✓") ? "var(--green)" : "var(--red)",
                    maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {feedback[key]}
                  </span>
                ) : null;
              })}
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
