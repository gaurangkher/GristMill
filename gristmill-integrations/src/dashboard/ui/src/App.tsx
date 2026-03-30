import { useState } from "react";
import OverviewPage from "./pages/OverviewPage.js";
import TrainerPage from "./pages/TrainerPage.js";
import PipelinesPage from "./pages/PipelinesPage.js";
import EcosystemPage from "./pages/EcosystemPage.js";
import MemoryPage from "./pages/MemoryPage.js";
import MetricsPage from "./pages/MetricsPage.js";
import WatchesPage from "./pages/WatchesPage.js";

type Tab = "overview" | "trainer" | "pipelines" | "ecosystem" | "memory" | "metrics" | "watches";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview",   label: "Overview" },
  { id: "trainer",    label: "Trainer" },
  { id: "pipelines",  label: "Pipelines" },
  { id: "ecosystem",  label: "Ecosystem" },
  { id: "memory",     label: "Memory" },
  { id: "metrics",    label: "Metrics" },
  { id: "watches",    label: "Watches" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <header style={{
        background: "var(--surface)",
        borderBottom: "1px solid var(--border)",
        padding: "0 24px",
        display: "flex",
        alignItems: "center",
        gap: 32,
        height: 52,
      }}>
        <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.02em" }}>
          ⚙ GristMill
        </span>
        <nav style={{ display: "flex", gap: 4 }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                background: tab === t.id ? "var(--accent-dim)" : "transparent",
                border: "none",
                color: tab === t.id ? "var(--accent)" : "var(--text-muted)",
                fontWeight: tab === t.id ? 600 : 400,
                padding: "6px 14px",
                borderRadius: "var(--radius)",
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main style={{ flex: 1, padding: 24 }}>
        {tab === "overview"  && <OverviewPage />}
        {tab === "trainer"   && <TrainerPage />}
        {tab === "pipelines" && <PipelinesPage />}
        {tab === "ecosystem" && <EcosystemPage />}
        {tab === "memory"    && <MemoryPage />}
        {tab === "metrics"   && <MetricsPage />}
        {tab === "watches"   && <WatchesPage />}
      </main>

      <footer style={{
        borderTop: "1px solid var(--border)",
        padding: "10px 24px",
        color: "var(--text-muted)",
        fontSize: 12,
        textAlign: "center",
      }}>
        GristMill v2 — local-first AI orchestration
      </footer>
    </div>
  );
}
