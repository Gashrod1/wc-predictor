import { useState, useEffect } from "react";
import type { BacktestMetrics, BacktestDetails, BacktestMatchDetail } from "../types";
import { fetchBacktest, fetchBacktestDetails } from "../api";

const TOURNAMENTS = ["WC2018", "WC2022", "WC2026"];

function MetricsTable({ name, m }: { name: string; m: BacktestMetrics }) {
  return (
    <div className="card">
      <h3>{name}</h3>
      <table>
        <tbody>
          <tr>
            <th>Précision vainqueur</th>
            <td>{(m.outcome_accuracy * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <th>Score exact</th>
            <td>{(m.exact_score_accuracy * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <th>Score dans top 3</th>
            <td>{(m.top3_score_accuracy * 100).toFixed(1)}%</td>
          </tr>
          <tr>
            <th>Brier score</th>
            <td>{m.brier_score.toFixed(4)}</td>
          </tr>
          <tr>
            <th>Log loss</th>
            <td>{m.log_loss.toFixed(4)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function winnerLabel(w: string, home: string, away: string): string {
  if (w === "home") return home;
  if (w === "away") return away;
  return "Nul";
}

function MatchRow({ m }: { m: BacktestMatchDetail }) {
  const correct = m.outcome_correct;
  return (
    <tr style={{ color: correct ? "var(--green, #22c55e)" : "var(--red, #ef4444)" }}>
      <td style={{ whiteSpace: "nowrap", paddingRight: "0.75rem" }}>{m.date}</td>
      <td style={{ paddingRight: "0.75rem" }}>
        {m.home_team} vs {m.away_team}
      </td>
      <td style={{ textAlign: "center", paddingRight: "0.75rem" }}>
        {m.predicted_score}
        <span style={{ fontSize: "0.75em", opacity: 0.7, marginLeft: "0.25rem" }}>
          ({winnerLabel(m.predicted_winner, m.home_team, m.away_team)})
        </span>
      </td>
      <td style={{ textAlign: "center", paddingRight: "0.75rem" }}>
        {m.actual_score}
      </td>
      <td style={{ textAlign: "center" }}>
        {correct ? "✓" : "✗"}
        {m.score_correct && (
          <span style={{ marginLeft: "0.35rem", fontSize: "0.75em", opacity: 0.8 }}>score exact</span>
        )}
      </td>
    </tr>
  );
}

function DetailsDropdown({ tournament }: { tournament: string }) {
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState<BacktestDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function toggle() {
    if (!open && !details && !loading) {
      setLoading(true);
      fetchBacktestDetails(tournament)
        .then(setDetails)
        .catch(() => setError("Impossible de charger le détail."))
        .finally(() => setLoading(false));
    }
    setOpen((o) => !o);
  }

  const correct = details ? details.matches.filter((m) => m.outcome_correct).length : 0;
  const total = details ? details.matches.length : 0;

  return (
    <div style={{ marginTop: "0.5rem" }}>
      <button
        onClick={toggle}
        style={{
          background: "none",
          border: "1px solid var(--border, #374151)",
          borderRadius: "6px",
          color: "inherit",
          cursor: "pointer",
          padding: "0.4rem 0.8rem",
          fontSize: "0.875rem",
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
        }}
      >
        <span>{open ? "▾" : "▸"}</span>
        Détail des prédictions
        {details && (
          <span style={{ marginLeft: "0.5rem", opacity: 0.7 }}>
            ({correct}/{total} correctes)
          </span>
        )}
      </button>

      {open && (
        <div style={{ marginTop: "0.5rem", overflowX: "auto" }}>
          {loading && <div style={{ padding: "0.5rem", opacity: 0.7 }}>Chargement…</div>}
          {error && <div style={{ color: "var(--red, #ef4444)", padding: "0.5rem" }}>{error}</div>}
          {details && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
              <thead>
                <tr style={{ opacity: 0.6, textAlign: "left" }}>
                  <th style={{ paddingRight: "0.75rem", paddingBottom: "0.35rem" }}>Date</th>
                  <th style={{ paddingRight: "0.75rem", paddingBottom: "0.35rem" }}>Match</th>
                  <th style={{ textAlign: "center", paddingRight: "0.75rem", paddingBottom: "0.35rem" }}>Prédit</th>
                  <th style={{ textAlign: "center", paddingRight: "0.75rem", paddingBottom: "0.35rem" }}>Réel</th>
                  <th style={{ textAlign: "center", paddingBottom: "0.35rem" }}>Résultat</th>
                </tr>
              </thead>
              <tbody>
                {details.matches.map((m, i) => (
                  <MatchRow key={i} m={m} />
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

export default function BacktestTab() {
  const [results, setResults] = useState<Record<string, BacktestMetrics>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all(TOURNAMENTS.map((t) => fetchBacktest(t).then((m) => [t, m] as const)))
      .then((pairs) => {
        const out: Record<string, BacktestMetrics> = {};
        for (const [t, m] of pairs) out[t] = m;
        setResults(out);
      })
      .catch(() => setError("Impossible de charger les métriques."))
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return <div className="loading">Calcul des métriques de backtesting (peut prendre ~1 min)…</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div>
      {TOURNAMENTS.map(
        (t) =>
          results[t] && (
            <div key={t}>
              <MetricsTable name={t} m={results[t]} />
              <DetailsDropdown tournament={t} />
            </div>
          ),
      )}
    </div>
  );
}
