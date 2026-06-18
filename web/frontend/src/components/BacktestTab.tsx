import { useState, useEffect } from "react";
import type { BacktestMetrics } from "../types";
import { fetchBacktest } from "../api";

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
      {TOURNAMENTS.map((t) => results[t] && <MetricsTable key={t} name={t} m={results[t]} />)}
    </div>
  );
}
