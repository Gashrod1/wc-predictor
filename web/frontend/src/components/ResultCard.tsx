import type { PredictResponse } from "../types";
import ProbabilityBar from "./ProbabilityBar";

interface Props {
  result: PredictResponse;
}

export default function ResultCard({ result }: Props) {
  const p = result.outcome_probabilities;
  const winnerName =
    result.predicted_winner === "home"
      ? result.home_team
      : result.predicted_winner === "away"
        ? result.away_team
        : "Match nul";
  const winnerPct =
    Math.max(p.home_win, p.draw, p.away_win) * 100;
  const confPct = Math.round(result.confidence * 100);

  return (
    <div className="card">
      <div className="winner-banner">
        Vainqueur probable : <span className="accent">{winnerName}</span> (
        {winnerPct.toFixed(1)}%)
      </div>

      <ProbabilityBar label={result.home_team} value={p.home_win} color="var(--home)" />
      <ProbabilityBar label="Match nul" value={p.draw} color="var(--draw)" />
      <ProbabilityBar label={result.away_team} value={p.away_win} color="var(--away)" />

      <h3 style={{ marginTop: 20, marginBottom: 8 }}>
        Score le plus probable : {result.most_likely_score}
      </h3>
      <ul className="score-list">
        {result.top_scores.map((s, i) => (
          <li key={s.score}>
            <span>
              {i + 1}. {s.score}
            </span>
            <span>{(s.probability * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>

      <div style={{ marginTop: 20 }}>
        <label>Confiance : {confPct}%</label>
        <div className="confidence-track">
          <div className="confidence-fill" style={{ width: `${confPct}%` }} />
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        Accord des modèles :{" "}
        {result.model_agreement ? (
          <span className="badge ok">✓ Oui</span>
        ) : (
          <span className="badge no">✗ Non</span>
        )}
      </div>
    </div>
  );
}
