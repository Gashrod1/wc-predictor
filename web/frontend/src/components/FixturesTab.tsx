import { useState, useEffect } from "react";
import type { FixtureItem } from "../types";
import { fetchFixtures } from "../api";

interface Props {
  onPredict: (home: string, away: string, stage: string) => void;
}

const STAGE_LABELS: Record<string, string> = {
  group: "Phase de groupes",
  round_of_32: "16es de finale",
  round_of_16: "8es de finale",
  quarter_final: "Quarts",
  semi_final: "Demi-finales",
  third_place: "Petite finale",
  final: "Finale",
};

function groupKey(f: FixtureItem): string {
  if (f.stage === "group") return `Journée ${f.matchday || "?"}`;
  return STAGE_LABELS[f.stage] ?? f.stage;
}

function FixtureResult({ f }: { f: FixtureItem }) {
  const correct = f.outcome_correct;

  if (f.status === "Joué" && f.actual_score) {
    return (
      <div className="fixture-result">
        <div
          className={`fixture-score-block ${correct === true ? "result-correct" : correct === false ? "result-wrong" : ""}`}
        >
          <span className="fixture-score actual-score">{f.actual_score}</span>
          <span className="fixture-score-label">réel</span>
        </div>
        {f.predicted_score && (
          <div className="fixture-score-block">
            <span className="fixture-score predicted-score">{f.predicted_score}</span>
            <span className="fixture-score-label">prédit</span>
          </div>
        )}
        {correct !== null && (
          <span className={`badge ${correct ? "ok" : "no"}`}>
            {correct ? "✓" : "✗"}
          </span>
        )}
      </div>
    );
  }

  if (f.status === "En direct") {
    return <span className="badge live">En direct</span>;
  }

  if (f.predictable) {
    return <span className="badge ok">Prédire →</span>;
  }

  return <span className="meta">à venir</span>;
}

export default function FixturesTab({ onPredict }: Props) {
  const [fixtures, setFixtures] = useState<FixtureItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchFixtures()
      .then((f) => setFixtures(f))
      .catch(() => setError("Impossible de charger le calendrier."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Chargement du calendrier…</div>;
  if (error) return <div className="error">{error}</div>;

  const groups: { title: string; items: FixtureItem[] }[] = [];
  for (const f of fixtures) {
    const title = groupKey(f);
    let g = groups.find((x) => x.title === title);
    if (!g) {
      g = { title, items: [] };
      groups.push(g);
    }
    g.items.push(f);
  }

  return (
    <div>
      {groups.map((g) => (
        <div key={g.title}>
          <h2 className="fixture-group-title">{g.title}</h2>
          {g.items.map((f, idx) => {
            const played = f.status === "Joué";
            const clickable = f.predictable && !played;
            const tbd = !f.predictable && !played;
            return (
              <div
                key={`${f.date}-${f.time}-${idx}`}
                className={`fixture ${tbd ? "disabled" : ""} ${
                  played && f.outcome_correct === true
                    ? "fixture-correct"
                    : played && f.outcome_correct === false
                      ? "fixture-wrong"
                      : ""
                }`}
                style={{ cursor: clickable ? "pointer" : "default" }}
                onClick={() =>
                  clickable && onPredict(f.home_team, f.away_team, f.stage)
                }
              >
                <div>
                  <div className="teams">
                    {f.home_team} vs {f.away_team}
                  </div>
                  <div className="meta">
                    {f.date} {f.time} · {f.city || "—"} {f.stadium ? `· ${f.stadium}` : ""}
                  </div>
                </div>
                <FixtureResult f={f} />
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
