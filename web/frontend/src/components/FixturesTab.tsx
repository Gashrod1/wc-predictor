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
          {g.items.map((f, idx) => (
            <div
              key={`${f.date}-${f.time}-${idx}`}
              className={`fixture ${f.predictable ? "" : "disabled"}`}
              onClick={() =>
                f.predictable && onPredict(f.home_team, f.away_team, f.stage)
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
              {f.predictable ? (
                <span className="badge ok">Prédire →</span>
              ) : (
                <span className="meta">à venir</span>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
