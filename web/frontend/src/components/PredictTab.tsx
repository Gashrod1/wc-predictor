import { useState, useEffect } from "react";
import type { TeamInfo, PredictResponse } from "../types";
import { predict } from "../api";
import TeamSelect from "./TeamSelect";
import ResultCard from "./ResultCard";

const STAGES = [
  { value: "group", label: "Phase de groupes" },
  { value: "round_of_32", label: "16es de finale" },
  { value: "round_of_16", label: "8es de finale" },
  { value: "quarter_final", label: "Quarts" },
  { value: "semi_final", label: "Demi-finales" },
  { value: "third_place", label: "Petite finale" },
  { value: "final", label: "Finale" },
];

interface Props {
  teams: TeamInfo[];
  prefill: { home: string; away: string; stage: string } | null;
}

export default function PredictTab({ teams, prefill }: Props) {
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [stage, setStage] = useState("group");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runPredict(h: string, a: string, s: string) {
    setLoading(true);
    setError("");
    try {
      const r = await predict(h, a, s);
      setResult(r);
    } catch (e) {
      setError("Erreur lors de la prédiction. L'API est-elle lancée ?");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (prefill) {
      setHome(prefill.home);
      setAway(prefill.away);
      setStage(prefill.stage);
      void runPredict(prefill.home, prefill.away, prefill.stage);
    }
  }, [prefill]);

  return (
    <div>
      <div className="card">
        <div className="row">
          <TeamSelect label="Équipe à domicile" teams={teams} value={home} onChange={setHome} />
          <TeamSelect label="Équipe à l'extérieur" teams={teams} value={away} onChange={setAway} />
          <div>
            <label>Phase</label>
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              {STAGES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div style={{ marginTop: 16 }}>
          <button
            className="primary"
            disabled={!home || !away || home === away || loading}
            onClick={() => runPredict(home, away, stage)}
          >
            {loading ? "Calcul…" : "Prédire"}
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}
      {result && <ResultCard result={result} />}
    </div>
  );
}
