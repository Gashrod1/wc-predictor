import { useState } from "react";
import type { TeamInfo, TeamDetail, H2HResponse } from "../types";
import { fetchTeam, fetchH2H } from "../api";
import TeamSelect from "./TeamSelect";

interface Props {
  teams: TeamInfo[];
}

function TeamColumn({ detail }: { detail: TeamDetail | null }) {
  if (!detail) return <div className="card">—</div>;
  return (
    <div className="card">
      <h3>{detail.name}</h3>
      <div className="stat-row">
        <span>ELO</span>
        <span>{Math.round(detail.elo)}</span>
      </div>
      <div className="stat-row">
        <span>Buts marqués (5)</span>
        <span>{detail.form_goals_scored.toFixed(2)}</span>
      </div>
      <div className="stat-row">
        <span>Buts encaissés (5)</span>
        <span>{detail.form_goals_conceded.toFixed(2)}</span>
      </div>
      <h4 style={{ marginTop: 12 }}>Derniers matchs</h4>
      {detail.last_matches.length === 0 && <div className="meta">Aucun historique</div>}
      {detail.last_matches.map((m, i) => (
        <div key={i} className="stat-row">
          <span>
            {m.home_team} {m.home_goals}-{m.away_goals} {m.away_team}
          </span>
          <span className="meta">{m.date}</span>
        </div>
      ))}
    </div>
  );
}

export default function TeamsTab({ teams }: Props) {
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [homeDetail, setHomeDetail] = useState<TeamDetail | null>(null);
  const [awayDetail, setAwayDetail] = useState<TeamDetail | null>(null);
  const [h2h, setH2H] = useState<H2HResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function compare() {
    setError("");
    setLoading(true);
    try {
      const [hd, ad, h] = await Promise.all([
        fetchTeam(home),
        fetchTeam(away),
        fetchH2H(home, away),
      ]);
      setHomeDetail(hd);
      setAwayDetail(ad);
      setH2H(h);
    } catch (e) {
      setError("Erreur lors du chargement des équipes.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card">
        <div className="row">
          <TeamSelect label="Équipe 1" teams={teams} value={home} onChange={setHome} />
          <TeamSelect label="Équipe 2" teams={teams} value={away} onChange={setAway} />
          <div style={{ alignSelf: "flex-end" }}>
            <button className="primary" disabled={!home || !away || home === away || loading} onClick={compare}>
              {loading ? "Chargement…" : "Comparer"}
            </button>
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {homeDetail && awayDetail && (
        <div className="comparator">
          <TeamColumn detail={homeDetail} />
          <div className="card" style={{ minWidth: 160 }}>
            <h4>Face-à-face</h4>
            {h2h && (
              <>
                <div className="stat-row">
                  <span>Victoires {h2h.home}</span>
                  <span>{(h2h.home_wins_pct * 100).toFixed(0)}%</span>
                </div>
                <div className="stat-row">
                  <span>Buts/match</span>
                  <span>{h2h.avg_goals.toFixed(1)}</span>
                </div>
                <div className="stat-row">
                  <span>Confrontations</span>
                  <span>{h2h.matches.length}</span>
                </div>
              </>
            )}
          </div>
          <TeamColumn detail={awayDetail} />
        </div>
      )}
    </div>
  );
}
