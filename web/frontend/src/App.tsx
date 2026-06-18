import { useState, useEffect } from "react";
import type { TeamInfo } from "./types";
import { fetchTeams } from "./api";
import PredictTab from "./components/PredictTab";
import FixturesTab from "./components/FixturesTab";
import TeamsTab from "./components/TeamsTab";
import BacktestTab from "./components/BacktestTab";

type Tab = "predict" | "fixtures" | "teams" | "backtest";

export interface Prefill {
  home: string;
  away: string;
  stage: string;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [teams, setTeams] = useState<TeamInfo[]>([]);
  const [prefill, setPrefill] = useState<Prefill | null>(null);

  useEffect(() => {
    fetchTeams()
      .then(setTeams)
      .catch(() => setTeams([]));
  }, []);

  function predictFixture(home: string, away: string, stage: string) {
    setPrefill({ home, away, stage });
    setTab("predict");
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          World Cup <span className="accent">Predictor</span>
        </h1>
      </header>

      <nav className="tabs">
        <button className={`tab-btn ${tab === "predict" ? "active" : ""}`} onClick={() => setTab("predict")}>
          Prédire
        </button>
        <button className={`tab-btn ${tab === "fixtures" ? "active" : ""}`} onClick={() => setTab("fixtures")}>
          Calendrier 2026
        </button>
        <button className={`tab-btn ${tab === "teams" ? "active" : ""}`} onClick={() => setTab("teams")}>
          Équipes
        </button>
        <button className={`tab-btn ${tab === "backtest" ? "active" : ""}`} onClick={() => setTab("backtest")}>
          Performance
        </button>
      </nav>

      {tab === "predict" && <PredictTab teams={teams} prefill={prefill} />}
      {tab === "fixtures" && <FixturesTab onPredict={predictFixture} />}
      {tab === "teams" && <TeamsTab teams={teams} />}
      {tab === "backtest" && <BacktestTab />}
    </div>
  );
}
