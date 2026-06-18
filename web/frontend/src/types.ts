export interface TeamInfo {
  name: string;
  elo: number;
}

export interface ScoreItem {
  score: string;
  probability: number;
}

export interface PredictResponse {
  home_team: string;
  away_team: string;
  stage: string;
  outcome_probabilities: { home_win: number; draw: number; away_win: number };
  predicted_winner: "home" | "draw" | "away";
  most_likely_score: string;
  top_scores: ScoreItem[];
  confidence: number;
  model_agreement: boolean;
}

export interface MatchSummary {
  date: string;
  home_team: string;
  away_team: string;
  home_goals: number;
  away_goals: number;
}

export interface TeamDetail {
  name: string;
  elo: number;
  form_goals_scored: number;
  form_goals_conceded: number;
  last_matches: MatchSummary[];
}

export interface H2HResponse {
  home: string;
  away: string;
  home_wins_pct: number;
  avg_goals: number;
  matches: MatchSummary[];
}

export interface FixtureItem {
  date: string;
  time: string;
  matchday: string;
  home_team: string;
  away_team: string;
  city: string;
  stadium: string;
  stage: string;
  predictable: boolean;
}

export interface BacktestMetrics {
  outcome_accuracy: number;
  exact_score_accuracy: number;
  top3_score_accuracy: number;
  brier_score: number;
  log_loss: number;
}
