"""Pydantic request/response models for the web API."""
from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Body for POST /api/predict."""

    home: str
    away: str
    stage: str = "group"


class ScoreItem(BaseModel):
    """A single scoreline probability."""

    score: str
    probability: float


class PredictResponse(BaseModel):
    """Response for POST /api/predict."""

    home_team: str
    away_team: str
    stage: str
    outcome_probabilities: dict[str, float]
    predicted_winner: str
    most_likely_score: str
    top_scores: list[ScoreItem]
    confidence: float
    model_agreement: bool
    model_divergence: float = 0.0
    scenario_dc: str = ""
    scenario_xgb: str = ""


class TeamInfo(BaseModel):
    """A team and its ELO, for listing."""

    name: str
    elo: float


class MatchSummary(BaseModel):
    """A compact past-match summary."""

    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


class TeamDetail(BaseModel):
    """Response for GET /api/team/{name}."""

    name: str
    elo: float
    form_goals_scored: float
    form_goals_conceded: float
    last_matches: list[MatchSummary]


class H2HResponse(BaseModel):
    """Response for GET /api/h2h."""

    home: str
    away: str
    home_wins_pct: float
    avg_goals: float
    matches: list[MatchSummary]


class FixtureItem(BaseModel):
    """A single fixture with optional result and prediction for played matches."""

    date: str
    time: str
    matchday: str
    home_team: str
    away_team: str
    city: str
    stadium: str
    stage: str
    predictable: bool
    status: str  # "Joué", "À jouer", "En direct"
    actual_score: str  # e.g. "2-0" or "" if not played
    predicted_score: str  # e.g. "1-0" or "" if not played
    outcome_correct: bool | None  # None if not played


class BacktestMetrics(BaseModel):
    """Response for GET /api/backtest/{tournament}."""

    outcome_accuracy: float
    exact_score_accuracy: float
    top3_score_accuracy: float
    brier_score: float
    log_loss: float


class BacktestMatchDetail(BaseModel):
    """Prediction vs actual for a single match."""

    date: str
    home_team: str
    away_team: str
    predicted_score: str
    actual_score: str
    predicted_winner: str
    actual_winner: str
    outcome_correct: bool
    score_correct: bool


class BacktestDetails(BaseModel):
    """Response for GET /api/backtest/{tournament}/details."""

    tournament: str
    matches: list[BacktestMatchDetail]
