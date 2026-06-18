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
    """A single fixture (no result)."""

    date: str
    time: str
    matchday: str
    home_team: str
    away_team: str
    city: str
    stadium: str
    stage: str
    predictable: bool


class BacktestMetrics(BaseModel):
    """Response for GET /api/backtest/{tournament}."""

    outcome_accuracy: float
    exact_score_accuracy: float
    top3_score_accuracy: float
    brier_score: float
    log_loss: float
