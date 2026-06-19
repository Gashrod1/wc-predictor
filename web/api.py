"""FastAPI web API for the World Cup predictor."""
from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data.features import build_match_features
from src.data.loader import (
    load_elo_ratings,
    load_historical_matches,
    resolve_team_name,
)
from src.evaluation.backtesting import run_backtest, run_backtest_details
from src.models.dixon_coles import DixonColesModel
from src.models.xgboost_classifier import XGBoostOutcomeClassifier
from src.models.ensemble import EnsemblePredictor
from src.prediction.predictor import load_or_train_predictor
from src.data.wc2026_sync import sync_wc2026_results, get_sync_status
from web.fixtures import load_fixtures
from web.schemas import (
    BacktestDetails,
    BacktestMatchDetail,
    BacktestMetrics,
    FixtureItem,
    H2HResponse,
    MatchSummary,
    PredictRequest,
    PredictResponse,
    TeamDetail,
    TeamInfo,
)

_DEFAULT_ELO = 1850.0
_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data once at startup; keep them in memory."""
    _state["predictor"] = load_or_train_predictor()
    _state["elo"] = load_elo_ratings()
    _state["history"] = load_historical_matches()
    _state["backtest_cache"] = {}
    yield
    _state.clear()


app = FastAPI(title="World Cup Predictor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _recent_matches(team: str, n: int = 5) -> list[MatchSummary]:
    """Return the last n matches for a team from history."""
    df: pd.DataFrame = _state["history"]  # type: ignore[assignment]
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    recent = df[mask].sort_values("date", ascending=False).head(n)
    out: list[MatchSummary] = []
    for _, row in recent.iterrows():
        out.append(
            MatchSummary(
                date=str(row["date"])[:10],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
            )
        )
    return out


@app.get("/api/teams", response_model=list[TeamInfo])
def get_teams() -> list[TeamInfo]:
    """Return all teams with ELO, sorted by ELO descending."""
    elo: dict[str, float] = _state["elo"]  # type: ignore[assignment]
    teams = [TeamInfo(name=name, elo=score) for name, score in elo.items()]
    teams.sort(key=lambda t: t.elo, reverse=True)
    return teams


@app.post("/api/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest) -> PredictResponse:
    """Predict a single match outcome."""
    predictor: EnsemblePredictor = _state["predictor"]  # type: ignore[assignment]
    home = resolve_team_name(req.home)
    away = resolve_team_name(req.away)
    result = predictor.predict(home, away, context={"stage": req.stage, "neutral": req.neutral})
    return PredictResponse(
        home_team=result.home_team,
        away_team=result.away_team,
        stage=req.stage,
        outcome_probabilities=result.outcome_probabilities,
        predicted_winner=result.predicted_winner,
        most_likely_score=result.most_likely_score,
        top_scores=result.top_scores,
        confidence=result.confidence,
        model_agreement=result.model_agreement,
        model_divergence=result.model_divergence,
        scenario_dc=result.scenario_dc,
        scenario_xgb=result.scenario_xgb,
    )


@app.get("/api/team/{name}", response_model=TeamDetail)
def get_team(name: str) -> TeamDetail:
    """Return ELO, recent form, and last matches for a team."""
    canonical = resolve_team_name(name)
    elo: dict[str, float] = _state["elo"]  # type: ignore[assignment]
    df: pd.DataFrame = _state["history"]  # type: ignore[assignment]

    mask = (df["home_team"] == canonical) | (df["away_team"] == canonical)
    recent = df[mask].sort_values("date", ascending=False).head(5)

    scored: list[float] = []
    conceded: list[float] = []
    for _, row in recent.iterrows():
        if row["home_team"] == canonical:
            scored.append(row["home_goals"])
            conceded.append(row["away_goals"])
        else:
            scored.append(row["away_goals"])
            conceded.append(row["home_goals"])

    return TeamDetail(
        name=canonical,
        elo=elo.get(canonical, _DEFAULT_ELO),
        form_goals_scored=float(np.mean(scored)) if scored else 0.0,
        form_goals_conceded=float(np.mean(conceded)) if conceded else 0.0,
        last_matches=_recent_matches(canonical),
    )


@app.get("/api/h2h", response_model=H2HResponse)
def get_h2h(home: str, away: str) -> H2HResponse:
    """Return head-to-head stats between two teams."""
    h = resolve_team_name(home)
    a = resolve_team_name(away)
    df: pd.DataFrame = _state["history"]  # type: ignore[assignment]

    mask = (
        ((df["home_team"] == h) & (df["away_team"] == a))
        | ((df["home_team"] == a) & (df["away_team"] == h))
    )
    h2h = df[mask].sort_values("date", ascending=False).head(8)

    home_wins = 0
    total_goals: list[float] = []
    matches: list[MatchSummary] = []
    for _, row in h2h.iterrows():
        total_goals.append(row["home_goals"] + row["away_goals"])
        if row["home_team"] == h and row["home_goals"] > row["away_goals"]:
            home_wins += 1
        elif row["away_team"] == h and row["away_goals"] > row["home_goals"]:
            home_wins += 1
        matches.append(
            MatchSummary(
                date=str(row["date"])[:10],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
            )
        )

    return H2HResponse(
        home=h,
        away=a,
        home_wins_pct=(home_wins / len(h2h)) if len(h2h) else 0.0,
        avg_goals=float(np.mean(total_goals)) if total_goals else 0.0,
        matches=matches,
    )


@app.get("/api/sync", response_model=dict)
def post_sync() -> dict:
    """Force-refresh WC2026 results from ESPN (bypasses the 30-min cache)."""
    added = sync_wc2026_results(force=True)
    status = get_sync_status()
    # Invalidate backtest cache so WC2026 metrics get recomputed
    if added > 0:
        cache: dict = _state["backtest_cache"]  # type: ignore[assignment]
        cache.pop("WC2026", None)
        cache.pop("WC2026:details", None)
    return {**status, "new_matches_added": added}


@app.get("/api/fixtures", response_model=list[FixtureItem])
def get_fixtures() -> list[FixtureItem]:
    """Return the WC2026 fixtures with prediction and result for played matches.

    Automatically syncs the latest results from ESPN before returning
    (30-min TTL cache — transparent to the caller).
    """
    # Sync fresh results; cache TTL prevents too-frequent requests
    added = sync_wc2026_results()
    if added > 0:
        # New results found — invalidate the WC2026 backtest cache
        cache: dict = _state["backtest_cache"]  # type: ignore[assignment]
        cache.pop("WC2026", None)
        cache.pop("WC2026:details", None)

    predictor: EnsemblePredictor = _state["predictor"]  # type: ignore[assignment]
    result = []
    for f in load_fixtures():
        if f["status"] == "Joué" and f["predictable"] and f["actual_score"]:
            try:
                home = resolve_team_name(f["home_team"])
                away = resolve_team_name(f["away_team"])
                pred = predictor.predict(home, away, context={"stage": f["stage"]})
                f["predicted_score"] = pred.most_likely_score
                parts = str(f["actual_score"]).split("-")
                hg, ag = int(parts[0]), int(parts[1])
                if hg > ag:
                    actual_winner = "home"
                elif hg < ag:
                    actual_winner = "away"
                else:
                    actual_winner = "draw"
                f["outcome_correct"] = pred.predicted_winner == actual_winner
            except Exception:
                pass
        result.append(FixtureItem(**f))
    return result


@app.get("/api/backtest/{tournament}", response_model=BacktestMetrics)
def get_backtest(tournament: str) -> BacktestMetrics:
    """Compute (and cache) backtest metrics for a tournament."""
    cache: dict[str, BacktestMetrics] = _state["backtest_cache"]  # type: ignore[assignment]
    if tournament in cache:
        return cache[tournament]

    all_df: pd.DataFrame = _state["history"]  # type: ignore[assignment]
    target_df = all_df[all_df["tournament"].str.contains(tournament, na=False)]
    if target_df.empty:
        raise HTTPException(status_code=404, detail=f"Unknown tournament: {tournament}")

    # ELO computed only from matches before this tournament (no leakage)
    elo = load_elo_ratings(as_of_tournament=tournament)
    train_df = all_df[~all_df["tournament"].str.contains(tournament, na=False)]
    if train_df.empty:
        train_df = all_df

    dc_model = DixonColesModel()
    dc_model.fit(train_df)

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for _, row in train_df.iterrows():
        rows.append(
            build_match_features(row["home_team"], row["away_team"], elo, train_df, stage=row["stage"])
        )
        if row["home_goals"] > row["away_goals"]:
            labels.append(2)
        elif row["home_goals"] == row["away_goals"]:
            labels.append(1)
        else:
            labels.append(0)

    xgb_model = XGBoostOutcomeClassifier()
    xgb_model.fit(pd.DataFrame(rows), pd.Series(labels))

    ensemble = EnsemblePredictor(dc_model=dc_model, xgb_model=xgb_model, historical_df=train_df)
    metrics = run_backtest(ensemble, target_df)

    result = BacktestMetrics(**metrics)
    cache[tournament] = result
    return result


@app.get("/api/backtest/{tournament}/details", response_model=BacktestDetails)
def get_backtest_details(tournament: str) -> BacktestDetails:
    """Compute (and cache) per-match backtest detail for a tournament."""
    cache: dict[str, object] = _state["backtest_cache"]  # type: ignore[assignment]
    details_key = f"{tournament}:details"
    if details_key in cache:
        return cache[details_key]  # type: ignore[return-value]

    all_df: pd.DataFrame = _state["history"]  # type: ignore[assignment]
    target_df = all_df[all_df["tournament"].str.contains(tournament, na=False)]
    if target_df.empty:
        raise HTTPException(status_code=404, detail=f"Unknown tournament: {tournament}")

    elo = load_elo_ratings(as_of_tournament=tournament)
    train_df = all_df[~all_df["tournament"].str.contains(tournament, na=False)]
    if train_df.empty:
        train_df = all_df

    dc_model = DixonColesModel()
    dc_model.fit(train_df)

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for _, row in train_df.iterrows():
        rows.append(
            build_match_features(row["home_team"], row["away_team"], elo, train_df, stage=row["stage"])
        )
        if row["home_goals"] > row["away_goals"]:
            labels.append(2)
        elif row["home_goals"] == row["away_goals"]:
            labels.append(1)
        else:
            labels.append(0)

    xgb_model = XGBoostOutcomeClassifier()
    xgb_model.fit(pd.DataFrame(rows), pd.Series(labels))

    ensemble = EnsemblePredictor(dc_model=dc_model, xgb_model=xgb_model, historical_df=train_df)
    match_rows = run_backtest_details(ensemble, target_df)

    details = BacktestDetails(
        tournament=tournament,
        matches=[BacktestMatchDetail(**r) for r in match_rows],
    )
    cache[details_key] = details
    return details
