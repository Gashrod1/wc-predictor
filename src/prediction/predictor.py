"""High-level prediction interface that loads trained models automatically."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import (
    load_historical_matches,
    load_elo_ratings,
    load_elo_trends,
    resolve_team_name,
)
from src.models.dixon_coles import DixonColesModel
from src.models.ensemble import EnsemblePredictor, PredictionResult
from src.models.xgboost_classifier import XGBoostOutcomeClassifier

_SAVED_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_or_train_predictor() -> EnsemblePredictor:
    """Load a saved ensemble predictor (with squad/chemistry), or train from scratch."""
    from src.data.squad_loader import SquadLoader
    from src.data.chemistry import ChemistryAnalyzer

    dc_path = _SAVED_DIR / "dixon_coles.joblib"
    xgb_path = _SAVED_DIR / "xgboost.joblib"

    if dc_path.exists() and xgb_path.exists():
        dc_model: DixonColesModel = joblib.load(dc_path)
        xgb_model: XGBoostOutcomeClassifier = joblib.load(xgb_path)
    else:
        dc_model, xgb_model = _train_models()

    elo_trends = load_elo_trends()

    return EnsemblePredictor(
        dc_model=dc_model,
        xgb_model=xgb_model,
        squad_loader=SquadLoader(),
        chemistry_analyzer=ChemistryAnalyzer(),
        elo_trends=elo_trends,
    )


def predict_match(
    home_team: str,
    away_team: str,
    stage: str = "group",
    neutral: bool = True,
) -> PredictionResult:
    """Predict a single match outcome using the ensemble."""
    predictor = load_or_train_predictor()
    return predictor.predict(
        resolve_team_name(home_team),
        resolve_team_name(away_team),
        context={"stage": stage, "neutral": neutral},
    )


def _train_models() -> tuple[DixonColesModel, XGBoostOutcomeClassifier]:
    """Train both models from historical + competitive data."""
    from src.data.features import build_match_features

    wc_df = load_historical_matches()
    elo = load_elo_ratings()
    elo_trends = load_elo_trends()

    # Dixon-Coles: trained on WC matches only (score distribution is WC-specific)
    dc_model = DixonColesModel()
    dc_model.fit(wc_df)

    # XGBoost: use ALL competitive matches for richer form/H2H features
    comp_path = _DATA_DIR / "historical" / "international_competitive.csv"
    try:
        comp_df = pd.read_csv(comp_path, parse_dates=["date"])
        # Combine WC + competitive, deduplicate
        all_df = pd.concat([wc_df, comp_df], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        ).sort_values("date").reset_index(drop=True)
    except Exception:
        all_df = wc_df

    feature_rows = []
    labels = []
    for _, row in all_df.iterrows():
        feats = build_match_features(
            row["home_team"], row["away_team"], elo, all_df,
            stage=str(row.get("stage", "group")),
            elo_trends=elo_trends,
            squad_loader=None,
            chemistry_analyzer=None,
        )
        feature_rows.append(feats)
        if row["home_goals"] > row["away_goals"]:
            labels.append(2)
        elif row["home_goals"] == row["away_goals"]:
            labels.append(1)
        else:
            labels.append(0)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels)

    xgb_model = XGBoostOutcomeClassifier()
    # With 14k+ matches, run hyperparameter tuning
    print("  Running XGBoost hyperparameter search (TimeSeriesSplit, 40 iterations)...")
    best_params = xgb_model.tune_hyperparameters(X, y, n_iter=40)
    if best_params:
        print(f"  Best params: {best_params}")

    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dc_model, _SAVED_DIR / "dixon_coles.joblib")
    joblib.dump(xgb_model, _SAVED_DIR / "xgboost.joblib")

    return dc_model, xgb_model
