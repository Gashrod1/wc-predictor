"""High-level prediction interface that loads trained models automatically."""
from __future__ import annotations

from pathlib import Path

import joblib

from src.data.loader import load_historical_matches, load_elo_ratings, resolve_team_name
from src.models.dixon_coles import DixonColesModel
from src.models.ensemble import EnsemblePredictor, PredictionResult
from src.models.xgboost_classifier import XGBoostOutcomeClassifier

_SAVED_DIR = Path(__file__).parent.parent.parent / "models" / "saved"


def load_or_train_predictor() -> EnsemblePredictor:
    """Load a saved ensemble predictor (with squad/chemistry), or train from scratch.

    Squad and chemistry analyzers are always instantiated fresh — they load
    their own caches on demand and do not need serialisation.

    Returns:
        Ready-to-use EnsemblePredictor.
    """
    from src.data.squad_loader import SquadLoader
    from src.data.chemistry import ChemistryAnalyzer

    dc_path = _SAVED_DIR / "dixon_coles.joblib"
    xgb_path = _SAVED_DIR / "xgboost.joblib"

    if dc_path.exists() and xgb_path.exists():
        dc_model: DixonColesModel = joblib.load(dc_path)
        xgb_model: XGBoostOutcomeClassifier = joblib.load(xgb_path)
    else:
        dc_model, xgb_model = _train_models()

    return EnsemblePredictor(
        dc_model=dc_model,
        xgb_model=xgb_model,
        squad_loader=SquadLoader(),
        chemistry_analyzer=ChemistryAnalyzer(),
    )


def predict_match(
    home_team: str,
    away_team: str,
    stage: str = "group",
    neutral: bool = True,
) -> PredictionResult:
    """Predict a single match outcome using the ensemble.

    Args:
        home_team: First team name (French aliases resolved automatically).
        away_team: Second team name.
        stage: Match stage (e.g. 'group', 'semi_final', 'final').
        neutral: If True (default), no home advantage applied — correct for all
            World Cup matches except USA/Canada/Mexico games on their own soil.

    Returns:
        PredictionResult with all prediction details.
    """
    predictor = load_or_train_predictor()
    return predictor.predict(
        resolve_team_name(home_team),
        resolve_team_name(away_team),
        context={"stage": stage, "neutral": neutral},
    )


def _train_models() -> tuple[DixonColesModel, XGBoostOutcomeClassifier]:
    """Train both models from historical data (without squad features).

    Squad features are NOT used during historical training because we do not
    have reliable squad data for 2018/2022 via API. The squad columns will be
    absent from the training DataFrame; XGBoost handles this via .get(f, 0.0).

    Returns:
        Fitted (DixonColesModel, XGBoostOutcomeClassifier) tuple.
    """
    import pandas as pd
    from src.data.features import build_match_features

    df = load_historical_matches()
    elo = load_elo_ratings()

    dc_model = DixonColesModel()
    dc_model.fit(df)

    feature_rows = []
    labels = []
    for _, row in df.iterrows():
        feats = build_match_features(
            row["home_team"], row["away_team"], elo, df,
            stage=row["stage"],
            squad_loader=None,          # no squad data for historical training
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
    xgb_model.fit(X, y)

    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dc_model, _SAVED_DIR / "dixon_coles.joblib")
    joblib.dump(xgb_model, _SAVED_DIR / "xgboost.joblib")

    return dc_model, xgb_model
