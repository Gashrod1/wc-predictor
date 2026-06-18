import pytest
import pandas as pd
import numpy as np
from src.evaluation.backtesting import run_backtest
from src.models.dixon_coles import DixonColesModel
from src.models.xgboost_classifier import XGBoostOutcomeClassifier
from src.models.ensemble import EnsemblePredictor
from src.data.loader import load_historical_matches, load_elo_ratings
from src.data.features import build_match_features


@pytest.fixture
def trained_ensemble():
    df = load_historical_matches()
    elo = load_elo_ratings()
    dc = DixonColesModel()
    dc.fit(df)
    rows, labels = [], []
    for _, row in df.iterrows():
        feats = build_match_features(row["home_team"], row["away_team"], elo, df, stage=row["stage"])
        rows.append(feats)
        labels.append(2 if row["home_goals"] > row["away_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0))
    xgb = XGBoostOutcomeClassifier()
    xgb.fit(pd.DataFrame(rows), pd.Series(labels))
    return EnsemblePredictor(dc_model=dc, xgb_model=xgb)


def test_run_backtest_returns_dict(trained_ensemble):
    df = load_historical_matches(tournament="WC2022")
    result = run_backtest(trained_ensemble, df.head(20))
    assert isinstance(result, dict)


def test_run_backtest_has_required_metrics(trained_ensemble):
    df = load_historical_matches(tournament="WC2022")
    result = run_backtest(trained_ensemble, df.head(20))
    for key in ["outcome_accuracy", "exact_score_accuracy", "top3_score_accuracy", "brier_score", "log_loss"]:
        assert key in result, f"Missing metric: {key}"


def test_run_backtest_accuracy_in_range(trained_ensemble):
    df = load_historical_matches(tournament="WC2022")
    result = run_backtest(trained_ensemble, df.head(20))
    assert 0.0 <= result["outcome_accuracy"] <= 1.0
    assert 0.0 <= result["brier_score"] <= 1.0
