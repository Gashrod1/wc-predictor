import pytest
import pandas as pd
from src.data.loader import load_historical_matches, load_elo_ratings, fetch_recent_form


def test_load_historical_matches_returns_dataframe():
    df = load_historical_matches()
    assert isinstance(df, pd.DataFrame)
    required_cols = {"date", "home_team", "away_team", "home_goals", "away_goals", "stage", "tournament"}
    assert required_cols.issubset(df.columns)


def test_load_historical_matches_has_rows():
    df = load_historical_matches()
    assert len(df) >= 64  # At least WC2022


def test_load_elo_ratings_returns_dict():
    ratings = load_elo_ratings()
    assert isinstance(ratings, dict)
    assert "France" in ratings
    assert ratings["France"] > 2000


def test_fetch_recent_form_no_api_key(monkeypatch):
    """Without API key, fetch_recent_form should return form from historical data."""
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    form = fetch_recent_form("France", n_matches=5)
    assert isinstance(form, pd.DataFrame)
    # May be empty if no recent matches, but must not raise


from src.data.features import build_match_features


def test_build_match_features_returns_all_keys():
    df = load_historical_matches()
    elo = load_elo_ratings()
    features = build_match_features("France", "Brazil", elo, df)
    expected_keys = {
        "elo_diff", "elo_home", "elo_away",
        "home_form_goals_scored", "home_form_goals_conceded", "home_form_xg",
        "away_form_goals_scored", "away_form_goals_conceded", "away_form_xg",
        "h2h_home_wins", "h2h_avg_goals", "is_knockout",
    }
    assert expected_keys.issubset(set(features.keys()))


def test_build_match_features_elo_diff():
    df = load_historical_matches()
    elo = load_elo_ratings()
    features = build_match_features("France", "Brazil", elo, df)
    assert features["elo_diff"] == pytest.approx(elo["France"] - elo["Brazil"], abs=1)


def test_build_match_features_is_knockout_flag():
    df = load_historical_matches()
    elo = load_elo_ratings()
    features_group = build_match_features("France", "Brazil", elo, df, stage="group")
    features_ko = build_match_features("France", "Brazil", elo, df, stage="semi_final")
    assert features_group["is_knockout"] == 0
    assert features_ko["is_knockout"] == 1


def test_build_match_features_unknown_team_uses_defaults():
    """Unknown team should not crash — use mean ELO and zero form."""
    df = load_historical_matches()
    elo = load_elo_ratings()
    features = build_match_features("Atlantis", "Brazil", elo, df)
    assert isinstance(features["elo_home"], float)


from src.models.xgboost_classifier import XGBoostOutcomeClassifier
import numpy as np


def test_xgboost_fit_predict():
    """Classifier should train and return probabilities summing to 1."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({
        "elo_diff": np.random.randn(n) * 100,
        "elo_home": np.random.uniform(1800, 2100, n),
        "elo_away": np.random.uniform(1800, 2100, n),
        "home_form_goals_scored": np.random.uniform(0.5, 3.0, n),
        "home_form_goals_conceded": np.random.uniform(0.5, 2.5, n),
        "home_form_xg": np.random.uniform(0.5, 2.5, n),
        "away_form_goals_scored": np.random.uniform(0.5, 3.0, n),
        "away_form_goals_conceded": np.random.uniform(0.5, 2.5, n),
        "away_form_xg": np.random.uniform(0.5, 2.5, n),
        "h2h_home_wins": np.random.uniform(0, 1, n),
        "h2h_avg_goals": np.random.uniform(1.5, 4.0, n),
        "is_knockout": np.random.randint(0, 2, n),
    })
    y = pd.Series(np.random.choice([0, 1, 2], size=n))
    clf = XGBoostOutcomeClassifier()
    clf.fit(X, y)
    features_dict = X.iloc[0].to_dict()
    probs = clf.predict_proba(features_dict)
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 0.01


def test_xgboost_feature_importance():
    """get_feature_importance should return a DataFrame with 'feature' and 'importance'."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({k: np.random.randn(n) for k in [
        "elo_diff", "elo_home", "elo_away",
        "home_form_goals_scored", "home_form_goals_conceded", "home_form_xg",
        "away_form_goals_scored", "away_form_goals_conceded", "away_form_xg",
        "h2h_home_wins", "h2h_avg_goals", "is_knockout",
    ]})
    y = pd.Series(np.random.choice([0, 1, 2], size=n))
    clf = XGBoostOutcomeClassifier()
    clf.fit(X, y)
    fi = clf.get_feature_importance()
    assert "feature" in fi.columns
    assert "importance" in fi.columns
    assert len(fi) == 12


from src.models.ensemble import EnsemblePredictor, PredictionResult
from src.models.dixon_coles import DixonColesModel
from src.models.xgboost_classifier import XGBoostOutcomeClassifier
from src.data.features import build_match_features
from src.data.loader import load_historical_matches, load_elo_ratings


def test_ensemble_predict_returns_prediction_result():
    df = load_historical_matches()
    elo = load_elo_ratings()
    dc = DixonColesModel()
    dc.fit(df)

    feature_rows = []
    labels = []
    for _, row in df.iterrows():
        feats = build_match_features(row["home_team"], row["away_team"], elo, df, stage=row["stage"])
        feature_rows.append(feats)
        if row["home_goals"] > row["away_goals"]:
            labels.append(2)
        elif row["home_goals"] == row["away_goals"]:
            labels.append(1)
        else:
            labels.append(0)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels)
    xgb = XGBoostOutcomeClassifier()
    xgb.fit(X, y)

    ensemble = EnsemblePredictor(dc_model=dc, xgb_model=xgb)
    result = ensemble.predict("France", "Brazil", context={"stage": "semi_final"})

    assert isinstance(result, PredictionResult)
    assert result.home_team == "France"
    assert result.away_team == "Brazil"
    assert result.predicted_winner in ("home", "draw", "away")
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_scores) == 5
    assert isinstance(result.model_agreement, bool)
