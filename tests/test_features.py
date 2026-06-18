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


import pandas as pd
from src.models.ensemble import EnsemblePredictor, PredictionResult
from src.models.dixon_coles import DixonColesModel
from src.models.xgboost_classifier import XGBoostOutcomeClassifier
from src.data.features import build_match_features
from src.data.loader import load_historical_matches, load_elo_ratings


def test_prediction_result_has_divergence_fields():
    df = load_historical_matches()
    elo = load_elo_ratings()
    dc = DixonColesModel()
    dc.fit(df)
    rows, labels = [], []
    for _, row in df.iterrows():
        rows.append(build_match_features(row["home_team"], row["away_team"], elo, df, stage=row["stage"]))
        labels.append(2 if row["home_goals"] > row["away_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0))
    xgb = XGBoostOutcomeClassifier()
    xgb.fit(pd.DataFrame(rows), pd.Series(labels))
    ensemble = EnsemblePredictor(dc_model=dc, xgb_model=xgb)
    result = ensemble.predict("France", "Brazil", context={"stage": "semi_final"})

    assert hasattr(result, "model_divergence"), "missing model_divergence"
    assert hasattr(result, "scenario_dc"), "missing scenario_dc"
    assert hasattr(result, "scenario_xgb"), "missing scenario_xgb"
    assert 0.0 <= result.model_divergence <= 1.0
    assert result.scenario_dc in ("home", "draw", "away")
    assert result.scenario_xgb in ("home", "draw", "away")


def test_confidence_penalised_when_models_diverge():
    """High divergence should lower confidence."""
    result_low = PredictionResult(
        home_team="A", away_team="B",
        outcome_probabilities={"home_win": 0.7, "draw": 0.2, "away_win": 0.1},
        predicted_winner="home", most_likely_score="1-0",
        top_scores=[], confidence=0.5, model_agreement=True,
        model_divergence=0.0, scenario_dc="home", scenario_xgb="home",
    )
    result_high = PredictionResult(
        home_team="A", away_team="B",
        outcome_probabilities={"home_win": 0.7, "draw": 0.2, "away_win": 0.1},
        predicted_winner="home", most_likely_score="1-0",
        top_scores=[], confidence=0.5, model_agreement=False,
        model_divergence=0.5, scenario_dc="home", scenario_xgb="away",
    )
    assert result_low.model_divergence == 0.0
    assert result_high.model_divergence == 0.5


def test_build_match_features_with_squad_loader():
    """With squad_loader provided, feature dict gains 6 squad keys."""
    from src.data.loader import load_historical_matches, load_elo_ratings
    from src.data.squad_loader import SquadLoader
    from src.data.chemistry import ChemistryAnalyzer
    from src.data.features import build_match_features
    df = load_historical_matches()
    elo = load_elo_ratings()
    loader = SquadLoader()
    analyzer = ChemistryAnalyzer()
    features = build_match_features("France", "Brazil", elo, df,
                                    squad_loader=loader,
                                    chemistry_analyzer=analyzer)
    squad_keys = {
        "squad_avg_club_elo", "squad_pct_top5_league", "squad_avg_age",
        "squad_market_value_m", "squad_n_in_form", "squad_elo_diff",
        "home_chemistry_score", "away_chemistry_score",
        "home_pass_network_density", "away_pass_network_density",
        "chemistry_diff",
    }
    assert squad_keys.issubset(set(features.keys()))


def test_build_match_features_without_squad_loader_still_works():
    """Without squad_loader, result is the original 12 features."""
    from src.data.loader import load_historical_matches, load_elo_ratings
    from src.data.features import build_match_features
    df = load_historical_matches()
    elo = load_elo_ratings()
    features = build_match_features("France", "Brazil", elo, df)
    assert len(features) == 12


def test_squad_features_failure_does_not_crash():
    """A squad loader that raises must be silently swallowed."""
    from src.data.loader import load_historical_matches, load_elo_ratings
    from src.data.features import build_match_features

    class FailingLoader:
        def get_match_squad_features(self, *a, **kw):
            raise RuntimeError("simulated failure")
        def get_squad(self, *a, **kw):
            raise RuntimeError("simulated failure")

    df = load_historical_matches()
    elo = load_elo_ratings()
    features = build_match_features("France", "Brazil", elo, df,
                                    squad_loader=FailingLoader())
    # Should still return at least the base 12 features
    assert len(features) >= 12
