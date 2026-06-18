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
