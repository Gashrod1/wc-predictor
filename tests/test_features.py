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
