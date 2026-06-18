"""Data loading utilities: CSV historical data and optional API-Football integration."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.elo import load_elo_from_csv

load_dotenv()

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_HISTORICAL_DIR = _DATA_DIR / "historical"


def load_historical_matches(
    tournament: str | None = None,
    year_from: int = 1990,
) -> pd.DataFrame:
    """Load historical World Cup match data from CSV files.

    Combines WC2018 and WC2022 CSV files. Filters by tournament prefix and year.

    Args:
        tournament: If provided, filter to rows where `tournament` contains this string.
        year_from: Only include matches from this year onward.

    Returns:
        DataFrame with columns: date, home_team, away_team, home_goals, away_goals,
        home_xg, away_xg, stage, tournament.
    """
    frames: list[pd.DataFrame] = []
    for csv_file in sorted(_HISTORICAL_DIR.glob("wc_*.csv")):
        try:
            df = pd.read_csv(csv_file, parse_dates=["date"])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(
            columns=[
                "date", "home_team", "away_team", "home_goals",
                "away_goals", "home_xg", "away_xg", "stage", "tournament",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # Add xG columns with fallback to goals if missing
    if "home_xg" not in combined.columns:
        combined["home_xg"] = combined["home_goals"].astype(float)
    if "away_xg" not in combined.columns:
        combined["away_xg"] = combined["away_goals"].astype(float)

    combined = combined[combined["date"].dt.year >= year_from]

    if tournament is not None:
        combined = combined[combined["tournament"].str.contains(tournament, na=False)]

    return combined.reset_index(drop=True)


def load_elo_ratings() -> dict[str, float]:
    """Load ELO ratings for national teams from the historical CSV.

    Returns:
        Dictionary mapping team name to ELO score.
    """
    elo_path = _HISTORICAL_DIR / "elo_ratings.csv"
    return load_elo_from_csv(str(elo_path))


def fetch_recent_form(team: str, n_matches: int = 5) -> pd.DataFrame:
    """Fetch recent match form for a team, using API-Football if key is available.

    Falls back to historical CSV data if no API key is configured.

    Args:
        team: National team name.
        n_matches: Number of recent matches to retrieve.

    Returns:
        DataFrame with columns: date, home_team, away_team, home_goals, away_goals,
        home_xg, away_xg, stage, tournament.
    """
    api_key = os.getenv("API_FOOTBALL_KEY")
    if api_key:
        try:
            return _fetch_api_form(team, n_matches, api_key)
        except Exception:
            pass  # Fall back to historical data

    return _fetch_historical_form(team, n_matches)


def _fetch_historical_form(team: str, n_matches: int) -> pd.DataFrame:
    """Return most recent n_matches from historical CSV for given team."""
    df = load_historical_matches()
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_matches = df[mask].sort_values("date", ascending=False)
    return team_matches.head(n_matches).reset_index(drop=True)


def _fetch_api_form(team: str, n_matches: int, api_key: str) -> pd.DataFrame:
    """Fetch recent form from API-Football v3."""
    headers = {"x-apisports-key": api_key}
    resp = requests.get(
        "https://v3.football.api-sports.io/teams",
        headers=headers,
        params={"name": team},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("response"):
        return pd.DataFrame()

    team_id = data["response"][0]["team"]["id"]
    fixtures_resp = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=headers,
        params={"team": team_id, "last": n_matches, "league": "1"},
        timeout=10,
    )
    fixtures_resp.raise_for_status()
    fixtures = fixtures_resp.json().get("response", [])

    rows = []
    for f in fixtures:
        rows.append(
            {
                "date": f["fixture"]["date"][:10],
                "home_team": f["teams"]["home"]["name"],
                "away_team": f["teams"]["away"]["name"],
                "home_goals": f["goals"]["home"] or 0,
                "away_goals": f["goals"]["away"] or 0,
                "home_xg": f["goals"]["home"] or 0,
                "away_xg": f["goals"]["away"] or 0,
                "stage": "international",
                "tournament": "API",
            }
        )
    return pd.DataFrame(rows)
