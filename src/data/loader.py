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

# French/common aliases → canonical English team names used in CSV data
TEAM_ALIASES: dict[str, str] = {
    "Argentine": "Argentina",
    "Brésil": "Brazil",
    "Allemagne": "Germany",
    "Espagne": "Spain",
    "Italie": "Italy",
    "Angleterre": "England",
    "Pays-Bas": "Netherlands",
    "Hollande": "Netherlands",
    "Suisse": "Switzerland",
    "Suède": "Sweden",
    "Danemark": "Denmark",
    "Croatie": "Croatia",
    "Pologne": "Poland",
    "Sénégal": "Senegal",
    "Maroc": "Morocco",
    "Corée du Sud": "South Korea",
    "Japon": "Japan",
    "Australie": "Australia",
    "Mexique": "Mexico",
    "Costa Rica": "Costa Rica",
    "États-Unis": "USA",
    "Cameroun": "Cameroon",
    "Tunisie": "Tunisia",
    "Ghana": "Ghana",
    "Équateur": "Ecuador",
    "Uruguay": "Uruguay",
    "Belgique": "Belgium",
    "Russie": "Russia",
    "Serbie": "Serbia",
    "Arabie Saoudite": "Saudi Arabia",
}


def resolve_team_name(name: str) -> str:
    """Resolve a team name alias to its canonical English form.

    Args:
        name: Team name, possibly in French or with alternate spelling.

    Returns:
        Canonical English team name.
    """
    return TEAM_ALIASES.get(name, name)
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


def load_elo_ratings(as_of_tournament: str | None = None) -> dict[str, float]:
    """Compute ELO ratings dynamically from all historical match results.

    Processes WC matches chronologically (oldest first) using the World
    Football ELO algorithm with goal-difference weighting and separate
    K-factors for group (40) vs knockout (60) stages.

    The CSV fallback (`elo_ratings.csv`) is only used for teams that never
    appeared in any WC match — it acts as a prior, not a source of truth.

    Args:
        as_of_tournament: If set (e.g. "WC2026"), exclude that tournament's
            matches from the computation. Useful for backtesting to avoid
            data leakage.

    Returns:
        Dictionary mapping team name to computed ELO rating.
    """
    from src.data.elo import compute_elo_from_matches, load_elo_from_csv

    # Load CSV as seed/fallback for teams not in any WC match
    elo_path = _HISTORICAL_DIR / "elo_ratings.csv"
    try:
        seed = load_elo_from_csv(str(elo_path))
    except Exception:
        seed = {}

    # Load all match history and exclude the target tournament if requested
    df = load_historical_matches()
    if as_of_tournament:
        df = df[~df["tournament"].str.contains(as_of_tournament, na=False)]

    if df.empty:
        return seed

    computed = compute_elo_from_matches(df, seed=seed)

    # Merge: computed ratings take precedence; CSV fills in the rest
    return {**seed, **computed}


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
