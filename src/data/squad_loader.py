"""Squad loader: cascade API-Football → football-data.org → CSV fallback → synthetic."""
from __future__ import annotations

import os
import time
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_FALLBACK_CSV = Path(__file__).parent.parent.parent / "data" / "historical" / "squads_fallback.csv"
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "squads"
_CACHE_TTL_SECONDS = 86400  # 24 hours

# Top 5 European league club keywords for squad_pct_top5_league
_TOP5_KEYWORDS = {
    # Premier League
    "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
    "Tottenham", "Newcastle", "Aston Villa", "West Ham", "Brighton",
    "Everton", "Brentford", "Fulham", "Wolves", "Wolverhampton", "Crystal Palace",
    # La Liga
    "Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla", "Real Betis",
    "Valencia", "Villarreal", "Athletic Bilbao", "Osasuna", "Real Sociedad",
    # Bundesliga
    "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
    "Eintracht Frankfurt", "Freiburg", "Hoffenheim", "Borussia Monchengladbach",
    "Werder Bremen", "Stuttgart",
    # Serie A
    "Juventus", "Inter Milan", "AC Milan", "Roma", "Napoli", "Lazio",
    "Fiorentina", "Atalanta", "Torino", "Sassuolo",
    # Ligue 1
    "Paris Saint-Germain", "Marseille", "Lyon", "Monaco", "Rennes", "Lille",
    "Nice", "Lens", "Nantes", "Montpellier",
}

# Approximate current ELO for major clubs (static reference, avoids HTTP at runtime)
_CLUB_ELO: dict[str, float] = {
    "Manchester City": 1980, "Real Madrid": 1970, "Bayern Munich": 1955,
    "Liverpool": 1945, "Chelsea": 1920, "Barcelona": 1915, "PSG": 1910,
    "Paris Saint-Germain": 1910, "Arsenal": 1905, "Atletico Madrid": 1900,
    "Manchester United": 1895, "Tottenham Hotspur": 1880, "Juventus": 1870,
    "Inter Milan": 1875, "AC Milan": 1870, "Napoli": 1865, "Borussia Dortmund": 1860,
    "RB Leipzig": 1850, "Bayer Leverkusen": 1845, "Eintracht Frankfurt": 1820,
    "Newcastle United": 1810, "Aston Villa": 1800, "West Ham United": 1790,
    "Brighton": 1785, "Fiorentina": 1780, "Roma": 1785, "Lazio": 1775,
    "Sevilla": 1780, "Real Betis": 1775, "Villarreal": 1770, "Atalanta": 1790,
    "Benfica": 1790, "Ajax": 1785, "Porto": 1780, "Sporting CP": 1760,
    "Celtic": 1730, "Rangers": 1725, "Marseille": 1760, "Lyon": 1750,
    "Monaco": 1755, "Rennes": 1720, "Freiburg": 1735, "Hoffenheim": 1710,
    "Werder Bremen": 1700, "Stuttgart": 1715, "Torino": 1700, "Sassuolo": 1695,
    "Osasuna": 1680, "Valencia": 1730, "Athletic Bilbao": 1740,
    "Real Valladolid": 1620, "Brentford": 1770, "Crystal Palace": 1745,
    "Everton": 1735, "Fulham": 1740, "Wolverhampton": 1750, "Wolves": 1750,
    "Borussia Monchengladbach": 1720, "Red Bull Salzburg": 1730,
    "Dinamo Zagreb": 1680, "Wydad AC": 1550, "Flamengo": 1710,
    "Palmeiras": 1680, "River Plate": 1690, "Boca Juniors": 1680,
    "Genk": 1680, "Standard Liege": 1650, "Toulouse": 1660, "Bari": 1600,
    "Queens Park Rangers": 1580, "Angers": 1620, "Besiktas": 1680,
    "Zenit": 1700, "Ferencvaros": 1660, "Atlanta United": 1540,
    "Qatar SC": 1450, "Ismaily SC": 1420, "Deportivo de la Coruna": 1560,
    "Osijek": 1600, "Hajduk Split": 1640, "Brest": 1660,
}
_DEFAULT_CLUB_ELO = 1500.0


class SquadLoader:
    """Loads national team squads from multiple sources with caching.

    Source cascade:
    1. API-Football (if API_FOOTBALL_KEY in env)
    2. football-data.org (if FOOTBALL_DATA_KEY in env)
    3. squads_fallback.csv (always available)
    4. Synthetic squad from national ELO

    All results cached to data/raw/squads/{team}.parquet for 24 hours.
    """

    def __init__(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._fallback_df: Optional[pd.DataFrame] = None

    def _load_fallback_csv(self) -> pd.DataFrame:
        """Load and cache the fallback CSV."""
        if self._fallback_df is None:
            try:
                self._fallback_df = pd.read_csv(_FALLBACK_CSV)
            except Exception:
                self._fallback_df = pd.DataFrame(
                    columns=["team", "player_name", "position", "club", "age", "caps", "market_value_eur"]
                )
        return self._fallback_df

    def _cache_path(self, team: str) -> Path:
        safe = team.replace(" ", "_").replace("/", "_")
        return _CACHE_DIR / f"{safe}.parquet"

    def _try_load_cache(self, team: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(team)
        if path.exists():
            if time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
                try:
                    return pd.read_parquet(path)
                except Exception:
                    pass
        return None

    def _save_cache(self, team: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_path(team), index=False)
        except Exception:
            pass  # cache failure is non-fatal

    def get_squad(self, team: str) -> pd.DataFrame:
        """Return a squad DataFrame for the given national team.

        Always returns a DataFrame with columns:
        player_name, position, club, age, caps, market_value_eur.
        Never raises; falls back to synthetic squad on all failures.

        Args:
            team: National team name (English canonical form).

        Returns:
            DataFrame with one row per player.
        """
        cached = self._try_load_cache(team)
        if cached is not None:
            return cached

        squad = self._try_api_football(team)
        if squad is None:
            squad = self._try_football_data(team)
        if squad is None:
            squad = self._try_csv_fallback(team)
        if squad is None or squad.empty:
            squad = self._synthetic_squad(team)

        self._save_cache(team, squad)
        return squad

    def _try_api_football(self, team: str) -> Optional[pd.DataFrame]:
        key = os.getenv("API_FOOTBALL_KEY", "").strip()
        if not key:
            return None
        try:
            headers = {"x-apisports-key": key}
            r = requests.get(
                "https://v3.football.api-sports.io/teams",
                headers=headers,
                params={"name": team, "type": "national"},
                timeout=8,
            )
            r.raise_for_status()
            teams = r.json().get("response", [])
            if not teams:
                return None
            team_id = teams[0]["team"]["id"]

            r2 = requests.get(
                "https://v3.football.api-sports.io/players/squads",
                headers=headers,
                params={"team": team_id},
                timeout=8,
            )
            r2.raise_for_status()
            players = r2.json().get("response", [{}])[0].get("players", [])
            if not players:
                return None

            rows = []
            for p in players:
                rows.append({
                    "player_name": p.get("name", ""),
                    "position": p.get("position", "MID"),
                    "club": p.get("club", {}).get("name", ""),
                    "age": p.get("age", 25),
                    "caps": 0,
                    "market_value_eur": 0,
                })
            return pd.DataFrame(rows)
        except Exception:
            return None

    def _try_football_data(self, team: str) -> Optional[pd.DataFrame]:
        key = os.getenv("FOOTBALL_DATA_KEY", "").strip()
        if not key:
            return None
        try:
            headers = {"X-Auth-Token": key}
            r = requests.get(
                "https://api.football-data.org/v4/competitions/WC/teams",
                headers=headers,
                timeout=8,
            )
            r.raise_for_status()
            teams = r.json().get("teams", [])
            match = next((t for t in teams if t["name"].lower() == team.lower()), None)
            if match is None:
                return None
            team_id = match["id"]

            r2 = requests.get(
                f"https://api.football-data.org/v4/teams/{team_id}/",
                headers=headers,
                timeout=8,
            )
            r2.raise_for_status()
            squad = r2.json().get("squad", [])
            if not squad:
                return None

            rows = []
            for p in squad:
                rows.append({
                    "player_name": p.get("name", ""),
                    "position": p.get("position", "Midfielder")[:3].upper(),
                    "club": p.get("currentTeam", {}).get("name", ""),
                    "age": 25,
                    "caps": 0,
                    "market_value_eur": 0,
                })
            return pd.DataFrame(rows)
        except Exception:
            return None

    def _try_csv_fallback(self, team: str) -> Optional[pd.DataFrame]:
        try:
            df = self._load_fallback_csv()
            subset = df[df["team"] == team].drop(columns=["team"], errors="ignore")
            return subset.reset_index(drop=True) if not subset.empty else None
        except Exception:
            return None

    def _synthetic_squad(self, team: str) -> pd.DataFrame:
        """Generate a plausible synthetic squad from national ELO."""
        from src.data.loader import load_elo_ratings
        try:
            elo = load_elo_ratings().get(team, 1850.0)
        except Exception:
            elo = 1850.0

        club_elo_estimate = elo * 0.92
        positions = ["GK", "GK", "GK", "DEF", "DEF", "DEF", "DEF", "DEF",
                     "MID", "MID", "MID", "MID", "MID", "MID", "MID",
                     "FW", "FW", "FW", "FW", "FW", "FW", "FW", "FW"]
        rows = []
        for i, pos in enumerate(positions):
            rows.append({
                "player_name": f"{team} Player {i+1}",
                "position": pos,
                "club": f"Synthetic Club ({team})",
                "age": 25,
                "caps": 20,
                "market_value_eur": 0,
            })
        df = pd.DataFrame(rows)
        df["_club_elo_estimate"] = club_elo_estimate
        return df

    @lru_cache(maxsize=64)
    def _club_elo(self, club: str) -> float:
        """Return approximate ELO for a club. Uses static dict, falls back to 1500."""
        if club in _CLUB_ELO:
            return _CLUB_ELO[club]
        for key, val in _CLUB_ELO.items():
            if key.lower() in club.lower() or club.lower() in key.lower():
                return val
        return _DEFAULT_CLUB_ELO

    def _squad_features(self, squad_df: pd.DataFrame) -> dict[str, float]:
        """Compute squad features from a squad DataFrame."""
        if squad_df.empty:
            return {
                "avg_club_elo": _DEFAULT_CLUB_ELO,
                "pct_top5_league": 0.0,
                "avg_age": 26.0,
                "market_value_m": 0.0,
                "n_in_form": 0.0,
            }

        # Check for synthetic squad (has _club_elo_estimate column)
        if "_club_elo_estimate" in squad_df.columns:
            avg_elo = float(squad_df["_club_elo_estimate"].iloc[0])
            return {
                "avg_club_elo": avg_elo,
                "pct_top5_league": 0.0,
                "avg_age": 26.0,
                "market_value_m": 0.0,
                "n_in_form": 0.0,
            }

        clubs = squad_df["club"].fillna("").astype(str).tolist()
        avg_elo = float(np.mean([self._club_elo(c) for c in clubs]))

        top5_count = sum(
            1 for c in clubs
            if any(kw.lower() in c.lower() for kw in _TOP5_KEYWORDS)
        )
        pct_top5 = top5_count / len(clubs) if clubs else 0.0

        ages = pd.to_numeric(squad_df["age"], errors="coerce").dropna()
        avg_age = float(ages.mean()) if not ages.empty else 26.0

        vals = pd.to_numeric(squad_df["market_value_eur"], errors="coerce").fillna(0.0)
        market_value_m = float(vals.sum() / 1_000_000)

        return {
            "avg_club_elo": avg_elo,
            "pct_top5_league": pct_top5,
            "avg_age": avg_age,
            "market_value_m": market_value_m,
            "n_in_form": 0.0,
        }

    def get_match_squad_features(
        self, home_team: str, away_team: str
    ) -> dict[str, float]:
        """Compute 6 squad-based features for a match.

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Dict with keys: squad_avg_club_elo, squad_pct_top5_league,
            squad_avg_age, squad_market_value_m, squad_n_in_form, squad_elo_diff.
        """
        home_sq = self.get_squad(home_team)
        away_sq = self.get_squad(away_team)
        h = self._squad_features(home_sq)
        a = self._squad_features(away_sq)
        return {
            "squad_avg_club_elo": h["avg_club_elo"],
            "squad_pct_top5_league": h["pct_top5_league"],
            "squad_avg_age": h["avg_age"],
            "squad_market_value_m": h["market_value_m"],
            "squad_n_in_form": h["n_in_form"],
            "squad_elo_diff": h["avg_club_elo"] - a["avg_club_elo"],
        }
