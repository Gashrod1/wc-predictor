"""Squad chemistry: StatsBomb pass-network + club-pairs fallback."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ChemistryAnalyzer:
    """Compute collective chemistry scores for national teams.

    Sources (in priority order):
    1. StatsBomb Open Data (WC 2022, competition_id=43, season_id=106) — real
       pass-network density and pass accuracy
    2. Club-pair heuristic — fraction of squad mates from the same club

    Weights when StatsBomb available: 40% club-pairs + 35% network density + 25% pass accuracy
    Fallback (club-pairs only): 100% club-pairs
    """

    def __init__(self) -> None:
        self._sb_cache: dict[str, dict[str, float]] = {}  # team → {density, pass_acc}
        self._sb_loaded = False

    def _try_load_statsbomb(self) -> None:
        """Attempt to load StatsBomb WC2022 data into cache. Silent on failure."""
        if self._sb_loaded:
            return
        self._sb_loaded = True  # don't retry on failure
        try:
            from statsbombpy import sb  # type: ignore[import]
            matches = sb.matches(competition_id=43, season_id=106)
            for _, match_row in matches.iterrows():
                for side, team_name in [("home", match_row["home_team"]), ("away", match_row["away_team"])]:
                    try:
                        events = sb.events(match_id=match_row["match_id"])
                        self._update_sb_cache(team_name, events)
                    except Exception:
                        continue
        except Exception:
            pass  # statsbombpy not installed or data unavailable

    def _update_sb_cache(self, team: str, events: pd.DataFrame) -> None:
        """Compute pass-network stats for a team from events DataFrame."""
        try:
            passes = events[
                (events["team"] == team) &
                (events["type"] == "Pass") &
                (events["pass_outcome"].isna())  # successful passes only
            ]
            total_attempted = events[
                (events["team"] == team) & (events["type"] == "Pass")
            ]
            if total_attempted.empty:
                return

            pass_accuracy = len(passes) / len(total_attempted)

            # Network density: unique passer→receiver pairs / max possible pairs among lineup
            if "player" in passes.columns and "pass_recipient" in passes.columns:
                pairs = set(zip(passes["player"].dropna(), passes["pass_recipient"].dropna()))
                players_involved = set(passes["player"].dropna()) | set(passes["pass_recipient"].dropna())
                n = len(players_involved)
                max_pairs = n * (n - 1) if n > 1 else 1
                density = len(pairs) / max_pairs
            else:
                density = 0.0

            if team not in self._sb_cache:
                self._sb_cache[team] = {"density": 0.0, "pass_acc": 0.0, "count": 0}

            # Running average across matches
            entry = self._sb_cache[team]
            c = entry["count"]
            entry["density"] = (entry["density"] * c + density) / (c + 1)
            entry["pass_acc"] = (entry["pass_acc"] * c + pass_accuracy) / (c + 1)
            entry["count"] = c + 1
        except Exception:
            pass

    @staticmethod
    def _club_pair_score(squad_df: pd.DataFrame) -> float:
        """Fraction of all player pairs who share a club."""
        if squad_df.empty or "club" not in squad_df.columns:
            return 0.0
        clubs = squad_df["club"].dropna().tolist()
        n = len(clubs)
        if n < 2:
            return 0.0
        max_pairs = n * (n - 1) / 2
        pair_count = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if clubs[i] == clubs[j] and clubs[i] != ""
        )
        return pair_count / max_pairs

    def _team_chemistry(self, team: str, squad_df: pd.DataFrame) -> tuple[float, float]:
        """Return (chemistry_score, pass_network_density) for a team."""
        club_pairs = self._club_pair_score(squad_df)

        if team in self._sb_cache:
            sb_entry = self._sb_cache[team]
            density = sb_entry["density"]
            pass_acc = sb_entry["pass_acc"]
            score = 0.40 * club_pairs + 0.35 * density + 0.25 * pass_acc
            return float(np.clip(score, 0.0, 1.0)), float(density)
        else:
            return float(np.clip(club_pairs, 0.0, 1.0)), 0.0

    def get_chemistry_features(
        self,
        home_team: str,
        away_team: str,
        home_squad_df: pd.DataFrame,
        away_squad_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute 5 chemistry features for a match.

        Attempts to load StatsBomb WC2022 data on first call (silent failure).

        Args:
            home_team: Home team name (used to look up StatsBomb data).
            away_team: Away team name.
            home_squad_df: Home squad DataFrame (needs 'club' column).
            away_squad_df: Away squad DataFrame (needs 'club' column).

        Returns:
            Dict with keys: home_chemistry_score, away_chemistry_score,
            home_pass_network_density, away_pass_network_density, chemistry_diff.
        """
        self._try_load_statsbomb()

        home_score, home_density = self._team_chemistry(home_team, home_squad_df)
        away_score, away_density = self._team_chemistry(away_team, away_squad_df)

        return {
            "home_chemistry_score": home_score,
            "away_chemistry_score": away_score,
            "home_pass_network_density": home_density,
            "away_pass_network_density": away_density,
            "chemistry_diff": home_score - away_score,
        }
