"""Feature engineering for match prediction models."""
from __future__ import annotations

import numpy as np
import pandas as pd

_KNOCKOUT_STAGES = {"round_of_32", "round_of_16", "quarter_final", "semi_final", "final", "third_place"}
_DEFAULT_ELO = 1850.0


def build_match_features(
    home_team: str,
    away_team: str,
    elo_ratings: dict[str, float],
    historical_df: pd.DataFrame,
    stage: str = "group",
    n_form: int = 5,
    n_h2h: int = 8,
    squad_loader: object | None = None,
    chemistry_analyzer: object | None = None,
    elo_trends: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build the feature vector for a match prediction.

    Args:
        home_team: Name of the home/first team.
        away_team: Name of the away/second team.
        elo_ratings: Dict mapping team name to ELO score.
        historical_df: Historical match DataFrame from loader.load_historical_matches().
        stage: Match stage string (e.g. 'group', 'semi_final').
        n_form: Number of recent matches to use for form.
        n_h2h: Number of head-to-head matches to use.
        squad_loader: Optional SquadLoader instance for squad features.
        chemistry_analyzer: Optional ChemistryAnalyzer instance for chemistry features.
        elo_trends: Optional dict mapping team name to ELO trend (change over recent
            days). Defaults to 0.0 per team when not provided.

    Returns:
        Dictionary with 15 base feature keys, plus optional squad/chemistry keys.
    """
    home_elo = elo_ratings.get(home_team, _DEFAULT_ELO)
    away_elo = elo_ratings.get(away_team, _DEFAULT_ELO)

    home_form = _team_form(home_team, historical_df, n_form)
    away_form = _team_form(away_team, historical_df, n_form)
    h2h = _head_to_head(home_team, away_team, historical_df, n_h2h)

    features: dict[str, float] = {
        "elo_diff": home_elo - away_elo,
        "elo_home": home_elo,
        "elo_away": away_elo,
        "home_form_goals_scored": home_form["goals_scored"],
        "home_form_goals_conceded": home_form["goals_conceded"],
        "home_form_xg": home_form["xg"],
        "away_form_goals_scored": away_form["goals_scored"],
        "away_form_goals_conceded": away_form["goals_conceded"],
        "away_form_xg": away_form["xg"],
        "h2h_home_wins": h2h["home_win_pct"],
        "h2h_avg_goals": h2h["avg_goals"],
        "is_knockout": 1 if stage in _KNOCKOUT_STAGES else 0,
        "elo_trend_home": float(elo_trends.get(home_team, 0.0)) if elo_trends else 0.0,
        "elo_trend_away": float(elo_trends.get(away_team, 0.0)) if elo_trends else 0.0,
        "elo_trend_diff": (
            float(elo_trends.get(home_team, 0.0)) - float(elo_trends.get(away_team, 0.0))
        ) if elo_trends else 0.0,
    }

    # --- Optional squad features ---
    if squad_loader is not None:
        try:
            sq_feats = squad_loader.get_match_squad_features(home_team, away_team)
            features.update(sq_feats)

            # --- Optional chemistry features (requires squad data) ---
            if chemistry_analyzer is not None:
                try:
                    home_sq = squad_loader.get_squad(home_team)
                    away_sq = squad_loader.get_squad(away_team)
                    chem_feats = chemistry_analyzer.get_chemistry_features(
                        home_team, away_team, home_sq, away_sq
                    )
                    features.update(chem_feats)
                except Exception:
                    features.update({
                        "home_chemistry_score": 0.0,
                        "away_chemistry_score": 0.0,
                        "home_pass_network_density": 0.0,
                        "away_pass_network_density": 0.0,
                        "chemistry_diff": 0.0,
                    })
        except Exception:
            features.update({
                "squad_avg_club_elo": 1850.0,
                "squad_pct_top5_league": 0.0,
                "squad_avg_age": 26.0,
                "squad_market_value_m": 0.0,
                "squad_n_in_form": 0.0,
                "squad_elo_diff": 0.0,
            })

    return features


def _team_form(
    team: str, df: pd.DataFrame, n: int
) -> dict[str, float]:
    """Compute average form stats over the last n matches for a team."""
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    recent = df[mask].sort_values("date", ascending=False).head(n)

    if recent.empty:
        return {"goals_scored": 1.2, "goals_conceded": 1.2, "xg": 1.2}

    scored = []
    conceded = []
    xg_vals = []

    for _, row in recent.iterrows():
        if row["home_team"] == team:
            scored.append(row["home_goals"])
            conceded.append(row["away_goals"])
            xg_vals.append(row.get("home_xg", row["home_goals"]))
        else:
            scored.append(row["away_goals"])
            conceded.append(row["home_goals"])
            xg_vals.append(row.get("away_xg", row["away_goals"]))

    return {
        "goals_scored": float(np.mean(scored)),
        "goals_conceded": float(np.mean(conceded)),
        "xg": float(np.mean(xg_vals)),
    }


def _head_to_head(
    home_team: str,
    away_team: str,
    df: pd.DataFrame,
    n: int,
) -> dict[str, float]:
    """Compute head-to-head stats between two teams."""
    mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    h2h = df[mask].sort_values("date", ascending=False).head(n)

    if h2h.empty:
        return {"home_win_pct": 0.33, "avg_goals": 2.5}

    home_wins = 0
    total_goals = []

    for _, row in h2h.iterrows():
        goals = row["home_goals"] + row["away_goals"]
        total_goals.append(goals)
        if row["home_team"] == home_team and row["home_goals"] > row["away_goals"]:
            home_wins += 1
        elif row["away_team"] == home_team and row["away_goals"] > row["home_goals"]:
            home_wins += 1

    return {
        "home_win_pct": home_wins / len(h2h),
        "avg_goals": float(np.mean(total_goals)),
    }
