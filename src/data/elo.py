"""ELO rating calculations for national football teams."""
from __future__ import annotations

import pandas as pd


def expected_score(elo_a: float, elo_b: float) -> float:
    """Return expected score for team A against team B using ELO formula.

    Args:
        elo_a: ELO rating of team A.
        elo_b: ELO rating of team B.

    Returns:
        Expected score in [0, 1], where 1 = win, 0.5 = draw, 0 = loss.
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    home_elo: float,
    away_elo: float,
    home_goals: int,
    away_goals: int,
    k_factor: float = 32.0,
) -> tuple[float, float]:
    """Update ELO ratings after a match.

    Args:
        home_elo: Current ELO of home team.
        away_elo: Current ELO of away team.
        home_goals: Goals scored by home team.
        away_goals: Goals scored by away team.
        k_factor: K-factor controlling rating volatility.

    Returns:
        Tuple of (new_home_elo, new_away_elo).
    """
    expected_home = expected_score(home_elo, away_elo)
    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals == away_goals:
        actual_home = 0.5
    else:
        actual_home = 0.0

    delta = k_factor * (actual_home - expected_home)
    return home_elo + delta, away_elo - delta


def load_elo_from_csv(path: str) -> dict[str, float]:
    """Load ELO ratings from a CSV file with columns 'team' and 'elo'.

    Args:
        path: Path to the CSV file.

    Returns:
        Dictionary mapping team name to ELO score.
    """
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["elo"].astype(float)))
