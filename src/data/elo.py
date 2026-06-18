"""ELO rating calculations for national football teams."""
from __future__ import annotations

import pandas as pd


# K-factors matching the World Football ELO methodology
_K_GROUP    = 40   # group stage / round of 32
_K_KNOCKOUT = 60   # round of 16, quarter, semi, third-place, final

# Stages considered knockout (higher K)
_KNOCKOUT_STAGES = {
    "round_of_16", "quarter_final", "semi_final", "third_place", "final",
    "knockout",
}

# Initial rating for a team with no prior history
_INITIAL_ELO = 1500.0


def _goal_diff_multiplier(goal_diff: int) -> float:
    """Goal-difference weighting factor (World Football ELO formula).

    A larger margin of victory provides stronger ELO evidence:
      1 goal  → 1.00
      2 goals → 1.50
      3 goals → 1.75
      N ≥ 4   → (11 + N) / 8

    Args:
        goal_diff: Absolute goal difference (always ≥ 0).

    Returns:
        Multiplier ≥ 1.0.
    """
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    if goal_diff == 3:
        return 1.75
    return (11 + goal_diff) / 8.0


def compute_elo_from_matches(
    df: pd.DataFrame,
    seed: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute ELO ratings from a chronological match history.

    Processes matches in date order, updating ratings after each game using
    the World Football ELO algorithm (goal-difference-weighted K-factor).

    K-factors:
        40 — group stage and round of 32
        60 — knockout rounds (R16, QF, SF, final)

    Args:
        df: DataFrame with columns date, home_team, away_team, home_goals,
            away_goals, stage. Rows need not be sorted — the function sorts
            by date internally.
        seed: Optional starting ratings for teams. Teams absent from both
            the seed dict and the match history start at 1500.

    Returns:
        Dict mapping each team name to its final ELO rating.
    """
    ratings: dict[str, float] = dict(seed) if seed else {}

    def _rating(team: str) -> float:
        return ratings.get(team, _INITIAL_ELO)

    sorted_df = df.sort_values("date", kind="stable")

    for _, row in sorted_df.iterrows():
        home  = str(row["home_team"])
        away  = str(row["away_team"])
        hg    = int(row["home_goals"])
        ag    = int(row["away_goals"])
        stage = str(row.get("stage", "group")).lower()

        r_h = _rating(home)
        r_a = _rating(away)

        k = _K_KNOCKOUT if stage in _KNOCKOUT_STAGES else _K_GROUP
        g = _goal_diff_multiplier(abs(hg - ag))

        e_h   = expected_score(r_h, r_a)
        s_h   = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = k * g * (s_h - e_h)

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    return ratings


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
