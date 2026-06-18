"""Parse the WC2026 fixtures CSV and map labels to model stages."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_FIXTURES_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "wc2026_fixtures.csv"


def map_stage(matchday: str, label: str) -> str:
    """Map a fixture's matchday / team label to a model stage string.

    Group-stage rows carry a numeric matchday (1, 2, 3). Knockout rows have an
    empty matchday and encode the round in the team label (e.g. 'Round of 32').

    Args:
        matchday: The matchday value as a string (may be empty).
        label: A team label that may contain the round name for knockout rows.

    Returns:
        One of: group, round_of_32, round_of_16, quarter_final, semi_final,
        third_place, final.
    """
    md = (matchday or "").strip()
    if md in {"1", "2", "3"}:
        return "group"

    text = label.lower()
    if "round of 32" in text:
        return "round_of_32"
    if "round of 16" in text:
        return "round_of_16"
    if "quarter" in text:
        return "quarter_final"
    if "semi" in text:
        return "semi_final"
    if "3rd place" in text or "third place" in text:
        return "third_place"
    if "final" in text:
        return "final"
    return "group"


def load_fixtures() -> list[dict[str, object]]:
    """Load WC2026 fixtures, excluding the actual result column.

    Returns:
        List of fixture dicts with keys: date, time, matchday, home_team,
        away_team, city, stadium, stage, predictable. The 'result' column is
        intentionally dropped. TBD placeholder rows have predictable=False.
    """
    df = pd.read_csv(_FIXTURES_PATH, dtype=str).fillna("")

    fixtures: list[dict[str, object]] = []
    for _, row in df.iterrows():
        home = row["home_team"].strip()
        away = row["away_team"].strip()
        predictable = not (home.startswith("TBD") or away.startswith("TBD"))
        stage = map_stage(row.get("matchday", ""), home + " " + away)
        fixtures.append(
            {
                "date": row.get("date", ""),
                "time": row.get("time", ""),
                "matchday": row.get("matchday", ""),
                "home_team": home,
                "away_team": away,
                "city": row.get("city", ""),
                "stadium": row.get("stadium", ""),
                "stage": stage,
                "predictable": predictable,
                "status": row.get("status", ""),
                "actual_score": row.get("result", ""),
                "predicted_score": "",
                "outcome_correct": None,
            }
        )
    return fixtures
