"""WC2026 live results sync from ESPN API (no authentication required).

Fetches finished match results from ESPN's unofficial soccer API and updates
data/historical/wc_2026_matches.csv automatically.

Cache TTL: 30 minutes — balances freshness vs. network calls.
The ESPN API has no rate limits for public use.

Usage:
    from src.data.wc2026_sync import sync_wc2026_results
    new_matches = sync_wc2026_results()   # returns count of newly added matches
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_DATA_DIR   = Path(__file__).parent.parent.parent / "data"
_CSV_PATH   = _DATA_DIR / "historical" / "wc_2026_matches.csv"
_CACHE_PATH = _DATA_DIR / "raw" / "wc2026_espn_cache.json"
_CACHE_TTL  = 1800  # 30 minutes

# ESPN scoreboard endpoint — no auth required
_ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)

# WC2026 runs June 11 – July 19 2026
_WC_START = "20260611"
_WC_END   = "20260719"

# Team name normalisation: ESPN display name → our canonical name
_TEAM_MAP: dict[str, str] = {
    "United States":          "USA",
    "Korea Republic":         "South Korea",
    "Korea DPR":              "North Korea",
    "Côte d'Ivoire":          "Ivory Coast",
    "IR Iran":                "Iran",
    "China PR":               "China",
    "Trinidad & Tobago":      "Trinidad and Tobago",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Cape Verde Islands":     "Cape Verde",
    "Czech Republic":         "Czech Republic",
    "Republic of Ireland":    "Ireland",
    # ESPN-specific names
    "Czechia":                "Czech Republic",
    "DR Congo":               "DR Congo",
    "Congo DR":               "DR Congo",
}


def _norm(name: str) -> str:
    return _TEAM_MAP.get(name.strip(), name.strip())


def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    return (time.time() - _CACHE_PATH.stat().st_mtime) < _CACHE_TTL


def _fetch_espn_day(date_str: str) -> list[dict]:
    """Fetch finished matches for a single day (YYYYMMDD) from ESPN."""
    try:
        r = requests.get(_ESPN_URL, params={"dates": date_str}, timeout=10)
        r.raise_for_status()
        events = r.json().get("events", [])
        rows = []
        for e in events:
            comp = e["competitions"][0]
            status_desc = e["status"]["type"]["description"]
            if status_desc not in ("Full Time", "Final", "FT", "AET", "Pen"):
                continue  # skip in-progress or not-started
            try:
                home_t = next(t for t in comp["competitors"] if t["homeAway"] == "home")
                away_t = next(t for t in comp["competitors"] if t["homeAway"] == "away")
                rows.append({
                    "date":       e["date"][:10],
                    "home_team":  _norm(home_t["team"]["displayName"]),
                    "away_team":  _norm(away_t["team"]["displayName"]),
                    "home_goals": int(home_t["score"]),
                    "away_goals": int(away_t["score"]),
                    "stage":      "group",   # refined below
                    "tournament": "WC2026",
                })
            except (KeyError, ValueError, StopIteration):
                continue
        return rows
    except Exception:
        return []


def _assign_knockout_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Mark matches after group stage (last 16 matches) as knockout."""
    if len(df) < 48:
        return df  # not enough matches to identify knockout stage yet
    sorted_df = df.sort_values("date")
    knockout_dates = sorted_df["date"].iloc[48:].unique()
    df = df.copy()
    df.loc[df["date"].isin(knockout_dates), "stage"] = "knockout"
    return df


def sync_wc2026_results(force: bool = False) -> int:
    """Fetch the latest WC2026 results and update the historical CSV.

    Skips the network call if the cache is fresh (< 30 min old), unless
    `force=True`.

    Args:
        force: If True, bypasses the TTL cache and always fetches fresh data.

    Returns:
        Number of newly added matches (0 if cache was fresh or no new results).
    """
    if not force and _cache_is_fresh():
        return 0

    # Build date range: WC start up to today + 1 (to catch matches just ended)
    today_str = datetime.utcnow().strftime("%Y%m%d")
    end_str   = min(today_str, _WC_END)

    # Generate all dates from WC start to today
    start_dt = datetime.strptime(_WC_START, "%Y%m%d")
    end_dt   = datetime.strptime(end_str, "%Y%m%d")
    dates = []
    d = start_dt
    while d <= end_dt:
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    # Fetch all days (ESPN returns empty list for days with no matches)
    all_rows: list[dict] = []
    for date_str in dates:
        all_rows.extend(_fetch_espn_day(date_str))

    if not all_rows:
        # Update cache timestamp even on empty result
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({"synced_at": time.time(), "n": 0}))
        return 0

    new_df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["date", "home_team", "away_team"]
    )

    # Load existing CSV
    try:
        existing = pd.read_csv(_CSV_PATH)
        before = len(existing)
    except FileNotFoundError:
        existing = pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_goals", "away_goals", "stage", "tournament"]
        )
        before = 0

    # Merge and deduplicate
    combined = pd.concat([new_df, existing], ignore_index=True).drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    ).sort_values("date").reset_index(drop=True)

    # Refine stage labels
    combined = _assign_knockout_stages(combined)

    # Save
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(_CSV_PATH, index=False)

    # Update cache
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps({
        "synced_at": time.time(),
        "n_matches": len(combined),
    }))

    added = len(combined) - before
    return max(added, 0)


def get_sync_status() -> dict:
    """Return info about the last sync (for API health endpoint)."""
    if not _CACHE_PATH.exists():
        return {"last_sync": None, "age_minutes": None, "n_matches": 0}
    try:
        data = json.loads(_CACHE_PATH.read_text())
        age = (time.time() - data.get("synced_at", 0)) / 60
        return {
            "last_sync": datetime.utcfromtimestamp(data["synced_at"]).isoformat() + "Z",
            "age_minutes": round(age, 1),
            "n_matches": data.get("n_matches", 0),
        }
    except Exception:
        return {"last_sync": None, "age_minutes": None, "n_matches": 0}
