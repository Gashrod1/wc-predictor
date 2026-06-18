"""Fetch historical World Cup match data from martj42/international_results (GitHub).

Downloads all available WC seasons and saves each as a CSV in data/historical/.
Already-loaded seasons (WC2018, WC2022, WC2026) are skipped.

Usage (inside Docker):
    docker compose run --rm app python scripts/fetch_training_data.py

No API key required — data comes from a public GitHub repository.
Source: https://github.com/martj42/international_results
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Public GitHub dataset (49 000+ international results since 1872)
_SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

_OUT_DIR = Path(__file__).parent.parent / "data" / "historical"

# Seasons to skip (already in our CSVs)
_ALREADY_LOADED = {"2018", "2022", "2026"}

# Earliest WC to import (older data has less reliable team identity)
_MIN_YEAR = 2002

# ──────────────────────────────────────────────────────────────
# Team name mapping: martj42 → our canonical names
# ──────────────────────────────────────────────────────────────
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
    "São Tomé and Príncipe":  "Sao Tome and Principe",
    "Czech Republic":         "Czech Republic",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of Ireland":    "Ireland",
}


def _norm(name: str) -> str:
    return _TEAM_MAP.get(name.strip(), name.strip())


# ──────────────────────────────────────────────────────────────
# Stage detection (martj42 has no stage column)
# A WC has 64 matches: 48 group + 8 R16 + 4 QF + 2 SF + 1 3rd + 1 F = 16 knockout
# We sort by date and label the last 16 as "knockout", rest as "group".
# ──────────────────────────────────────────────────────────────
def _assign_stages(df: pd.DataFrame, n_knockout: int = 16) -> pd.Series:
    sorted_dates = df["date"].sort_values()
    cutoff = sorted_dates.iloc[-(n_knockout)]   # date of first knockout match
    return df["date"].apply(lambda d: "knockout" if d >= cutoff else "group")


def _process_wc_year(df: pd.DataFrame, year: str) -> pd.DataFrame:
    mask = (
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].str.startswith(year))
    )
    wc = df[mask].copy()
    if wc.empty:
        return wc

    wc["stage"] = _assign_stages(wc)
    wc["tournament"] = f"WC{year}"
    wc = wc.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    wc["home_team"] = wc["home_team"].apply(_norm)
    wc["away_team"] = wc["away_team"].apply(_norm)

    # Drop matches where goals are missing (shouldn't happen for completed WC)
    wc = wc.dropna(subset=["home_goals", "away_goals"])
    wc["home_goals"] = wc["home_goals"].astype(int)
    wc["away_goals"] = wc["away_goals"].astype(int)

    return wc[["date", "home_team", "away_team", "home_goals", "away_goals",
               "stage", "tournament"]].reset_index(drop=True)


def main() -> None:
    print("Fetching martj42/international_results …")
    try:
        resp = requests.get(_SOURCE_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Could not download dataset: {e}")
        sys.exit(1)

    df = pd.read_csv(StringIO(resp.text))
    print(f"Downloaded {len(df):,} total records.")

    # Find available WC years in the dataset
    wc_df = df[df["tournament"] == "FIFA World Cup"]
    available_years = sorted(
        y for y in wc_df["date"].str[:4].unique()
        if int(y) >= _MIN_YEAR and y not in _ALREADY_LOADED
    )
    print(f"WC seasons to import: {available_years}\n")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    imported = 0

    for year in available_years:
        wc = _process_wc_year(df, year)
        if wc.empty:
            print(f"  WC{year}: no data — skipped")
            continue

        out_path = _OUT_DIR / f"wc_{year}_matches.csv"
        wc.to_csv(out_path, index=False)
        print(f"  WC{year}: {len(wc)} matches → {out_path.name}")
        imported += 1

    print(f"\nDone — {imported} season(s) imported.")
    print("Run 'docker compose run --rm app python cli.py train' to retrain models.")


if __name__ == "__main__":
    main()
