"""Train and serialize both models to models/saved/."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console()

_SAVED_DIR = Path("models") / "saved"
_DATA_DIR = Path("data")


def main() -> None:
    """Train Dixon-Coles and XGBoost models on all available data."""
    from src.data.loader import load_historical_matches, load_elo_ratings, load_elo_trends
    from src.data.features import build_match_features
    from src.models.dixon_coles import DixonColesModel
    from src.models.xgboost_classifier import XGBoostOutcomeClassifier

    _SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # 1. Load all data sources
    # ──────────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading data...[/bold cyan]")
    wc_df = load_historical_matches()
    elo = load_elo_ratings()
    elo_trends = load_elo_trends()

    # Competitive international matches (qualifiers, EURO, Copa, etc.)
    comp_path = _DATA_DIR / "historical" / "international_competitive.csv"
    try:
        comp_df = pd.read_csv(comp_path, parse_dates=["date"])
        console.print(f"  WC matches: [bold]{len(wc_df)}[/bold]")
        console.print(f"  Competitive matches: [bold]{len(comp_df):,}[/bold]")
        # Merge: WC + competitive, dedup, sort by date
        all_df = pd.concat([wc_df, comp_df], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        ).sort_values("date").reset_index(drop=True)
    except FileNotFoundError:
        console.print("[yellow]  No competitive CSV found — run scripts/fetch_training_data.py first[/yellow]")
        console.print("[yellow]  Falling back to WC-only data[/yellow]")
        all_df = wc_df
        comp_df = pd.DataFrame()

    console.print(
        f"  Total for XGBoost training: [bold]{len(all_df):,} matches[/bold]"
        f" ({all_df['tournament'].nunique()} tournaments)"
    )
    console.print(f"  ELO trends computed for {len(elo_trends)} teams")

    # ──────────────────────────────────────────────────────────────
    # 2. Dixon-Coles (WC matches only — score distribution is WC-specific)
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Training Dixon-Coles model (WC matches)...[/bold cyan]")
    dc_model = DixonColesModel()
    dc_model.fit(wc_df)
    console.print(f"  Teams modelled: {len(dc_model.attack_params)}")
    console.print(f"  rho = {dc_model.rho:.4f}, home_advantage = {dc_model.home_advantage:.4f}")

    # ──────────────────────────────────────────────────────────────
    # 3. XGBoost features (all competitive matches)
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Building XGBoost features...[/bold cyan]")
    feature_rows = []
    labels = []
    for _, row in track(all_df.iterrows(), total=len(all_df), description="Engineering features"):
        feats = build_match_features(
            row["home_team"], row["away_team"], elo, all_df,
            stage=str(row.get("stage", "group")),
            elo_trends=elo_trends,
            squad_loader=None,
            chemistry_analyzer=None,
        )
        feature_rows.append(feats)
        if row["home_goals"] > row["away_goals"]:
            labels.append(2)
        elif row["home_goals"] == row["away_goals"]:
            labels.append(1)
        else:
            labels.append(0)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels)
    console.print(
        f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features"
    )

    # ──────────────────────────────────────────────────────────────
    # 4. XGBoost hyperparameter tuning + training
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Training XGBoost classifier...[/bold cyan]")
    xgb_model = XGBoostOutcomeClassifier()

    if len(X) >= 300:
        console.print(
            f"  [dim]Running hyperparameter search"
            f" (RandomizedSearchCV, TimeSeriesSplit n=3, 40 iterations)…[/dim]"
        )
        best_params = xgb_model.tune_hyperparameters(X, y, n_iter=40)
        if best_params:
            console.print(f"  Best params found: {best_params}")
    else:
        console.print("  [yellow]Too few samples for tuning — using defaults[/yellow]")
        xgb_model.fit(X, y)

    fi = xgb_model.get_feature_importance()
    console.print("  Top 8 features:")
    for _, feat_row in fi.head(8).iterrows():
        console.print(
            f"    {feat_row['feature']:<40} {feat_row['importance']:.4f}"
        )

    # ──────────────────────────────────────────────────────────────
    # 5. Save
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Saving models...[/bold cyan]")
    joblib.dump(dc_model, _SAVED_DIR / "dixon_coles.joblib")
    joblib.dump(xgb_model, _SAVED_DIR / "xgboost.joblib")
    console.print(f"  Saved to {_SAVED_DIR}/")

    console.print("\n[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    main()
