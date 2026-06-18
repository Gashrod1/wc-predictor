"""Train and serialize both models to models/saved/."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console
from rich.progress import track

console = Console()

_SAVED_DIR = Path("models") / "saved"


def main() -> None:
    """Train Dixon-Coles and XGBoost models on all historical data."""
    from src.data.loader import load_historical_matches, load_elo_ratings
    from src.data.features import build_match_features
    from src.models.dixon_coles import DixonColesModel
    from src.models.xgboost_classifier import XGBoostOutcomeClassifier

    _SAVED_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Loading historical data...[/bold cyan]")
    df = load_historical_matches()
    elo = load_elo_ratings()
    console.print(f"  Loaded {len(df)} matches from {df['tournament'].nunique()} tournaments")

    console.print("\n[bold cyan]Training Dixon-Coles model...[/bold cyan]")
    dc_model = DixonColesModel()
    dc_model.fit(df)
    console.print(f"  Teams modelled: {len(dc_model.attack_params)}")
    console.print(f"  rho = {dc_model.rho:.4f}, home_advantage = {dc_model.home_advantage:.4f}")

    console.print("\n[bold cyan]Building XGBoost features...[/bold cyan]")
    feature_rows = []
    labels = []
    for _, row in track(df.iterrows(), total=len(df), description="Engineering features"):
        feats = build_match_features(
            row["home_team"], row["away_team"], elo, df, stage=row["stage"]
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

    console.print("\n[bold cyan]Training XGBoost classifier...[/bold cyan]")
    xgb_model = XGBoostOutcomeClassifier()
    xgb_model.fit(X, y)

    fi = xgb_model.get_feature_importance()
    console.print("  Top 5 features:")
    for _, feat_row in fi.head(5).iterrows():
        console.print(f"    {feat_row['feature']:<35} {feat_row['importance']:.4f}")

    console.print("\n[bold cyan]Saving models...[/bold cyan]")
    joblib.dump(dc_model, _SAVED_DIR / "dixon_coles.joblib")
    joblib.dump(xgb_model, _SAVED_DIR / "xgboost.joblib")
    console.print(f"  Saved to {_SAVED_DIR}/")

    console.print("\n[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    main()
