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
    from src.utils.wandb_logger import wandb_init, wandb_log, wandb_log_table, wandb_finish

    _SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # 1. Load all data sources
    # ──────────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading data...[/bold cyan]")
    wc_df = load_historical_matches()
    elo = load_elo_ratings()
    elo_trends = load_elo_trends()

    comp_path = _DATA_DIR / "historical" / "international_competitive.csv"
    try:
        comp_df = pd.read_csv(comp_path, parse_dates=["date"])
        console.print(f"  WC matches: [bold]{len(wc_df)}[/bold]")
        console.print(f"  Competitive matches: [bold]{len(comp_df):,}[/bold]")
        all_df = pd.concat([wc_df, comp_df], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        ).sort_values("date").reset_index(drop=True)
    except FileNotFoundError:
        console.print("[yellow]  No competitive CSV — run scripts/fetch_training_data.py[/yellow]")
        all_df = wc_df
        comp_df = pd.DataFrame()

    n_wc = len(wc_df)
    n_comp = len(comp_df) if not comp_df.empty else 0
    n_total = len(all_df)
    n_tournaments = all_df["tournament"].nunique()

    console.print(
        f"  Total for XGBoost: [bold]{n_total:,} matches[/bold] ({n_tournaments} tournaments)"
    )
    console.print(f"  ELO trends: {len(elo_trends)} teams")

    # ──────────────────────────────────────────────────────────────
    # 2. Init wandb run
    # ──────────────────────────────────────────────────────────────
    run_config = {
        "n_wc_matches": n_wc,
        "n_competitive_matches": n_comp,
        "n_total_matches": n_total,
        "n_tournaments": n_tournaments,
        "n_teams_with_elo_trend": len(elo_trends),
        "dc_training": "WC only",
        "xgb_training": "WC + competitive",
        "elo_algorithm": "World Football ELO (tournament K-factors)",
    }
    wandb_init(config=run_config, job_type="train", tags=["dixon-coles", "xgboost", "ensemble"])

    # ──────────────────────────────────────────────────────────────
    # 3. Dixon-Coles (WC matches only)
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Training Dixon-Coles model (WC matches)...[/bold cyan]")
    dc_model = DixonColesModel()
    dc_model.fit(wc_df)
    console.print(f"  Teams modelled: {len(dc_model.attack_params)}")
    console.print(f"  rho = {dc_model.rho:.4f}, home_advantage = {dc_model.home_advantage:.4f}")

    wandb_log({
        "dc/n_teams": len(dc_model.attack_params),
        "dc/rho": dc_model.rho,
        "dc/home_advantage": dc_model.home_advantage,
    })

    # ──────────────────────────────────────────────────────────────
    # 4. XGBoost feature engineering
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
    console.print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # Log label distribution
    from collections import Counter
    label_counts = Counter(labels)
    wandb_log({
        "data/n_home_wins":  label_counts[2],
        "data/n_draws":      label_counts[1],
        "data/n_away_wins":  label_counts[0],
        "data/n_features":   X.shape[1],
        "data/n_samples":    X.shape[0],
        "data/home_win_pct": label_counts[2] / len(labels),
        "data/draw_pct":     label_counts[1] / len(labels),
        "data/away_win_pct": label_counts[0] / len(labels),
    })

    # ──────────────────────────────────────────────────────────────
    # 5. XGBoost hyperparameter tuning + training
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Training XGBoost classifier...[/bold cyan]")
    xgb_model = XGBoostOutcomeClassifier()

    if len(X) >= 300:
        console.print(
            "  [dim]Hyperparameter search (RandomizedSearchCV + TimeSeriesSplit, 40 iter)…[/dim]"
        )
        best_params = xgb_model.tune_hyperparameters(X, y, n_iter=40)
        if best_params:
            console.print(f"  Best params: {best_params}")
            wandb_log({"xgb/best_" + k: v for k, v in best_params.items()})
    else:
        console.print("  [yellow]Too few samples — using defaults[/yellow]")
        xgb_model.fit(X, y)

    # Log feature importances
    fi = xgb_model.get_feature_importance()
    console.print("  Top 8 features:")
    for _, feat_row in fi.head(8).iterrows():
        console.print(f"    {feat_row['feature']:<40} {feat_row['importance']:.4f}")

    wandb_log_table(
        name="feature_importances",
        columns=["feature", "importance"],
        rows=fi.values.tolist(),
    )
    # Also log as individual metrics for easy charting
    for _, feat_row in fi.iterrows():
        wandb_log({f"importance/{feat_row['feature']}": feat_row["importance"]})

    # ──────────────────────────────────────────────────────────────
    # 6. Quick in-sample metrics (WC only — the real test)
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Computing in-sample metrics on WC data...[/bold cyan]")
    try:
        from src.models.ensemble import EnsemblePredictor
        ensemble = EnsemblePredictor(
            dc_model=dc_model,
            xgb_model=xgb_model,
            elo_trends=elo_trends,
            historical_df=wc_df,
        )
        correct = 0
        total = 0
        for _, row in wc_df.iterrows():
            try:
                result = ensemble.predict(
                    row["home_team"], row["away_team"],
                    context={"stage": row["stage"], "neutral": True},
                )
                actual = (
                    "home" if row["home_goals"] > row["away_goals"]
                    else ("away" if row["home_goals"] < row["away_goals"] else "draw")
                )
                if result.predicted_winner == actual:
                    correct += 1
                total += 1
            except Exception:
                continue
        in_sample_acc = correct / total if total > 0 else 0.0
        console.print(f"  In-sample accuracy (WC): {in_sample_acc*100:.1f}% ({correct}/{total})")
        wandb_log({"metrics/in_sample_accuracy_wc": in_sample_acc})
    except Exception as e:
        console.print(f"  [yellow]In-sample check skipped: {e}[/yellow]")

    # ──────────────────────────────────────────────────────────────
    # 7. Log top ELO rankings
    # ──────────────────────────────────────────────────────────────
    top_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:20]
    wandb_log_table(
        name="elo_rankings",
        columns=["rank", "team", "elo"],
        rows=[[i + 1, t, round(v, 1)] for i, (t, v) in enumerate(top_elo)],
    )
    top_trends = sorted(elo_trends.items(), key=lambda x: x[1], reverse=True)[:15]
    bottom_trends = sorted(elo_trends.items(), key=lambda x: x[1])[:15]
    wandb_log_table(
        name="elo_trend_top15",
        columns=["team", "trend_180d"],
        rows=[[t, round(v, 2)] for t, v in top_trends],
    )
    wandb_log_table(
        name="elo_trend_bottom15",
        columns=["team", "trend_180d"],
        rows=[[t, round(v, 2)] for t, v in bottom_trends],
    )

    # ──────────────────────────────────────────────────────────────
    # 8. Save models
    # ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Saving models...[/bold cyan]")
    joblib.dump(dc_model, _SAVED_DIR / "dixon_coles.joblib")
    joblib.dump(xgb_model, _SAVED_DIR / "xgboost.joblib")
    console.print(f"  Saved to {_SAVED_DIR}/")

    wandb_finish()
    console.print("\n[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    main()
