"""World Cup Predictor CLI — predict, backtest, train."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _make_confidence_bar(confidence: float, width: int = 10) -> str:
    filled = round(confidence * width)
    return "█" * filled + "░" * (width - filled)


def _winner_name(scenario: str, home: str, away: str) -> str:
    """Translate 'home'/'draw'/'away' to a readable team name."""
    return {"home": home, "draw": "Match nul", "away": away}.get(scenario, scenario)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run a match prediction and display results."""
    from src.prediction.predictor import predict_match

    home = args.home
    away = args.away
    stage = args.stage or "group"

    console.print(f"\n[dim]Predicting {home} vs {away} ({stage})...[/dim]")

    # neutral=True par défaut (terrain neutre WC). --host désactive pour USA/CAN/MEX à domicile.
    neutral = not getattr(args, "host", False)
    result = predict_match(home, away, stage=stage, neutral=neutral)
    probs = result.outcome_probabilities

    home_pct = probs["home_win"] * 100
    draw_pct = probs["draw"] * 100
    away_pct = probs["away_win"] * 100

    winner_label = {"home": home, "draw": "Draw", "away": away}[result.predicted_winner]
    winner_pct = max(home_pct, draw_pct, away_pct)

    lines = [
        f"  [bold]{home}[/bold] vs [bold]{away}[/bold] — {stage.replace('_', ' ').title()}",
        "",
        f"  Vainqueur probable : [bold green]{winner_label}[/bold green] ({winner_pct:.1f}%)",
        f"  {home}: {home_pct:.1f}%  |  Nul: {draw_pct:.1f}%  |  {away}: {away_pct:.1f}%",
        "",
        f"  Score le plus probable : [bold]{result.most_likely_score}[/bold] ({result.top_scores[0]['probability']*100:.1f}%)",
        "  Top 5 scores :",
    ]
    for i, s in enumerate(result.top_scores, 1):
        lines.append(f"    {i}. {s['score']:<6}→  {s['probability']*100:.1f}%")

    conf_bar = _make_confidence_bar(result.confidence)

    # --- Accord / Désaccord des modèles ---
    div = result.model_divergence
    if div > 0.20:
        # Show both model scenarios side by side
        lines += [
            "",
            f"  [yellow]⚠ Les modèles divergent (écart {div*100:.0f}%)[/yellow]",
            f"  Dixon-Coles : [cyan]{_winner_name(result.scenario_dc, home, away)}[/cyan]",
            f"  XGBoost     : [magenta]{_winner_name(result.scenario_xgb, home, away)}[/magenta]",
        ]
    elif result.model_agreement:
        lines.append("")
        lines.append("  [green]✓ Accord des modèles[/green]")
    else:
        lines.append("")
        lines.append("  [yellow]✗ Désaccord des modèles[/yellow]")

    lines += [
        "",
        f"  Confiance : {conf_bar} {result.confidence*100:.0f}%",
    ]

    content = "\n".join(lines)
    console.print(Panel(content, border_style="blue", padding=(0, 1)))

    if args.json:
        output = {
            "home_team": result.home_team,
            "away_team": result.away_team,
            "stage": stage,
            "outcome_probabilities": result.outcome_probabilities,
            "predicted_winner": result.predicted_winner,
            "most_likely_score": result.most_likely_score,
            "top_scores": result.top_scores,
            "confidence": result.confidence,
            "model_agreement": result.model_agreement,
            "model_divergence": result.model_divergence,
            "scenario_dc": result.scenario_dc,
            "scenario_xgb": result.scenario_xgb,
        }
        console.print_json(json.dumps(output, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtesting and display metrics."""
    import pandas as pd
    from src.data.loader import load_historical_matches, load_elo_ratings
    from src.data.features import build_match_features
    from src.models.dixon_coles import DixonColesModel
    from src.models.xgboost_classifier import XGBoostOutcomeClassifier
    from src.models.ensemble import EnsemblePredictor
    from src.evaluation.backtesting import run_backtest

    tournament = args.tournament

    console.print(f"\n[bold cyan]Running backtest on {tournament}...[/bold cyan]")

    all_df = load_historical_matches()
    target_df = load_historical_matches(tournament=tournament)

    if target_df.empty:
        console.print(f"[red]No data found for tournament: {tournament}[/red]")
        sys.exit(1)

    # ELO computed from matches BEFORE the target tournament (no leakage)
    elo = load_elo_ratings(as_of_tournament=tournament)

    # Train on data excluding the target tournament
    train_df = all_df[~all_df["tournament"].str.contains(tournament, na=False)]
    if train_df.empty:
        train_df = all_df

    console.print("  Training models on held-out data...")
    dc_model = DixonColesModel()
    dc_model.fit(train_df)

    rows, labels = [], []
    for _, row in train_df.iterrows():
        feats = build_match_features(row["home_team"], row["away_team"], elo, all_df, stage=row["stage"])
        rows.append(feats)
        labels.append(2 if row["home_goals"] > row["away_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0))

    xgb_model = XGBoostOutcomeClassifier()
    xgb_model.fit(pd.DataFrame(rows), pd.Series(labels))

    ensemble = EnsemblePredictor(dc_model=dc_model, xgb_model=xgb_model)

    console.print(f"  Predicting {len(target_df)} matches...")
    metrics = run_backtest(ensemble, target_df)

    table = Table(title=f"Backtest Results — {tournament}", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    acc_color = "green" if metrics["outcome_accuracy"] >= 0.5 else "yellow"
    table.add_row("Outcome Accuracy", f"[{acc_color}]{metrics['outcome_accuracy']*100:.1f}%[/{acc_color}]")
    table.add_row("Exact Score Accuracy", f"{metrics['exact_score_accuracy']*100:.1f}%")
    table.add_row("Top-3 Score Accuracy", f"{metrics['top3_score_accuracy']*100:.1f}%")
    table.add_row("Brier Score", f"{metrics['brier_score']:.4f}")
    table.add_row("Log Loss", f"{metrics['log_loss']:.4f}")

    console.print(table)


def cmd_train(_args: argparse.Namespace) -> None:
    """Retrain all models."""
    import train as train_module
    train_module.main()


def main() -> None:
    """Entry point for the World Cup Match Predictor CLI."""
    parser = argparse.ArgumentParser(
        description="World Cup Football Match Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict a match outcome")
    predict_parser.add_argument("--home", required=True, help="First team name")
    predict_parser.add_argument("--away", required=True, help="Second team name")
    predict_parser.add_argument("--stage", default="group", help="Match stage (default: group)")
    predict_parser.add_argument("--json", action="store_true", help="Also output raw JSON")
    predict_parser.add_argument(
        "--host", action="store_true",
        help="First team plays at home (disables neutral-venue mode — use for USA/Canada/Mexico group games)"
    )

    bt_parser = subparsers.add_parser("backtest", help="Run backtesting on a tournament")
    bt_parser.add_argument("--tournament", required=True, help="Tournament ID e.g. WC2022")

    subparsers.add_parser("train", help="Retrain models from historical data")

    args = parser.parse_args()
    {"predict": cmd_predict, "backtest": cmd_backtest, "train": cmd_train}[args.command](args)


if __name__ == "__main__":
    main()
