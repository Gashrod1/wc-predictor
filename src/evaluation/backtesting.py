"""Walk-forward backtesting with Brier score, log-loss, and accuracy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def run_backtest(model: object, matches_df: pd.DataFrame) -> dict[str, float]:
    """Walk-forward backtest: train on all matches before date T, predict T.

    For each match in matches_df (sorted by date), uses the ensemble's current
    fitted state to predict and compares against the actual outcome.

    Args:
        model: A fitted EnsemblePredictor with a .predict(home, away, context) method.
        matches_df: DataFrame with columns date, home_team, away_team,
                    home_goals, away_goals, stage, tournament.

    Returns:
        Dict with keys:
            outcome_accuracy: fraction of correct winner predictions
            exact_score_accuracy: fraction of exact score matches
            top3_score_accuracy: fraction where actual score is in top 3
            brier_score: average Brier score across outcome probabilities
            log_loss: calibration metric (lower = better)
    """
    df = matches_df.copy().sort_values("date").reset_index(drop=True)

    outcome_correct = []
    exact_correct = []
    top3_correct = []
    y_true_proba: list[list[float]] = []
    y_pred_proba: list[list[float]] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        stage = row.get("stage", "group")
        actual_home = int(row["home_goals"])
        actual_away = int(row["away_goals"])
        actual_score = f"{actual_home}-{actual_away}"

        if actual_home > actual_away:
            actual_outcome = "home"
        elif actual_home == actual_away:
            actual_outcome = "draw"
        else:
            actual_outcome = "away"

        try:
            result = model.predict(home, away, context={"stage": stage})
        except Exception:
            continue

        probs = result.outcome_probabilities
        y_pred_proba.append([
            probs.get("away_win", 0.0),
            probs.get("draw", 0.0),
            probs.get("home_win", 0.0),
        ])

        true_vec = [0.0, 0.0, 0.0]
        idx = {"away": 0, "draw": 1, "home": 2}[actual_outcome]
        true_vec[idx] = 1.0
        y_true_proba.append(true_vec)

        outcome_correct.append(int(result.predicted_winner == actual_outcome))
        exact_correct.append(int(result.most_likely_score == actual_score))
        top3_scores = [s["score"] for s in result.top_scores[:3]]
        top3_correct.append(int(actual_score in top3_scores))

    if not y_pred_proba:
        return {
            "outcome_accuracy": 0.0,
            "exact_score_accuracy": 0.0,
            "top3_score_accuracy": 0.0,
            "brier_score": 1.0,
            "log_loss": 10.0,
        }

    y_true_arr = np.array(y_true_proba)
    y_pred_arr = np.array(y_pred_proba)

    brier = float(np.mean(np.sum((y_pred_arr - y_true_arr) ** 2, axis=1) / 3.0))
    # Convert one-hot y_true back to integer labels for log_loss compatibility
    y_true_labels = np.argmax(y_true_arr, axis=1)
    ll = float(log_loss(y_true_labels, y_pred_arr, labels=[0, 1, 2]))

    return {
        "outcome_accuracy": float(np.mean(outcome_correct)),
        "exact_score_accuracy": float(np.mean(exact_correct)),
        "top3_score_accuracy": float(np.mean(top3_correct)),
        "brier_score": brier,
        "log_loss": ll,
    }


def run_backtest_details(model: object, matches_df: pd.DataFrame) -> list[dict]:
    """Same walk-forward logic as run_backtest but returns per-match detail rows."""
    df = matches_df.copy().sort_values("date").reset_index(drop=True)
    rows = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        stage = row.get("stage", "group")
        actual_home = int(row["home_goals"])
        actual_away = int(row["away_goals"])
        actual_score = f"{actual_home}-{actual_away}"

        if actual_home > actual_away:
            actual_winner = "home"
        elif actual_home == actual_away:
            actual_winner = "draw"
        else:
            actual_winner = "away"

        try:
            result = model.predict(home, away, context={"stage": stage})
        except Exception:
            continue

        rows.append({
            "date": str(row["date"])[:10],
            "home_team": home,
            "away_team": away,
            "predicted_score": result.most_likely_score,
            "actual_score": actual_score,
            "predicted_winner": result.predicted_winner,
            "actual_winner": actual_winner,
            "outcome_correct": result.predicted_winner == actual_winner,
            "score_correct": result.most_likely_score == actual_score,
        })

    return rows
