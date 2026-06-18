"""Weighted ensemble combining Dixon-Coles and XGBoost predictions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.features import build_match_features
from src.data.loader import load_historical_matches, load_elo_ratings
from src.models.dixon_coles import DixonColesModel
from src.models.xgboost_classifier import XGBoostOutcomeClassifier

_DC_WEIGHT = 0.55
_XGB_WEIGHT = 0.45


@dataclass
class PredictionResult:
    """Container for a single match prediction.

    Attributes:
        home_team: Name of the home team.
        away_team: Name of the away team.
        outcome_probabilities: Dict with keys home_win, draw, away_win.
        predicted_winner: "home", "draw", or "away".
        most_likely_score: E.g. "2-1".
        top_scores: List of dicts with keys score and probability.
        confidence: Float in [0.1, 0.95] penalised by model divergence.
        model_agreement: True if both models agree on the winner.
        model_divergence: Max absolute probability difference across outcomes
            between DC and XGB models. 0.0 = perfect agreement.
        scenario_dc: Winner predicted by Dixon-Coles alone.
        scenario_xgb: Winner predicted by XGBoost alone.
    """

    home_team: str
    away_team: str
    outcome_probabilities: dict[str, float]
    predicted_winner: str
    most_likely_score: str
    top_scores: list[dict[str, object]]
    confidence: float
    model_agreement: bool
    model_divergence: float = 0.0
    scenario_dc: str = ""
    scenario_xgb: str = ""


class EnsemblePredictor:
    """Weighted ensemble of Dixon-Coles and XGBoost outcome predictors.

    Weights can be either the fixed defaults (0.55 / 0.45) or learned via
    `fit_weights()` which trains a LogisticRegression on historical predictions.

    Args:
        dc_model: Fitted DixonColesModel instance.
        xgb_model: Fitted XGBoostOutcomeClassifier instance.
        dc_weight: Weight for Dixon-Coles (default 0.55), used when no
            weight model has been fitted.
        xgb_weight: Weight for XGBoost (default 0.45), used when no
            weight model has been fitted.
        squad_loader: Optional SquadLoader instance for squad-based features.
        chemistry_analyzer: Optional ChemistryAnalyzer instance.
        historical_df: Historical match data for feature engineering. If None,
            loads all available data via load_historical_matches(). Pass a
            filtered DataFrame (e.g. excluding the target tournament) to avoid
            data leakage during backtesting.
    """

    def __init__(
        self,
        dc_model: DixonColesModel,
        xgb_model: XGBoostOutcomeClassifier,
        dc_weight: float = _DC_WEIGHT,
        xgb_weight: float = _XGB_WEIGHT,
        squad_loader: Any = None,
        chemistry_analyzer: Any = None,
        historical_df: pd.DataFrame | None = None,
    ) -> None:
        self.dc_model = dc_model
        self.xgb_model = xgb_model
        self.dc_weight = dc_weight
        self.xgb_weight = xgb_weight
        self._elo_ratings = load_elo_ratings()
        self._historical_df = historical_df if historical_df is not None else load_historical_matches()
        self._weight_model: Any = None  # set by fit_weights()
        self._squad_loader: Any = squad_loader
        self._chemistry_analyzer: Any = chemistry_analyzer

    def fit_weights(
        self,
        historical_df: pd.DataFrame,
        elo_ratings: dict[str, float],
    ) -> None:
        """Learn optimal blending weights via LogisticRegression.

        For each historical match, computes DC and XGB probabilities separately
        and uses them as a 6-feature input (3 DC + 3 XGB) to a multinomial
        LogisticRegression. The fitted model replaces the fixed-weight blend
        in subsequent calls to predict().

        Args:
            historical_df: Historical matches DataFrame.
            elo_ratings: Dict mapping team name to ELO score.
        """
        from sklearn.linear_model import LogisticRegression

        X_rows: list[list[float]] = []
        y_rows: list[int] = []

        for _, row in historical_df.iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            try:
                dc_p = self.dc_model.predict_outcome_probabilities(home, away)
                feats = build_match_features(home, away, elo_ratings, historical_df, stage=str(row["stage"]))
                xgb_p = self.xgb_model.predict_proba(feats)
                X_rows.append([
                    dc_p["home_win"], dc_p["draw"], dc_p["away_win"],
                    xgb_p["home_win"], xgb_p["draw"], xgb_p["away_win"],
                ])
                if row["home_goals"] > row["away_goals"]:
                    y_rows.append(2)
                elif row["home_goals"] == row["away_goals"]:
                    y_rows.append(1)
                else:
                    y_rows.append(0)
            except Exception:
                continue  # skip rows where prediction fails (unknown teams)

        if len(X_rows) < 10:
            return  # not enough data to fit reliably; keep fixed weights

        lr = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", multi_class="multinomial")
        lr.fit(np.array(X_rows), np.array(y_rows))
        self._weight_model = lr

    def predict(
        self,
        home_team: str,
        away_team: str,
        context: dict[str, object] | None = None,
    ) -> PredictionResult:
        """Generate an ensemble prediction for a match.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            context: Optional dict with key 'stage' (e.g. 'semi_final').

        Returns:
            PredictionResult with combined probabilities, divergence metadata,
            and per-model scenarios.
        """
        stage = (context or {}).get("stage", "group")

        dc_probs = self.dc_model.predict_outcome_probabilities(home_team, away_team)
        top_scores = self.dc_model.predict_top_scores(home_team, away_team, top_n=5)

        features = build_match_features(
            home_team, away_team, self._elo_ratings, self._historical_df,
            stage=str(stage),
            squad_loader=self._squad_loader,
            chemistry_analyzer=self._chemistry_analyzer,
        )
        xgb_probs = self.xgb_model.predict_proba(features)

        # --- Blend ---
        if self._weight_model is not None:
            features_arr = np.array([[
                dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"],
                xgb_probs["home_win"], xgb_probs["draw"], xgb_probs["away_win"],
            ]])
            lr_proba = self._weight_model.predict_proba(features_arr)[0]
            classes: list[int] = list(self._weight_model.classes_)
            label_map = {0: "away_win", 1: "draw", 2: "home_win"}
            blended: dict[str, float] = {label_map[c]: float(p) for c, p in zip(classes, lr_proba)}
        else:
            blended = {
                "home_win": self.dc_weight * dc_probs["home_win"] + self.xgb_weight * xgb_probs["home_win"],
                "draw":     self.dc_weight * dc_probs["draw"]     + self.xgb_weight * xgb_probs["draw"],
                "away_win": self.dc_weight * dc_probs["away_win"] + self.xgb_weight * xgb_probs["away_win"],
            }
            total = sum(blended.values())
            blended = {k: v / total for k, v in blended.items()}

        predicted_winner = max(blended, key=blended.get).replace("_win", "")

        # --- Per-model scenarios ---
        dc_winner = max(dc_probs, key=dc_probs.get).replace("_win", "")
        xgb_winner = max(xgb_probs, key=xgb_probs.get).replace("_win", "")
        model_agreement = dc_winner == xgb_winner

        # --- Divergence: max absolute difference across 3 outcomes ---
        model_divergence = float(max(
            abs(dc_probs["home_win"] - xgb_probs["home_win"]),
            abs(dc_probs["draw"]     - xgb_probs["draw"]),
            abs(dc_probs["away_win"] - xgb_probs["away_win"]),
        ))

        # --- Confidence: entropy-based, penalised by divergence ---
        probs_arr = np.array(list(blended.values()))
        entropy = -np.sum(probs_arr * np.log(probs_arr + 1e-9))
        max_entropy = np.log(3)
        raw_confidence = float(1.0 - entropy / max_entropy)
        confidence = float(np.clip(raw_confidence * (1.0 - 0.3 * model_divergence), 0.1, 0.95))

        # Pick the most likely score whose implied outcome matches predicted_winner
        def _score_outcome(s: str) -> str:
            h, a = s.split("-")
            if int(h) > int(a):
                return "home"
            if int(h) < int(a):
                return "away"
            return "draw"

        all_scores = self.dc_model.predict_top_scores(home_team, away_team, top_n=100)
        matching = [s for s in all_scores if _score_outcome(str(s["score"])) == predicted_winner]
        most_likely_score = matching[0]["score"] if matching else "1-1"

        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            outcome_probabilities=blended,
            predicted_winner=predicted_winner,
            most_likely_score=most_likely_score,
            top_scores=top_scores,
            confidence=confidence,
            model_agreement=model_agreement,
            model_divergence=model_divergence,
            scenario_dc=dc_winner,
            scenario_xgb=xgb_winner,
        )
