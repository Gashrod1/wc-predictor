"""Weighted ensemble combining Dixon-Coles and XGBoost predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
        confidence: Float in [0, 1] based on probability entropy.
        model_agreement: True if both models agree on the winner.
    """

    home_team: str
    away_team: str
    outcome_probabilities: dict[str, float]
    predicted_winner: str
    most_likely_score: str
    top_scores: list[dict[str, object]]
    confidence: float
    model_agreement: bool


class EnsemblePredictor:
    """Weighted ensemble of Dixon-Coles and XGBoost outcome predictors.

    Args:
        dc_model: Fitted DixonColesModel instance.
        xgb_model: Fitted XGBoostOutcomeClassifier instance.
        dc_weight: Weight for Dixon-Coles predictions (default 0.55).
        xgb_weight: Weight for XGBoost predictions (default 0.45).
    """

    def __init__(
        self,
        dc_model: DixonColesModel,
        xgb_model: XGBoostOutcomeClassifier,
        dc_weight: float = _DC_WEIGHT,
        xgb_weight: float = _XGB_WEIGHT,
    ) -> None:
        self.dc_model = dc_model
        self.xgb_model = xgb_model
        self.dc_weight = dc_weight
        self.xgb_weight = xgb_weight
        self._elo_ratings = load_elo_ratings()
        self._historical_df = load_historical_matches()

    def predict(
        self, home_team: str, away_team: str, context: dict[str, object] | None = None
    ) -> PredictionResult:
        """Generate an ensemble prediction for a match.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            context: Optional dict with keys like 'stage' (e.g. 'semi_final').

        Returns:
            PredictionResult with combined probabilities and metadata.
        """
        stage = (context or {}).get("stage", "group")

        dc_probs = self.dc_model.predict_outcome_probabilities(home_team, away_team)
        top_scores = self.dc_model.predict_top_scores(home_team, away_team, top_n=5)

        features = build_match_features(
            home_team, away_team, self._elo_ratings, self._historical_df, stage=str(stage)
        )
        xgb_probs = self.xgb_model.predict_proba(features)

        blended = {
            "home_win": (
                self.dc_weight * dc_probs["home_win"]
                + self.xgb_weight * xgb_probs["home_win"]
            ),
            "draw": (
                self.dc_weight * dc_probs["draw"]
                + self.xgb_weight * xgb_probs["draw"]
            ),
            "away_win": (
                self.dc_weight * dc_probs["away_win"]
                + self.xgb_weight * xgb_probs["away_win"]
            ),
        }
        total = sum(blended.values())
        blended = {k: v / total for k, v in blended.items()}

        predicted_winner = max(blended, key=blended.get)
        predicted_winner = predicted_winner.replace("_win", "")

        probs_arr = np.array(list(blended.values()))
        entropy = -np.sum(probs_arr * np.log(probs_arr + 1e-9))
        max_entropy = np.log(3)
        confidence = float(1.0 - entropy / max_entropy)

        dc_winner = max(dc_probs, key=dc_probs.get).replace("_win", "")
        xgb_winner = max(xgb_probs, key=xgb_probs.get).replace("_win", "")
        model_agreement = dc_winner == xgb_winner

        most_likely_score = top_scores[0]["score"] if top_scores else "1-1"

        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            outcome_probabilities=blended,
            predicted_winner=predicted_winner,
            most_likely_score=most_likely_score,
            top_scores=top_scores,
            confidence=confidence,
            model_agreement=model_agreement,
        )
