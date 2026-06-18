"""Weighted ensemble combining Dixon-Coles and XGBoost predictions."""
from __future__ import annotations

from dataclasses import dataclass
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
        home_team: Name of the first team.
        away_team: Name of the second team.
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

    def _blend(
        self,
        dc_probs: dict[str, float],
        xgb_probs: dict[str, float],
    ) -> dict[str, float]:
        """Weighted blend of DC and XGB outcome probabilities."""
        if self._weight_model is not None:
            features_arr = np.array([[
                dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"],
                xgb_probs["home_win"], xgb_probs["draw"], xgb_probs["away_win"],
            ]])
            lr_proba = self._weight_model.predict_proba(features_arr)[0]
            classes: list[int] = list(self._weight_model.classes_)
            label_map = {0: "away_win", 1: "draw", 2: "home_win"}
            return {label_map[c]: float(p) for c, p in zip(classes, lr_proba)}
        else:
            raw = {
                "home_win": self.dc_weight * dc_probs["home_win"] + self.xgb_weight * xgb_probs["home_win"],
                "draw":     self.dc_weight * dc_probs["draw"]     + self.xgb_weight * xgb_probs["draw"],
                "away_win": self.dc_weight * dc_probs["away_win"] + self.xgb_weight * xgb_probs["away_win"],
            }
            total = sum(raw.values())
            return {k: v / total for k, v in raw.items()}

    def _enrich_with_live_form(
        self, home_team: str, away_team: str
    ) -> pd.DataFrame:
        """Prepend fresh API form data for both teams to the historical DataFrame.

        Returns the combined DataFrame (API rows first so they rank higher in
        recency sorting inside _team_form). Falls back to self._historical_df
        on any API error.
        """
        try:
            from src.data.loader import get_team_form_cached
            home_form = get_team_form_cached(home_team)
            away_form = get_team_form_cached(away_team)
            if home_form.empty and away_form.empty:
                return self._historical_df
            extra = pd.concat(
                [df for df in (home_form, away_form) if not df.empty],
                ignore_index=True,
            )
            # Add any columns the historical_df has but form rows don't
            for col in self._historical_df.columns:
                if col not in extra.columns:
                    extra[col] = None
            combined = pd.concat([extra, self._historical_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["date", "home_team", "away_team"], keep="first"
            )
            return combined.sort_values("date").reset_index(drop=True)
        except Exception:
            return self._historical_df

    def _predict_directed(
        self,
        home_team: str,
        away_team: str,
        stage: str,
        neutral: bool,
        hist_df: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], list[dict[str, object]]]:
        """Compute raw (non-symmetrised) per-model and blended probabilities.

        Args:
            hist_df: Optional enriched historical DataFrame. Falls back to
                self._historical_df if not provided.

        Returns:
            (dc_probs, xgb_probs, blended, top_scores)
        """
        df = hist_df if hist_df is not None else self._historical_df

        dc_probs = self.dc_model.predict_outcome_probabilities(home_team, away_team, neutral=neutral)
        top_scores = self.dc_model.predict_top_scores(home_team, away_team, top_n=5, neutral=neutral)

        features = build_match_features(
            home_team, away_team, self._elo_ratings, df,
            stage=stage,
            squad_loader=self._squad_loader,
            chemistry_analyzer=self._chemistry_analyzer,
        )
        xgb_probs = self.xgb_model.predict_proba(features)
        blended = self._blend(dc_probs, xgb_probs)

        return dc_probs, xgb_probs, blended, top_scores

    @staticmethod
    def _flip(probs: dict[str, float]) -> dict[str, float]:
        """Swap home_win and away_win (perspective flip for neutral symmetry)."""
        return {
            "home_win": probs["away_win"],
            "draw":     probs["draw"],
            "away_win": probs["home_win"],
        }

    @staticmethod
    def _avg(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
        """Average two probability dicts."""
        return {k: (a[k] + b[k]) / 2 for k in a}

    def predict(
        self,
        home_team: str,
        away_team: str,
        context: dict[str, object] | None = None,
    ) -> PredictionResult:
        """Generate an ensemble prediction for a match.

        When neutral=True (default), the prediction is symmetrised: the model
        is run once with home_team listed first and once with away_team listed
        first, and the results are averaged. This removes any 'listed first'
        bias from both the Dixon-Coles rho correction and the XGBoost
        order-dependent features (elo_diff, home/away form).

        Args:
            home_team: First team name.
            away_team: Second team name.
            context: Optional dict with keys:
                - 'stage': match stage string (default 'group')
                - 'neutral': bool, True = no home advantage (default True)

        Returns:
            PredictionResult with symmetric probabilities on neutral venues.
        """
        ctx = context or {}
        stage = str(ctx.get("stage", "group"))
        # Default neutral=True: at the World Cup no team plays at home
        # (except USA/Canada/Mexico as hosts — set neutral=False for those games)
        neutral = bool(ctx.get("neutral", True))

        # Enrich historical_df with fresh form data from API (if key configured).
        # This updates home/away form features with real recent matches rather
        # than just WC-only historical averages. Cache TTL = 24h → ≤2 API calls
        # per team pair per day.
        enriched_df = self._enrich_with_live_form(home_team, away_team)

        dc_ab, xgb_ab, blend_ab, top_scores = self._predict_directed(
            home_team, away_team, stage, neutral, hist_df=enriched_df
        )

        if neutral:
            # Run prediction in the reverse direction, then flip perspective.
            # Averaging eliminates the ordering bias from both DC (rho asymmetry)
            # and XGBoost (directional features like elo_diff, home_form_*).
            dc_ba, xgb_ba, blend_ba, _ = self._predict_directed(
                away_team, home_team, stage, neutral, hist_df=enriched_df
            )
            dc_probs  = self._avg(dc_ab,    self._flip(dc_ba))
            xgb_probs = self._avg(xgb_ab,   self._flip(xgb_ba))
            blended   = self._avg(blend_ab,  self._flip(blend_ba))
        else:
            dc_probs  = dc_ab
            xgb_probs = xgb_ab
            blended   = blend_ab

        predicted_winner = max(blended, key=blended.get).replace("_win", "")

        # --- Per-model scenarios ---
        dc_winner  = max(dc_probs,  key=dc_probs.get).replace("_win", "")
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
