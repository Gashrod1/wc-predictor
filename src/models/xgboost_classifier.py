"""Calibrated XGBoost classifier for W/D/L outcome prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

_FEATURE_COLS = [
    "elo_diff", "elo_home", "elo_away",
    "home_form_goals_scored", "home_form_goals_conceded", "home_form_xg",
    "away_form_goals_scored", "away_form_goals_conceded", "away_form_xg",
    "h2h_home_wins", "h2h_avg_goals", "is_knockout",
]


class XGBoostOutcomeClassifier:
    """Calibrated XGBoost for V/N/D classification.

    Label encoding: 0 = away_win, 1 = draw, 2 = home_win.
    Probabilities are calibrated with Platt scaling (sigmoid).
    """

    def __init__(self) -> None:
        base = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self._model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._feature_names: list[str] = _FEATURE_COLS.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the calibrated XGBoost classifier.

        Args:
            X: Feature DataFrame with the 12 feature columns.
            y: Target Series encoded as 0=away_win, 1=draw, 2=home_win.
        """
        self._feature_names = [c for c in _FEATURE_COLS if c in X.columns]
        X_arr = X[self._feature_names].values
        self._model.fit(X_arr, y.values)

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        """Return win/draw/loss probabilities for a single match.

        Args:
            features: Dict with the 12 feature keys.

        Returns:
            Dict with keys 'home_win', 'draw', 'away_win'.
        """
        row = np.array([[features.get(f, 0.0) for f in self._feature_names]])
        probs = self._model.predict_proba(row)[0]
        classes = list(self._model.classes_)
        prob_map = {c: p for c, p in zip(classes, probs)}
        return {
            "away_win": float(prob_map.get(0, 0.0)),
            "draw": float(prob_map.get(1, 0.0)),
            "home_win": float(prob_map.get(2, 0.0)),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the underlying XGBoost estimator.

        Returns:
            DataFrame with columns 'feature' and 'importance', sorted descending.
        """
        try:
            base_estimator = self._model.calibrated_classifiers_[0].estimator
            importances = base_estimator.feature_importances_
        except (AttributeError, IndexError):
            importances = np.ones(len(self._feature_names)) / len(self._feature_names)

        return (
            pd.DataFrame({"feature": self._feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
