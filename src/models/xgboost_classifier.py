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
    # Squad features (optional — 0.0 when squad data unavailable)
    "squad_avg_club_elo", "squad_pct_top5_league", "squad_avg_age",
    "squad_market_value_m", "squad_n_in_form", "squad_elo_diff",
    # Chemistry features (optional — 0.0 when chemistry data unavailable)
    "home_chemistry_score", "away_chemistry_score",
    "home_pass_network_density", "away_pass_network_density",
    "chemistry_diff",
    # ELO trend / momentum features (0.0 when trend data unavailable)
    "elo_trend_home", "elo_trend_away", "elo_trend_diff",
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

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 40) -> dict:
        """Find optimal hyperparameters via randomised search with time-series CV.

        Uses 3-fold TimeSeriesSplit to prevent data leakage (always trains on
        older data, tests on newer). Fits the tuned model on the full dataset
        after search. Only makes sense with 500+ training samples.

        Args:
            X: Feature DataFrame.
            y: Target labels (0=away, 1=draw, 2=home).
            n_iter: Number of random parameter combinations to try.

        Returns:
            Best hyperparameter dict found.
        """
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

        if len(X) < 300:
            # Too little data for reliable cross-validation — use defaults
            self.fit(X, y)
            return {}

        feature_names = [c for c in _FEATURE_COLS if c in X.columns]
        X_arr = X[feature_names].fillna(0.0).values
        y_arr = y.values

        param_dist = {
            "n_estimators":      [300, 500, 700, 1000],
            "max_depth":         [3, 4, 5, 6],
            "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample":         [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
            "min_child_weight":  [1, 3, 5, 7],
            "reg_alpha":         [0.0, 0.1, 0.5, 1.0],
            "reg_lambda":        [1.0, 2.0, 5.0],
            "gamma":             [0.0, 0.1, 0.3],
        }

        base = XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            base,
            param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_log_loss",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_arr, y_arr)
        best = search.best_params_

        # Re-fit with best params + Platt calibration on full data
        tuned_base = XGBClassifier(
            **best,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self._feature_names = feature_names
        self._model = CalibratedClassifierCV(tuned_base, method="sigmoid", cv=3)
        self._model.fit(X_arr, y_arr)

        return best

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
