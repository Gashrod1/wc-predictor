"""Dixon-Coles (1997) bivariate Poisson model with temporal weighting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


_MAX_GOALS = 8
_XI = 0.0065  # temporal decay half-life ~300 days


def _tau(home_goals: int, away_goals: int, lambda_h: float, mu_a: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring matches."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * mu_a * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class DixonColesModel:
    """Bivariate Poisson model with Dixon-Coles low-score correction.

    Attributes:
        attack_params: Dict mapping team name to attack strength.
        defense_params: Dict mapping team name to defense weakness.
        home_advantage: Multiplicative home advantage parameter.
        rho: Dixon-Coles correction for 0-0, 1-0, 0-1, 1-1 scorelines.
    """

    attack_params: dict[str, float] = field(default_factory=dict)
    defense_params: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 1.1
    rho: float = -0.1
    _teams: list[str] = field(default_factory=list, repr=False)
    _reference_team: Optional[str] = field(default=None, repr=False)

    def fit(self, matches_df: pd.DataFrame) -> None:
        """Estimate attack/defense parameters via L-BFGS-B optimization.

        Uses temporal weighting w(t) = exp(-xi * delta_t) where delta_t is
        days since the match and xi=0.0065 (half-life ~300 days).

        Args:
            matches_df: DataFrame with columns date, home_team, away_team,
                        home_goals, away_goals.
        """
        df = matches_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        latest = df["date"].max()

        df["weight"] = np.exp(-_XI * (latest - df["date"]).dt.days)

        self._teams = sorted(
            set(df["home_team"].tolist() + df["away_team"].tolist())
        )
        self._reference_team = self._teams[0]
        n = len(self._teams)
        team_idx = {t: i for i, t in enumerate(self._teams)}

        # Initial params: [attack_0..n-1, defense_0..n-1, log_home_adv, rho]
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.1      # log home advantage
        x0[2 * n + 1] = -0.1  # rho

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n]
            defense = params[n : 2 * n]
            log_home_adv = params[2 * n]
            rho_ = params[2 * n + 1]

            ll = 0.0
            for _, row in df.iterrows():
                hi = team_idx[row["home_team"]]
                ai = team_idx[row["away_team"]]
                hg = int(row["home_goals"])
                ag = int(row["away_goals"])
                w = row["weight"]

                lambda_h = np.exp(attack[hi] - defense[ai] + log_home_adv)
                mu_a = np.exp(attack[ai] - defense[hi])

                t = _tau(hg, ag, lambda_h, mu_a, rho_)
                if t <= 0:
                    t = 1e-10

                ll += w * (
                    np.log(t)
                    + poisson.logpmf(hg, lambda_h)
                    + poisson.logpmf(ag, mu_a)
                )
            return -ll

        constraints = [
            {
                "type": "eq",
                "fun": lambda p: p[team_idx[self._reference_team]],
            }
        ]
        bounds = (
            [(-3, 3)] * n
            + [(-3, 3)] * n
            + [(-0.5, 0.5)]
            + [(-0.5, 0.0)]
        )

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        params = result.x
        self.attack_params = {t: params[i] for i, t in enumerate(self._teams)}
        self.defense_params = {t: params[n + i] for i, t in enumerate(self._teams)}
        self.home_advantage = float(np.exp(params[2 * n]))
        self.rho = float(params[2 * n + 1])

    def _get_lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Compute expected goals for both teams using fitted parameters."""
        mean_attack = float(np.mean(list(self.attack_params.values()))) if self.attack_params else 0.0
        mean_defense = float(np.mean(list(self.defense_params.values()))) if self.defense_params else 0.0

        atk_h = self.attack_params.get(home_team, mean_attack)
        def_h = self.defense_params.get(home_team, mean_defense)
        atk_a = self.attack_params.get(away_team, mean_attack)
        def_a = self.defense_params.get(away_team, mean_defense)

        lambda_h = np.exp(atk_h - def_a + np.log(self.home_advantage))
        mu_a = np.exp(atk_a - def_h)
        return float(lambda_h), float(mu_a)

    def predict_score_distribution(
        self, home_team: str, away_team: str
    ) -> np.ndarray:
        """Return (max_goals x max_goals) matrix of score probabilities.

        Cell [i, j] = P(home scores i, away scores j). Dixon-Coles correction
        applied to cells where i+j <= 2.

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            8x8 numpy array of joint score probabilities.
        """
        lambda_h, mu_a = self._get_lambdas(home_team, away_team)
        matrix = np.zeros((_MAX_GOALS, _MAX_GOALS))

        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                p = poisson.pmf(i, lambda_h) * poisson.pmf(j, mu_a)
                if i + j <= 2:
                    p *= _tau(i, j, lambda_h, mu_a, self.rho)
                matrix[i, j] = p

        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def predict_top_scores(
        self, home_team: str, away_team: str, top_n: int = 5
    ) -> list[dict[str, object]]:
        """Return the top_n most probable exact scorelines.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            top_n: Number of scores to return.

        Returns:
            List of dicts with keys 'score' (str) and 'probability' (float),
            sorted descending by probability.
        """
        matrix = self.predict_score_distribution(home_team, away_team)
        scores = []
        for i in range(_MAX_GOALS):
            for j in range(_MAX_GOALS):
                scores.append({"score": f"{i}-{j}", "probability": float(matrix[i, j])})
        scores.sort(key=lambda x: x["probability"], reverse=True)
        return scores[:top_n]

    def predict_outcome_probabilities(
        self, home_team: str, away_team: str
    ) -> dict[str, float]:
        """Return win/draw/loss probabilities for the match.

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Dict with keys 'home_win', 'draw', 'away_win', each in [0, 1].
        """
        matrix = self.predict_score_distribution(home_team, away_team)
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())
        return {"home_win": home_win, "draw": draw, "away_win": away_win}
