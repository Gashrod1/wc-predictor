import pytest
import numpy as np
import pandas as pd
from src.models.dixon_coles import DixonColesModel


@pytest.fixture
def sample_matches():
    """Minimal match dataset for testing."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-11-20", "2022-11-21", "2022-11-22", "2022-11-23",
                 "2022-11-24", "2022-11-25", "2022-11-26", "2022-11-27",
                 "2022-11-28", "2022-11-29", "2022-12-03", "2022-12-18"]
            ),
            "home_team": ["France", "Brazil", "Argentina", "Germany",
                          "France", "Brazil", "Argentina", "Germany",
                          "Spain", "England", "France", "Argentina"],
            "away_team": ["Germany", "Argentina", "France", "Brazil",
                          "Spain", "England", "Germany", "France",
                          "Brazil", "Argentina", "Brazil", "France"],
            "home_goals": [2, 1, 2, 0, 1, 3, 2, 1, 1, 2, 1, 3],
            "away_goals": [1, 0, 1, 2, 0, 1, 0, 0, 2, 1, 0, 3],
            "stage": ["group"] * 10 + ["semi_final", "final"],
            "tournament": ["WC2022"] * 12,
        }
    )


def test_fit_runs_without_error(sample_matches):
    model = DixonColesModel()
    model.fit(sample_matches)
    assert model.attack_params is not None
    assert model.defense_params is not None


def test_predict_score_distribution_shape(sample_matches):
    model = DixonColesModel()
    model.fit(sample_matches)
    matrix = model.predict_score_distribution("France", "Brazil")
    assert matrix.shape == (8, 8)


def test_predict_score_distribution_sums_to_one(sample_matches):
    model = DixonColesModel()
    model.fit(sample_matches)
    matrix = model.predict_score_distribution("France", "Brazil")
    assert abs(matrix.sum() - 1.0) < 0.02


def test_predict_top_scores_format(sample_matches):
    model = DixonColesModel()
    model.fit(sample_matches)
    top = model.predict_top_scores("France", "Brazil", top_n=5)
    assert len(top) == 5
    assert "score" in top[0]
    assert "probability" in top[0]
    assert top[0]["probability"] >= top[1]["probability"]


def test_predict_outcome_probabilities_sum(sample_matches):
    model = DixonColesModel()
    model.fit(sample_matches)
    probs = model.predict_outcome_probabilities("France", "Brazil")
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 0.01


def test_unknown_team_fallback(sample_matches):
    """Unknown team should use mean attack/defense, not crash."""
    model = DixonColesModel()
    model.fit(sample_matches)
    probs = model.predict_outcome_probabilities("Atlantis", "Brazil")
    assert probs["home_win"] + probs["draw"] + probs["away_win"] > 0.99
