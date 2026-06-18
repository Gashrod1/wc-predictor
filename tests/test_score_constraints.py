import numpy as np
import pytest
from src.utils.score_constraints import clip_lambda, shrink_to_prior, zero_high_scoring_cells


def test_clip_lambda_within_bounds():
    assert clip_lambda(1.5) == pytest.approx(1.5)


def test_clip_lambda_clips_high():
    assert clip_lambda(5.0) == pytest.approx(3.2)


def test_clip_lambda_clips_low():
    assert clip_lambda(0.1) == pytest.approx(0.3)


def test_shrink_to_prior_full_weight():
    # weight=1.0 → returns prior
    assert shrink_to_prior(3.0, prior=1.31, weight=1.0) == pytest.approx(1.31)


def test_shrink_to_prior_zero_weight():
    # weight=0.0 → returns value unchanged
    assert shrink_to_prior(3.0, prior=1.31, weight=0.0) == pytest.approx(3.0)


def test_shrink_to_prior_default_weight():
    expected = 0.85 * 2.0 + 0.15 * 1.31
    assert shrink_to_prior(2.0, prior=1.31) == pytest.approx(expected, rel=1e-6)


def test_zero_high_scoring_cells_zeroes_extreme():
    matrix = np.ones((6, 6))
    result = zero_high_scoring_cells(matrix, max_total=7)
    # Cell [4][4] has total 8 > 7 → should be 0
    assert result[4, 4] == pytest.approx(0.0)
    # Cell [3][4] has total 7 → should NOT be zeroed
    assert result[3, 4] > 0.0


def test_zero_high_scoring_cells_sums_to_one():
    matrix = np.random.rand(6, 6)
    result = zero_high_scoring_cells(matrix, max_total=7)
    assert abs(result.sum() - 1.0) < 1e-9


def test_zero_high_scoring_cells_preserves_zero_matrix():
    matrix = np.zeros((6, 6))
    result = zero_high_scoring_cells(matrix, max_total=7)
    # All zeros → result remains all zeros (no division)
    assert result.sum() == pytest.approx(0.0)
