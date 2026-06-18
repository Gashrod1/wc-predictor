import pytest
from src.data.elo import expected_score, update_elo, load_elo_from_csv
import pandas as pd
import os


def test_expected_score_equal_elo():
    """Equal ELO teams should each have 0.5 expected score."""
    assert abs(expected_score(1800, 1800) - 0.5) < 1e-6


def test_expected_score_higher_wins():
    """Higher ELO team should have expected score > 0.5."""
    assert expected_score(2000, 1800) > 0.5


def test_update_elo_winner_gains():
    """Winning team should gain ELO."""
    new_home, new_away = update_elo(1800, 1800, home_goals=2, away_goals=0)
    assert new_home > 1800
    assert new_away < 1800


def test_update_elo_draw_equalizes():
    """Draw between equal teams should not change ELO significantly."""
    new_home, new_away = update_elo(1800, 1800, home_goals=1, away_goals=1)
    assert abs(new_home - 1800) < 5
    assert abs(new_away - 1800) < 5


def test_load_elo_from_csv(tmp_path):
    """Should load ELO ratings into a dict keyed by team name."""
    csv_path = tmp_path / "elo.csv"
    csv_path.write_text("team,elo\nFrance,2100\nBrazil,2150\n")
    ratings = load_elo_from_csv(str(csv_path))
    assert ratings["France"] == 2100.0
    assert ratings["Brazil"] == 2150.0
