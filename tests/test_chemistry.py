import pytest
import pandas as pd
from src.data.chemistry import ChemistryAnalyzer


@pytest.fixture
def analyzer():
    return ChemistryAnalyzer()


def test_get_chemistry_features_returns_dict(analyzer):
    """Always returns a dict with the 5 required keys."""
    home_squad = pd.DataFrame({
        "player_name": ["P1", "P2", "P3"],
        "club": ["Bayern Munich", "Bayern Munich", "Arsenal"],
        "position": ["GK", "DEF", "MID"],
    })
    away_squad = pd.DataFrame({
        "player_name": ["P4", "P5", "P6"],
        "club": ["Real Madrid", "Real Madrid", "Liverpool"],
        "position": ["GK", "DEF", "MID"],
    })
    feats = analyzer.get_chemistry_features("France", "Brazil", home_squad, away_squad)
    for key in ["home_chemistry_score", "away_chemistry_score",
                "home_pass_network_density", "away_pass_network_density",
                "chemistry_diff"]:
        assert key in feats, f"Missing key: {key}"


def test_chemistry_score_bounded(analyzer):
    """Chemistry scores must be in [0, 1]."""
    home_squad = pd.DataFrame({
        "player_name": [f"P{i}" for i in range(11)],
        "club": ["Bayern Munich"] * 11,
        "position": ["MID"] * 11,
    })
    away_squad = pd.DataFrame({
        "player_name": [f"Q{i}" for i in range(11)],
        "club": ["Real Madrid"] * 5 + ["Liverpool"] * 6,
        "position": ["MID"] * 11,
    })
    feats = analyzer.get_chemistry_features("France", "Brazil", home_squad, away_squad)
    assert 0.0 <= feats["home_chemistry_score"] <= 1.0
    assert 0.0 <= feats["away_chemistry_score"] <= 1.0


def test_chemistry_diff_equals_home_minus_away(analyzer):
    home_squad = pd.DataFrame({
        "player_name": ["P1", "P2"],
        "club": ["Bayern Munich", "Bayern Munich"],
        "position": ["DEF", "MID"],
    })
    away_squad = pd.DataFrame({
        "player_name": ["P3", "P4"],
        "club": ["Arsenal", "Chelsea"],
        "position": ["DEF", "MID"],
    })
    feats = analyzer.get_chemistry_features("France", "Brazil", home_squad, away_squad)
    expected_diff = feats["home_chemistry_score"] - feats["away_chemistry_score"]
    assert abs(feats["chemistry_diff"] - expected_diff) < 1e-9


def test_empty_squads_dont_crash(analyzer):
    """Empty squads must return valid zeros, not raise."""
    feats = analyzer.get_chemistry_features(
        "Unknown1", "Unknown2",
        pd.DataFrame(columns=["player_name", "club", "position"]),
        pd.DataFrame(columns=["player_name", "club", "position"]),
    )
    assert feats["home_chemistry_score"] == pytest.approx(0.0)
    assert feats["away_chemistry_score"] == pytest.approx(0.0)
