import pytest
import pandas as pd
from src.data.squad_loader import SquadLoader


@pytest.fixture
def loader():
    return SquadLoader()


def test_csv_fallback_returns_dataframe(loader):
    """CSV fallback returns a DataFrame with required columns."""
    df = loader.get_squad("France")
    assert isinstance(df, pd.DataFrame)
    assert "player_name" in df.columns
    assert "position" in df.columns
    assert "club" in df.columns
    assert "age" in df.columns
    assert "market_value_eur" in df.columns
    assert len(df) > 0


def test_csv_fallback_france_has_mbappe(loader):
    """Sanity check: Mbappé is in the France fallback squad."""
    df = loader.get_squad("France")
    assert any("Mbappe" in name or "Mbappé" in name for name in df["player_name"])


def test_synthetic_fallback_for_unknown_team(loader):
    """Unknown teams get a synthetic squad of 23 players."""
    df = loader.get_squad("Atlantis FC Unknown")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 23
    assert "player_name" in df.columns


def test_get_match_squad_features_returns_dict(loader):
    """get_match_squad_features returns all 6 expected feature keys."""
    feats = loader.get_match_squad_features("France", "Brazil")
    expected_keys = {
        "squad_avg_club_elo",
        "squad_pct_top5_league",
        "squad_avg_age",
        "squad_market_value_m",
        "squad_n_in_form",
        "squad_elo_diff",
    }
    assert expected_keys.issubset(set(feats.keys()))


def test_get_match_squad_features_numeric(loader):
    """All squad features are finite floats."""
    import math
    feats = loader.get_match_squad_features("France", "Brazil")
    for k, v in feats.items():
        assert isinstance(v, (int, float)), f"{k} is not numeric"
        assert math.isfinite(v), f"{k} is not finite"
