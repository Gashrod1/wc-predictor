import pytest
from web.fixtures import load_fixtures, map_stage


def test_map_stage_group_from_matchday():
    assert map_stage("1", "") == "group"
    assert map_stage("2", "") == "group"
    assert map_stage("3", "") == "group"


def test_map_stage_knockout_from_label():
    assert map_stage("", "TBD Home (Round of 32 #1)") == "round_of_32"
    assert map_stage("", "TBD Home (Round of 16 #2)") == "round_of_16"
    assert map_stage("", "TBD Home (Quarter-finals #1)") == "quarter_final"
    assert map_stage("", "TBD Home (Semi-finals #1)") == "semi_final"
    assert map_stage("", "TBD Home (3rd Place Final #1)") == "third_place"
    assert map_stage("", "TBD Home (Final #1)") == "final"


def test_load_fixtures_returns_list_without_result():
    fixtures = load_fixtures()
    assert isinstance(fixtures, list)
    assert len(fixtures) >= 64
    for f in fixtures:
        assert "result" not in f


def test_load_fixtures_has_expected_keys():
    fixtures = load_fixtures()
    f = fixtures[0]
    for key in ["date", "time", "matchday", "home_team", "away_team", "city", "stadium", "stage", "predictable"]:
        assert key in f


def test_load_fixtures_marks_tbd_not_predictable():
    fixtures = load_fixtures()
    tbd = [f for f in fixtures if f["home_team"].startswith("TBD")]
    assert tbd, "expected some TBD fixtures"
    assert all(f["predictable"] is False for f in tbd)


def test_load_fixtures_marks_real_match_predictable():
    fixtures = load_fixtures()
    real = [f for f in fixtures if not f["home_team"].startswith("TBD")]
    assert real
    assert all(f["predictable"] is True for f in real)
