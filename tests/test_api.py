import pytest
from fastapi.testclient import TestClient
from web.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_teams_endpoint(client):
    resp = client.get("/api/teams")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0] and "elo" in data[0]


def test_predict_endpoint(client):
    resp = client.post("/api/predict", json={"home": "France", "away": "Brazil", "stage": "final"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["home_team"] == "France"
    assert data["away_team"] == "Brazil"
    probs = data["outcome_probabilities"]
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 0.01
    assert len(data["top_scores"]) == 5
    assert data["predicted_winner"] in ("home", "draw", "away")


def test_predict_french_alias(client):
    resp = client.post("/api/predict", json={"home": "France", "away": "Argentine", "stage": "final"})
    assert resp.status_code == 200
    assert resp.json()["away_team"] == "Argentina"


def test_fixtures_endpoint(client):
    resp = client.get("/api/fixtures")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) >= 64
    for f in data:
        assert "result" not in f
    tbd = [f for f in data if f["home_team"].startswith("TBD")]
    assert all(f["predictable"] is False for f in tbd)


def test_team_detail_endpoint(client):
    resp = client.get("/api/team/France")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "France"
    assert data["elo"] > 2000


def test_h2h_endpoint(client):
    resp = client.get("/api/h2h", params={"home": "France", "away": "Argentina"})
    assert resp.status_code == 200
    data = resp.json()
    assert "home_wins_pct" in data and "avg_goals" in data


def test_backtest_endpoint(client):
    resp = client.get("/api/backtest/WC2022")
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["outcome_accuracy"] <= 1.0


def test_backtest_unknown_tournament(client):
    resp = client.get("/api/backtest/WC1900")
    assert resp.status_code == 404
