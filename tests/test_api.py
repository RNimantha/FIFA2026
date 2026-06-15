"""API tests — uses TestClient (no live server needed)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Single TestClient for the whole module — lifespan runs once on enter."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict/match
# ---------------------------------------------------------------------------

def test_predict_known_teams(client):
    payload = {
        "home_team": "Brazil",
        "away_team": "Argentina",
        "tournament": "FIFA World Cup",
        "neutral_venue": True,
    }
    r = client.post("/predict/match", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "home_win_probability" in data
    assert "draw_probability" in data
    assert "away_win_probability" in data
    total = data["home_win_probability"] + data["draw_probability"] + data["away_win_probability"]
    assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1.0"
    assert data["confidence"] in {"high", "medium", "low"}
    assert "-" in data["most_likely_score"]


def test_predict_probabilities_sum_to_one(client):
    """Property test across several matchups."""
    matchups = [
        ("France", "England"),
        ("Spain", "Germany"),
        ("Japan", "South Korea"),
        ("Morocco", "Senegal"),
    ]
    for home, away in matchups:
        r = client.post("/predict/match", json={"home_team": home, "away_team": away})
        assert r.status_code == 200
        d = r.json()
        total = d["home_win_probability"] + d["draw_probability"] + d["away_win_probability"]
        assert abs(total - 1.0) < 0.01, f"{home} vs {away}: probs sum to {total}"


def test_predict_unknown_team_returns_200_with_defaults(client):
    """Unknown teams should NOT raise 500 — fallback to default ELO."""
    r = client.post("/predict/match", json={"home_team": "Unknown FC", "away_team": "Brazil"})
    assert r.status_code == 200
    data = r.json()
    assert data["home_elo"] == 1000.0  # default ELO


def test_predict_home_advantage(client):
    """ELO-equal teams at home should have slightly higher win prob than away."""
    r_home = client.post("/predict/match", json={
        "home_team": "Denmark", "away_team": "Switzerland", "neutral_venue": False,
    })
    r_neutral = client.post("/predict/match", json={
        "home_team": "Denmark", "away_team": "Switzerland", "neutral_venue": True,
    })
    assert r_home.status_code == 200
    assert r_neutral.status_code == 200
    home_prob_home = r_home.json()["home_win_probability"]
    home_prob_neutral = r_neutral.json()["home_win_probability"]
    assert home_prob_home >= home_prob_neutral, "Home advantage not reflected in predictions"


# ---------------------------------------------------------------------------
# /teams
# ---------------------------------------------------------------------------

def test_team_stats_known_team(client):
    r = client.get("/teams/Brazil/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["team"] == "Brazil"
    assert data["current_elo"] > 1000
    assert "form_last5" in data
    assert data["elo_rank"] >= 1


def test_team_stats_unknown_team(client):
    r = client.get("/teams/UnknownTeamXYZ/stats")
    assert r.status_code == 404


def test_list_teams(client):
    r = client.get("/teams/")
    assert r.status_code == 200
    data = r.json()
    assert data["total_teams"] > 100
    teams = [t["team"] for t in data["teams"]]
    assert "Brazil" in teams
    assert "Argentina" in teams


# ---------------------------------------------------------------------------
# /simulate/tournament
# ---------------------------------------------------------------------------

def test_simulate_tournament_small(client):
    r = client.post("/simulate/tournament", json={
        "tournament": "FIFA World Cup 2026",
        "n_simulations": 10,
    }, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert data["simulation_count"] == 10
    assert len(data["champion_probabilities"]) > 0
    total = sum(data["champion_probabilities"].values())
    assert abs(total - 1.0) < 0.05, f"Champion probs sum to {total}"


def test_simulate_unsupported_tournament(client):
    r = client.post("/simulate/tournament", json={
        "tournament": "UEFA Euro 2024",
        "n_simulations": 10,
    })
    assert r.status_code == 400


def test_precomputed_endpoint(client):
    r = client.get("/simulate/wc2026/precomputed")
    assert r.status_code in {200, 404}
