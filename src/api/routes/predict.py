"""POST /predict/match — single match outcome prediction."""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from scipy.stats import poisson

from src.api.schemas import MatchPredictionRequest, MatchPredictionResponse
from src.features.tournament import TOURNAMENT_WEIGHTS
from src.simulation.monte_carlo import (
    FEATURE_ORDER,
    INITIAL_ELO,
    _DEF_GC,
    _DEF_GF,
    _DEF_GOALS,
    _DEF_PPG,
    _DEF_WIN_RATE,
    HOME_ADVANTAGE,
)

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = logging.getLogger(__name__)

MAX_GOALS_SCORE = 7  # for most-likely score grid
MODEL_VERSION = "xgb_v1_20260615"


def _confidence(max_prob: float) -> str:
    if max_prob >= 0.55:
        return "high"
    if max_prob >= 0.45:
        return "medium"
    return "low"


def _most_likely_score(lambda_h: float, lambda_a: float) -> str:
    best_prob, best = 0.0, (1, 1)
    for h in range(MAX_GOALS_SCORE + 1):
        for a in range(MAX_GOALS_SCORE + 1):
            p = float(poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a))
            if p > best_prob:
                best_prob, best = p, (h, a)
    return f"{best[0]}-{best[1]}"


def _build_features(req: MatchPredictionRequest, snap) -> np.ndarray:
    """Build feature array from snapshot + request. No state mutation."""
    home, away = req.home_team, req.away_team
    neutral = req.neutral_venue

    # ELO
    h_elo = snap.elo.get(home, INITIAL_ELO)
    a_elo = snap.elo.get(away, INITIAL_ELO)

    # Form helpers
    def form_stats(team: str, w: int) -> tuple[float, float, float]:
        hist = list(snap.form.get(team, []))
        recent = hist[-w:] if len(hist) >= w else hist
        if not recent:
            return _DEF_PPG, _DEF_GF, _DEF_GC
        return (
            float(np.mean([r[2] for r in recent])),
            float(np.mean([r[0] for r in recent])),
            float(np.mean([r[1] for r in recent])),
        )

    def rolling30(team: str) -> float:
        hist = list(snap.form.get(team, []))
        return float(np.mean([r[0] for r in hist])) if hist else _DEF_GF

    def h2h_stats() -> tuple[int, float, float, float, float]:
        key = frozenset({home, away})
        history = snap.h2h.get(key, [])
        n = len(history)
        if n == 0:
            return 0, _DEF_WIN_RATE, _DEF_WIN_RATE, _DEF_GOALS, _DEF_GOALS
        h_scores, a_scores = [], []
        for (past_home, phs, pas) in history:
            if past_home == home:
                h_scores.append(phs)
                a_scores.append(pas)
            else:
                h_scores.append(pas)
                a_scores.append(phs)
        hw = sum(h > a for h, a in zip(h_scores, a_scores))
        dr = sum(h == a for h, a in zip(h_scores, a_scores))
        return n, hw / n, dr / n, float(np.mean(h_scores)), float(np.mean(a_scores))

    h_ppg5, h_gs5, h_gc5 = form_stats(home, 5)
    a_ppg5, a_gs5, a_gc5 = form_stats(away, 5)
    h_ppg10, _, _ = form_stats(home, 10)
    a_ppg10, _, _ = form_stats(away, 10)
    h2h_n, h2h_hwr, h2h_dr, h2h_hga, h2h_aga = h2h_stats()
    tw = TOURNAMENT_WEIGHTS.get(req.tournament, 0.5)

    return np.array([[
        h_elo, a_elo, h_elo - a_elo,
        h_ppg5, a_ppg5, h_gs5, a_gs5, h_gc5, a_gc5,
        h_ppg10, a_ppg10,
        h2h_hwr, h2h_dr, float(h2h_n), h2h_hga, h2h_aga,
        float(neutral), tw,
        rolling30(home), rolling30(away),
    ]], dtype=np.float32)


@router.post("/match", response_model=MatchPredictionResponse)
def predict_match(req: MatchPredictionRequest, request: Request) -> MatchPredictionResponse:
    snap = request.app.state.snapshot
    clf = request.app.state.clf
    home_model = request.app.state.home_model
    away_model = request.app.state.away_model

    known_teams = set(snap.elo.keys())
    for team in [req.home_team, req.away_team]:
        if team not in known_teams:
            logger.warning("Unknown team '%s' — using default ELO.", team)

    X = _build_features(req, snap)

    # Classifier probabilities: [P(Away Win), P(Draw), P(Home Win)]
    clf_proba = clf.predict_proba(X)[0]

    # Poisson expected goals
    lambda_h = float(np.clip(home_model.model.predict(X)[0], 0.01, 15.0))
    lambda_a = float(np.clip(away_model.model.predict(X)[0], 0.01, 15.0))

    # Ensemble blend (0.6 clf + 0.4 Poisson) — mirrors ensemble.py
    from src.models.ensemble import _poisson_outcome_probs
    poisson_proba = _poisson_outcome_probs(
        np.array([lambda_h]), np.array([lambda_a])
    )[0]
    blended = 0.6 * clf_proba + 0.4 * poisson_proba
    blended /= blended.sum()

    away_win_p, draw_p, home_win_p = float(blended[0]), float(blended[1]), float(blended[2])
    max_p = max(home_win_p, draw_p, away_win_p)

    return MatchPredictionResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        home_win_probability=round(home_win_p, 4),
        draw_probability=round(draw_p, 4),
        away_win_probability=round(away_win_p, 4),
        predicted_home_goals=round(lambda_h, 2),
        predicted_away_goals=round(lambda_a, 2),
        most_likely_score=_most_likely_score(lambda_h, lambda_a),
        home_elo=round(snap.elo.get(req.home_team, INITIAL_ELO), 1),
        away_elo=round(snap.elo.get(req.away_team, INITIAL_ELO), 1),
        confidence=_confidence(max_p),
        model_version=MODEL_VERSION,
    )
