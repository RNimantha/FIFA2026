"""GET /teams/{team}/stats — current team ELO, form, and context."""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import FormStats, TeamStatsResponse
from src.simulation.monte_carlo import INITIAL_ELO, _DEF_GC, _DEF_GF, _DEF_PPG

router = APIRouter(prefix="/teams", tags=["teams"])
logger = logging.getLogger(__name__)


def _form_stats(hist: list, window: int) -> FormStats:
    recent = hist[-window:] if len(hist) >= window else hist
    if not recent:
        return FormStats(ppg=_DEF_PPG, goals_scored_avg=_DEF_GF, goals_conceded_avg=_DEF_GC)
    return FormStats(
        ppg=round(float(np.mean([r[2] for r in recent])), 3),
        goals_scored_avg=round(float(np.mean([r[0] for r in recent])), 3),
        goals_conceded_avg=round(float(np.mean([r[1] for r in recent])), 3),
    )


@router.get("/{team}/stats", response_model=TeamStatsResponse)
def get_team_stats(team: str, request: Request) -> TeamStatsResponse:
    snap = request.app.state.snapshot

    if team not in snap.elo and team not in snap.form:
        # Try case-insensitive match
        matches = [t for t in snap.elo if t.lower() == team.lower()]
        if not matches:
            raise HTTPException(
                status_code=404,
                detail=f"Team '{team}' not found. Check spelling or team name standardization.",
            )
        team = matches[0]

    hist = list(snap.form.get(team, []))
    current_elo = float(snap.elo.get(team, INITIAL_ELO))

    # ELO rank among all known teams
    all_elos = sorted(snap.elo.values(), reverse=True)
    elo_rank = all_elos.index(current_elo) + 1

    g30 = float(np.mean([r[0] for r in hist[-30:]])) if hist else _DEF_GF

    return TeamStatsResponse(
        team=team,
        current_elo=round(current_elo, 1),
        elo_rank=elo_rank,
        form_last5=_form_stats(hist, 5),
        form_last10=_form_stats(hist, 10),
        goals_rolling_30=round(g30, 3),
    )


@router.get("/", tags=["teams"])
def list_teams(request: Request) -> dict:
    """Return all known teams sorted by ELO."""
    snap = request.app.state.snapshot
    ranked = sorted(snap.elo.items(), key=lambda x: -x[1])
    return {
        "total_teams": len(ranked),
        "teams": [{"team": t, "elo": round(float(e), 1)} for t, e in ranked],
    }
