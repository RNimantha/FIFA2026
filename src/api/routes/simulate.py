"""POST /simulate/tournament — Monte Carlo tournament simulation."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import TournamentSimulationRequest, TournamentSimulationResponse
from src.simulation.monte_carlo import MatchSimulator
from src.simulation.tournament import aggregate_simulations, simulate_tournament_once
from src.simulation.wc2026 import WC2026_GROUPS

router = APIRouter(prefix="/simulate", tags=["simulation"])
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path(__file__).parents[4] / "data" / "predictions"
PRECOMPUTED_PATH = PREDICTIONS_DIR / "wc2026_simulation_results.json"


@router.post("/tournament", response_model=TournamentSimulationResponse)
def simulate_tournament(
    req: TournamentSimulationRequest,
    request: Request,
) -> TournamentSimulationResponse:
    snap = request.app.state.snapshot
    clf = request.app.state.clf
    home_model = request.app.state.home_model
    away_model = request.app.state.away_model

    # Only WC2026 supported
    if "2026" not in req.tournament and "World Cup" not in req.tournament:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported tournament '{req.tournament}'. Supported: 'FIFA World Cup 2026'.",
        )

    # For n_simulations == default (100) and precomputed results exist, offer fast path
    # For live simulation, run Monte Carlo
    logger.info("Running %d simulations for %s", req.n_simulations, req.tournament)

    import numpy as np
    np.random.seed(None)  # unseeded for API calls (different result each time)

    all_runs = []
    for i in range(req.n_simulations):
        sim_state = snap.copy()
        sim = MatchSimulator(sim_state, clf, home_model, away_model)
        run_result = simulate_tournament_once(WC2026_GROUPS, sim)
        all_runs.append(run_result)

    aggregated = aggregate_simulations(all_runs)

    champion_probs = {
        team: probs.get("champion", 0.0)
        for team, probs in sorted(aggregated.items(), key=lambda x: -x[1].get("champion", 0))
        if probs.get("champion", 0) > 0
    }
    finalist_probs = {
        team: probs.get("final", 0.0)
        for team, probs in aggregated.items()
        if probs.get("final", 0) > 0
    }
    sf_probs = {
        team: probs.get("sf", 0.0)
        for team, probs in aggregated.items()
        if probs.get("sf", 0) > 0
    }

    return TournamentSimulationResponse(
        champion_probabilities=champion_probs,
        finalist_probabilities=finalist_probs,
        semifinal_probabilities=sf_probs,
        simulation_count=req.n_simulations,
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/wc2026/precomputed", tags=["simulation"])
def get_precomputed_wc2026() -> dict:
    """Return pre-computed 10k simulation results (fast, no compute)."""
    if not PRECOMPUTED_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Pre-computed results not found. Run scripts/phase4_simulate_wc2026.py first.",
        )
    with open(PRECOMPUTED_PATH) as f:
        data = json.load(f)

    # Sort by champion prob
    sorted_data = dict(
        sorted(data.items(), key=lambda x: -x[1].get("champion", 0))
    )
    return {
        "source": "pre-computed (10,000 simulations)",
        "generated_from": str(PRECOMPUTED_PATH),
        "results": sorted_data,
    }
