"""
FIFA World Cup 2026 group configuration and simulation runner.

NOTE: Groups below are approximate. Verify against the official FIFA draw
(December 5, 2024) before relying on specific group-stage predictions.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Official FIFA WC 2026 group draw (source: FIFA, June 2026)
# Team names use dataset-standardized forms matching training data
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland"],
    "C": ["Haiti", "Scotland", "Brazil", "Morocco"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Ivory Coast", "Ecuador", "Germany", "Curacao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Iran", "New Zealand", "Belgium", "Egypt"],
    "H": ["Saudi Arabia", "Uruguay", "Spain", "Cape Verde"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["Ghana", "Panama", "England", "Croatia"],
}

N_SIMULATIONS = 10_000
RANDOM_SEED = 42


def run_wc2026_simulation(
    snapshot,          # TeamStateSnapshot
    clf,               # OutcomeClassifier
    home_model,        # GoalsPredictor
    away_model,        # GoalsPredictor
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Run Monte Carlo simulation of WC2026.
    Returns aggregated probability dict.
    """
    from src.simulation.monte_carlo import MatchSimulator
    from src.simulation.tournament import aggregate_simulations, simulate_tournament_once

    np.random.seed(seed)

    all_runs: list[dict[str, str]] = []
    log_every = max(1, n_simulations // 10)

    for i in range(n_simulations):
        if i % log_every == 0:
            logger.info("Simulation %d / %d", i, n_simulations)

        # Fresh state copy per simulation so runs are independent
        sim_state = snapshot.copy()
        sim = MatchSimulator(sim_state, clf, home_model, away_model)
        run_result = simulate_tournament_once(WC2026_GROUPS, sim)
        all_runs.append(run_result)

    return aggregate_simulations(all_runs)


def save_results(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved simulation results → %s", out_path)


def print_champion_table(results: dict, top_n: int = 20) -> None:
    ranked = sorted(results.items(), key=lambda x: -x[1].get("champion", 0))
    print(f"\n{'Team':<22} {'Champion':>9} {'Final':>7} {'Semi':>7} {'QF':>7} {'R16':>6} {'R32':>6}")
    print("-" * 70)
    for team, probs in ranked[:top_n]:
        print(
            f"{team:<22}"
            f" {probs.get('champion', 0):>8.1%}"
            f" {probs.get('final', 0):>7.1%}"
            f" {probs.get('sf', 0):>7.1%}"
            f" {probs.get('qf', 0):>7.1%}"
            f" {probs.get('r16', 0):>6.1%}"
            f" {probs.get('r32', 0):>6.1%}"
        )
