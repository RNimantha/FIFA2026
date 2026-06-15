"""Phase 4: Monte Carlo WC2026 simulation — 10,000 tournament runs."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from src.models.base import BasePredictor
from src.simulation.monte_carlo import TeamStateSnapshot
from src.simulation.wc2026 import (
    N_SIMULATIONS,
    WC2026_GROUPS,
    print_champion_table,
    run_wc2026_simulation,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4")

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).parents[1] / "models"
PREDICTIONS_DIR = Path(__file__).parents[1] / "data" / "predictions"
TIMESTAMP = datetime.now().strftime("%Y%m%d")


def load_latest_model(prefix: str):
    candidates = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model found matching '{prefix}_*.pkl' in {MODELS_DIR}")
    path = candidates[0]
    logger.info("Loading %s", path.name)
    return BasePredictor.load(path)


def verify_teams(snapshot: TeamStateSnapshot) -> None:
    all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
    known = set(snapshot.elo.keys())
    unknown = [t for t in all_teams if t not in known]
    if unknown:
        logger.warning(
            "%d WC2026 teams not in historical data (will use default ELO=1000): %s",
            len(unknown), unknown,
        )
    else:
        logger.info("All 48 WC2026 teams found in historical data.")

    logger.info("ELO preview (WC2026 teams):")
    team_elos = [(t, snapshot.elo.get(t, 1000.0)) for t in all_teams]
    for team, elo in sorted(team_elos, key=lambda x: -x[1])[:10]:
        logger.info("  %-22s  ELO=%.0f", team, elo)


def main() -> None:
    # 1. Load models
    logger.info("=== Step 1: Load models ===")
    clf = load_latest_model("xgb_classifier")
    home_model = load_latest_model("xgb_home_goals")
    away_model = load_latest_model("xgb_away_goals")

    # 2. Build team state snapshot from full historical data
    logger.info("=== Step 2: Build team state snapshot ===")
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    snapshot = TeamStateSnapshot.from_dataframe(df)
    verify_teams(snapshot)

    # 3. Run Monte Carlo simulation
    logger.info("=== Step 3: Monte Carlo simulation (%d runs) ===", N_SIMULATIONS)
    t0 = perf_counter()
    results = run_wc2026_simulation(snapshot, clf, home_model, away_model, N_SIMULATIONS)
    elapsed = perf_counter() - t0
    logger.info("Simulation complete in %.1fs (%.1f ms/run)", elapsed, elapsed / N_SIMULATIONS * 1000)

    # 4. Save results
    logger.info("=== Step 4: Save ===")
    out_path = PREDICTIONS_DIR / f"wc2026_simulation_results_{TIMESTAMP}.json"
    save_results(results, out_path)

    # Also save as the canonical output file
    save_results(results, PREDICTIONS_DIR / "wc2026_simulation_results.json")

    # 5. Print summary
    logger.info("=== Phase 4 Complete ===")
    print("\n" + "=" * 70)
    print(f"FIFA WORLD CUP 2026 — MONTE CARLO SIMULATION ({N_SIMULATIONS:,} runs)")
    print("=" * 70)
    print_champion_table(results, top_n=20)

    # Spot-check probabilities sum to ~100%
    champ_total = sum(v.get("champion", 0) for v in results.values())
    print(f"\nTotal champion probability (should be ~1.0): {champ_total:.4f}")


if __name__ == "__main__":
    main()
