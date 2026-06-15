"""
Manual trigger for the FIFA daily validation agent.

Usage:
    python scripts/run_daily_agent.py                        # full pipeline for today (UTC)
    python scripts/run_daily_agent.py --date 2026-06-15     # full pipeline for specific date
    python scripts/run_daily_agent.py --live                 # fetch & append any new completed results now
    python scripts/run_daily_agent.py --schedule            # start blocking scheduler (2h live + 23:00 daily)
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

from src.agent.scheduler import run_pipeline, run_live_results_update, start_scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="FIFA Daily Validation Agent")
    parser.add_argument(
        "--date",
        default=None,
        help="Date to process in YYYY-MM-DD format (default: today UTC)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch any newly completed match results and update completed_matches_results.json",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Start blocking scheduler (live results every 2h + full pipeline at 23:00 UTC)",
    )
    args = parser.parse_args()

    if args.schedule:
        start_scheduler()
    elif args.live:
        run_live_results_update()
    else:
        run_pipeline(date=args.date)


if __name__ == "__main__":
    main()
