"""
APScheduler daily pipeline: runs at 23:00 UTC after match days end.
Pipeline: load predictions → fetch actuals → generate feedback → maybe retrain → notify API.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

from src.agent import validator, feedback_engine, retrainer
from src.agent.validator import MatchQuery

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path(__file__).parents[2] / "data" / "predictions"
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DAILY_RUN_HOUR = 23
DAILY_RUN_MINUTE = 0
TIMEZONE = "UTC"


def _load_today_matches(date: str) -> list[MatchQuery]:
    """Load match list from data/predictions/{date}_predictions.json."""
    path = PREDICTIONS_DIR / f"{date}_predictions.json"
    if not path.exists():
        logger.warning("No prediction file found for %s — using empty match list.", date)
        return []
    with open(path) as f:
        preds = json.load(f)
    return [
        MatchQuery(
            home_team=p["home_team"],
            away_team=p["away_team"],
            tournament=p.get("tournament", "FIFA World Cup"),
            date=date,
        )
        for p in preds
    ]


def _notify_api(feedback) -> None:
    """POST feedback summary to FastAPI /internal/daily-update."""
    try:
        payload = {
            "date": feedback.date,
            "accuracy": feedback.accuracy,
            "total_matches": feedback.total_matches,
            "correct": feedback.correct_predictions,
            "avg_goal_mae": feedback.avg_goal_mae,
            "log_loss": feedback.log_loss,
            "calibration": feedback.model_confidence_calibration,
            "suggestions": feedback.improvement_suggestions,
        }
        resp = requests.post(f"{API_BASE_URL}/internal/daily-update", json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("API notified successfully.")
        else:
            logger.warning("API notification returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Could not notify API: %s", exc)


def run_pipeline(date: str | None = None) -> None:
    """
    Run full daily validation pipeline for a given date (default: today UTC).
    Steps: load predictions → fetch actuals → feedback → retrain → notify.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("Daily pipeline START | date=%s", date)
    logger.info("=" * 60)

    # Step 1: Load today's prediction file
    today_matches = _load_today_matches(date)
    predictions_raw = []
    pred_path = PREDICTIONS_DIR / f"{date}_predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            predictions_raw = json.load(f)
    logger.info("Step 1 complete: loaded %d predictions", len(predictions_raw))

    # Step 2: Fetch real results
    try:
        actuals = validator.run(today_matches, date)
    except EnvironmentError as exc:
        logger.error("STOP: %s. Set OPENAI_API_KEY in environment or .env file.", exc)
        return
    logger.info("Step 2 complete: fetched %d actual results", len(actuals))

    if not actuals:
        logger.warning("No actual results — skipping feedback + retrain.")
        return

    # Step 3: Generate feedback report
    feedback = feedback_engine.compare(predictions_raw, actuals, date)
    logger.info(
        "Step 3 complete: acc=%.3f | mistakes=%d | calibration=%s",
        feedback.accuracy, len(feedback.mistakes), feedback.model_confidence_calibration,
    )

    # Stop condition: accuracy < 30% may indicate data pipeline issue
    if feedback.accuracy < 0.30 and feedback.total_matches >= 5:
        logger.error(
            "STOP CONDITION: accuracy=%.3f < 0.30 on %d matches — possible data pipeline issue. "
            "Human review required.",
            feedback.accuracy, feedback.total_matches,
        )
        return

    # Step 4: Maybe retrain
    retrainer.maybe_retrain(feedback)
    logger.info("Step 4 complete: retrainer decision made")

    # Step 5: Notify API
    _notify_api(feedback)
    logger.info("Step 5 complete: API notified")

    logger.info(
        "Daily pipeline COMPLETE | date=%s | acc=%.3f | matches=%d/%d correct",
        date, feedback.accuracy, feedback.correct_predictions, feedback.total_matches,
    )


def _scheduled_job() -> None:
    run_pipeline()


def run_live_results_update() -> None:
    """
    Fetch any newly completed match results and append to completed_matches_results.json.
    Runs every 2 hours so the frontend feed stays current throughout match days.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from scripts.save_daily_predictions import WC2026_SCHEDULE

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Live results update — checking for new completed matches up to %s", today)
    try:
        n = validator.update_completed_results(WC2026_SCHEDULE, today)
        logger.info("Live results update complete: +%d new results", n)
    except EnvironmentError as exc:
        logger.error("OPENAI_API_KEY missing: %s", exc)
    except Exception as exc:
        logger.exception("Live results update failed: %s", exc)


def start_scheduler() -> None:
    """
    Start blocking APScheduler with two jobs:
      - Every 2 hours: fetch newly completed match results (live feed).
      - Daily at 23:00 UTC: full feedback + retrain pipeline.
    """
    scheduler = BlockingScheduler(timezone=TIMEZONE)

    # Live results: poll every 2 hours, starting immediately
    scheduler.add_job(
        run_live_results_update,
        trigger="interval",
        hours=2,
        id="live_results_update",
        next_run_time=datetime.now(timezone.utc),  # run immediately on start
    )

    # Full daily pipeline: feedback, retrain, notify API
    scheduler.add_job(
        _scheduled_job,
        trigger="cron",
        hour=DAILY_RUN_HOUR,
        minute=DAILY_RUN_MINUTE,
        id="daily_validation",
    )

    logger.info(
        "Scheduler started — live results every 2h, full pipeline at %02d:%02d %s. "
        "Press Ctrl+C to stop.",
        DAILY_RUN_HOUR, DAILY_RUN_MINUTE, TIMEZONE,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
