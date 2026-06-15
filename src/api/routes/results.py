"""GET /results/daily and GET /results/accuracy — daily prediction vs actual results."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import AccuracyResponse, DailyResultsResponse, MatchResultItem

router = APIRouter(prefix="/results", tags=["results"])
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path(__file__).parents[3] / "data" / "predictions"
ACTUALS_DIR = Path(__file__).parents[3] / "data" / "actuals"
MODELS_DIR = Path(__file__).parents[3] / "models"


def _latest_model_version() -> str:
    active_file = MODELS_DIR / "active_model.json"
    if active_file.exists():
        with open(active_file) as f:
            data = json.load(f)
        return data.get("version", "xgb_v1_20260615")
    # Fall back to newest pkl
    candidates = sorted(MODELS_DIR.glob("xgb_classifier_*.pkl"), reverse=True)
    if candidates:
        return candidates[0].stem
    return "xgb_v1_20260615"


def _load_predictions(date: str) -> list[dict]:
    path = PREDICTIONS_DIR / f"{date}_predictions.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_actuals(date: str) -> list[dict]:
    path = ACTUALS_DIR / f"{date}_actuals.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_feedback(date: str) -> dict | None:
    path = ACTUALS_DIR / f"{date}_feedback.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@router.get("/daily", response_model=DailyResultsResponse)
def get_daily_results(
    date: str = Query(default=None, description="Date in YYYY-MM-DD format (default: today UTC)"),
) -> DailyResultsResponse:
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    predictions = _load_predictions(date)
    actuals = _load_actuals(date)
    feedback = _load_feedback(date)

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {date}. Run save_daily_predictions for that date first.",
        )

    actual_map = {(a["home_team"], a["away_team"]): a for a in actuals}

    items: list[MatchResultItem] = []
    for pred in predictions:
        key = (pred["home_team"], pred["away_team"])
        actual = actual_map.get(key)

        pred_result = pred.get("predicted_result", "H")
        actual_result = actual["result"] if actual else None
        correct: Optional[bool] = None
        if actual_result is not None and actual.get("confidence", "low") != "low":
            correct = pred_result == actual_result

        items.append(MatchResultItem(
            home_team=pred["home_team"],
            away_team=pred["away_team"],
            predicted_result=pred_result,
            predicted_home_goals=pred.get("predicted_home_goals", 0.0),
            predicted_away_goals=pred.get("predicted_away_goals", 0.0),
            actual_result=actual_result,
            actual_home_goals=actual["home_score"] if actual else None,
            actual_away_goals=actual["away_score"] if actual else None,
            correct=correct,
            home_win_prob=pred.get("home_win_prob", 0.0),
            draw_prob=pred.get("draw_prob", 0.0),
            away_win_prob=pred.get("away_win_prob", 0.0),
        ))

    daily_accuracy = feedback["accuracy"] if feedback else None
    daily_goal_mae = feedback["avg_goal_mae"] if feedback else None

    return DailyResultsResponse(
        date=date,
        matches=items,
        daily_accuracy=daily_accuracy,
        daily_goal_mae=daily_goal_mae,
    )


@router.get("/accuracy", response_model=AccuracyResponse)
def get_accuracy(
    days: int = Query(default=7, ge=1, le=90, description="Number of past days to aggregate"),
) -> AccuracyResponse:
    from datetime import timedelta

    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(days)]

    total = 0
    correct = 0
    goal_maes: list[float] = []
    log_losses: list[float] = []

    for date in dates:
        feedback = _load_feedback(date)
        if not feedback:
            continue
        total += feedback.get("total_matches", 0)
        correct += feedback.get("correct_predictions", 0)
        if feedback.get("avg_goal_mae") is not None:
            goal_maes.append(feedback["avg_goal_mae"])
        if feedback.get("log_loss") is not None:
            log_losses.append(feedback["log_loss"])

    if total == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No feedback data available for the last {days} days.",
        )

    accuracy = round(correct / total, 4)
    avg_goal_mae = round(sum(goal_maes) / len(goal_maes), 4) if goal_maes else 0.0
    avg_log_loss = round(sum(log_losses) / len(log_losses), 4) if log_losses else 0.0

    # Trend: compare first half vs second half of the period
    half = days // 2
    recent_dates = dates[:half]
    older_dates = dates[half:]

    def _period_acc(period_dates: list[str]) -> float:
        t, c = 0, 0
        for d in period_dates:
            fb = _load_feedback(d)
            if fb:
                t += fb.get("total_matches", 0)
                c += fb.get("correct_predictions", 0)
        return c / t if t > 0 else 0.0

    recent_acc = _period_acc(recent_dates)
    older_acc = _period_acc(older_dates)
    if recent_acc > older_acc + 0.03:
        trend = "improving"
    elif recent_acc < older_acc - 0.03:
        trend = "declining"
    else:
        trend = "stable"

    return AccuracyResponse(
        period_days=days,
        total_matches=total,
        correct=correct,
        accuracy=accuracy,
        avg_goal_mae=avg_goal_mae,
        avg_log_loss=avg_log_loss,
        trend=trend,
        model_version=_latest_model_version(),
    )


@router.get("/completed", tags=["results"])
def get_completed_results(group: Optional[str] = Query(default=None, description="Filter by group (A-L)")):
    """
    Return all completed match results fetched by the validation agent.
    Reads from data/actuals/completed_matches_results.json.
    """
    path = ACTUALS_DIR / "completed_matches_results.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No completed results yet. Run: python3 -c \"from src.agent import ...\" or scripts/run_daily_agent.py",
        )
    with open(path) as f:
        results = json.load(f)

    # Optionally filter by group using the schedule match list
    if group:
        group = group.upper()
        # Load group membership from schedule
        from src.simulation.wc2026 import WC2026_GROUPS
        group_teams = set(WC2026_GROUPS.get(group, []))
        results = [
            r for r in results
            if r["home_team"] in group_teams or r["away_team"] in group_teams
        ]

    # Sort by date desc (most recent first)
    results = sorted(results, key=lambda r: r["date"], reverse=True)

    return {
        "total": len(results),
        "matches": results,
    }
