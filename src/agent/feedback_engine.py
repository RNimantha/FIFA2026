"""
Compare today's predictions vs actual results.
Generates structured FeedbackReport with LLM improvement suggestions.
Saves to data/actuals/YYYY-MM-DD_feedback.json.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parents[2] / ".env")

from src.agent.validator import ActualResult

logger = logging.getLogger(__name__)

ACTUALS_DIR = Path(__file__).parents[2] / "data" / "actuals"
PREDICTIONS_DIR = Path(__file__).parents[2] / "data" / "predictions"


class MistakeSummary(BaseModel):
    home_team: str
    away_team: str
    predicted: str      # "H" / "D" / "A"
    actual: str
    predicted_prob: float
    error_type: Literal["overconfident_wrong", "wrong_direction", "goals_off"]


class FeedbackReport(BaseModel):
    date: str
    total_matches: int
    correct_predictions: int
    accuracy: float
    avg_goal_mae: float
    log_loss: float
    mistakes: list[MistakeSummary]
    improvement_suggestions: list[str]
    model_confidence_calibration: Literal["overconfident", "underconfident", "well-calibrated"]


def _result_code_to_label(code: str) -> str:
    return {"H": "Home Win", "D": "Draw", "A": "Away Win"}.get(code, code)


def _calc_log_loss(predictions: list[dict], actuals: list[ActualResult]) -> float:
    """Binary log-loss averaged over all matched predictions."""
    eps = 1e-7
    losses = []
    actual_map = {(a.home_team, a.away_team): a for a in actuals}

    for pred in predictions:
        key = (pred["home_team"], pred["away_team"])
        actual = actual_map.get(key)
        if actual is None or actual.confidence == "low":
            continue

        code = actual.result
        if code == "H":
            p = pred.get("home_win_prob", 1 / 3)
        elif code == "D":
            p = pred.get("draw_prob", 1 / 3)
        else:
            p = pred.get("away_win_prob", 1 / 3)

        p = max(eps, min(1 - eps, p))
        losses.append(-math.log(p))

    return round(float(sum(losses) / len(losses)), 4) if losses else 0.0


def _calibration(predictions: list[dict], actuals: list[ActualResult]) -> Literal["overconfident", "underconfident", "well-calibrated"]:
    """Rough calibration check based on avg max-confidence vs accuracy."""
    actual_map = {(a.home_team, a.away_team): a for a in actuals}
    total, correct = 0, 0
    avg_conf = 0.0

    for pred in predictions:
        key = (pred["home_team"], pred["away_team"])
        actual = actual_map.get(key)
        if actual is None or actual.confidence == "low":
            continue
        pred_result = pred.get("predicted_result", "H")
        max_p = max(
            pred.get("home_win_prob", 0),
            pred.get("draw_prob", 0),
            pred.get("away_win_prob", 0),
        )
        avg_conf += max_p
        total += 1
        if pred_result == actual.result:
            correct += 1

    if total == 0:
        return "well-calibrated"
    accuracy = correct / total
    avg_conf /= total

    gap = avg_conf - accuracy
    if gap > 0.15:
        return "overconfident"
    if gap < -0.10:
        return "underconfident"
    return "well-calibrated"


def _llm_suggestions(mistakes: list[MistakeSummary]) -> list[str]:
    """Call GPT-4o (no web search) to generate 3 improvement suggestions."""
    if not mistakes:
        return ["Model performed perfectly — no improvement suggestions."]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM suggestions")
        return ["Set OPENAI_API_KEY to enable AI improvement suggestions."]

    mistakes_text = "\n".join(
        f"- {m.home_team} vs {m.away_team}: predicted {m.predicted} "
        f"(prob={m.predicted_prob:.2f}), actual {m.actual} [error_type={m.error_type}]"
        for m in mistakes[:10]  # cap at 10
    )

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sports analytics expert reviewing a football prediction model's errors. "
                        "Output must be valid JSON — a JSON array of exactly 3 strings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The model made these errors today:\n{mistakes_text}\n\n"
                        "Current model features: ELO, form-5, form-10, H2H, tournament weight, neutral venue.\n"
                        "Suggest 3 specific, actionable improvements to the feature engineering or model "
                        "parameters to reduce these errors. Output as a JSON array of strings. No preamble."
                    ),
                },
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions[:3]]
    except Exception as exc:
        logger.warning("LLM suggestion call failed: %s", exc)

    return ["Could not generate suggestions — check logs for OpenAI errors."]


def compare(predictions: list[dict], actuals: list[ActualResult], date: str) -> FeedbackReport:
    """
    Compare predictions vs actuals, compute metrics, generate LLM suggestions.
    Saves FeedbackReport to data/actuals/{date}_feedback.json.
    """
    actual_map = {(a.home_team, a.away_team): a for a in actuals}
    mistakes: list[MistakeSummary] = []
    correct = 0
    goal_abs_errors: list[float] = []

    for pred in predictions:
        key = (pred["home_team"], pred["away_team"])
        actual = actual_map.get(key)
        if actual is None or actual.confidence == "low":
            continue

        pred_result = pred.get("predicted_result", "H")
        pred_h_goals = pred.get("predicted_home_goals", 1.0)
        pred_a_goals = pred.get("predicted_away_goals", 1.0)

        # Goal MAE
        goal_abs_errors.append(abs(pred_h_goals - actual.home_score))
        goal_abs_errors.append(abs(pred_a_goals - actual.away_score))

        if pred_result == actual.result:
            correct += 1
        else:
            # Determine error type
            pred_prob_map = {
                "H": pred.get("home_win_prob", 1 / 3),
                "D": pred.get("draw_prob", 1 / 3),
                "A": pred.get("away_win_prob", 1 / 3),
            }
            pred_prob = pred_prob_map.get(pred_result, 1 / 3)

            if pred_prob > 0.60:
                error_type = "overconfident_wrong"
            elif (pred_result in ("H", "A") and actual.result in ("H", "A")
                  and pred_result != actual.result):
                error_type = "wrong_direction"
            else:
                error_type = "goals_off"

            mistakes.append(MistakeSummary(
                home_team=pred["home_team"],
                away_team=pred["away_team"],
                predicted=pred_result,
                actual=actual.result,
                predicted_prob=round(pred_prob, 3),
                error_type=error_type,
            ))

    total = len([p for p in predictions
                 if actual_map.get((p["home_team"], p["away_team"])) is not None
                 and actual_map[(p["home_team"], p["away_team"])].confidence != "low"])
    accuracy = round(correct / total, 4) if total > 0 else 0.0
    avg_goal_mae = round(sum(goal_abs_errors) / len(goal_abs_errors), 4) if goal_abs_errors else 0.0
    log_loss_val = _calc_log_loss(predictions, actuals)
    calibration = _calibration(predictions, actuals)
    suggestions = _llm_suggestions(mistakes)

    report = FeedbackReport(
        date=date,
        total_matches=total,
        correct_predictions=correct,
        accuracy=accuracy,
        avg_goal_mae=avg_goal_mae,
        log_loss=log_loss_val,
        mistakes=mistakes,
        improvement_suggestions=suggestions,
        model_confidence_calibration=calibration,
    )

    out_path = ACTUALS_DIR / f"{date}_feedback.json"
    with open(out_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    logger.info("Feedback saved → %s | acc=%.3f | mistakes=%d", out_path, accuracy, len(mistakes))

    return report


def load_predictions(date: str) -> list[dict]:
    """Load daily prediction file from data/predictions/."""
    path = PREDICTIONS_DIR / f"{date}_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with open(path) as f:
        return json.load(f)
