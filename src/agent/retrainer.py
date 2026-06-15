"""
Incremental model retraining with quality gate.
Triggered by FeedbackReport after each daily validation run.
NEVER deletes old models. NEVER saves a model worse than current by >0.02 accuracy.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.feedback_engine import FeedbackReport

logger = logging.getLogger(__name__)

ACTUALS_DIR = Path(__file__).parents[2] / "data" / "actuals"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).parents[2] / "models"
EXPERIMENT_LOG = MODELS_DIR / "experiment_log.json"
ACTIVE_MODEL_FILE = MODELS_DIR / "active_model.json"

CONSECUTIVE_POOR_THRESHOLD = 3
POOR_ACCURACY_THRESHOLD = 0.50
MIN_NEW_ACTUALS = 5
QUALITY_GATE_TOLERANCE = 0.02  # new model must not be worse by more than this


def _load_experiment_log() -> list[dict]:
    if not EXPERIMENT_LOG.exists():
        return []
    with open(EXPERIMENT_LOG) as f:
        return json.load(f)


def _save_experiment_log(entries: list[dict]) -> None:
    with open(EXPERIMENT_LOG, "w") as f:
        json.dump(entries, f, indent=2)


def _get_active_model_accuracy() -> float:
    """Read current active model's test accuracy from experiment_log.json."""
    entries = _load_experiment_log()
    if not entries:
        return 0.0
    # Use the last logged accuracy (most recent model)
    for entry in reversed(entries):
        if "test_accuracy" in entry:
            return float(entry["test_accuracy"])
    return 0.0


def _count_consecutive_poor_days() -> int:
    """Count consecutive days (most recent) with accuracy < threshold."""
    feedback_files = sorted(ACTUALS_DIR.glob("*_feedback.json"), reverse=True)
    count = 0
    for fp in feedback_files[:10]:
        with open(fp) as f:
            report = json.load(f)
        if report.get("accuracy", 1.0) < POOR_ACCURACY_THRESHOLD:
            count += 1
        else:
            break
    return count


def _load_all_actuals_for_features() -> pd.DataFrame:
    """Load all high/medium confidence actuals as a DataFrame for feature engineering."""
    rows = []
    for fp in sorted(ACTUALS_DIR.glob("*_actuals.json")):
        with open(fp) as f:
            actuals = json.load(f)
        for a in actuals:
            if a.get("confidence", "low") == "low":
                continue
            h_score = a["home_score"]
            a_score = a["away_score"]
            if h_score > a_score:
                result = 2  # Home Win
            elif h_score == a_score:
                result = 1  # Draw
            else:
                result = 0  # Away Win

            rows.append({
                "date": pd.to_datetime(a["date"]),
                "home_team": a["home_team"],
                "away_team": a["away_team"],
                "home_score": h_score,
                "away_score": a_score,
                "tournament": "FIFA World Cup",  # WC2026 context
                "neutral": True,
                "result": result,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _incremental_retrain(new_data: pd.DataFrame) -> None:
    """
    Run feature engineering on new matches, append to features.parquet,
    retrain models, and save only if quality gate passes.
    """
    from src.features.engineer import build_features, FEATURES, TARGETS
    from src.models.classifier import OutcomeClassifier
    from src.models.score_predictor import GoalsPredictor
    from src.models.evaluator import evaluate_classifier, evaluate_regressor

    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        logger.error("features.parquet not found — cannot retrain. Run Phase 2 first.")
        return

    # Load existing features
    existing = pd.read_parquet(features_path)

    # Build features for new data
    logger.info("Running feature engineering on %d new matches...", len(new_data))
    new_features = build_features(new_data)

    # Append (do NOT overwrite)
    combined = pd.concat([existing, new_features], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(features_path, index=False)
    logger.info(
        "features.parquet updated: %d → %d rows (+%d new)",
        len(existing), len(combined), len(new_features),
    )

    # Split for evaluation
    test_mask = combined["date"] >= pd.Timestamp("2022-11-01")
    train_df = combined[~test_mask]
    test_df = combined[test_mask]

    if len(train_df) < 100 or len(test_df) < 10:
        logger.warning("Insufficient data for retraining (train=%d, test=%d).", len(train_df), len(test_df))
        return

    X_train = train_df[FEATURES].values
    X_test = test_df[FEATURES].values
    y_train_clf = train_df["result"].values
    y_test_clf = test_df["result"].values
    y_train_h = train_df["home_score"].values
    y_test_h = test_df["home_score"].values
    y_train_a = train_df["away_score"].values
    y_test_a = test_df["away_score"].values

    # Current model accuracy (quality gate baseline)
    current_accuracy = _get_active_model_accuracy()
    logger.info("Current model accuracy: %.4f — quality gate: >= %.4f", current_accuracy, current_accuracy - QUALITY_GATE_TOLERANCE)

    # Train new classifier
    clf = OutcomeClassifier()
    clf.fit(X_train, y_train_clf)
    clf_metrics = evaluate_classifier(clf, X_test, y_test_clf)
    new_accuracy = clf_metrics["accuracy"]
    logger.info("New classifier accuracy: %.4f", new_accuracy)

    if new_accuracy < current_accuracy - QUALITY_GATE_TOLERANCE:
        logger.warning(
            "Quality gate FAILED: new accuracy %.4f < %.4f (current - tolerance). Not saving.",
            new_accuracy, current_accuracy - QUALITY_GATE_TOLERANCE,
        )
        return

    # Train score regressors
    home_model = GoalsPredictor(label="home_goals")
    away_model = GoalsPredictor(label="away_goals")
    home_model.fit(X_train, y_train_h)
    away_model.fit(X_train, y_train_a)
    home_metrics = evaluate_regressor(home_model, X_test, y_test_h)
    away_metrics = evaluate_regressor(away_model, X_test, y_test_a)

    # Version and save
    ts = datetime.utcnow().strftime("%Y%m%d")
    existing_versions = sorted(MODELS_DIR.glob(f"xgb_classifier_{ts}_v*.pkl"))
    v = len(existing_versions) + 1
    clf_path = MODELS_DIR / f"xgb_classifier_{ts}_v{v}.pkl"
    home_path = MODELS_DIR / f"xgb_home_goals_{ts}_v{v}.pkl"
    away_path = MODELS_DIR / f"xgb_away_goals_{ts}_v{v}.pkl"

    clf.save(clf_path)
    home_model.save(home_path)
    away_model.save(away_path)
    logger.info("New models saved: v%d | acc=%.4f", v, new_accuracy)

    # Update active model pointer
    active = {
        "classifier": str(clf_path),
        "home_goals": str(home_path),
        "away_goals": str(away_path),
        "version": f"xgb_v{v}_{ts}",
        "updated_at": datetime.utcnow().isoformat(),
        "test_accuracy": new_accuracy,
    }
    with open(ACTIVE_MODEL_FILE, "w") as f:
        json.dump(active, f, indent=2)

    # Log to experiment_log.json
    log_entries = _load_experiment_log()
    log_entries.append({
        "timestamp": datetime.utcnow().isoformat(),
        "trigger": "incremental_retrain",
        "version": f"v{v}_{ts}",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_accuracy": new_accuracy,
        "classifier_metrics": clf_metrics,
        "home_goals_metrics": home_metrics,
        "away_goals_metrics": away_metrics,
    })
    _save_experiment_log(log_entries)
    logger.info("Experiment log updated.")


def _full_retrain() -> None:
    """Run scripts/train.py as a subprocess (full retrain)."""
    train_script = Path(__file__).parents[2] / "scripts" / "phase3_model_training.py"
    logger.info("Triggering full retrain via %s", train_script)
    result = subprocess.run(
        [sys.executable, str(train_script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("Full retrain failed:\n%s", result.stderr)
    else:
        logger.info("Full retrain complete.\n%s", result.stdout[-500:])


def maybe_retrain(feedback: FeedbackReport) -> None:
    """
    Decide whether to retrain based on FeedbackReport.

    Logic:
    - accuracy < threshold for 3 consecutive days → full retrain
    - new_actuals_count >= MIN_NEW_ACTUALS → incremental retrain
    - else → skip
    """
    if not (PROCESSED_DIR / "features.parquet").exists():
        logger.error("features.parquet missing — skipping retraining.")
        return

    consecutive_poor = _count_consecutive_poor_days()
    logger.info("Consecutive poor days: %d | threshold: %d", consecutive_poor, CONSECUTIVE_POOR_THRESHOLD)

    if consecutive_poor >= CONSECUTIVE_POOR_THRESHOLD:
        logger.warning("3+ consecutive poor-accuracy days — triggering FULL retrain.")
        _full_retrain()
        return

    # Count new high-confidence actuals
    new_data = _load_all_actuals_for_features()
    if new_data.empty or len(new_data) < MIN_NEW_ACTUALS:
        logger.info(
            "Not enough new data for retraining today (%d rows, need %d).",
            len(new_data) if not new_data.empty else 0, MIN_NEW_ACTUALS,
        )
        return

    logger.info("Triggering incremental retrain with %d new matches.", len(new_data))
    _incremental_retrain(new_data)
