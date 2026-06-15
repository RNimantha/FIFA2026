"""Phase 3: Train, evaluate, and save all models."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parents[1]))

import joblib
import pandas as pd

from src.features.engineer import FEATURES
from src.models.classifier import OutcomeClassifier
from src.models.ensemble import EnsemblePredictor
from src.models.evaluator import (
    evaluate_classifier,
    evaluate_regressor,
    generate_calibration_plot,
)
from src.models.score_predictor import GoalsPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3")

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d")
RANDOM_SEED = 42


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "features_val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "features_test.parquet")
    logger.info("Loaded | train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


def train_classifier(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[OutcomeClassifier, dict]:
    logger.info("=== Training outcome classifier ===")
    X_train = train[FEATURES]
    y_train = train["result"]
    X_val = val[FEATURES]
    y_val = val["result"]

    clf = OutcomeClassifier()

    # TimeSeriesSplit CV on training data only
    logger.info("Running 5-fold TimeSeriesSplit CV on training data...")
    t0 = perf_counter()
    cv_metrics = clf.cross_validate(X_train, y_train, n_splits=5)
    logger.info("CV done in %.1fs", perf_counter() - t0)

    # Final fit on full training set
    logger.info("Fitting on full training set...")
    clf.fit(X_train, y_train)

    # Evaluate on val set
    y_val_pred = clf.predict(X_val)
    y_val_proba = clf.predict_proba(X_val)
    val_metrics = evaluate_classifier(y_val, y_val_pred, y_val_proba, label="val")

    return clf, {**cv_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}


def train_score_models(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[GoalsPredictor, GoalsPredictor, dict]:
    logger.info("=== Training score predictors ===")
    X_train = train[FEATURES]
    X_val = val[FEATURES]

    home_model = GoalsPredictor(label="home")
    away_model = GoalsPredictor(label="away")

    home_model.fit(X_train, train["home_score"])
    away_model.fit(X_train, train["away_score"])

    # Val metrics
    home_val_metrics = evaluate_regressor(val["home_score"], home_model.predict(X_val), "val_home")
    away_val_metrics = evaluate_regressor(val["away_score"], away_model.predict(X_val), "val_away")

    metrics = {
        **{f"home_{k}": v for k, v in home_val_metrics.items()},
        **{f"away_{k}": v for k, v in away_val_metrics.items()},
    }
    return home_model, away_model, metrics


def evaluate_on_test(
    clf: OutcomeClassifier,
    home_model: GoalsPredictor,
    away_model: GoalsPredictor,
    test: pd.DataFrame,
) -> dict:
    logger.info("=== Final evaluation on held-out test set ===")
    X_test = test[FEATURES]
    y_test = test["result"]

    # Classifier
    clf_pred = clf.predict(X_test)
    clf_proba = clf.predict_proba(X_test)
    clf_metrics = evaluate_classifier(y_test, clf_pred, clf_proba, label="test_classifier")

    # Score regressors
    home_metrics = evaluate_regressor(test["home_score"], home_model.predict(X_test), "test_home_goals")
    away_metrics = evaluate_regressor(test["away_score"], away_model.predict(X_test), "test_away_goals")

    # Ensemble
    logger.info("--- Ensemble ---")
    ensemble = EnsemblePredictor(clf, home_model, away_model)
    ens_pred = ensemble.predict(X_test)
    ens_proba = ensemble.predict_proba(X_test)
    ens_metrics = evaluate_classifier(y_test, ens_pred, ens_proba, label="test_ensemble")

    # Calibration plot
    generate_calibration_plot(
        y_test, ens_proba,
        MODELS_DIR / f"calibration_ensemble_{TIMESTAMP}.png",
        model_label="Ensemble",
    )

    return {
        **{f"clf_{k}": v for k, v in clf_metrics.items()},
        **{f"home_{k}": v for k, v in home_metrics.items()},
        **{f"away_{k}": v for k, v in away_metrics.items()},
        **{f"ensemble_{k}": v for k, v in ens_metrics.items()},
    }


def save_models(
    clf: OutcomeClassifier,
    home_model: GoalsPredictor,
    away_model: GoalsPredictor,
) -> dict[str, str]:
    paths = {
        "classifier": MODELS_DIR / f"xgb_classifier_{TIMESTAMP}_v1.pkl",
        "home_goals": MODELS_DIR / f"xgb_home_goals_{TIMESTAMP}_v1.pkl",
        "away_goals": MODELS_DIR / f"xgb_away_goals_{TIMESTAMP}_v1.pkl",
    }
    clf.save(paths["classifier"])
    home_model.save(paths["home_goals"])
    away_model.save(paths["away_goals"])

    for name, path in paths.items():
        logger.info("Saved %s → %s", name, path)

    return {k: str(v) for k, v in paths.items()}


def log_experiment(hyperparams: dict, metrics: dict, model_paths: dict) -> None:
    log_path = MODELS_DIR / "experiment_log.json"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_version": f"v1_{TIMESTAMP}",
        "hyperparameters": hyperparams,
        "metrics": metrics,
        "model_paths": model_paths,
    }

    existing = []
    if log_path.exists():
        with open(log_path) as f:
            data = json.load(f)
            existing = data.get("experiments", [])

    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump({"experiments": existing}, f, indent=2)
    logger.info("Experiment logged → %s", log_path)


def main() -> None:
    train, val, test = load_splits()

    # Train
    clf, clf_metrics = train_classifier(train, val)
    home_model, away_model, score_metrics = train_score_models(train, val)

    # Test evaluation
    test_metrics = evaluate_on_test(clf, home_model, away_model, test)

    # Save models
    model_paths = save_models(clf, home_model, away_model)

    # Log experiment
    all_metrics = {**clf_metrics, **score_metrics, **test_metrics}
    hyperparams = {
        "classifier": {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        "score_predictor": {
            "objective": "count:poisson", "n_estimators": 300,
            "max_depth": 5, "learning_rate": 0.05,
        },
        "ensemble": {"classifier_weight": 0.6, "score_model_weight": 0.4},
        "cv": {"n_splits": 5, "method": "TimeSeriesSplit"},
        "random_seed": RANDOM_SEED,
    }
    log_experiment(hyperparams, all_metrics, model_paths)

    _print_summary(all_metrics)


def _print_summary(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    print("\n--- CV (training data) ---")
    print(f"  Accuracy:   {metrics.get('cv_accuracy_mean', 0):.4f} ± {metrics.get('cv_accuracy_std', 0):.4f}")
    print(f"  F1 macro:   {metrics.get('cv_f1_macro_mean', 0):.4f}")
    print(f"  Log-loss:   {metrics.get('cv_logloss_mean', 0):.4f}")
    print("\n--- Test set: Classifier ---")
    print(f"  Accuracy:   {metrics.get('clf_accuracy', 0):.4f}  (baseline={metrics.get('clf_baseline_accuracy', 0):.4f})")
    print(f"  F1 macro:   {metrics.get('clf_f1_macro', 0):.4f}  (target >0.48)")
    print(f"  Log-loss:   {metrics.get('clf_log_loss', 0):.4f}  (target <0.95)")
    print(f"  Brier:      {metrics.get('clf_brier_score', 0):.4f}")
    print("\n--- Test set: Ensemble ---")
    print(f"  Accuracy:   {metrics.get('ensemble_accuracy', 0):.4f}")
    print(f"  F1 macro:   {metrics.get('ensemble_f1_macro', 0):.4f}")
    print(f"  Log-loss:   {metrics.get('ensemble_log_loss', 0):.4f}")
    print("\n--- Test set: Goal regressors ---")
    print(f"  Home MAE:   {metrics.get('home_mae', 0):.4f}  (target <0.9)")
    print(f"  Away MAE:   {metrics.get('away_mae', 0):.4f}  (target <0.9)")
    print()
    hits = []
    if metrics.get('clf_accuracy', 0) > 0.55:
        hits.append("✓ Accuracy > 55%")
    else:
        hits.append("✗ Accuracy < 55%")
    if metrics.get('clf_log_loss', 1) < 0.95:
        hits.append("✓ Log-loss < 0.95")
    else:
        hits.append("✗ Log-loss ≥ 0.95")
    if metrics.get('clf_f1_macro', 0) > 0.48:
        hits.append("✓ F1 macro > 0.48")
    else:
        hits.append("✗ F1 macro < 0.48")
    if metrics.get('home_mae', 1) < 0.9:
        hits.append("✓ Home MAE < 0.9")
    else:
        hits.append("✗ Home MAE ≥ 0.9")
    print("Target metrics:")
    for h in hits:
        print(f"  {h}")
    print("=" * 60)


if __name__ == "__main__":
    main()
