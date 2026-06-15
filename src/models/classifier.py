"""XGBoost outcome classifier: 0=Away Win, 1=Draw, 2=Home Win."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from xgboost import XGBClassifier

from src.models.base import BasePredictor

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


class OutcomeClassifier(BasePredictor):

    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OutcomeClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (n_samples, 3) array: [P(Away Win), P(Draw), P(Home Win)]."""
        return self.model.predict_proba(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv = cross_validate(
            self.model,
            X, y,
            cv=tscv,
            scoring=["accuracy", "f1_macro", "neg_log_loss"],
            return_train_score=False,
            n_jobs=-1,
        )
        results = {
            "cv_accuracy_mean": float(np.mean(cv["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(cv["test_f1_macro"])),
            "cv_logloss_mean": float(-np.mean(cv["test_neg_log_loss"])),
        }
        logger.info(
            "CV results | accuracy=%.4f±%.4f  f1_macro=%.4f  logloss=%.4f",
            results["cv_accuracy_mean"], results["cv_accuracy_std"],
            results["cv_f1_macro_mean"], results["cv_logloss_mean"],
        )
        return results
