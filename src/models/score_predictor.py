"""XGBoost Poisson regressors for home and away goal prediction."""

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.base import BasePredictor

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


class GoalsPredictor(BasePredictor):
    """Predicts expected goals for one side (home or away)."""

    def __init__(self, label: str = "home") -> None:
        self.label = label
        self.model = XGBRegressor(
            objective="count:poisson",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GoalsPredictor":
        self.model.fit(X, y)
        logger.info(
            "GoalsPredictor[%s] trained | train_mae=%.3f",
            self.label,
            float(np.mean(np.abs(self.model.predict(X) - y))),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns expected goals (lambda for Poisson distribution)."""
        preds = self.model.predict(X)
        return np.clip(preds, 0.01, 15.0)  # Poisson lambda must be positive
