"""Ensemble: blend XGBoost classifier probs with Poisson score-model probs."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.classifier import OutcomeClassifier
from src.models.score_predictor import GoalsPredictor

logger = logging.getLogger(__name__)

# Blend weight: how much to weight the classifier vs Poisson score model
CLASSIFIER_WEIGHT = 0.6
SCORE_MODEL_WEIGHT = 0.4
MAX_GOALS = 10  # cap for Poisson probability sum


def _poisson_outcome_probs(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> np.ndarray:
    """
    Convert (lambda_home, lambda_away) to (P_away_win, P_draw, P_home_win).
    Uses truncated Poisson sum up to MAX_GOALS each side.
    Returns shape (n_samples, 3).
    """
    n = len(lambda_home)
    p_home = np.zeros(n)
    p_draw = np.zeros(n)
    p_away = np.zeros(n)

    goals = np.arange(MAX_GOALS + 1)
    for h in goals:
        for a in goals:
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

    # Stack: columns match classifier order [Away Win, Draw, Home Win]
    return np.column_stack([p_away, p_draw, p_home])


class EnsemblePredictor:
    """Blends OutcomeClassifier with GoalsPredictor-derived Poisson probabilities."""

    def __init__(
        self,
        classifier: OutcomeClassifier,
        home_goals: GoalsPredictor,
        away_goals: GoalsPredictor,
    ) -> None:
        self.classifier = classifier
        self.home_goals = home_goals
        self.away_goals = away_goals

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns (n_samples, 3): [P(Away Win), P(Draw), P(Home Win)].
        Blends classifier probs with Poisson-derived probs.
        """
        clf_proba = self.classifier.predict_proba(X)  # (n, 3)

        lambda_h = self.home_goals.predict(X)
        lambda_a = self.away_goals.predict(X)
        poisson_proba = _poisson_outcome_probs(lambda_h, lambda_a)  # (n, 3)

        blended = CLASSIFIER_WEIGHT * clf_proba + SCORE_MODEL_WEIGHT * poisson_proba

        # Renormalize (floating point safety)
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted class: 0=Away Win, 1=Draw, 2=Home Win."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (expected_home_goals, expected_away_goals)."""
        return self.home_goals.predict(X), self.away_goals.predict(X)
