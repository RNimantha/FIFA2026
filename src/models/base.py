"""Abstract base class for all prediction models."""

import abc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BasePredictor(abc.ABC):
    """Common interface: fit, predict, save, load."""

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor":
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BasePredictor":
        return joblib.load(path)
