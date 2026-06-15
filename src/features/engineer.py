"""Master feature engineering pipeline. Combines all feature modules."""

import logging

import pandas as pd

from src.features.elo import add_elo_features
from src.features.form import add_form_features
from src.features.h2h import add_h2h_features
from src.features.tournament import add_tournament_features

logger = logging.getLogger(__name__)

FEATURES: list[str] = [
    # ELO
    "home_elo", "away_elo", "elo_diff",
    # Form (5-match window)
    "home_form5_ppg", "away_form5_ppg",
    "home_form5_goals_scored", "away_form5_goals_scored",
    "home_form5_goals_conceded", "away_form5_goals_conceded",
    # Form (10-match window)
    "home_form10_ppg", "away_form10_ppg",
    # H2H
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_total_matches",
    "h2h_home_goals_avg", "h2h_away_goals_avg",
    # Context
    "is_neutral_venue", "tournament_weight",
    "home_goals_rolling_30", "away_goals_rolling_30",
]

TARGETS: list[str] = ["result", "home_score", "away_score"]
META: list[str] = ["date", "home_team", "away_team", "tournament"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full feature engineering pipeline on df (must be sorted by date).
    Returns DataFrame with META + FEATURES + TARGETS columns only.
    """
    logger.info("build_features START | rows=%d", len(df))

    df = df.sort_values("date").reset_index(drop=True)

    df = add_elo_features(df)
    df = add_form_features(df)
    df = add_h2h_features(df)
    df = add_tournament_features(df)

    _validate_no_nan(df)

    keep = META + FEATURES + TARGETS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    result = df[keep].copy()
    logger.info("build_features DONE | shape=%s", result.shape)
    return result


def _validate_no_nan(df: pd.DataFrame) -> None:
    nan_cols = {c: int(df[c].isna().sum()) for c in FEATURES if c in df.columns and df[c].isna().any()}
    if nan_cols:
        raise ValueError(f"NaN values found in feature columns: {nan_cols}")
    logger.info("NaN validation passed — all %d feature columns clean", len(FEATURES))
