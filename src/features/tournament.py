"""Tournament weight and venue context features."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TOURNAMENT_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.9,
    "Copa América": 0.9,
    "AFC Asian Cup": 0.85,
    "Africa Cup of Nations": 0.85,
    "FIFA World Cup qualification": 0.75,
    "UEFA Euro qualification": 0.75,
    "Copa América qualification": 0.70,
    "AFC Asian Cup qualification": 0.70,
    "Africa Cup of Nations qualification": 0.70,
    "CONCACAF Gold Cup": 0.80,
    "CONCACAF Championship": 0.80,
    "UEFA Nations League": 0.70,
    "OFC Nations Cup": 0.75,
    "FIFA Confederations Cup": 0.90,
    "Friendly": 0.30,
}
DEFAULT_WEIGHT = 0.50


def add_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tournament_weight and is_neutral_venue columns."""
    df = df.copy()
    df["tournament_weight"] = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
    df["is_neutral_venue"] = df["neutral"].astype(int)

    unmapped = df.loc[~df["tournament"].isin(TOURNAMENT_WEIGHTS), "tournament"].unique()
    if len(unmapped) > 0:
        logger.info(
            "Tournament weight: %d unmapped tournaments → default %.2f. Top: %s",
            len(unmapped), DEFAULT_WEIGHT, unmapped[:5],
        )
    return df
