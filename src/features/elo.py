"""Rolling ELO rating system. Zero data leakage — pre-match ELOs recorded before update."""

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

INITIAL_ELO = 1000.0
HOME_ADVANTAGE = 100.0

K_FACTOR_MAP: dict[str, int] = {
    "FIFA World Cup": 60,
    "FIFA Confederations Cup": 50,
    "Confederations Cup": 50,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "Copa América qualification": 40,
    "AFC Asian Cup qualification": 40,
    "Africa Cup of Nations qualification": 40,
    "CONCACAF Championship qualification": 40,
    "UEFA Euro": 35,
    "Copa América": 35,
    "AFC Asian Cup": 35,
    "Africa Cup of Nations": 35,
    "CONCACAF Gold Cup": 35,
    "UEFA Nations League": 35,
    "OFC Nations Cup": 35,
    "COSAFA Cup": 30,
    "CECAFA Cup": 30,
    "Friendly": 20,
}
DEFAULT_K = 30


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home_elo, away_elo, elo_diff columns.
    df must be sorted by date (ascending). Mutates a copy.
    """
    df = df.copy()

    elo: dict[str, float] = defaultdict(lambda: INITIAL_ELO)

    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in df.itertuples(index=False):
        h_elo = elo[row.home_team]
        a_elo = elo[row.away_team]

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Home advantage applied only to expected-score calculation, not stored ELO
        h_eff = h_elo + (0.0 if row.neutral else HOME_ADVANTAGE)
        e_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_eff) / 400.0))
        e_a = 1.0 - e_h

        if row.home_score > row.away_score:
            s_h, s_a = 1.0, 0.0
        elif row.home_score == row.away_score:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        k = K_FACTOR_MAP.get(row.tournament, DEFAULT_K)
        elo[row.home_team] = h_elo + k * (s_h - e_h)
        elo[row.away_team] = a_elo + k * (s_a - e_a)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    logger.info(
        "ELO features added | home_elo range=[%.0f, %.0f]",
        df["home_elo"].min(), df["home_elo"].max(),
    )
    return df
