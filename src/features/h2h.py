"""Head-to-head historical features. Uses only past matches between a given pair."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEF_WIN_RATE = 1 / 3
_DEF_GOALS = 1.3


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add h2h_* columns. df must be sorted by date (ascending).
    All stats are from current match's home-team perspective.
    """
    df = df.copy()

    # key = frozenset({home_team, away_team})
    # value = list of (home_team_in_that_match, home_score, away_score)
    h2h: dict[frozenset, list] = defaultdict(list)

    totals: list[int] = []
    home_win_rates: list[float] = []
    draw_rates: list[float] = []
    home_goals_avgs: list[float] = []
    away_goals_avgs: list[float] = []

    for row in df.itertuples(index=False):
        key = frozenset({row.home_team, row.away_team})
        history = h2h[key]
        n = len(history)

        if n == 0:
            totals.append(0)
            home_win_rates.append(_DEF_WIN_RATE)
            draw_rates.append(_DEF_WIN_RATE)
            home_goals_avgs.append(_DEF_GOALS)
            away_goals_avgs.append(_DEF_GOALS)
        else:
            # Translate all past matches into current home team's perspective
            cur_home = row.home_team
            h_scores: list[float] = []
            a_scores: list[float] = []
            for (past_home, past_hs, past_as) in history:
                if past_home == cur_home:
                    h_scores.append(past_hs)
                    a_scores.append(past_as)
                else:
                    h_scores.append(past_as)
                    a_scores.append(past_hs)

            home_wins = sum(h > a for h, a in zip(h_scores, a_scores))
            draws = sum(h == a for h, a in zip(h_scores, a_scores))

            totals.append(n)
            home_win_rates.append(home_wins / n)
            draw_rates.append(draws / n)
            home_goals_avgs.append(float(np.mean(h_scores)))
            away_goals_avgs.append(float(np.mean(a_scores)))

        # Append current match AFTER recording stats (no leakage)
        h2h[key].append((row.home_team, row.home_score, row.away_score))

    df["h2h_total_matches"] = totals
    df["h2h_home_win_rate"] = home_win_rates
    df["h2h_draw_rate"] = draw_rates
    df["h2h_home_goals_avg"] = home_goals_avgs
    df["h2h_away_goals_avg"] = away_goals_avgs

    logger.info(
        "H2H features added | pairs=%d | avg_h2h_matches=%.1f",
        len(h2h),
        float(np.mean([len(v) for v in h2h.values()])),
    )
    return df
