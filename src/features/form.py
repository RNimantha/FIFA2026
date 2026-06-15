"""Rolling form features. All lookups use only matches before current date."""

import logging
from collections import defaultdict, deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOWS = [5, 10]
DEQUE_MAX = 30  # also used for rolling_30 features

# Defaults for teams with no prior history
_DEF_PPG = 1.0
_DEF_GOALS_FOR = 1.5
_DEF_GOALS_AGAINST = 1.2


def _form_stats(history: list[tuple], window: int) -> tuple[float, float, float]:
    """Compute (ppg, goals_scored_avg, goals_conceded_avg) from last `window` records."""
    recent = history[-window:] if len(history) >= window else history
    if not recent:
        return _DEF_PPG, _DEF_GOALS_FOR, _DEF_GOALS_AGAINST
    gs = [r[0] for r in recent]
    gc = [r[1] for r in recent]
    pts = [r[2] for r in recent]
    return float(np.mean(pts)), float(np.mean(gs)), float(np.mean(gc))


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add form5/form10 and rolling_30 goal features.
    df must be sorted by date (ascending).
    """
    df = df.copy()

    # Each entry: (goals_scored, goals_conceded, points) from team's own perspective
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=DEQUE_MAX))

    col_arrays: dict[str, list] = {
        "home_form5_ppg": [],
        "home_form5_goals_scored": [],
        "home_form5_goals_conceded": [],
        "away_form5_ppg": [],
        "away_form5_goals_scored": [],
        "away_form5_goals_conceded": [],
        "home_form10_ppg": [],
        "away_form10_ppg": [],
        "home_goals_rolling_30": [],
        "away_goals_rolling_30": [],
    }

    for row in df.itertuples(index=False):
        h_hist = list(form[row.home_team])
        a_hist = list(form[row.away_team])

        for w in WINDOWS:
            h_ppg, h_gs, h_gc = _form_stats(h_hist, w)
            a_ppg, a_gs, a_gc = _form_stats(a_hist, w)

            col_arrays[f"home_form{w}_ppg"].append(h_ppg)
            col_arrays[f"away_form{w}_ppg"].append(a_ppg)
            if w == 5:
                col_arrays["home_form5_goals_scored"].append(h_gs)
                col_arrays["home_form5_goals_conceded"].append(h_gc)
                col_arrays["away_form5_goals_scored"].append(a_gs)
                col_arrays["away_form5_goals_conceded"].append(a_gc)

        # Rolling 30 (full deque)
        col_arrays["home_goals_rolling_30"].append(
            float(np.mean([r[0] for r in h_hist])) if h_hist else _DEF_GOALS_FOR
        )
        col_arrays["away_goals_rolling_30"].append(
            float(np.mean([r[0] for r in a_hist])) if a_hist else _DEF_GOALS_FOR
        )

        # Points
        if row.home_score > row.away_score:
            h_pts, a_pts = 3, 0
        elif row.home_score == row.away_score:
            h_pts, a_pts = 1, 1
        else:
            h_pts, a_pts = 0, 3

        form[row.home_team].append((row.home_score, row.away_score, h_pts))
        form[row.away_team].append((row.away_score, row.home_score, a_pts))

    for col, values in col_arrays.items():
        df[col] = values

    logger.info("Form features added | %d columns", len(col_arrays))
    return df
