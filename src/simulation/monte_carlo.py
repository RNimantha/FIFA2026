"""
Team state snapshot + match simulator.
Rebuilds ELO/form/H2H from historical data, then simulates future matches
with live state updates (no data leakage across simulated matches).
"""

import copy
import logging
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.features.elo import K_FACTOR_MAP, HOME_ADVANTAGE, INITIAL_ELO
from src.features.tournament import TOURNAMENT_WEIGHTS

logger = logging.getLogger(__name__)

_DEF_PPG = 1.0
_DEF_GF = 1.5
_DEF_GC = 1.2
_DEF_WIN_RATE = 1 / 3
_DEF_GOALS = 1.3
DEFAULT_K = 30
FORM_MAXLEN = 30

# Must match FEATURES order in engineer.py exactly
FEATURE_ORDER = [
    "home_elo", "away_elo", "elo_diff",
    "home_form5_ppg", "away_form5_ppg",
    "home_form5_goals_scored", "away_form5_goals_scored",
    "home_form5_goals_conceded", "away_form5_goals_conceded",
    "home_form10_ppg", "away_form10_ppg",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_total_matches",
    "h2h_home_goals_avg", "h2h_away_goals_avg",
    "is_neutral_venue", "tournament_weight",
    "home_goals_rolling_30", "away_goals_rolling_30",
]


class TeamStateSnapshot:
    """Current ELO, form, H2H for all teams after processing historical data."""

    def __init__(self) -> None:
        self.elo: dict[str, float] = defaultdict(lambda: float(INITIAL_ELO))
        self.form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_MAXLEN))
        self.h2h: dict[frozenset, list] = defaultdict(list)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TeamStateSnapshot":
        """Build snapshot by replaying all historical matches in chronological order."""
        snap = cls()
        df = df.sort_values("date")
        n = len(df)
        logger.info("Building team state snapshot from %d historical matches...", n)

        for row in df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            h_elo = snap.elo[home]
            a_elo = snap.elo[away]

            # ELO update
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
            snap.elo[home] = h_elo + k * (s_h - e_h)
            snap.elo[away] = a_elo + k * (s_a - e_a)

            # Form update
            if row.home_score > row.away_score:
                h_pts, a_pts = 3, 0
            elif row.home_score == row.away_score:
                h_pts, a_pts = 1, 1
            else:
                h_pts, a_pts = 0, 3
            snap.form[home].append((float(row.home_score), float(row.away_score), h_pts))
            snap.form[away].append((float(row.away_score), float(row.home_score), a_pts))

            # H2H update
            key = frozenset({home, away})
            snap.h2h[key].append((home, float(row.home_score), float(row.away_score)))

        logger.info("Snapshot built | %d teams tracked", len(snap.elo))
        return snap

    def copy(self) -> "TeamStateSnapshot":
        """Deep copy — must preserve defaultdict so unknown teams get defaults."""
        new = TeamStateSnapshot.__new__(TeamStateSnapshot)
        new.elo = defaultdict(lambda: float(INITIAL_ELO), self.elo)
        new.form = defaultdict(
            lambda: deque(maxlen=FORM_MAXLEN),
            {k: deque(v, maxlen=FORM_MAXLEN) for k, v in self.form.items()},
        )
        new.h2h = defaultdict(list, {k: list(v) for k, v in self.h2h.items()})
        return new


class MatchSimulator:
    """
    Simulates individual matches using trained models + live team state.
    State (ELO, form, H2H) updates after each simulated match.
    """

    def __init__(
        self,
        snapshot: TeamStateSnapshot,
        clf,      # OutcomeClassifier (for feature parity; goals come from score models)
        home_model,  # GoalsPredictor
        away_model,  # GoalsPredictor
    ) -> None:
        self.state = snapshot
        self.home_model = home_model
        self.away_model = away_model

    # ------------------------------------------------------------------
    # Feature construction (numpy — avoids pandas DataFrame overhead)
    # ------------------------------------------------------------------

    def _form_stats(self, team: str, window: int) -> tuple[float, float, float]:
        hist = list(self.state.form.get(team, []))
        recent = hist[-window:] if len(hist) >= window else hist
        if not recent:
            return _DEF_PPG, _DEF_GF, _DEF_GC
        gs = [r[0] for r in recent]
        gc = [r[1] for r in recent]
        pts = [r[2] for r in recent]
        return float(np.mean(pts)), float(np.mean(gs)), float(np.mean(gc))

    def _rolling30(self, team: str) -> float:
        hist = list(self.state.form.get(team, []))
        return float(np.mean([r[0] for r in hist])) if hist else _DEF_GF

    def _h2h_stats(self, home: str, away: str) -> tuple[int, float, float, float, float]:
        key = frozenset({home, away})
        history = self.state.h2h.get(key, [])
        n = len(history)
        if n == 0:
            return 0, _DEF_WIN_RATE, _DEF_WIN_RATE, _DEF_GOALS, _DEF_GOALS
        h_scores, a_scores = [], []
        for (past_home, past_hs, past_as) in history:
            if past_home == home:
                h_scores.append(past_hs)
                a_scores.append(past_as)
            else:
                h_scores.append(past_as)
                a_scores.append(past_hs)
        hw = sum(h > a for h, a in zip(h_scores, a_scores))
        dr = sum(h == a for h, a in zip(h_scores, a_scores))
        return n, hw / n, dr / n, float(np.mean(h_scores)), float(np.mean(a_scores))

    def _build_feature_array(
        self, home: str, away: str, tournament: str = "FIFA World Cup", neutral: bool = True
    ) -> np.ndarray:
        h_elo = self.state.elo.get(home, INITIAL_ELO)
        a_elo = self.state.elo.get(away, INITIAL_ELO)

        h_ppg5, h_gs5, h_gc5 = self._form_stats(home, 5)
        a_ppg5, a_gs5, a_gc5 = self._form_stats(away, 5)
        h_ppg10, _, _ = self._form_stats(home, 10)
        a_ppg10, _, _ = self._form_stats(away, 10)

        h2h_n, h2h_hwr, h2h_dr, h2h_hga, h2h_aga = self._h2h_stats(home, away)
        tw = TOURNAMENT_WEIGHTS.get(tournament, 0.5)

        return np.array([[
            h_elo, a_elo, h_elo - a_elo,
            h_ppg5, a_ppg5, h_gs5, a_gs5, h_gc5, a_gc5,
            h_ppg10, a_ppg10,
            h2h_hwr, h2h_dr, float(h2h_n), h2h_hga, h2h_aga,
            float(neutral), tw,
            self._rolling30(home), self._rolling30(away),
        ]], dtype=np.float32)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_match(
        self, home: str, away: str, tournament: str = "FIFA World Cup", neutral: bool = True
    ) -> tuple[int, int]:
        """Returns (home_goals, away_goals). Updates state after result."""
        X = self._build_feature_array(home, away, tournament, neutral)
        lambda_h = float(np.clip(self.home_model.model.predict(X)[0], 0.01, 15.0))
        lambda_a = float(np.clip(self.away_model.model.predict(X)[0], 0.01, 15.0))
        goals_h = int(np.random.poisson(lambda_h))
        goals_a = int(np.random.poisson(lambda_a))
        self._update_state(home, away, goals_h, goals_a, tournament, neutral)
        return goals_h, goals_a

    def _update_state(
        self, home: str, away: str, hg: int, ag: int, tournament: str, neutral: bool
    ) -> None:
        h_elo = self.state.elo.get(home, INITIAL_ELO)
        a_elo = self.state.elo.get(away, INITIAL_ELO)
        h_eff = h_elo + (0.0 if neutral else HOME_ADVANTAGE)
        e_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_eff) / 400.0))
        if hg > ag:
            s_h, s_a, h_pts, a_pts = 1.0, 0.0, 3, 0
        elif hg == ag:
            s_h, s_a, h_pts, a_pts = 0.5, 0.5, 1, 1
        else:
            s_h, s_a, h_pts, a_pts = 0.0, 1.0, 0, 3
        k = K_FACTOR_MAP.get(tournament, DEFAULT_K)
        self.state.elo[home] = h_elo + k * (s_h - e_h)
        self.state.elo[away] = a_elo + k * (1.0 - s_h - (1.0 - e_h))
        self.state.form[home].append((float(hg), float(ag), h_pts))
        self.state.form[away].append((float(ag), float(hg), a_pts))
        key = frozenset({home, away})
        self.state.h2h[key].append((home, float(hg), float(ag)))
