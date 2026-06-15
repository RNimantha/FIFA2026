"""
Save model predictions for all scheduled WC2026 matches on a given date.
Run this BEFORE match day so the daily agent has predictions to compare.

Usage:
    python scripts/save_daily_predictions.py                    # today UTC
    python scripts/save_daily_predictions.py --date 2026-06-20 # specific date
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path(__file__).parents[1] / "data" / "predictions"
PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).parents[1] / "models"

# Official WC2026 group stage schedule — dataset-friendly team names
T = "FIFA World Cup"
"FIFA World Cup"
WC2026_SCHEDULE: dict[str, list[tuple[str, str, str]]] = {
    "2026-06-12": [("Mexico", "South Africa", T), ("South Korea", "Czech Republic", T)],
    "2026-06-13": [("Canada", "Bosnia-Herzegovina", T), ("USA", "Paraguay", T)],
    "2026-06-14": [("Qatar", "Switzerland", T), ("Brazil", "Morocco", T), ("Haiti", "Scotland", T), ("Australia", "Turkey", T), ("Germany", "Curacao", T)],
    "2026-06-15": [("Ivory Coast", "Ecuador", T), ("Netherlands", "Japan", T), ("Sweden", "Tunisia", T), ("Spain", "Cape Verde", T)],
    "2026-06-16": [("Belgium", "Egypt", T), ("Saudi Arabia", "Uruguay", T), ("Iran", "New Zealand", T)],
    "2026-06-17": [("France", "Senegal", T), ("Iraq", "Norway", T), ("Argentina", "Algeria", T), ("Austria", "Jordan", T), ("Portugal", "DR Congo", T)],
    "2026-06-18": [("England", "Croatia", T), ("Ghana", "Panama", T), ("Uzbekistan", "Colombia", T), ("Czech Republic", "South Africa", T)],
    "2026-06-19": [("Switzerland", "Bosnia-Herzegovina", T), ("Canada", "Qatar", T), ("Mexico", "South Korea", T)],
    "2026-06-20": [("USA", "Australia", T), ("Scotland", "Morocco", T), ("Brazil", "Haiti", T), ("Turkey", "Paraguay", T), ("Netherlands", "Sweden", T)],
    "2026-06-21": [("Germany", "Ivory Coast", T), ("Ecuador", "Curacao", T), ("Tunisia", "Japan", T), ("Spain", "Saudi Arabia", T)],
    "2026-06-22": [("Belgium", "Iran", T), ("Uruguay", "Cape Verde", T), ("New Zealand", "Egypt", T), ("Argentina", "Austria", T)],
    "2026-06-23": [("France", "Iraq", T), ("Norway", "Senegal", T), ("Jordan", "Algeria", T), ("Portugal", "Uzbekistan", T)],
    "2026-06-24": [("England", "Ghana", T), ("Panama", "Croatia", T), ("Colombia", "DR Congo", T)],
    "2026-06-25": [("Switzerland", "Canada", T), ("Bosnia-Herzegovina", "Qatar", T), ("Morocco", "Haiti", T), ("Scotland", "Brazil", T), ("South Africa", "South Korea", T), ("Czech Republic", "Mexico", T)],
    "2026-06-26": [("Curacao", "Ivory Coast", T), ("Ecuador", "Germany", T), ("Tunisia", "Netherlands", T), ("Japan", "Sweden", T), ("Turkey", "USA", T), ("Paraguay", "Australia", T)],
    "2026-06-27": [("Norway", "France", T), ("Senegal", "Iraq", T), ("Cape Verde", "Saudi Arabia", T), ("Uruguay", "Spain", T), ("New Zealand", "Belgium", T), ("Egypt", "Iran", T)],
    "2026-06-28": [("Panama", "England", T), ("Croatia", "Ghana", T), ("Colombia", "Portugal", T), ("DR Congo", "Uzbekistan", T), ("Algeria", "Austria", T), ("Jordan", "Argentina", T)],
}


def _load_models():
    from src.models.base import BasePredictor

    def _latest(prefix):
        candidates = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No model matching '{prefix}_*.pkl'")
        return candidates[0]

    clf = BasePredictor.load(_latest("xgb_classifier"))
    home_model = BasePredictor.load(_latest("xgb_home_goals"))
    away_model = BasePredictor.load(_latest("xgb_away_goals"))
    return clf, home_model, away_model


def _build_snapshot():
    import pandas as pd
    from src.simulation.monte_carlo import TeamStateSnapshot

    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    return TeamStateSnapshot.from_dataframe(df)


def _predict_match(req_dict: dict, snap, clf, home_model, away_model) -> dict:
    """Use the same prediction logic as the API route."""
    import numpy as np
    from scipy.stats import poisson
    from src.simulation.monte_carlo import INITIAL_ELO, _DEF_GC, _DEF_GF, _DEF_GOALS, _DEF_PPG, _DEF_WIN_RATE
    from src.features.tournament import TOURNAMENT_WEIGHTS
    from src.models.ensemble import _poisson_outcome_probs

    home = req_dict["home_team"]
    away = req_dict["away_team"]
    neutral = req_dict.get("neutral_venue", True)
    tournament = req_dict.get("tournament", "FIFA World Cup")

    h_elo = snap.elo.get(home, INITIAL_ELO)
    a_elo = snap.elo.get(away, INITIAL_ELO)

    def form_stats(team, w):
        hist = list(snap.form.get(team, []))
        recent = hist[-w:] if len(hist) >= w else hist
        if not recent:
            return _DEF_PPG, _DEF_GF, _DEF_GC
        return (
            float(np.mean([r[2] for r in recent])),
            float(np.mean([r[0] for r in recent])),
            float(np.mean([r[1] for r in recent])),
        )

    def rolling30(team):
        hist = list(snap.form.get(team, []))
        return float(np.mean([r[0] for r in hist])) if hist else _DEF_GF

    def h2h_stats():
        key = frozenset({home, away})
        history = snap.h2h.get(key, [])
        n = len(history)
        if n == 0:
            return 0, _DEF_WIN_RATE, _DEF_WIN_RATE, _DEF_GOALS, _DEF_GOALS
        h_scores, a_scores = [], []
        for (ph, phs, pas) in history:
            if ph == home:
                h_scores.append(phs); a_scores.append(pas)
            else:
                h_scores.append(pas); a_scores.append(phs)
        hw = sum(h > a for h, a in zip(h_scores, a_scores))
        dr = sum(h == a for h, a in zip(h_scores, a_scores))
        return n, hw / n, dr / n, float(np.mean(h_scores)), float(np.mean(a_scores))

    h_ppg5, h_gs5, h_gc5 = form_stats(home, 5)
    a_ppg5, a_gs5, a_gc5 = form_stats(away, 5)
    h_ppg10, _, _ = form_stats(home, 10)
    a_ppg10, _, _ = form_stats(away, 10)
    h2h_n, h2h_hwr, h2h_dr, h2h_hga, h2h_aga = h2h_stats()
    tw = TOURNAMENT_WEIGHTS.get(tournament, 0.5)

    X = np.array([[
        h_elo, a_elo, h_elo - a_elo,
        h_ppg5, a_ppg5, h_gs5, a_gs5, h_gc5, a_gc5,
        h_ppg10, a_ppg10,
        h2h_hwr, h2h_dr, float(h2h_n), h2h_hga, h2h_aga,
        float(neutral), tw,
        rolling30(home), rolling30(away),
    ]], dtype=np.float32)

    clf_proba = clf.predict_proba(X)[0]
    lambda_h = float(np.clip(home_model.model.predict(X)[0], 0.01, 15.0))
    lambda_a = float(np.clip(away_model.model.predict(X)[0], 0.01, 15.0))
    poisson_proba = _poisson_outcome_probs(np.array([lambda_h]), np.array([lambda_a]))[0]
    blended = 0.6 * clf_proba + 0.4 * poisson_proba
    blended /= blended.sum()

    away_p, draw_p, home_p = float(blended[0]), float(blended[1]), float(blended[2])
    max_p = max(home_p, draw_p, away_p)
    pred_result = "H" if home_p == max_p else "D" if draw_p == max_p else "A"

    return {
        "home_team": home,
        "away_team": away,
        "tournament": tournament,
        "neutral_venue": neutral,
        "predicted_result": pred_result,
        "home_win_prob": round(home_p, 4),
        "draw_prob": round(draw_p, 4),
        "away_win_prob": round(away_p, 4),
        "predicted_home_goals": round(lambda_h, 2),
        "predicted_away_goals": round(lambda_a, 2),
    }


def save_predictions(date: str) -> None:
    matches = WC2026_SCHEDULE.get(date, [])
    if not matches:
        logger.warning("No scheduled matches found for %s in WC2026_SCHEDULE.", date)
        return

    logger.info("Loading models and snapshot...")
    clf, home_model, away_model = _load_models()
    snap = _build_snapshot()
    logger.info("Ready. Predicting %d matches for %s...", len(matches), date)

    predictions = []
    for home, away, tournament in matches:
        pred = _predict_match(
            {"home_team": home, "away_team": away, "tournament": tournament, "neutral_venue": True},
            snap, clf, home_model, away_model,
        )
        predictions.append(pred)
        logger.info("  %s vs %s → %s (%.0f%%/%.0f%%/%.0f%%)",
                    home, away, pred["predicted_result"],
                    pred["home_win_prob"] * 100, pred["draw_prob"] * 100, pred["away_win_prob"] * 100)

    out_path = PREDICTIONS_DIR / f"{date}_predictions.json"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Saved %d predictions → %s", len(predictions), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    args = parser.parse_args()
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_predictions(date)


if __name__ == "__main__":
    main()
