"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.models.base import BasePredictor
from src.simulation.monte_carlo import TeamStateSnapshot

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "models"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _latest_model(prefix: str) -> Path:
    candidates = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model matching '{prefix}_*.pkl' in {MODELS_DIR}")
    return candidates[0]


def _start_background_scheduler():
    """Start APScheduler in background thread inside the FastAPI process."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — live results agent will not run.")
        return None

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from src.agent.scheduler import run_live_results_update, run_pipeline

        sched = BackgroundScheduler(timezone="UTC")
        # Every 2 hours: fetch newly completed match results
        sched.add_job(run_live_results_update, "interval", hours=2,
                      id="live_results", next_run_time=__import__('datetime').datetime.utcnow())
        # Daily 23:00 UTC: full feedback + retrain pipeline
        sched.add_job(run_pipeline, "cron", hour=23, minute=0, id="daily_pipeline")
        sched.start()
        logger.info("Background scheduler started — live results every 2h, pipeline at 23:00 UTC.")
        return sched
    except Exception as exc:
        logger.warning("Scheduler failed to start: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    app.state.clf = BasePredictor.load(_latest_model("xgb_classifier"))
    app.state.home_model = BasePredictor.load(_latest_model("xgb_home_goals"))
    app.state.away_model = BasePredictor.load(_latest_model("xgb_away_goals"))
    logger.info("Models loaded.")

    logger.info("Building team state snapshot...")
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    app.state.snapshot = TeamStateSnapshot.from_dataframe(df)
    logger.info("Snapshot ready. %d teams tracked.", len(app.state.snapshot.elo))

    app.state.scheduler = _start_background_scheduler()

    yield

    if app.state.scheduler:
        app.state.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


app = FastAPI(
    title="FIFA Match Predictor",
    description="Predict FIFA international match outcomes and simulate World Cup 2026.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from src.api.routes.predict import router as predict_router
from src.api.routes.results import router as results_router
from src.api.routes.simulate import router as simulate_router
from src.api.routes.teams import router as teams_router

app.include_router(predict_router)
app.include_router(simulate_router)
app.include_router(teams_router)
app.include_router(results_router)


FRONTEND_DIR = Path(__file__).parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

@app.get("/", tags=["health"])
def root():
    if FRONTEND_DIR.exists():
        return RedirectResponse("/ui/index.html")
    return {"status": "ok", "service": "FIFA Match Predictor", "version": "1.0.0"}


@app.post("/internal/daily-update", tags=["internal"])
def daily_update(payload: dict) -> dict:
    """Receive daily feedback summary from the validation agent."""
    logger.info(
        "Daily update received: date=%s acc=%.3f matches=%d/%d",
        payload.get("date"), payload.get("accuracy", 0),
        payload.get("correct", 0), payload.get("total_matches", 0),
    )
    # Store latest daily stats in app state for real-time dashboard access
    app.state.latest_daily_update = payload
    return {"status": "ok", "received": payload.get("date")}


@app.get("/health", tags=["health"])
def health(request):
    snap = request.app.state.snapshot
    return {
        "status": "healthy",
        "teams_tracked": len(snap.elo),
        "models": ["xgb_classifier", "xgb_home_goals", "xgb_away_goals"],
    }
