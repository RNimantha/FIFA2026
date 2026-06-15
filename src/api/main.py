"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

    yield
    # Cleanup (none needed)


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


@app.get("/", tags=["health"])
def root():
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
