"""Pydantic request/response schemas for all API endpoints."""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# /predict/match
# ---------------------------------------------------------------------------

class MatchPredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["Brazil"])
    away_team: str = Field(..., examples=["Argentina"])
    tournament: str = Field(default="FIFA World Cup", examples=["FIFA World Cup"])
    neutral_venue: bool = Field(default=True)
    match_date: Optional[date] = Field(default=None)

    @field_validator("home_team", "away_team")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class MatchPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_home_goals: float
    predicted_away_goals: float
    most_likely_score: str
    home_elo: float
    away_elo: float
    confidence: str  # "high" | "medium" | "low"
    model_version: str


# ---------------------------------------------------------------------------
# /simulate/tournament
# ---------------------------------------------------------------------------

class TournamentSimulationRequest(BaseModel):
    tournament: str = Field(default="FIFA World Cup 2026", examples=["FIFA World Cup 2026"])
    n_simulations: int = Field(default=100, ge=10, le=1000)


class TournamentSimulationResponse(BaseModel):
    champion_probabilities: dict[str, float]
    finalist_probabilities: dict[str, float]
    semifinal_probabilities: dict[str, float]
    simulation_count: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# /teams/{team}/stats
# ---------------------------------------------------------------------------

class FormStats(BaseModel):
    ppg: float
    goals_scored_avg: float
    goals_conceded_avg: float


class TeamStatsResponse(BaseModel):
    team: str
    current_elo: float
    elo_rank: int
    form_last5: FormStats
    form_last10: FormStats
    goals_rolling_30: float


# ---------------------------------------------------------------------------
# /results/daily  &  /results/accuracy
# ---------------------------------------------------------------------------

class MatchResultItem(BaseModel):
    home_team: str
    away_team: str
    predicted_result: str        # "H" | "D" | "A"
    predicted_home_goals: float
    predicted_away_goals: float
    actual_result: Optional[str]        # None if not yet available
    actual_home_goals: Optional[int]
    actual_away_goals: Optional[int]
    correct: Optional[bool]
    home_win_prob: float
    draw_prob: float
    away_win_prob: float


class DailyResultsResponse(BaseModel):
    date: str
    matches: list[MatchResultItem]
    daily_accuracy: Optional[float]
    daily_goal_mae: Optional[float]


class AccuracyResponse(BaseModel):
    period_days: int
    total_matches: int
    correct: int
    accuracy: float
    avg_goal_mae: float
    avg_log_loss: float
    trend: str          # "improving" | "declining" | "stable"
    model_version: str
