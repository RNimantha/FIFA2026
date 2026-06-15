# Claude Code Prompt — FIFA Self-Learning Validation Agent

> Paste this directly into Claude Code terminal inside your `fifa-match-predictor/` repo.

---

```
Read CLAUDE.md first. Then build the following system exactly as described. Do not skip steps. Do not add features beyond what is specified. After each major step, output: ✅ [what was completed].

---

## OBJECTIVE

Build a FIFA Match Validation & Self-Learning Agent that:
1. Runs daily after real match results are known
2. Uses OpenAI GPT-4o (via web search tool) to fetch actual results
3. Compares actual results against the model's predictions
4. Generates structured feedback on what the model got wrong
5. Triggers model retraining using the updated dataset
6. Updates the frontend dashboard with live actual results and accuracy metrics

---

## CONTEXT (carry forward from CLAUDE.md)

- Project: `fifa-match-predictor/`
- Existing models: XGBoost classifier (Win/Draw/Loss) + Poisson score regressors
- Models saved at: `models/*.pkl`
- Processed data at: `data/processed/features.parquet`
- Frontend: React + FastAPI (already exists — only UPDATE, do not rebuild)
- Stack: Python 3.11, FastAPI, OpenAI SDK, APScheduler, SQLite (for daily run logs)
- OpenAI API key: read from env var `OPENAI_API_KEY` — NEVER hardcode

---

## STARTING STATE

```
fifa-match-predictor/
├── CLAUDE.md
├── src/
│   ├── models/          ← trained models already exist
│   ├── features/        ← feature engineering pipeline exists
│   └── api/             ← FastAPI app exists (routes/predict.py, routes/teams.py)
├── data/
│   ├── processed/
│   │   └── features.parquet
│   └── predictions/     ← daily prediction logs live here (create if missing)
└── frontend/            ← React dashboard exists
```

---

## TARGET STATE (what must exist when done)

```
fifa-match-predictor/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── validator.py          ← OpenAI agent: fetches real results via web search
│   │   ├── feedback_engine.py    ← compares prediction vs actual, generates feedback JSON
│   │   ├── retrainer.py          ← triggers incremental model retraining
│   │   └── scheduler.py          ← APScheduler daily job (runs at 23:00 UTC)
│   └── api/
│       └── routes/
│           └── results.py        ← NEW: GET /results/daily, GET /results/accuracy
├── data/
│   ├── predictions/
│   │   └── YYYY-MM-DD_predictions.json   ← saved before each match day
│   └── actuals/
│       └── YYYY-MM-DD_actuals.json       ← saved by validator after match day
├── frontend/src/
│   └── components/
│       └── ResultsTracker.tsx    ← NEW component: shows predictions vs actuals + accuracy
└── scripts/
    └── run_daily_agent.py        ← manual trigger: python scripts/run_daily_agent.py
```

---

## BUILD STEPS — Execute in this exact order

### STEP 1 — Install dependencies
Add to requirements.txt and install:
```
openai>=1.30.0
apscheduler>=3.10.0
```
Run: pip install openai apscheduler

✅ Output: "Dependencies installed"

---

### STEP 2 — Build `src/agent/validator.py`

This module uses the OpenAI Responses API with built-in web_search tool to fetch real FIFA match results.

Requirements:
- Use model: `gpt-4o`
- Use tool: `{"type": "web_search_preview"}` (OpenAI built-in, no key needed)
- Input: list of matches played today `[{"home": "Brazil", "away": "Argentina", "tournament": "FIFA WC 2026", "date": "2026-06-15"}]`
- For each match, construct query: `"Brazil vs Argentina FIFA World Cup 2026 result score June 15"`
- Parse the web search response to extract: `home_score`, `away_score`, `result` (H/D/A)
- Return: `list[ActualResult]` where each has `{date, home_team, away_team, home_score, away_score, result, source_url, confidence}`
- If result not found (match postponed/not played): set `confidence="low"`, skip feedback for that match
- Save output to `data/actuals/YYYY-MM-DD_actuals.json`

Pydantic schemas:
```python
class MatchQuery(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    date: str  # YYYY-MM-DD

class ActualResult(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    result: Literal["H", "D", "A"]  # Home win, Draw, Away win
    source_url: str
    confidence: Literal["high", "medium", "low"]
```

Error handling:
- Wrap every OpenAI call in try/except
- Log all API responses to `logs/validator_YYYY-MM-DD.log`
- If rate limited: retry once after 10 seconds, then skip and log warning

✅ Output: "validator.py built — fetches results via OpenAI web search"

---

### STEP 3 — Build `src/agent/feedback_engine.py`

Compares today's predictions (loaded from `data/predictions/YYYY-MM-DD_predictions.json`) against actuals from validator.

Input:
```python
predictions = [
    {
        "home_team": "Brazil", "away_team": "Argentina",
        "predicted_result": "H", "home_win_prob": 0.52,
        "draw_prob": 0.26, "away_win_prob": 0.22,
        "predicted_home_goals": 1.8, "predicted_away_goals": 1.2
    }
]
actuals = [ActualResult(...)]  # from validator
```

Output — `FeedbackReport` (saved to `data/actuals/YYYY-MM-DD_feedback.json`):
```python
class FeedbackReport(BaseModel):
    date: str
    total_matches: int
    correct_predictions: int
    accuracy: float
    avg_goal_mae: float  # Mean Absolute Error on goals
    log_loss: float      # Log-loss of probability predictions
    mistakes: list[MistakeSummary]
    improvement_suggestions: list[str]  # LLM-generated
    model_confidence_calibration: str   # "overconfident" | "underconfident" | "well-calibrated"

class MistakeSummary(BaseModel):
    home_team: str
    away_team: str
    predicted: str
    actual: str
    predicted_prob: float  # probability assigned to the predicted outcome
    error_type: str        # "overconfident_wrong" | "wrong_direction" | "goals_off"
```

After calculating metrics, make ONE OpenAI call (gpt-4o, NO web search) to generate `improvement_suggestions`:
```
System: You are a sports analytics expert reviewing a football prediction model's errors.
User: The model made these errors today: {mistakes_summary}. 
Current model features: ELO, form-5, form-10, H2H, tournament weight, neutral venue.
Suggest 3 specific, actionable improvements to the feature engineering or model parameters 
to reduce these errors. Output as a JSON array of strings. No preamble.
```

✅ Output: "feedback_engine.py built — compares predictions vs actuals, generates improvement report"

---

### STEP 4 — Build `src/agent/retrainer.py`

Triggered by feedback_engine after each daily report.

Logic:
```
IF feedback.accuracy < 0.50 for 3 consecutive days:
    → trigger full retraining (runs scripts/train.py)
ELSE IF new_actuals_count >= 5:
    → trigger incremental update: append new matches to features.parquet, retrain with updated data
ELSE:
    → log "Not enough new data for retraining today" and skip
```

Retraining steps:
1. Load `data/actuals/` — collect all confirmed actuals with `confidence != "low"`
2. Run feature engineering on new matches using existing `src/features/engineer.py`
3. Append to `data/processed/features.parquet` (do NOT overwrite — append only)
4. Retrain XGBoost classifier and both score regressors
5. Evaluate on held-out test set (matches after 2022-11-01)
6. Save new model ONLY if new_accuracy >= old_accuracy - 0.02 (do not save a worse model)
7. Version new model: `models/xgb_classifier_{YYYYMMDD}_v{N}.pkl`
8. Update `models/active_model.json` to point to new model path
9. Log all metrics to `models/experiment_log.json`

NEVER delete old model files. NEVER retrain if `data/processed/features.parquet` doesn't exist.

✅ Output: "retrainer.py built — incremental retrain with quality gate"

---

### STEP 5 — Build `src/agent/scheduler.py`

Use APScheduler to run the full agent pipeline daily.

```python
from apscheduler.schedulers.blocking import BlockingScheduler

DAILY_RUN_TIME = "23:00"  # UTC — after most match days end
TIMEZONE = "UTC"

Pipeline order (run sequentially, stop if any step fails):
1. Load today's prediction file from data/predictions/
2. validator.run(today_matches) → actuals
3. feedback_engine.compare(predictions, actuals) → feedback_report
4. retrainer.maybe_retrain(feedback_report)
5. POST feedback_report to FastAPI /internal/daily-update (updates frontend state)
6. Log "Daily pipeline complete" with summary stats
```

Also expose a manual trigger: `python scripts/run_daily_agent.py --date 2026-06-15`

✅ Output: "scheduler.py built — daily pipeline at 23:00 UTC"

---

### STEP 6 — Add API routes `src/api/routes/results.py`

Add two new GET endpoints to the existing FastAPI app:

```
GET /results/daily?date=2026-06-15
Response: {
  "date": "2026-06-15",
  "matches": [
    {
      "home_team": "Brazil",
      "away_team": "Argentina",
      "predicted_result": "H",
      "predicted_home_goals": 1.8,
      "predicted_away_goals": 1.2,
      "actual_result": "D",
      "actual_home_goals": 1,
      "actual_away_goals": 1,
      "correct": false,
      "home_win_prob": 0.52,
      "draw_prob": 0.26,
      "away_win_prob": 0.22
    }
  ],
  "daily_accuracy": 0.43,
  "daily_goal_mae": 0.7
}

GET /results/accuracy?days=7
Response: {
  "period_days": 7,
  "total_matches": 32,
  "correct": 18,
  "accuracy": 0.5625,
  "avg_goal_mae": 0.81,
  "avg_log_loss": 0.94,
  "trend": "improving",
  "model_version": "xgb_classifier_20260615_v2"
}
```

Register routes in `src/api/main.py` — do NOT modify any existing routes.

✅ Output: "results.py routes added and registered"

---

### STEP 7 — Build `frontend/src/components/ResultsTracker.tsx`

Add a new React component to the existing dashboard (do NOT rebuild the dashboard, only add this component).

Component displays:
- Date selector (default: today)
- Table: Home Team | Away Team | Predicted | Actual | ✅/❌ | Confidence
- Color coding: green row = correct, red row = wrong
- Summary bar: "Today: 4/7 correct (57%) | Goal MAE: 0.8"
- 7-day accuracy trend line chart (use recharts — already in project dependencies)
- "Model last retrained: {date}" badge

Data source: fetch from `GET /results/daily` and `GET /results/accuracy`
Polling: refresh every 60 seconds (matches update throughout the day)
Error state: show "Results not yet available" if API returns 404

Integration: Import and render `<ResultsTracker />` in the existing dashboard's main page. Add it below the prediction form. Do NOT remove or modify existing components.

✅ Output: "ResultsTracker.tsx built and integrated into dashboard"

---

## FORBIDDEN ACTIONS

- Do NOT modify `data/raw/` — ever
- Do NOT hardcode OPENAI_API_KEY — always read from environment
- Do NOT rebuild the existing FastAPI app or React dashboard from scratch
- Do NOT modify existing prediction routes (`/predict/match`, `/simulate/tournament`)
- Do NOT push to git
- Do NOT run `npm run build` or deploy
- Do NOT overwrite model files — always create versioned copies
- Do NOT save a retrained model with lower accuracy than the current active model

---

## STOP CONDITIONS — Pause and ask for human review when:

- The OpenAI web search returns conflicting scores from different sources for the same match
- The retrainer would delete or overwrite the currently active model
- A new external dependency is needed that is not in requirements.txt
- The feedback report shows accuracy dropping below 30% (possible data pipeline issue)
- Any database migration or schema change would be required
- An error in any step cannot be resolved in 2 retries

---

## DONE WHEN — All of these are true:

- [ ] `src/agent/` directory exists with validator.py, feedback_engine.py, retrainer.py, scheduler.py
- [ ] `python scripts/run_daily_agent.py --date 2026-06-15` runs without errors
- [ ] `GET /results/daily?date=2026-06-15` returns valid JSON
- [ ] `GET /results/accuracy?days=7` returns valid JSON
- [ ] `ResultsTracker.tsx` renders in the dashboard without breaking existing UI
- [ ] All agent logs write to `logs/` directory
- [ ] No API keys hardcoded anywhere in the codebase

Output a final summary listing every file created or modified.
```

---

> ⚠️ **Agentic Tool Warning:** This prompt gives Claude Code real filesystem and terminal access. Before pasting:
> - Confirm `OPENAI_API_KEY` is set in your `.env` file
> - Confirm you are inside the `fifa-match-predictor/` directory
> - Review the Forbidden Actions list — especially the model versioning rules
> - The scheduler will run at 23:00 UTC daily once started; stop it with `Ctrl+C` or kill the process