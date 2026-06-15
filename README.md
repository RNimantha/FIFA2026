# FIFA World Cup 2026 — Match Predictor

A production-grade machine learning system for predicting FIFA international match outcomes, simulating the World Cup 2026 tournament, and tracking real match results via a self-learning validation agent.

---

## Features

- **Match outcome prediction** — Win / Draw / Loss probabilities using XGBoost ensemble
- **Score prediction** — Expected goals per team (Poisson regression)
- **ELO ratings** — Rolling ELO calculated over 150+ years of international data (1872–present)
- **Monte Carlo simulator** — 10,000-simulation World Cup bracket predictor
- **Self-learning agent** — Fetches real match results via OpenAI web search, compares to predictions, triggers model retraining
- **Live results feed** — Auto-updating dashboard showing completed WC2026 match results
- **REST API** — FastAPI backend with match prediction, team stats, and tournament simulation endpoints

---

## Model Performance

| Metric | Baseline (Always Home Win) | Model |
|--------|---------------------------|-------|
| Accuracy | 48.3% | **59.4%** |
| F1-macro | — | **0.474** |
| Log-loss | — | **0.883** |
| Brier Score | — | **0.173** |
| Home Goals MAE | 1.23 | **1.04** |
| Away Goals MAE | 0.96 | **0.85** |

Trained on matches from 1950–2017. Validated on 2018–2022. Tested on Qatar WC 2022 onward.

---

## Tech Stack

- **ML:** XGBoost, scikit-learn, LightGBM
- **Data:** pandas, numpy, kagglehub
- **API:** FastAPI, uvicorn, Pydantic v2
- **Agent:** OpenAI GPT-4o (web search), APScheduler
- **Frontend:** Vanilla JS, HTML/CSS (no framework)
- **Viz:** matplotlib, seaborn, plotly

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-proj-...
```

### 3. Download and process data
```bash
python scripts/train.py
```
This downloads the Kaggle dataset (`martj42/international-football-results-from-1872-to-2017`), engineers features, trains all models, and saves them to `models/`.

---

## Running Locally

### Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Start the frontend
```bash
python3 -m http.server 3030 --directory frontend
# Open http://localhost:3030
```

### API docs
```
http://localhost:8000/docs
```

---

## WC2026 Match Day Workflow

```bash
# Morning — save predictions for today's scheduled matches
python3 scripts/save_daily_predictions.py

# After matches finish — fetch real results and update live feed
python3 scripts/run_daily_agent.py --live

# Or run the scheduler (auto-fetches every 2h + full pipeline at 23:00 UTC)
python3 scripts/run_daily_agent.py --schedule
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/match` | Predict outcome + score for any two teams |
| `POST` | `/simulate/tournament` | Monte Carlo tournament simulation |
| `GET` | `/teams/` | List all teams with ELO ratings |
| `GET` | `/teams/{team}/stats` | Team form, ELO, H2H stats |
| `GET` | `/results/completed` | All completed WC2026 results |
| `GET` | `/results/daily?date=` | Predictions vs actuals for a date |
| `GET` | `/results/accuracy?days=7` | Rolling model accuracy metrics |

### Example — predict a match
```bash
curl -X POST http://localhost:8000/predict/match \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "Argentina",
    "tournament": "FIFA World Cup",
    "neutral_venue": true,
    "match_date": "2026-07-19"
  }'
```

```json
{
  "home_win_probability": 0.38,
  "draw_probability": 0.27,
  "away_win_probability": 0.35,
  "predicted_home_goals": 1.6,
  "predicted_away_goals": 1.5,
  "most_likely_score": "1-1",
  "home_elo": 2034,
  "away_elo": 2089
}
```

---

## Project Structure

```
├── src/
│   ├── data/           # Data loading, cleaning, validation
│   ├── features/       # ELO, form, H2H, tournament weighting
│   ├── models/         # XGBoost classifier + Poisson regressors + ensemble
│   ├── simulation/     # Monte Carlo simulator, WC2026 bracket
│   ├── agent/          # Validator, feedback engine, retrainer, scheduler
│   └── api/            # FastAPI routes and schemas
├── scripts/
│   ├── train.py                    # End-to-end training pipeline
│   ├── save_daily_predictions.py   # Save predictions for a match day
│   └── run_daily_agent.py          # Run validation agent
├── frontend/           # Dashboard UI (HTML/CSS/JS)
├── data/
│   ├── processed/      # Feature-engineered parquet files
│   ├── predictions/    # Daily model predictions (JSON)
│   └── actuals/        # Real match results fetched by agent (JSON)
├── models/             # Trained model pkl files + experiment log
└── notebooks/          # EDA + training notebooks
```

---

## Self-Learning Agent

The validation agent runs automatically to keep the model improving:

1. **Validator** — Searches the web for real match scores using GPT-4o
2. **Feedback engine** — Compares predictions vs actuals, calculates accuracy/MAE/log-loss
3. **Retrainer** — Triggers incremental retraining if enough new data; quality gate prevents regression
4. **Scheduler** — Polls every 2 hours during tournament; full pipeline runs at 23:00 UTC

```
Fetch results → Compare predictions → Analyse mistakes → Retrain if warranted → Update live feed
```

---

## Data

**Source:** [Kaggle — martj42/international-football-results-from-1872-to-2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

**Train / Val / Test split (temporal):**
- Training: matches before 2018-01-01
- Validation: 2018-01-01 → 2022-11-01
- Test: 2022-11-01 onward (Qatar WC onward)

---

## Author

**Rashmika (Nimantha Bandara)** — Lead Data Scientist  
Built for FIFA World Cup 2026 prediction and tournament simulation.
