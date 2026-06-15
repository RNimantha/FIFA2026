# CLAUDE.md — FIFA Match Prediction System
## Project: `fifa-match-predictor`
### Engineer: Rashmika (Nimantha Bandara) | Lead Data Scientist

---

## 🎯 Project Mission

Build a **production-grade FIFA international match prediction system** using 150+ years of international football data (1872–present). The system must predict:
1. **Match outcomes** — Win / Draw / Loss (3-class classification)
2. **Score predictions** — Expected goals for each team
3. **Tournament bracket simulation** — FIFA World Cup 2026 path predictor

---

## 📦 Dataset

**Source:** Kaggle — `martj42/international-football-results-from-1872-to-2017`

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "martj42/international-football-results-from-1872-to-2017",
    ""
)
```

**Key columns expected:**
| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Match date |
| `home_team` | str | Home team name |
| `away_team` | str | Away team name |
| `home_score` | int | Goals scored by home team |
| `away_score` | int | Goals scored by away team |
| `tournament` | str | Tournament name (FIFA WC, Friendly, etc.) |
| `city` | str | Host city |
| `country` | str | Host country |
| `neutral` | bool | True if played at neutral venue |

---

## 🏗️ Project Structure

```
fifa-match-predictor/
├── CLAUDE.md                   ← This file (always read first)
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                    ← Original Kaggle data (never modify)
│   ├── processed/              ← Feature-engineered datasets
│   └── predictions/            ← Output predictions
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py           ← Kaggle data loading + caching
│   │   ├── cleaner.py          ← Data cleaning + validation
│   │   └── validator.py        ← Schema + quality checks
│   │
│   ├── features/
│   │   ├── engineer.py         ← Main feature engineering pipeline
│   │   ├── elo.py              ← ELO rating system implementation
│   │   ├── form.py             ← Recent form features (last N matches)
│   │   ├── h2h.py              ← Head-to-head historical stats
│   │   └── tournament.py       ← Tournament weight & importance features
│   │
│   ├── models/
│   │   ├── base.py             ← Abstract base model class
│   │   ├── classifier.py       ← Win/Draw/Loss classifier
│   │   ├── score_predictor.py  ← Goals regression models
│   │   ├── ensemble.py         ← Model stacking/blending
│   │   └── evaluator.py        ← Metrics, cross-validation, reporting
│   │
│   ├── simulation/
│   │   ├── tournament.py       ← Tournament bracket simulator
│   │   ├── monte_carlo.py      ← Monte Carlo match simulations (N=10000)
│   │   └── wc2026.py           ← FIFA World Cup 2026 specific predictor
│   │
│   └── api/
│       ├── main.py             ← FastAPI application entry point
│       ├── routes/
│       │   ├── predict.py      ← POST /predict/match
│       │   ├── simulate.py     ← POST /simulate/tournament
│       │   └── teams.py        ← GET /teams/{team}/stats
│       └── schemas.py          ← Pydantic request/response models
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_wc2026_simulation.ipynb
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
└── scripts/
    ├── train.py                ← CLI: python scripts/train.py
    ├── evaluate.py             ← CLI: python scripts/evaluate.py
    └── simulate_wc2026.py      ← CLI: python scripts/simulate_wc2026.py
```

---

## 🔑 Core Rules (Always Follow)

### Rule 1: Data Integrity
- **NEVER modify files in `data/raw/`** — treat as immutable source of truth
- Always load raw data → apply transformations → save to `data/processed/`
- Log every data transformation with shape before/after
- Validate data at boundaries using `src/data/validator.py`

### Rule 2: Temporal Honesty (Critical for Sports ML)
```
TRAINING DATA:  matches before 2018-01-01
VALIDATION DATA: matches 2018-01-01 to 2022-11-01
TEST DATA:       matches after 2022-11-01 (Qatar WC onward)
```
**NEVER use future data to predict past matches.** All features must use only information available at match time (`date < match_date`).

### Rule 3: Feature Engineering First
Build features before models. Order:
1. ELO ratings (calculated rolling, no lookahead)
2. Form features (last 5/10 matches)
3. Head-to-head stats (historical only)
4. Tournament weighting
5. Venue/neutrality factors

### Rule 4: Model Versioning
- Save every trained model with timestamp: `models/xgb_classifier_20260615_v1.pkl`
- Log all hyperparameters in `models/experiment_log.json`
- Never overwrite a trained model — always version it

### Rule 5: Reproducibility
```python
RANDOM_SEED = 42  # Use everywhere: train_test_split, models, sampling
```

---

## ⚙️ Feature Engineering Specification

### A. ELO Rating System
```
Initial ELO:     1000 for all teams
K-factor:        
  - World Cup matches:    60
  - Confederations Cup:   50
  - WC Qualifying:        40
  - Continental Cup:      35
  - Friendlies:           20

Expected score:  E = 1 / (1 + 10^((opponent_elo - team_elo) / 400))
ELO update:      new_elo = old_elo + K * (actual_score - expected_score)
Home advantage:  +100 ELO points for home team (unless neutral=True)
```

### B. Form Features (Rolling Window)
For each team at match time, calculate over last N=5 and N=10 matches:
- `form_wins`, `form_draws`, `form_losses`
- `form_goals_scored_avg`, `form_goals_conceded_avg`
- `form_points_per_game` (W=3, D=1, L=0)
- `form_clean_sheets`
- `form_goal_diff_avg`

### C. Head-to-Head Features
All historical matches between home_team vs away_team before current date:
- `h2h_total_matches`
- `h2h_home_win_rate`, `h2h_draw_rate`, `h2h_away_win_rate`
- `h2h_home_goals_avg`, `h2h_away_goals_avg`
- `h2h_last_3_results` (encoded: e.g., [W, D, L] → [1, 0, -1])

### D. Tournament Weight Features
```python
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.9,
    "Copa América": 0.9,
    "AFC Asian Cup": 0.85,
    "Africa Cup of Nations": 0.85,
    "FIFA World Cup qualification": 0.75,
    "UEFA Nations League": 0.7,
    "Friendly": 0.3,
}
```

### E. Final Feature Vector (per match)
```python
FEATURES = [
    # ELO
    "home_elo", "away_elo", "elo_diff",
    # Form (5-match window)
    "home_form5_ppg", "away_form5_ppg",
    "home_form5_goals_scored", "away_form5_goals_scored",
    "home_form5_goals_conceded", "away_form5_goals_conceded",
    # Form (10-match window)
    "home_form10_ppg", "away_form10_ppg",
    # H2H
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_total_matches",
    "h2h_home_goals_avg", "h2h_away_goals_avg",
    # Context
    "is_neutral_venue", "tournament_weight",
    "home_goals_rolling_30", "away_goals_rolling_30",
]

TARGET_OUTCOME = "result"        # 0=Away Win, 1=Draw, 2=Home Win
TARGET_HOME_GOALS = "home_score" # Regression
TARGET_AWAY_GOALS = "away_score" # Regression
```

---

## 🤖 Model Specification

### Model 1: Match Outcome Classifier
```python
# Primary: XGBoost (best for tabular + handles class imbalance)
# Backup: LightGBM, RandomForest
# Evaluation: Accuracy, F1-macro, Log-loss, Brier Score

XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)
```

### Model 2: Score Predictor (2 separate regressors)
```python
# Home goals regressor + Away goals regressor
# Use Poisson regression as baseline, XGBRegressor as main
# Evaluation: MAE, RMSE, R²

XGBRegressor(
    objective="count:poisson",  # Goals are count data → use Poisson
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
)
```

### Model 3: Ensemble
```python
# Blend classifier probabilities with score-derived probabilities
# Final P(Home Win) = 0.6 * classifier_prob + 0.4 * score_model_prob
```

### Evaluation Protocol
```
Cross-validation:  TimeSeriesSplit(n_splits=5)  ← MUST use time-based CV
Baseline:          Always predict home win (naive baseline)
Target metrics:
  - Accuracy > 55% (human expert baseline ~57%)
  - Log-loss < 1.0
  - Calibration curve: predicted probs should match actual frequencies
```

---

## 🏆 Tournament Simulator

### Monte Carlo Simulation
```python
N_SIMULATIONS = 10_000  # Run 10k simulations per tournament

def simulate_match(team_a: str, team_b: str, neutral: bool) -> tuple[int, int]:
    """
    Returns (goals_a, goals_b) sampled from Poisson distributions.
    Expected goals from score predictor model.
    """
    lambda_a = home_goals_model.predict(features(team_a, team_b, neutral))
    lambda_b = away_goals_model.predict(features(team_a, team_b, neutral))
    goals_a = np.random.poisson(lambda_a)
    goals_b = np.random.poisson(lambda_b)
    return goals_a, goals_b
```

### World Cup 2026 Groups (Pre-configured)
```python
WC2026_GROUPS = {
    "A": ["USA", "Mexico", "Canada", "..."],  # Fill actual draw groups
    "B": ["Brazil", "...", "...", "..."],
    # ... all 12 groups (48 teams format)
}
```

Output: `{team: {champion_prob, semifinal_prob, quarterfinal_prob, group_exit_prob}}`

---

## 🚀 API Specification

### POST `/predict/match`
```json
Request:
{
  "home_team": "Brazil",
  "away_team": "Argentina",
  "tournament": "FIFA World Cup",
  "neutral_venue": true,
  "match_date": "2026-07-15"
}

Response:
{
  "home_win_probability": 0.42,
  "draw_probability": 0.28,
  "away_win_probability": 0.30,
  "predicted_home_goals": 1.8,
  "predicted_away_goals": 1.4,
  "most_likely_score": "2-1",
  "home_elo": 2034,
  "away_elo": 2089,
  "confidence": "high",
  "model_version": "xgb_v1_20260615"
}
```

### POST `/simulate/tournament`
```json
Request:
{
  "tournament": "FIFA World Cup 2026",
  "n_simulations": 10000
}

Response:
{
  "champion_probabilities": {
    "Brazil": 0.18,
    "France": 0.14,
    "Argentina": 0.13,
    ...
  },
  "simulation_count": 10000,
  "generated_at": "2026-06-15T10:00:00Z"
}
```

---

## 🔧 Tech Stack

```
Python:       3.11+
ML:           scikit-learn, xgboost, lightgbm
Data:         pandas, numpy, kagglehub
Validation:   pydantic
API:          fastapi, uvicorn
Viz:          matplotlib, seaborn, plotly
Testing:      pytest
Notebooks:    jupyter
Tracking:     mlflow (optional)
```

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
kagglehub>=0.2.0
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
pytest>=7.4.0
python-dotenv>=1.0.0
joblib>=1.3.0
```

---

## 📋 Build Phases (Sequential)

### Phase 1: Data Foundation ✅
```bash
# Goal: Clean, validated dataset ready for feature engineering
Tasks:
  □ Load Kaggle dataset via kagglehub
  □ Inspect shape, dtypes, nulls, date range
  □ Standardize team names (e.g., "South Korea" vs "Korea Republic")
  □ Filter to competitive matches (drop very old data pre-1950 for main model)
  □ Create train/val/test splits by date
  □ Save to data/processed/matches_clean.parquet
Output: data/processed/matches_clean.parquet
```

### Phase 2: Feature Engineering ✅
```bash
# Goal: Build all features with ZERO data leakage
Tasks:
  □ Implement rolling ELO calculator (src/features/elo.py)
  □ Implement form features (src/features/form.py)
  □ Implement H2H features (src/features/h2h.py)
  □ Add tournament weights (src/features/tournament.py)
  □ Combine into single feature matrix (src/features/engineer.py)
  □ Validate: no NaN in final feature matrix
  □ Save to data/processed/features.parquet
Output: data/processed/features.parquet
```

### Phase 3: Model Training ✅
```bash
# Goal: Trained, evaluated, versioned models
Tasks:
  □ Train outcome classifier (XGBoost) with TimeSeriesSplit CV
  □ Train home goals regressor (Poisson XGB)
  □ Train away goals regressor (Poisson XGB)
  □ Evaluate all models on held-out test set
  □ Generate calibration plots
  □ Save models with versions
  □ Log all metrics to experiment_log.json
Output: models/*.pkl, models/experiment_log.json
```

### Phase 4: Simulation Engine ✅
```bash
# Goal: Working Monte Carlo tournament simulator
Tasks:
  □ Implement match simulator using Poisson sampling
  □ Implement group stage simulator
  □ Implement knockout bracket simulator
  □ Add WC2026 group draw configuration
  □ Run 10k simulations → generate champion probabilities
Output: data/predictions/wc2026_simulation_results.json
```

### Phase 5: API ✅
```bash
# Goal: FastAPI serving predictions
Tasks:
  □ Create FastAPI app with /predict/match endpoint
  □ Add /simulate/tournament endpoint
  □ Add /teams/{team}/stats endpoint
  □ Add input validation with Pydantic
  □ Add error handling for unknown teams
  □ Test with pytest
Output: Running API on localhost:8000
```

---

## ⚠️ Known Gotchas

### Team Name Standardization (Critical)
The dataset has inconsistent team names across eras. Always normalize:
```python
TEAM_NAME_MAP = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    # Add more as discovered
}
```

### Class Imbalance
Home wins dominate (~46%), away wins (~27%), draws (~27%). Use:
```python
scale_pos_weight  # for XGBoost
# Or: class_weight='balanced' 
# Or: SMOTE (careful — apply only on training fold, never validation)
```

### Data Sparsity for New Teams
Teams with < 20 historical matches → use regional average ELO as fallback.

### Friendly Matches
Weight friendlies very low or exclude entirely from training — teams rotate squads heavily and results are misleading.

---

## 🧪 Test Strategy

```python
# tests/test_features.py — Key assertions
def test_no_data_leakage():
    """All features for match on date D use only data before D."""
    pass

def test_elo_sum_conservation():
    """Total ELO points across all teams should remain roughly constant."""
    pass

def test_feature_completeness():
    """Final feature matrix has zero NaN values."""
    pass

def test_prediction_probabilities_sum_to_one():
    """P(win) + P(draw) + P(loss) = 1.0 for all predictions."""
    pass
```

---

## 📊 Success Metrics

| Metric | Baseline (Always Home Win) | Target |
|--------|---------------------------|--------|
| Accuracy | ~46% | >55% |
| Log-loss | ~1.05 | <0.95 |
| F1-macro | ~0.21 | >0.48 |
| MAE (goals) | ~1.2 | <0.9 |
| Calibration | Poor | Good (ECE < 0.08) |

---

## 💬 Claude Code Workflow

When using Claude Code terminal, always start with:
```
"Read CLAUDE.md first. I am on Phase [X]. Current task: [specific task].
Do not skip ahead to other phases."
```

For debugging:
```
"Read CLAUDE.md. The following error occurred in [file]: [error].
Fix it following the architecture and rules defined in CLAUDE.md."
```

For feature requests:
```
"Read CLAUDE.md. Add [feature] following the existing patterns in src/.
Maintain temporal integrity — no data leakage."
```

---

*Last Updated: June 2026 | Version: 1.0*
*Dataset: martj42/international-football-results-from-1872-to-2017*
*Target: FIFA World Cup 2026 Champion Prediction*