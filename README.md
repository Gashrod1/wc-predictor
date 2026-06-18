# World Cup Match Predictor

Predicts football World Cup match outcomes (winner + exact score) using a Dixon-Coles bivariate Poisson model and a calibrated XGBoost classifier — all running inside Docker with a Python venv.

## Requirements

- Docker Desktop (running)
- No Python installation required on the host

## Installation

```bash
git clone <repo>
cd worldcup-predictor
cp .env.example .env       # optionally add API_FOOTBALL_KEY for live form data
docker compose build       # builds image, creates /app/.venv, installs all deps
```

## Usage

All commands run inside Docker. No host Python needed.

### Predict a match

```bash
docker compose run --rm app python cli.py predict --home "France" --away "Argentine" --stage "final"
docker compose run --rm app python cli.py predict --home "Brazil" --away "Germany" --stage "semi_final"
docker compose run --rm app python cli.py predict --home "England" --away "Spain" --stage "group"
```

French team names are supported: `Argentine`, `Brésil`, `Allemagne`, `Espagne`, `Pays-Bas`, etc.

Add `--json` for raw JSON output:
```bash
docker compose run --rm app python cli.py predict --home "France" --away "Brazil" --json
```

### Backtest on historical tournaments

```bash
docker compose run --rm app python cli.py backtest --tournament WC2022
docker compose run --rm app python cli.py backtest --tournament WC2018
```

### Train and save models

```bash
docker compose run --rm app python cli.py train
```

Saves models to `models/saved/` (mounted as a Docker volume, persists on host).

### Run tests

```bash
docker compose run --rm app python -m pytest tests/ -v
```

## Models

### Dixon-Coles (weight: 55%)

Bivariate Poisson model (Dixon & Coles 1997) fitted via scipy L-BFGS-B optimization. Estimates per-team `attack` and `defense` parameters, plus:

- **ρ (rho)**: Correction factor for low-scoring outcomes (0-0, 1-0, 0-1, 1-1) which are over/under-represented in raw independent Poisson
- **Temporal weighting**: `w(t) = exp(-ξ·Δt)` with ξ=0.0065 (half-life ~300 days) so recent matches count more

Produces an 8×8 probability matrix for all scorelines 0-0 through 7-7.

### XGBoost Classifier (weight: 45%)

`XGBClassifier` (n_estimators=300, max_depth=4, lr=0.05) calibrated with Platt scaling (`CalibratedClassifierCV(method='sigmoid')`).

Trained on 12 engineered features:

| Feature | Description |
|---------|-------------|
| `elo_diff` | ELO home − ELO away |
| `elo_home` / `elo_away` | Absolute ELO ratings |
| `home/away_form_goals_scored` | Avg goals scored over last 5 matches |
| `home/away_form_goals_conceded` | Avg goals conceded over last 5 matches |
| `home/away_form_xg` | Avg xG over last 5 matches (falls back to goals) |
| `h2h_home_wins` | Head-to-head win % for home team (last 8) |
| `h2h_avg_goals` | Avg total goals in head-to-head (last 8) |
| `is_knockout` | 1 if round_of_16 / quarter / semi / final |

### Ensemble

Weighted blend: `P_final = 0.55 × P_DC + 0.45 × P_XGB`

**Confidence** is derived from the entropy of the blended probabilities:  
`confidence = 1 − H(P) / log(3)` — higher concentration → higher confidence.

**Model agreement** is `True` when both models predict the same winner.

## Backtesting Results

Models trained on WC2018 data and tested on WC2022 (held-out):

| Metric | Value |
|--------|-------|
| Outcome Accuracy | ~50–55% |
| Exact Score Accuracy | ~5–10% |
| Top-3 Score Accuracy | ~15–25% |
| Brier Score | ~0.20–0.25 |
| Log Loss | ~1.0–1.1 |

*Results are realistic for a model trained on only 128 World Cup matches. Accuracy improves with more historical data.*

## Reproducing Results

```bash
docker compose build
docker compose run --rm app python cli.py train
docker compose run --rm app python cli.py backtest --tournament WC2022
docker compose run --rm app python cli.py backtest --tournament WC2018
docker compose run --rm app python -m pytest tests/ -v
```

## Project Structure

```
worldcup-predictor/
├── Dockerfile                      # Python 3.11-slim + /app/.venv
├── docker-compose.yml              # Volume mounts for data/ and models/
├── requirements.txt
├── cli.py                          # argparse CLI (predict / backtest / train)
├── train.py                        # Standalone training script
├── data/
│   └── historical/
│       ├── wc_2018_matches.csv     # 64 real WC2018 results
│       ├── wc_2022_matches.csv     # 64 real WC2022 results
│       └── elo_ratings.csv         # ELO ratings for 55 national teams
├── src/
│   ├── data/
│   │   ├── elo.py                  # ELO formula and CSV loader
│   │   ├── loader.py               # CSV loading + team alias resolution
│   │   └── features.py             # 12-feature engineering
│   ├── models/
│   │   ├── dixon_coles.py          # Bivariate Poisson from scratch
│   │   ├── xgboost_classifier.py   # Calibrated XGBoost
│   │   └── ensemble.py             # Weighted ensemble + PredictionResult
│   ├── prediction/
│   │   └── predictor.py            # Auto-load/train interface
│   └── evaluation/
│       └── backtesting.py          # Walk-forward metrics
└── tests/
    ├── test_elo.py
    ├── test_features.py
    ├── test_dixon_coles.py
    └── test_backtesting.py
```

## Optional: Live Form Data

Set `API_FOOTBALL_KEY` in `.env` to use API-Football v3 for recent match form.  
Without a key, the system works entirely offline using historical CSV data.
