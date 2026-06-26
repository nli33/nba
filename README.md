# NBA Predictor

An NBA game prediction pipeline: ingesting data from nba_api, building processed features from raw data, baseline predictors, and more advanced models (WIP)

## Data Pipeline

Fetch raw NBA API data for one season:

```bash
uv run nba-ingest 2025-26
```

Overwrite existing raw parquet files:

```bash
uv run nba-ingest 2025-26 --overwrite
```

Build processed game, team-feature, and model-game parquet files:

```bash
uv run nba-process 2025-26
```

Overwrite existing processed parquet files:

```bash
uv run nba-process 2025-26 --overwrite
```

## Baseline Prediction

Predict one game with a registered baseline predictor:

```bash
uv run nba-predict season_to_date_net_rating 0022500018
```

Available baseline predictors:

```text
home_team
season_to_date_net_rating
season_to_date_win_pct
rolling_10_net_rating
```

Evaluate one baseline predictor over a full season:

```bash
uv run nba-evaluate season_to_date_net_rating 2025-26
```

Print game-by-game results for one baseline predictor:

```bash
uv run nba-evaluate season_to_date_net_rating 2025-26 --details
```

Evaluate all baseline predictors over a full season:

```bash
uv run nba-evaluate-baselines 2025-26
```

Print game-by-game results for all baseline predictors:

```bash
uv run nba-evaluate-baselines 2025-26 --details
```

## Logistic Regression

Train a logistic regression model on one processed season and save it:

```bash
uv run nba-train-logistic 2024-25 --output models/logistic_regression_2024-25.pkl
```

The training command prints the number of rows used/dropped, the intercept, and learned
coefficients sorted by absolute magnitude.

By default, training uses the built-in preliminary feature set. To train with explicit
feature columns instead:

```bash
uv run nba-train-logistic 2024-25 \
  --features DIFF_SEASON_TO_DATE_NET_RATING DIFF_ROLLING_10_NET_RATING \
  --output models/logistic_regression_custom.pkl
```

For longer or repeatable feature sets, use a text file with one column per line:

```bash
uv run nba-train-logistic 2024-25 \
  --features-file configs/features.txt \
  --output models/logistic_regression_custom.pkl
```

Saved model artifacts store the feature columns used at training time, so evaluation and
single-game prediction do not need feature arguments.

Inspect a saved model later:

```bash
uv run nba-inspect-logistic models/logistic_regression_2024-25.pkl
```

Evaluate the saved model on a different season:

```bash
uv run nba-evaluate-logistic models/logistic_regression_2024-25.pkl 2025-26
```

Print game-by-game results for the saved model:

```bash
uv run nba-evaluate-logistic models/logistic_regression_2024-25.pkl 2025-26 --details
```

Predict one game with the saved model:

```bash
uv run nba-predict-logistic models/logistic_regression_2024-25.pkl 0022500018
```

Single-game logistic predictions include `P(home wins)` when all required pre-game
features are available.

Run drop-one-feature ablation across rolling historical splits:

```bash
uv run nba-ablate-logistic \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --features-file configs/features.txt
```

Create `configs/features.txt` locally with one feature column per line. Blank lines
and lines starting with `#` are ignored.

The command prints progress while fitting models, then outputs one final report with
average accuracy, correct-pick deltas, log loss, and Brier score.
