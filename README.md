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

By default, training uses the built-in feature set (`DEFAULT_FEATURE_COLUMNS`): team
strength/form (season-to-date and rolling win%, point differential, net rating),
schedule/rest, cross-season Elo carryover, and pre-tip player-availability (declared
inactives) features. The availability features are the strongest single lever over
team-aggregate form (see `docs/related-work-nba-prediction.md`) and are populated from
the 2005-06 season on; training on earlier seasons requires an explicit feature set.

To train with explicit feature columns instead:

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

Run walk-forward probability diagnostics over the same rolling season splits:

```bash
uv run nba-diagnose-logistic \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 \
  --min-train-seasons 5
```

This collects out-of-fold home-win probabilities across the expanding chronological
splits and reports three views of them: **calibration** (expected calibration error and
the Murphy `reliability - resolution + uncertainty` decomposition of the Brier score),
**selective prediction** (accuracy by confidence decile — the model is much more accurate
on the games it is most sure about), and **upset structure** (where the model's wrong
picks fall on the confidence axis). It accepts the same `--features` / `--features-file`
arguments as the other commands.

Add `--plots-dir DIR` to also write `calibration.png`, `selective_prediction.png`, and
`upset_structure.png`. Plotting needs the optional `viz` extra:

```bash
uv pip install -e '.[viz]'
uv run nba-diagnose-logistic \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 \
  --min-train-seasons 5 \
  --plots-dir reports/diagnostics
```

Run the same ablation with repeated randomized train/eval splits instead of the
chronological season split:

```bash
uv run nba-ablate-logistic \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --features-file configs/features.txt \
  --split-strategy randomized \
  --test-size 0.2 \
  --random-repeats 10 \
  --random-seed 0
```

## Random Forest

Train a random forest model on one processed season and save it:

```bash
uv run nba-train-random-forest 2024-25 --output models/random_forest_2024-25.pkl
```

The training command prints the number of rows used/dropped and feature importances
sorted from highest to lowest. These are the random forest's built-in impurity-based
feature importances.

Random forest training supports the same feature arguments as logistic regression:

```bash
uv run nba-train-random-forest 2024-25 \
  --features-file configs/features.txt \
  --output models/random_forest_custom.pkl
```

Inspect, evaluate, or predict with a saved random forest model:

```bash
uv run nba-inspect-random-forest models/random_forest_2024-25.pkl
uv run nba-evaluate-random-forest models/random_forest_2024-25.pkl 2025-26
uv run nba-predict-random-forest models/random_forest_2024-25.pkl 0022500018
```

## LSTM

The LSTM is a sequence model. Instead of a single pooled feature vector per game,
it consumes, for each matchup, the chronological sequence of each team's own prior
pre-game feature vectors (from `team_game_features.parquet`). Sequences use only
games strictly before the predicted game, so game `t` is predicted from history
ending at `t-1`.

Train an LSTM on one or more processed seasons and save it:

```bash
uv run nba-train-lstm 2023-24 --output models/lstm_2023-24.pkl
uv run nba-train-lstm 2021-22 2022-23 2023-24 --output models/lstm_multi.pkl
```

Sequence models benefit from multiple training seasons, so training accepts several
seasons at once. Training uses the built-in sequence feature set (team-centric
counterparts of the default `DIFF_*` features). Override with explicit columns or a
feature file, and tune the sequence hyperparameters:

```bash
uv run nba-train-lstm 2023-24 \
  --features-file configs/sequence_features.txt \
  --sequence-length 15 \
  --min-history 5 \
  --hidden-size 32 \
  --epochs 20 \
  --output models/lstm_custom.pkl
```

Feature columns are the team-level (non-prefixed) columns from
`team_game_features.parquet`, for example `SEASON_TO_DATE_NET_RATING`.

Inspect, evaluate, or predict with a saved LSTM model:

```bash
uv run nba-inspect-lstm models/lstm_multi.pkl
uv run nba-evaluate-lstm models/lstm_multi.pkl 2024-25
uv run nba-predict-lstm models/lstm_multi.pkl 0022500018
```

Games where either team has fewer than `--min-history` prior games in the season
receive no prediction (mirroring the missing-feature behavior of the other models).
