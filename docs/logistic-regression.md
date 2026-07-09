# Logistic Regression

A `StandardScaler` + `LogisticRegression` pipeline over the `DIFF_*` pre-game features.
This is the project's strongest and default model.

## Train

Train on one processed season and save it:

```bash
uv run nba-train-logistic 2024-25 --output models/logistic_regression_2024-25.pkl
```

The training command prints the number of rows used/dropped, the intercept, and learned
coefficients sorted by absolute magnitude.

By default, training uses the built-in feature set (`DEFAULT_FEATURE_COLUMNS`): team
strength/form (season-to-date and rolling win%, point differential, net rating),
schedule/rest, cross-season Elo carryover, and pre-tip player-availability (declared
inactives) features. The availability features are the strongest single lever over
team-aggregate form (see [related work](related-work-nba-prediction.md)) and are
populated from the 2005-06 season on; training on earlier seasons requires an explicit
feature set. See [feature sets](feature-sets.md) for the `configs/v3|v4|v5.txt` presets.

Train with explicit feature columns:

```bash
uv run nba-train-logistic 2024-25 \
  --features DIFF_SEASON_TO_DATE_NET_RATING DIFF_ROLLING_10_NET_RATING \
  --output models/logistic_regression_custom.pkl
```

Or a feature file (one column per line; blank lines and `#` comments ignored):

```bash
uv run nba-train-logistic 2024-25 \
  --features-file configs/v5.txt \
  --output models/logistic_regression_custom.pkl
```

Saved model artifacts store the feature columns used at training time, so evaluation and
single-game prediction do not need feature arguments.

## Inspect, evaluate, predict

```bash
uv run nba-inspect-logistic models/logistic_regression_2024-25.pkl
uv run nba-evaluate-logistic models/logistic_regression_2024-25.pkl 2025-26
uv run nba-evaluate-logistic models/logistic_regression_2024-25.pkl 2025-26 --details
uv run nba-predict-logistic models/logistic_regression_2024-25.pkl 0022500018
```

Single-game predictions include `P(home wins)` when all required pre-game features are
available.

## Ablation

Run drop-one-feature ablation across rolling historical splits:

```bash
uv run nba-ablate-logistic \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --features-file configs/v5.txt
```

It prints progress while fitting, then a final report with average accuracy, correct-pick
deltas, log loss, and Brier score.

Use repeated randomized train/eval splits instead of the chronological season split (for
comparison only — the chronological split is the honest one):

```bash
uv run nba-ablate-logistic \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --features-file configs/v5.txt \
  --split-strategy randomized \
  --test-size 0.2 \
  --random-repeats 10 \
  --random-seed 0
```

## Diagnostics

Run walk-forward probability diagnostics over the same rolling season splits:

```bash
uv run nba-diagnose-logistic \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 \
  --min-train-seasons 5
```

This collects out-of-fold home-win probabilities across the expanding chronological
splits and reports three views:

- **Calibration** — expected calibration error and the Murphy
  `reliability - resolution + uncertainty` decomposition of the Brier score.
- **Selective prediction** — accuracy by confidence decile (the model is far more accurate
  on the games it is most sure about).
- **Upset structure** — where the model's wrong picks fall on the confidence axis.

It accepts the same `--features` / `--features-file` arguments as the other commands.

Add `--plots-dir DIR` to also write `calibration.png`, `selective_prediction.png`, and
`upset_structure.png`. Plotting needs the optional `viz` extra:

```bash
uv pip install -e '.[viz]'
uv run nba-diagnose-logistic \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25 \
  --min-train-seasons 5 \
  --plots-dir reports/diagnostics
```
