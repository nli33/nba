# Random Forest

A `RandomForestClassifier` over the same `DIFF_*` pre-game features as the logistic model.
Useful as a model-family cross-check; in practice it lands at the same accuracy as
logistic regression (this task is feature-limited, not model-limited).

## Train

```bash
uv run nba-train-random-forest 2024-25 --output models/random_forest_2024-25.pkl
```

The training command prints the number of rows used/dropped and the random forest's
built-in impurity-based feature importances, sorted from highest to lowest.

It supports the same feature arguments as logistic regression:

```bash
uv run nba-train-random-forest 2024-25 \
  --features-file configs/v5.txt \
  --output models/random_forest_custom.pkl
```

## Inspect, evaluate, predict

```bash
uv run nba-inspect-random-forest models/random_forest_2024-25.pkl
uv run nba-evaluate-random-forest models/random_forest_2024-25.pkl 2025-26
uv run nba-predict-random-forest models/random_forest_2024-25.pkl 0022500018
```
