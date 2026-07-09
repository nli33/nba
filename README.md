# NBA Predictor

[![Tests](https://github.com/nli33/nba/actions/workflows/tests.yml/badge.svg)](https://github.com/nli33/nba/actions/workflows/tests.yml)

A pre-game NBA game-prediction pipeline: ingest data from the NBA API, build leakage-safe
pre-game features, and train and evaluate baseline, logistic-regression, random-forest, and
LSTM win predictors — with a walk-forward diagnostics and market-benchmark suite.

**Constraints (the golden rule):** predictions use only information available before tip-off,
betting odds are used only as a benchmark (never a feature), and evaluation is strictly
chronological (walk-forward), never a shuffled split. The realistic accuracy ceiling under
these rules is ~68-70%; see [related work](docs/related-work-nba-prediction.md).

## Install

```bash
uv sync                     # runtime + dev tools
uv pip install -e '.[viz]'  # optional: matplotlib, for diagnostic plots
```

## Quickstart

```bash
# 1. ingest + process one season
uv run nba-ingest 2024-25
uv run nba-process 2024-25

# 2. train the default (logistic) model and evaluate it on a later season
uv run nba-train-logistic 2023-24 --output models/lr.pkl
uv run nba-evaluate-logistic models/lr.pkl 2024-25

# 3. predict a single game
uv run nba-predict-logistic models/lr.pkl 0022400360
```

Models default to the 21-feature `DEFAULT_FEATURE_COLUMNS` (team strength/form, schedule/rest,
Elo carryover, and pre-tip player-availability). Swap in a preset with
`--features-file configs/v5.txt`; see [feature sets](docs/feature-sets.md).

## Documentation

| Topic | Docs |
|---|---|
| Ingest raw data and build processed features | [docs/data-pipeline.md](docs/data-pipeline.md) |
| Baseline heuristics | [docs/baselines.md](docs/baselines.md) |
| Logistic regression (train / evaluate / ablate / diagnose) | [docs/logistic-regression.md](docs/logistic-regression.md) |
| Random forest | [docs/random-forest.md](docs/random-forest.md) |
| LSTM sequence model | [docs/lstm.md](docs/lstm.md) |
| Feature-set presets (v3 / v4 / v5) | [docs/feature-sets.md](docs/feature-sets.md) |
| Odds benchmarking (closing lines, CLV) | [docs/odds-benchmarking.md](docs/odds-benchmarking.md) |
| Related work and the accuracy ceiling | [docs/related-work-nba-prediction.md](docs/related-work-nba-prediction.md) |

## Development

```bash
uv run pytest -q            # tests
uv run ruff check src tests # lint
uv run mypy src             # type-check
```
