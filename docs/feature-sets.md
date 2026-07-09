# Feature Sets

The models default to `DEFAULT_FEATURE_COLUMNS` (in `nba_predictor/models/features.py`), a
21-feature set of `DIFF_*` columns. The `configs/` directory tracks the availability-feature
lineage as reusable `--features-file` presets:

| Config | Features | Contents |
|---|---|---|
| `configs/v3.txt` | 13 | Modern-era baseline: team strength/form + schedule/rest + Elo carryover. No player availability. |
| `configs/v4.txt` | 16 | v3 + the three **volume** inactive signals (count, prior minutes, prior fantasy). |
| `configs/v5.txt` | 21 | v4 + five **value-aware** inactive signals. Equals `DEFAULT_FEATURE_COLUMNS`. |

Use any of them with any model:

```bash
uv run nba-train-logistic 2024-25 --features-file configs/v5.txt
```

## Why availability matters

Adding pre-tip declared-inactive features (v3 → v5) is the single biggest honest lever
over team-aggregate form, lifting walk-forward accuracy roughly 2 points with no
train/test gap. See [related work](related-work-nba-prediction.md) for the empirical
confirmation and the ~68-70% ceiling discussion.

## Custom feature files

A feature file lists one column name per line; blank lines and lines starting with `#` are
ignored. Column names are the processed `model_games.parquet` columns
(`HOME_`/`AWAY_`/`DIFF_` prefixed); the LSTM instead uses the non-prefixed team-level
columns from `team_game_features.parquet`. `configs/paper_enhanced_four_factors_lr.txt` is
a tracked example using the weighted four-factors columns.
