# LSTM

A sequence model. Instead of a single pooled feature vector per game, it consumes, for
each matchup, the chronological sequence of each team's own prior pre-game feature vectors
(from `team_game_features.parquet`). Sequences use only games strictly before the predicted
game, so game `t` is predicted from history ending at `t-1`.

## Train

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

## Early stopping and validation

By default training holds out the most recent `--validation-fraction` (0.15) of the
training games — the latest games by date, a leakage-free validation split — and keeps the
epoch with the lowest validation log loss, early-stopping after `--patience` (5) epochs
without improvement. `--epochs` is therefore the **maximum** number of epochs. This avoids
over- or under-training the network; the training report prints how many epochs were kept
and the validation log loss.

For an untouched final estimate, evaluate on a season later than every training season.
Set `--validation-fraction 0` to train for exactly `--epochs` with no validation:

```bash
uv run nba-train-lstm 2021-22 2022-23 2023-24 \
  --epochs 40 --validation-fraction 0.15 --patience 5 \
  --output models/lstm_multi.pkl
```

## Inspect, evaluate, predict

```bash
uv run nba-inspect-lstm models/lstm_multi.pkl
uv run nba-evaluate-lstm models/lstm_multi.pkl 2024-25
uv run nba-predict-lstm models/lstm_multi.pkl 0022500018
```

Games where either team has fewer than `--min-history` prior games in the season receive
no prediction (mirroring the missing-feature behavior of the other models).
