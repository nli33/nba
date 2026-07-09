# Data Pipeline

Ingest raw NBA API data and build the processed feature parquet files the models train on.

## Ingest raw data

Fetch raw NBA API data for one season:

```bash
uv run nba-ingest 2025-26
```

Overwrite existing raw parquet files:

```bash
uv run nba-ingest 2025-26 --overwrite
```

Fetch declared-inactive (pre-tip availability) data for a season:

```bash
uv run nba-ingest-inactives 2025-26
```

## Build processed features

Build processed game, team-feature, and model-game parquet files:

```bash
uv run nba-process 2025-26
```

Overwrite existing processed parquet files:

```bash
uv run nba-process 2025-26 --overwrite
```

Processing writes three parquet files per season under `data/processed/<season>/`:

- `model_games.parquet` — one row per game with `HOME_`/`AWAY_`/`DIFF_` feature columns (the table the logistic and random-forest models train on).
- `team_game_features.parquet` — one row per team-game with the non-prefixed team-level features (the sequence input for the LSTM).
- `games.parquet` — processed game metadata.

Player-availability (declared inactives) features are populated from the **2005-06** season on; earlier seasons lack them and require an explicit feature set.
