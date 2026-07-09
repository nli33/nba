# Baseline Predictors

Simple, registered heuristics used as reference points for the trained models.

Available baseline predictors:

```text
home_team
season_to_date_net_rating
season_to_date_win_pct
rolling_10_net_rating
```

## Predict a single game

```bash
uv run nba-predict season_to_date_net_rating 0022500018
```

## Evaluate over a season

Evaluate one baseline predictor over a full season:

```bash
uv run nba-evaluate season_to_date_net_rating 2025-26
```

Print game-by-game results:

```bash
uv run nba-evaluate season_to_date_net_rating 2025-26 --details
```

## Evaluate all baselines at once

```bash
uv run nba-evaluate-baselines 2025-26
```

Print game-by-game results for all baselines:

```bash
uv run nba-evaluate-baselines 2025-26 --details
```
