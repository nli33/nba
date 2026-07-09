# Odds Benchmarking

Betting odds are used **only as a benchmark** — to score the models against the market —
and are **never** a model feature. This follows the project's golden rule that the same
system must be runnable live before tip-off.

## Closing lines (`nba_predictor.data.odds`)

`odds.py` loads historical **closing** moneylines from the community SBR-derived odds
database and converts them to a de-vigged market win probability for the home team.

```python
from pathlib import Path
from nba_predictor.data.odds import download_odds_db, load_season_odds

db = download_odds_db(Path("data/odds/OddsData.sqlite"))   # cached after first download
odds = load_season_odds("2024-25", db)
# columns: GAME_DATE, HOME_TEAM_ABBREVIATION, AWAY_TEAM_ABBREVIATION,
#          ML_HOME, ML_AWAY, MARKET_P_HOME  (de-vigged implied P(home win))
```

Join `odds` to a season's `model_games.parquet` on
`["GAME_DATE", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"]` to compare model
probabilities against the market's on identical games (accuracy, log loss, Brier,
calibration). On 2024-25 the closing line scores ~69.9% and the logistic model ~68.6% —
the market is the sharper forecaster, and a forecast-encompassing test shows it subsumes
the models' information.

## Opening-line snapshots (`nba_predictor.data.odds_api`)

`odds_api.py` fetches a point-in-time **snapshot** from The Odds API historical endpoint so
opening vs closing lines can be compared for closing-line value (CLV). The historical
endpoint requires a **paid** Odds API plan; supply the key via the `ODDS_API_KEY`
environment variable.

```python
from nba_predictor.data.odds_api import snapshot_market_probs

# de-vigged market P(home win) at a given ISO8601 UTC timestamp
opener = snapshot_market_probs("2024-12-19T13:00:00Z")   # reads ODDS_API_KEY
```

The pure `parse_h2h_snapshot(payload)` has no network dependency and is unit-tested. Use an
opening snapshot (morning of) and a closing snapshot (near tip) to measure whether the
model can beat the softer opening line — the only realistic route to a betting edge, since
the closing line already encompasses the models.
