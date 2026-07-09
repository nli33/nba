"""Ingest historical odds *snapshots* from The Odds API for CLV analysis.

Unlike the single closing line in ``odds.py``, this pulls a point-in-time
snapshot (e.g. the opening line the morning before tip, or the closing line at
tip) so we can measure Closing Line Value. Odds remain a benchmark, never a
model feature.

The historical endpoint requires a paid Odds API plan; pass the key via the
``ODDS_API_KEY`` environment variable. The pure ``parse_h2h_snapshot`` function
has no network dependency and is unit-tested.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from nba_predictor.data.odds import TEAM_NAME_TO_ABBREVIATION

HISTORICAL_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"

# The Odds API spells a few teams differently from the odds.py closing-line source.
ODDS_API_NAME_ALIASES = {
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
}


def team_abbreviation(name: str) -> str:
    if name in TEAM_NAME_TO_ABBREVIATION:
        return TEAM_NAME_TO_ABBREVIATION[name]
    if name in ODDS_API_NAME_ALIASES:
        return ODDS_API_NAME_ALIASES[name]
    raise ValueError(f"Unknown team name from odds API: {name!r}")


def _devig(home_decimal: float, away_decimal: float) -> float:
    """De-vigged home win probability from a two-way decimal-odds market."""
    implied_home = 1.0 / home_decimal
    implied_away = 1.0 / away_decimal
    return implied_home / (implied_home + implied_away)


def parse_h2h_snapshot(payload: dict) -> pd.DataFrame:
    """One row per game: teams, commence time, and de-vigged market P(home win).

    ``payload`` is the decoded historical snapshot (``{"timestamp", "data": [...]}``)
    with decimal-format h2h prices. Prices are averaged across all bookmakers in
    the snapshot before de-vigging.
    """
    rows = []
    for event in payload["data"]:
        home, away = event["home_team"], event["away_team"]
        home_prices, away_prices = [], []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                by_name = {o["name"]: o["price"] for o in market["outcomes"]}
                if home in by_name and away in by_name:
                    home_prices.append(by_name[home])
                    away_prices.append(by_name[away])
        if not home_prices:
            continue
        avg_home = sum(home_prices) / len(home_prices)
        avg_away = sum(away_prices) / len(away_prices)
        rows.append(
            {
                "COMMENCE_TIME": pd.to_datetime(event["commence_time"]),
                "HOME_TEAM_ABBREVIATION": team_abbreviation(home),
                "AWAY_TEAM_ABBREVIATION": team_abbreviation(away),
                "MARKET_P_HOME": _devig(avg_home, avg_away),
            }
        )
    return pd.DataFrame(rows)


def fetch_snapshot(timestamp_iso: str, api_key: str | None = None) -> dict:
    """Fetch the h2h snapshot at (or just before) ``timestamp_iso`` (ISO8601 UTC)."""
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise ValueError("Set ODDS_API_KEY (paid Odds API plan) to fetch snapshots")
    query = urllib.parse.urlencode(
        {
            "apiKey": key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "date": timestamp_iso,
        }
    )
    with urllib.request.urlopen(f"{HISTORICAL_URL}?{query}") as response:
        return json.loads(response.read().decode())


def snapshot_market_probs(timestamp_iso: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch and parse a snapshot into tidy per-game market probabilities."""
    return parse_h2h_snapshot(fetch_snapshot(timestamp_iso, api_key))
