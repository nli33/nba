"""Load historical closing betting odds for benchmarking (never a model feature).

Odds are used only to score our models against the market, per the project's
golden rule. Source: the community-maintained SBR-derived odds database in
kyleskom/NBA-Machine-Learning-Sports-Betting (one SQLite table per season with
American moneylines). We convert the moneylines to a de-vigged market win
probability for the home team.
"""

from __future__ import annotations

import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd


ODDS_DB_URL = (
    "https://raw.githubusercontent.com/kyleskom/"
    "NBA-Machine-Learning-Sports-Betting/master/Data/OddsData.sqlite"
)

# Full team names as they appear in the odds database -> our abbreviations.
TEAM_NAME_TO_ABBREVIATION = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def american_moneyline_to_probability(moneyline: float) -> float:
    """Convert an American moneyline to its implied win probability (with vig)."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100.0)
    return 100.0 / (moneyline + 100.0)


def download_odds_db(destination: Path) -> Path:
    """Download the odds SQLite database if it is not already cached locally."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ODDS_DB_URL, destination)
    return destination


def load_season_odds(season: str, db_path: Path) -> pd.DataFrame:
    """Return one row per game: teams, moneylines, and de-vigged market P(home win).

    ``db_path`` is the local OddsData.sqlite (see ``download_odds_db``). The season
    table (e.g. ``"2024-25"``) holds American moneylines for each game.
    """
    with sqlite3.connect(db_path) as connection:
        raw = pd.read_sql(f'select * from "{season}"', connection)

    odds = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(raw["Date"]),
            "HOME_TEAM_ABBREVIATION": raw["Home"].map(TEAM_NAME_TO_ABBREVIATION),
            "AWAY_TEAM_ABBREVIATION": raw["Away"].map(TEAM_NAME_TO_ABBREVIATION),
            "ML_HOME": raw["ML_Home"].astype(float),
            "ML_AWAY": raw["ML_Away"].astype(float),
        }
    )
    unmapped = odds[["HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION"]].isna().any(axis=1)
    if unmapped.any():
        raise ValueError(f"Unmapped team names in {season} odds: {raw.loc[unmapped, ['Home', 'Away']].values.tolist()}")

    implied_home = odds["ML_HOME"].map(american_moneyline_to_probability)
    implied_away = odds["ML_AWAY"].map(american_moneyline_to_probability)
    # Remove the bookmaker's vig by normalising the two implied probabilities to 1.
    odds["MARKET_P_HOME"] = implied_home / (implied_home + implied_away)
    return odds
