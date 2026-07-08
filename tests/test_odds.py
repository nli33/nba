from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from nba_predictor.data import odds


def test_american_moneyline_to_probability() -> None:
    assert odds.american_moneyline_to_probability(-200) == pytest.approx(2 / 3)
    assert odds.american_moneyline_to_probability(150) == pytest.approx(0.4)
    assert odds.american_moneyline_to_probability(-110) == pytest.approx(110 / 210)


def test_load_season_odds_maps_teams_and_devigs(tmp_path: Path) -> None:
    db_path = tmp_path / "odds.sqlite"
    raw = pd.DataFrame(
        [
            {"Date": "2024-10-22", "Home": "Boston Celtics", "Away": "New York Knicks",
             "ML_Home": -110, "ML_Away": -110},
            {"Date": "2024-10-23", "Home": "LA Clippers", "Away": "Denver Nuggets",
             "ML_Home": -200, "ML_Away": 170},
        ]
    )
    with sqlite3.connect(db_path) as connection:
        raw.to_sql("2024-25", connection, index=False)

    result = odds.load_season_odds("2024-25", db_path)

    assert result["HOME_TEAM_ABBREVIATION"].tolist() == ["BOS", "LAC"]
    assert result["AWAY_TEAM_ABBREVIATION"].tolist() == ["NYK", "DEN"]
    # A -110/-110 pick'em de-vigs to exactly 0.5.
    assert result["MARKET_P_HOME"].iloc[0] == pytest.approx(0.5)
    # The favorite's de-vigged probability sits between its raw implied prob and 1.
    assert 0.5 < result["MARKET_P_HOME"].iloc[1] < 2 / 3
