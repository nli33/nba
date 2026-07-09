from __future__ import annotations

import pytest

from nba_predictor.data import odds_api


def test_team_abbreviation_handles_odds_api_aliases() -> None:
    assert odds_api.team_abbreviation("Boston Celtics") == "BOS"
    assert odds_api.team_abbreviation("Los Angeles Clippers") == "LAC"
    with pytest.raises(ValueError, match="Unknown team"):
        odds_api.team_abbreviation("Springfield Isotopes")


def test_parse_h2h_snapshot_averages_books_and_devigs() -> None:
    payload = {
        "timestamp": "2024-12-19T12:00:00Z",
        "data": [
            {
                "home_team": "Detroit Pistons",
                "away_team": "Utah Jazz",
                "commence_time": "2024-12-20T00:00:00Z",
                "bookmakers": [
                    {"markets": [{"key": "h2h", "outcomes": [
                        {"name": "Detroit Pistons", "price": 1.5},
                        {"name": "Utah Jazz", "price": 2.5},
                    ]}]},
                    {"markets": [{"key": "h2h", "outcomes": [
                        {"name": "Detroit Pistons", "price": 1.5},
                        {"name": "Utah Jazz", "price": 2.5},
                    ]}]},
                ],
            }
        ],
    }
    df = odds_api.parse_h2h_snapshot(payload)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["HOME_TEAM_ABBREVIATION"] == "DET"
    assert row["AWAY_TEAM_ABBREVIATION"] == "UTA"
    # implied: home 1/1.5=0.667, away 1/2.5=0.4; de-vig -> 0.667/1.067
    assert row["MARKET_P_HOME"] == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.5))


def test_parse_h2h_snapshot_skips_events_without_h2h() -> None:
    payload = {"data": [{"home_team": "Miami Heat", "away_team": "Orlando Magic",
                         "commence_time": "2024-12-20T00:00:00Z", "bookmakers": []}]}
    assert odds_api.parse_h2h_snapshot(payload).empty
