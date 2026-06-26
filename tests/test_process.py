from __future__ import annotations

import math

import pandas as pd
import pytest

from nba_predictor.data import process


def sample_team_game_logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": "G1",
                "GAME_DATE": "2025-10-01",
                "SEASON": "2025-26",
                "SEASON_TYPE": "regular_season",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MATCHUP": "AAA vs. BBB",
                "PTS": 100,
                "WL": "W",
                "PLUS_MINUS": 10,
                "FGA": 80,
                "FTA": 20,
                "OREB": 10,
                "TOV": 12,
            },
            {
                "GAME_ID": "G1",
                "GAME_DATE": "2025-10-01",
                "SEASON": "2025-26",
                "SEASON_TYPE": "regular_season",
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "BBB",
                "MATCHUP": "BBB @ AAA",
                "PTS": 90,
                "WL": "L",
                "PLUS_MINUS": -10,
                "FGA": 82,
                "FTA": 15,
                "OREB": 8,
                "TOV": 14,
            },
            {
                "GAME_ID": "G2",
                "GAME_DATE": "2025-10-02",
                "SEASON": "2025-26",
                "SEASON_TYPE": "regular_season",
                "TEAM_ID": 3,
                "TEAM_ABBREVIATION": "CCC",
                "MATCHUP": "CCC vs. AAA",
                "PTS": 95,
                "WL": "L",
                "PLUS_MINUS": -10,
                "FGA": 78,
                "FTA": 18,
                "OREB": 9,
                "TOV": 11,
            },
            {
                "GAME_ID": "G2",
                "GAME_DATE": "2025-10-02",
                "SEASON": "2025-26",
                "SEASON_TYPE": "regular_season",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MATCHUP": "AAA @ CCC",
                "PTS": 105,
                "WL": "W",
                "PLUS_MINUS": 10,
                "FGA": 83,
                "FTA": 22,
                "OREB": 11,
                "TOV": 10,
            },
            {
                "GAME_ID": "G3",
                "GAME_DATE": "2025-10-05",
                "SEASON": "2025-26",
                "SEASON_TYPE": "regular_season",
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "BBB",
                "MATCHUP": "BBB vs. AAA",
                "PTS": 110,
                "WL": "W",
                "PLUS_MINUS": 10,
                "FGA": 85,
                "FTA": 19,
                "OREB": 12,
                "TOV": 9,
            },
            {
                "GAME_ID": "G3",
                "GAME_DATE": "2025-10-05",
                "SEASON": "2025-26",
                "SEASON_TYPE": "regular_season",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MATCHUP": "AAA @ BBB",
                "PTS": 100,
                "WL": "L",
                "PLUS_MINUS": -10,
                "FGA": 81,
                "FTA": 21,
                "OREB": 7,
                "TOV": 13,
            },
        ]
    ).assign(GAME_DATE=lambda data: pd.to_datetime(data["GAME_DATE"]))


def test_build_games_uses_matchup_sides() -> None:
    games = process.build_games(sample_team_game_logs())

    assert games["GAME_ID"].tolist() == ["G1", "G2", "G3"]
    assert games["HOME_TEAM_ABBREVIATION"].tolist() == ["AAA", "CCC", "BBB"]
    assert games["AWAY_TEAM_ABBREVIATION"].tolist() == ["BBB", "AAA", "AAA"]
    assert games["HOME_PTS"].tolist() == [100, 95, 110]
    assert games["AWAY_PTS"].tolist() == [90, 105, 100]
    assert games["HOME_WIN"].tolist() == [1, 0, 1]


def test_build_team_game_features_are_pre_game_values() -> None:
    logs = sample_team_game_logs()
    games = process.build_games(logs)
    features = process.build_team_game_features(logs, games, previous_games=None)

    aaa_game_1 = features.query("GAME_ID == 'G1' and TEAM_ID == 1").iloc[0]
    aaa_game_2 = features.query("GAME_ID == 'G2' and TEAM_ID == 1").iloc[0]
    aaa_game_3 = features.query("GAME_ID == 'G3' and TEAM_ID == 1").iloc[0]
    bbb_game_3 = features.query("GAME_ID == 'G3' and TEAM_ID == 2").iloc[0]

    assert math.isnan(aaa_game_1["SEASON_TO_DATE_WIN_PCT"])
    assert aaa_game_1["ELO_FLAT_PRE_GAME"] == process.START_ELO
    assert aaa_game_1["ELO_CARRYOVER_PRE_GAME"] == process.START_ELO

    expected_aaa_elo_after_g1, _ = process.updated_elos(
        process.START_ELO,
        process.START_ELO,
        home_win=1,
    )
    assert aaa_game_2["DAYS_REST"] == 1
    assert aaa_game_2["IS_BACK_TO_BACK"] == 1
    assert aaa_game_2["IS_3_IN_4"] == 0
    assert aaa_game_2["SEASON_TO_DATE_WIN_PCT"] == 1.0
    assert aaa_game_2["SEASON_TO_DATE_POINT_DIFF"] == 10.0
    assert aaa_game_2["ROLLING_5_POINT_DIFF"] == 10.0
    assert aaa_game_2["ELO_FLAT_PRE_GAME"] == pytest.approx(expected_aaa_elo_after_g1)

    assert aaa_game_3["DAYS_REST"] == 3
    assert aaa_game_3["IS_BACK_TO_BACK"] == 0
    assert aaa_game_3["IS_3_IN_4"] == 0
    assert aaa_game_3["SEASON_TO_DATE_WIN_PCT"] == 1.0
    assert aaa_game_3["SEASON_TO_DATE_POINT_DIFF"] == 10.0
    assert aaa_game_3["ROLLING_5_WIN_PCT"] == 1.0

    assert bbb_game_3["DAYS_REST"] == 4
    assert bbb_game_3["SEASON_TO_DATE_WIN_PCT"] == 0.0
    assert bbb_game_3["SEASON_TO_DATE_POINT_DIFF"] == -10.0
    assert bbb_game_3["STRENGTH_OF_SCHEDULE_ELO_TO_DATE"] == process.START_ELO


def test_build_model_games_adds_home_minus_away_diffs() -> None:
    logs = sample_team_game_logs()
    games = process.build_games(logs)
    features = process.build_team_game_features(logs, games, previous_games=None)
    model_games = process.build_model_games(games, features)

    game_3 = model_games.query("GAME_ID == 'G3'").iloc[0]

    assert game_3["HOME_TEAM_ABBREVIATION"] == "BBB"
    assert game_3["AWAY_TEAM_ABBREVIATION"] == "AAA"
    assert game_3["HOME_SEASON_TO_DATE_WIN_PCT"] == 0.0
    assert game_3["AWAY_SEASON_TO_DATE_WIN_PCT"] == 1.0
    assert game_3["DIFF_SEASON_TO_DATE_WIN_PCT"] == -1.0
    assert game_3["DIFF_SEASON_TO_DATE_POINT_DIFF"] == -20.0
