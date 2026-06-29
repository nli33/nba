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
                "FGM": 40,
                "FGA": 80,
                "FG3M": 10,
                "FG3A": 30,
                "FTA": 20,
                "OREB": 10,
                "DREB": 30,
                "AST": 25,
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
                "FGM": 35,
                "FGA": 82,
                "FG3M": 8,
                "FG3A": 25,
                "FTA": 15,
                "OREB": 8,
                "DREB": 28,
                "AST": 20,
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
                "FGM": 37,
                "FGA": 78,
                "FG3M": 9,
                "FG3A": 24,
                "FTA": 18,
                "OREB": 9,
                "DREB": 29,
                "AST": 22,
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
                "FGM": 42,
                "FGA": 83,
                "FG3M": 11,
                "FG3A": 31,
                "FTA": 22,
                "OREB": 11,
                "DREB": 31,
                "AST": 27,
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
                "FGM": 43,
                "FGA": 85,
                "FG3M": 12,
                "FG3A": 32,
                "FTA": 19,
                "OREB": 12,
                "DREB": 32,
                "AST": 26,
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
                "FGM": 39,
                "FGA": 81,
                "FG3M": 9,
                "FG3A": 27,
                "FTA": 21,
                "OREB": 7,
                "DREB": 27,
                "AST": 24,
                "TOV": 13,
            },
        ]
    ).assign(GAME_DATE=lambda data: pd.to_datetime(data["GAME_DATE"]))


def sample_player_game_logs() -> pd.DataFrame:
    rows = []
    player_id = 1
    team_players = {
        "G1": [
            (1, "AAA", 36, 40, 10, 20, 5, 10),
            (1, "AAA", 34, 35, 8, 18, 4, 8),
            (1, "AAA", 30, 20, 5, 12, 3, 5),
            (1, "AAA", 20, 10, -3, 7, 2, 1),
            (2, "BBB", 35, 30, -8, 15, 3, 6),
            (2, "BBB", 33, 25, -6, 13, 2, 4),
            (2, "BBB", 28, 20, -4, 11, 1, 2),
            (2, "BBB", 24, 12, 1, 8, 2, 1),
        ],
        "G2": [
            (3, "CCC", 34, 32, -8, 16, 3, 7),
            (3, "CCC", 32, 26, -4, 14, 2, 5),
            (1, "AAA", 37, 45, 12, 21, 6, 9),
            (1, "AAA", 35, 36, 9, 17, 4, 7),
        ],
        "G3": [
            (2, "BBB", 38, 42, 11, 22, 5, 8),
            (2, "BBB", 34, 31, 6, 14, 3, 5),
            (1, "AAA", 36, 34, -5, 18, 4, 6),
            (1, "AAA", 30, 24, -8, 12, 3, 4),
        ],
    }
    game_dates = {"G1": "2025-10-01", "G2": "2025-10-02", "G3": "2025-10-05"}

    for game_id, players in team_players.items():
        for team_id, abbreviation, minutes, fantasy, plus_minus, fga, fta, tov in players:
            rows.append(
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": game_dates[game_id],
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "TEAM_ID": team_id,
                    "TEAM_ABBREVIATION": abbreviation,
                    "MIN": minutes,
                    "FGA": fga,
                    "FTA": fta,
                    "TOV": tov,
                    "PLUS_MINUS": plus_minus,
                    "NBA_FANTASY_PTS": fantasy,
                }
            )
            player_id += 1

    return pd.DataFrame(rows).assign(GAME_DATE=lambda data: pd.to_datetime(data["GAME_DATE"]))


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
    assert aaa_game_1["CONSECUTIVE_ROAD_GAMES"] == 0

    expected_aaa_elo_after_g1, _ = process.updated_elos(
        process.START_ELO,
        process.START_ELO,
        home_win=1,
    )
    assert aaa_game_2["DAYS_REST"] == 1
    assert aaa_game_2["IS_BACK_TO_BACK"] == 1
    assert aaa_game_2["IS_3_IN_4"] == 0
    assert aaa_game_2["CONSECUTIVE_ROAD_GAMES"] == 1
    assert aaa_game_2["SEASON_TO_DATE_WIN_PCT"] == 1.0
    assert aaa_game_2["SEASON_TO_DATE_POINT_DIFF"] == 10.0
    assert aaa_game_2["ROLLING_5_POINT_DIFF"] == 10.0
    assert aaa_game_2["ELO_FLAT_PRE_GAME"] == pytest.approx(expected_aaa_elo_after_g1)
    assert aaa_game_2["SEASON_TO_DATE_OFF_EFG_PCT"] == pytest.approx(45 / 80)
    assert aaa_game_2["SEASON_TO_DATE_DEF_EFG_PCT"] == pytest.approx(39 / 82)
    assert aaa_game_2["SEASON_TO_DATE_OFF_TOV_RATE"] == pytest.approx(12 / 90.8)
    assert aaa_game_2["SEASON_TO_DATE_DEF_TOV_RATE"] == pytest.approx(14 / 94.6)
    assert aaa_game_2["SEASON_TO_DATE_OFF_OREB_RATE"] == pytest.approx(10 / 38)
    assert aaa_game_2["SEASON_TO_DATE_DEF_OREB_RATE"] == pytest.approx(8 / 38)
    assert aaa_game_2["SEASON_TO_DATE_OFF_FT_RATE"] == pytest.approx(20 / 80)
    assert aaa_game_2["SEASON_TO_DATE_DEF_FT_RATE"] == pytest.approx(15 / 82)
    assert aaa_game_2["SEASON_TO_DATE_PACE"] == pytest.approx((90.8 + 94.6) / 2)
    assert aaa_game_2["SEASON_TO_DATE_FG3A_RATE"] == pytest.approx(30 / 80)
    assert aaa_game_2["SEASON_TO_DATE_OPP_FG3A_RATE"] == pytest.approx(25 / 82)
    assert aaa_game_2["SEASON_TO_DATE_AST_RATE"] == pytest.approx(25 / 40)
    assert aaa_game_2["SEASON_TO_DATE_OPP_AST_RATE"] == pytest.approx(20 / 35)
    assert aaa_game_2["ROLLING_5_OFF_EFG_PCT"] == pytest.approx(45 / 80)

    assert aaa_game_3["DAYS_REST"] == 3
    assert aaa_game_3["IS_BACK_TO_BACK"] == 0
    assert aaa_game_3["IS_3_IN_4"] == 0
    assert aaa_game_3["CONSECUTIVE_ROAD_GAMES"] == 2
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
    assert game_3["AWAY_CONSECUTIVE_ROAD_GAMES"] == 2
    assert game_3["DIFF_CONSECUTIVE_ROAD_GAMES"] == -2
    assert "DIFF_SEASON_TO_DATE_OFF_EFG_PCT" in model_games.columns
    assert "DIFF_ROLLING_10_PACE" in model_games.columns


def test_build_team_game_features_can_use_prior_player_logs() -> None:
    logs = sample_team_game_logs()
    games = process.build_games(logs)
    features = process.build_team_game_features(
        logs,
        games,
        previous_games=None,
        player_game_logs=sample_player_game_logs(),
    )
    model_games = process.build_model_games(games, features)

    aaa_game_1 = features.query("GAME_ID == 'G1' and TEAM_ID == 1").iloc[0]
    aaa_game_2 = features.query("GAME_ID == 'G2' and TEAM_ID == 1").iloc[0]

    assert math.isnan(aaa_game_1["ROLLING_5_PLAYER_TOP3_MIN_SHARE"])
    assert aaa_game_2["ROLLING_5_PLAYER_ROTATION_PLAYERS_10_MIN"] == 4
    assert aaa_game_2["ROLLING_5_PLAYER_TOP3_MIN_SHARE"] == pytest.approx(100 / 120)
    assert aaa_game_2["ROLLING_5_PLAYER_TOP5_MIN_SHARE"] == 1.0
    assert aaa_game_2["ROLLING_5_PLAYER_TOP3_FANTASY_SHARE"] == pytest.approx(95 / 105)
    assert aaa_game_2["ROLLING_5_PLAYER_MIN_WEIGHTED_FANTASY_PER_36"] == pytest.approx(
        105 * 36 / 120
    )
    assert aaa_game_2["ROLLING_5_PLAYER_MIN_WEIGHTED_PLUS_MINUS_PER_36"] == pytest.approx(
        20 * 36 / 120
    )
    assert "DIFF_ROLLING_5_PLAYER_TOP3_FANTASY_SHARE" in model_games.columns
