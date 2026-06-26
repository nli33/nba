"""Build canonical game rows from team game logs."""

from __future__ import annotations

import pandas as pd


def add_matchup_sides(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    is_away_matchup = data["MATCHUP"].str.contains(" @ ", regex=False)
    is_home_matchup = data["MATCHUP"].str.contains(" vs. ", regex=False)
    if (is_away_matchup == is_home_matchup).any():
        raise ValueError("Expected every MATCHUP to contain exactly one of ' @ ' or ' vs. '")

    at_matchups = data["MATCHUP"].str.split(" @ ", expand=True)
    vs_matchups = data["MATCHUP"].str.split(" vs. ", expand=True)

    data["AWAY_TEAM_ABBREVIATION_FROM_MATCHUP"] = at_matchups[0].where(
        is_away_matchup, vs_matchups[1]
    )
    data["HOME_TEAM_ABBREVIATION_FROM_MATCHUP"] = at_matchups[1].where(
        is_away_matchup, vs_matchups[0]
    )
    data["IS_HOME"] = (
        data["TEAM_ABBREVIATION"] == data["HOME_TEAM_ABBREVIATION_FROM_MATCHUP"]
    )
    data["IS_AWAY"] = (
        data["TEAM_ABBREVIATION"] == data["AWAY_TEAM_ABBREVIATION_FROM_MATCHUP"]
    )
    if (data["IS_HOME"] == data["IS_AWAY"]).any():
        raise ValueError("Expected every team row to match exactly one matchup side")

    return data


def build_games(team_game_logs: pd.DataFrame) -> pd.DataFrame:
    data = add_matchup_sides(team_game_logs)

    keys = ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE"]
    home = data.loc[data["IS_HOME"], keys + ["TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]]
    away = data.loc[data["IS_AWAY"], keys + ["TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]]

    games = home.rename(
        columns={
            "TEAM_ID": "HOME_TEAM_ID",
            "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION",
            "PTS": "HOME_PTS",
            "WL": "HOME_WL",
        }
    ).merge(
        away.rename(
            columns={
                "TEAM_ID": "AWAY_TEAM_ID",
                "TEAM_ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
                "PTS": "AWAY_PTS",
                "WL": "AWAY_WL",
            }
        ),
        on=keys,
        validate="one_to_one",
    )

    games["HOME_WIN"] = (games["HOME_WL"] == "W").astype(int)
    return games[
        [
            "GAME_ID",
            "GAME_DATE",
            "SEASON",
            "SEASON_TYPE",
            "HOME_TEAM_ID",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ID",
            "AWAY_TEAM_ABBREVIATION",
            "HOME_PTS",
            "AWAY_PTS",
            "HOME_WIN",
        ]
    ].sort_values(["GAME_DATE", "GAME_ID"], ignore_index=True)
