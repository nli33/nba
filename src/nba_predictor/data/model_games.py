"""Build model-ready game rows from home and away team features."""

from __future__ import annotations

import pandas as pd


def build_model_games(games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in team_features.columns
        if column
        not in {
            "GAME_DATE",
            "SEASON",
            "SEASON_TYPE",
            "TEAM_ABBREVIATION",
            "IS_HOME",
        }
    ]

    home_features = team_features.loc[team_features["IS_HOME"], feature_columns].rename(
        columns={column: f"HOME_{column}" for column in feature_columns if column != "GAME_ID"}
    )
    away_features = team_features.loc[~team_features["IS_HOME"], feature_columns].rename(
        columns={column: f"AWAY_{column}" for column in feature_columns if column != "GAME_ID"}
    )

    model_games = games[
        [
            "GAME_ID",
            "GAME_DATE",
            "SEASON",
            "SEASON_TYPE",
            "HOME_TEAM_ID",
            "HOME_TEAM_ABBREVIATION",
            "AWAY_TEAM_ID",
            "AWAY_TEAM_ABBREVIATION",
            "HOME_WIN",
        ]
    ].merge(
        home_features,
        left_on=["GAME_ID", "HOME_TEAM_ID"],
        right_on=["GAME_ID", "HOME_TEAM_ID"],
        validate="one_to_one",
    ).merge(
        away_features,
        left_on=["GAME_ID", "AWAY_TEAM_ID"],
        right_on=["GAME_ID", "AWAY_TEAM_ID"],
        validate="one_to_one",
    )

    for column in home_features.columns:
        if not column.startswith("HOME_") or column == "HOME_TEAM_ID":
            continue

        away_column = column.replace("HOME_", "AWAY_", 1)
        if away_column in model_games.columns:
            model_games[column.replace("HOME_", "DIFF_", 1)] = (
                model_games[column] - model_games[away_column]
            )

    return model_games.sort_values(["GAME_DATE", "GAME_ID"], ignore_index=True)
