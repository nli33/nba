"""Build team-level pre-game features."""

from __future__ import annotations

import pandas as pd

from nba_predictor.data.elo import build_elo_features
from nba_predictor.data.games import add_matchup_sides


ROLLING_WINDOWS = (5, 10)


def add_estimated_possessions(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["EST_POSSESSIONS"] = data["FGA"] + (0.44 * data["FTA"]) - data["OREB"] + data["TOV"]
    data["OPP_EST_POSSESSIONS"] = (
        data.groupby("GAME_ID")["EST_POSSESSIONS"].transform("sum")
        - data["EST_POSSESSIONS"]
    )
    return data


def net_rating(
    points_for: pd.Series,
    possessions_for: pd.Series,
    points_against: pd.Series,
    possessions_against: pd.Series,
) -> pd.Series:
    return offensive_rating(points_for, possessions_for) - defensive_rating(
        points_against, possessions_against
    )


def offensive_rating(points_for: pd.Series, possessions_for: pd.Series) -> pd.Series:
    return 100 * points_for / possessions_for.where(possessions_for != 0)


def defensive_rating(points_against: pd.Series, possessions_against: pd.Series) -> pd.Series:
    return 100 * points_against / possessions_against.where(possessions_against != 0)


def build_team_game_features(
    team_game_logs: pd.DataFrame,
    games: pd.DataFrame,
    previous_games: pd.DataFrame | None,
) -> pd.DataFrame:
    data = add_estimated_possessions(add_matchup_sides(team_game_logs))
    data["WIN"] = (data["WL"] == "W").astype(int)
    data["POINT_DIFF"] = data["PLUS_MINUS"]
    data["PTS_AGAINST"] = data["PTS"] - data["PLUS_MINUS"]
    data = data.merge(
        build_elo_features(games, previous_games),
        on=["GAME_ID", "TEAM_ID"],
        validate="one_to_one",
    )
    data = data.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"], ignore_index=True)

    by_team = data.groupby("TEAM_ID", group_keys=False)
    data["DAYS_REST"] = by_team["GAME_DATE"].diff().dt.days
    data["IS_BACK_TO_BACK"] = data["DAYS_REST"].eq(1).astype(int)
    data["IS_3_IN_4"] = (
        (data["GAME_DATE"] - by_team["GAME_DATE"].shift(2)).dt.days <= 3
    ).astype(int)

    rolling_sources = {
        "WIN_PCT": "WIN",
        "PTS_FOR": "PTS",
        "PTS_AGAINST": "PTS_AGAINST",
        "POINT_DIFF": "POINT_DIFF",
    }
    for window in ROLLING_WINDOWS:
        for feature_name, source_column in rolling_sources.items():
            data[f"ROLLING_{window}_{feature_name}"] = by_team[
                source_column
            ].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )

    data["SEASON_TO_DATE_WIN_PCT"] = by_team["WIN"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    data["SEASON_TO_DATE_POINT_DIFF"] = by_team["POINT_DIFF"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    season_to_date_sums = by_team[
        ["PTS", "EST_POSSESSIONS", "PTS_AGAINST", "OPP_EST_POSSESSIONS"]
    ].transform(lambda values: values.shift(1).expanding().sum())
    data["SEASON_TO_DATE_NET_RATING"] = net_rating(
        season_to_date_sums["PTS"],
        season_to_date_sums["EST_POSSESSIONS"],
        season_to_date_sums["PTS_AGAINST"],
        season_to_date_sums["OPP_EST_POSSESSIONS"],
    )
    data["SEASON_TO_DATE_OFF_RATING"] = offensive_rating(
        season_to_date_sums["PTS"],
        season_to_date_sums["EST_POSSESSIONS"],
    )
    data["SEASON_TO_DATE_DEF_RATING"] = defensive_rating(
        season_to_date_sums["PTS_AGAINST"],
        season_to_date_sums["OPP_EST_POSSESSIONS"],
    )

    columns = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "IS_HOME",
        "DAYS_REST",
        "IS_BACK_TO_BACK",
        "IS_3_IN_4",
        "ELO_FLAT_PRE_GAME",
        "ELO_CARRYOVER_PRE_GAME",
        "STRENGTH_OF_SCHEDULE_ELO_TO_DATE",
        "SEASON_TO_DATE_WIN_PCT",
        "SEASON_TO_DATE_POINT_DIFF",
        "SEASON_TO_DATE_NET_RATING",
        "SEASON_TO_DATE_OFF_RATING",
        "SEASON_TO_DATE_DEF_RATING",
    ]
    for window in ROLLING_WINDOWS:
        rolling_sums = by_team[
            ["PTS", "EST_POSSESSIONS", "PTS_AGAINST", "OPP_EST_POSSESSIONS"]
        ].transform(lambda values: values.shift(1).rolling(window, min_periods=1).sum())
        data[f"ROLLING_{window}_NET_RATING"] = net_rating(
            rolling_sums["PTS"],
            rolling_sums["EST_POSSESSIONS"],
            rolling_sums["PTS_AGAINST"],
            rolling_sums["OPP_EST_POSSESSIONS"],
        )
        data[f"ROLLING_{window}_OFF_RATING"] = offensive_rating(
            rolling_sums["PTS"],
            rolling_sums["EST_POSSESSIONS"],
        )
        data[f"ROLLING_{window}_DEF_RATING"] = defensive_rating(
            rolling_sums["PTS_AGAINST"],
            rolling_sums["OPP_EST_POSSESSIONS"],
        )
        columns.extend(
            [
                f"ROLLING_{window}_WIN_PCT",
                f"ROLLING_{window}_PTS_FOR",
                f"ROLLING_{window}_PTS_AGAINST",
                f"ROLLING_{window}_POINT_DIFF",
                f"ROLLING_{window}_NET_RATING",
                f"ROLLING_{window}_OFF_RATING",
                f"ROLLING_{window}_DEF_RATING",
            ]
        )

    return data[columns].sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"], ignore_index=True)
