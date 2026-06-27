"""Build team-level pre-game features."""

from __future__ import annotations

import pandas as pd

from nba_predictor.data.elo import build_elo_features
from nba_predictor.data.games import add_matchup_sides


ROLLING_WINDOWS = (5, 10)
BOX_SCORE_SUM_COLUMNS = [
    "PTS",
    "EST_POSSESSIONS",
    "PTS_AGAINST",
    "OPP_EST_POSSESSIONS",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTA",
    "OREB",
    "DREB",
    "AST",
    "TOV",
    "OPP_FGM",
    "OPP_FGA",
    "OPP_FG3M",
    "OPP_FG3A",
    "OPP_FTA",
    "OPP_OREB",
    "OPP_DREB",
    "OPP_AST",
    "OPP_TOV",
]
OPPONENT_BOX_SCORE_COLUMNS = [
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTA",
    "OREB",
    "DREB",
    "AST",
    "TOV",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0)


def add_estimated_possessions(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["EST_POSSESSIONS"] = (
        data["FGA"] + (0.44 * data["FTA"]) - data["OREB"] + data["TOV"]
    )
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


def defensive_rating(
    points_against: pd.Series, possessions_against: pd.Series
) -> pd.Series:
    return 100 * points_against / possessions_against.where(possessions_against != 0)


def add_opponent_box_score_stats(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    opponent_totals = data.groupby("GAME_ID")[OPPONENT_BOX_SCORE_COLUMNS].transform(
        "sum"
    )
    for column in OPPONENT_BOX_SCORE_COLUMNS:
        data[f"OPP_{column}"] = opponent_totals[column] - data[column]
    return data


def add_rate_features(data: pd.DataFrame, prefix: str, source: pd.DataFrame) -> None:
    data[f"{prefix}_OFF_EFG_PCT"] = safe_divide(
        source["FGM"] + (0.5 * source["FG3M"]),
        source["FGA"],
    )
    data[f"{prefix}_DEF_EFG_PCT"] = safe_divide(
        source["OPP_FGM"] + (0.5 * source["OPP_FG3M"]),
        source["OPP_FGA"],
    )
    data[f"{prefix}_OFF_TOV_RATE"] = safe_divide(
        source["TOV"], source["EST_POSSESSIONS"]
    )
    data[f"{prefix}_DEF_TOV_RATE"] = safe_divide(
        source["OPP_TOV"],
        source["OPP_EST_POSSESSIONS"],
    )
    data[f"{prefix}_OFF_OREB_RATE"] = safe_divide(
        source["OREB"],
        source["OREB"] + source["OPP_DREB"],
    )
    data[f"{prefix}_DEF_OREB_RATE"] = safe_divide(
        source["OPP_OREB"],
        source["OPP_OREB"] + source["DREB"],
    )
    data[f"{prefix}_OFF_FT_RATE"] = safe_divide(source["FTA"], source["FGA"])
    data[f"{prefix}_DEF_FT_RATE"] = safe_divide(source["OPP_FTA"], source["OPP_FGA"])
    data[f"{prefix}_PACE"] = safe_divide(
        source["EST_POSSESSIONS"] + source["OPP_EST_POSSESSIONS"],
        pd.Series(2.0, index=source.index),
    )
    data[f"{prefix}_FG3A_RATE"] = safe_divide(source["FG3A"], source["FGA"])
    data[f"{prefix}_OPP_FG3A_RATE"] = safe_divide(source["OPP_FG3A"], source["OPP_FGA"])
    data[f"{prefix}_AST_RATE"] = safe_divide(source["AST"], source["FGM"])
    data[f"{prefix}_OPP_AST_RATE"] = safe_divide(source["OPP_AST"], source["OPP_FGM"])


def build_team_game_features(
    team_game_logs: pd.DataFrame,
    games: pd.DataFrame,
    previous_games: pd.DataFrame | None,
) -> pd.DataFrame:
    data = add_opponent_box_score_stats(
        add_estimated_possessions(add_matchup_sides(team_game_logs))
    )
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
    road_trip_segments = data["IS_HOME"].astype(int).groupby(data["TEAM_ID"]).cumsum()
    data["CONSECUTIVE_ROAD_GAMES"] = (
        data["IS_AWAY"]
        .astype(int)
        .groupby([data["TEAM_ID"], road_trip_segments])
        .cumsum()
    )

    rolling_sources = {
        "WIN_PCT": "WIN",
        "PTS_FOR": "PTS",
        "PTS_AGAINST": "PTS_AGAINST",
        "POINT_DIFF": "POINT_DIFF",
    }
    for window in ROLLING_WINDOWS:
        for feature_name, source_column in rolling_sources.items():
            data[f"ROLLING_{window}_{feature_name}"] = by_team[source_column].transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            )

    data["SEASON_TO_DATE_WIN_PCT"] = by_team["WIN"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    data["SEASON_TO_DATE_POINT_DIFF"] = by_team["POINT_DIFF"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    season_to_date_sums = by_team[BOX_SCORE_SUM_COLUMNS].transform(
        lambda values: values.shift(1).expanding().sum()
    )
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
    add_rate_features(data, "SEASON_TO_DATE", season_to_date_sums)

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
        "CONSECUTIVE_ROAD_GAMES",
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
        rolling_sums = by_team[BOX_SCORE_SUM_COLUMNS].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).sum()
        )
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
        add_rate_features(data, f"ROLLING_{window}", rolling_sums)
        columns.extend(
            [
                f"ROLLING_{window}_WIN_PCT",
                f"ROLLING_{window}_PTS_FOR",
                f"ROLLING_{window}_PTS_AGAINST",
                f"ROLLING_{window}_POINT_DIFF",
                f"ROLLING_{window}_NET_RATING",
                f"ROLLING_{window}_OFF_RATING",
                f"ROLLING_{window}_DEF_RATING",
                f"ROLLING_{window}_OFF_EFG_PCT",
                f"ROLLING_{window}_DEF_EFG_PCT",
                f"ROLLING_{window}_OFF_TOV_RATE",
                f"ROLLING_{window}_DEF_TOV_RATE",
                f"ROLLING_{window}_OFF_OREB_RATE",
                f"ROLLING_{window}_DEF_OREB_RATE",
                f"ROLLING_{window}_OFF_FT_RATE",
                f"ROLLING_{window}_DEF_FT_RATE",
                f"ROLLING_{window}_PACE",
                f"ROLLING_{window}_FG3A_RATE",
                f"ROLLING_{window}_OPP_FG3A_RATE",
                f"ROLLING_{window}_AST_RATE",
                f"ROLLING_{window}_OPP_AST_RATE",
            ]
        )

    columns.extend(
        [
            "SEASON_TO_DATE_OFF_EFG_PCT",
            "SEASON_TO_DATE_DEF_EFG_PCT",
            "SEASON_TO_DATE_OFF_TOV_RATE",
            "SEASON_TO_DATE_DEF_TOV_RATE",
            "SEASON_TO_DATE_OFF_OREB_RATE",
            "SEASON_TO_DATE_DEF_OREB_RATE",
            "SEASON_TO_DATE_OFF_FT_RATE",
            "SEASON_TO_DATE_DEF_FT_RATE",
            "SEASON_TO_DATE_PACE",
            "SEASON_TO_DATE_FG3A_RATE",
            "SEASON_TO_DATE_OPP_FG3A_RATE",
            "SEASON_TO_DATE_AST_RATE",
            "SEASON_TO_DATE_OPP_AST_RATE",
        ]
    )

    return data[columns].sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM_ID"], ignore_index=True
    )
