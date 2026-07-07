"""Build team-level pre-game player rotation features."""

from __future__ import annotations

import pandas as pd


ROLLING_PLAYER_WINDOWS = (5, 10)
PLAYER_NUMERIC_COLUMNS = [
    "MIN",
    "TOV",
    "FGA",
    "FTA",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
]
# Extra box columns needed only for Game Score (availability valuation), coerced
# separately so they don't force every player-log caller to supply them.
GAME_SCORE_INPUT_COLUMNS = ["PTS", "FGM", "FTM", "OREB", "DREB", "AST", "STL", "BLK", "PF"]
# Games below this league-wide Game Score percentile are treated as replacement level
# for value-over-replacement; computed from prior games only (expanding), so early
# season/small samples fall back to 0 via REPLACEMENT_MIN_PERIODS below.
REPLACEMENT_PERCENTILE = 0.1
REPLACEMENT_MIN_PERIODS = 20
PLAYER_GAME_METRIC_COLUMNS = [
    "PLAYER_ROTATION_PLAYERS_10_MIN",
    "PLAYER_TOP3_MIN_SHARE",
    "PLAYER_TOP5_MIN_SHARE",
    "PLAYER_TOP3_FANTASY_SHARE",
    "PLAYER_MIN_WEIGHTED_FANTASY_PER_36",
    "PLAYER_MIN_WEIGHTED_PLUS_MINUS_PER_36",
    "PLAYER_MIN_WEIGHTED_USAGE_PROXY_PER_36",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0)


def aligned_series(values: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(values.to_numpy(), index=index)


def top_group_sum(data: pd.DataFrame, column: str, count: int) -> pd.Series:
    group_columns = ["GAME_ID", "TEAM_ID"]
    ranked = data.sort_values([*group_columns, column], ascending=[True, True, False])
    return (
        ranked.loc[ranked.groupby(group_columns).cumcount() < count]
        .groupby(group_columns)[column]
        .sum()
    )


def prepare_player_game_logs(player_game_logs: pd.DataFrame) -> pd.DataFrame:
    data = player_game_logs.copy()
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    for column in PLAYER_NUMERIC_COLUMNS:
        data[column] = (
            pd.to_numeric(data[column], errors="coerce").fillna(0.0).astype("float64")
        )
    return data


def add_player_game_metrics(player_game_logs: pd.DataFrame) -> pd.DataFrame:
    data = prepare_player_game_logs(player_game_logs)
    data["USAGE_PROXY"] = data["FGA"] + (0.44 * data["FTA"]) + data["TOV"]
    data["FANTASY_PER_36"] = safe_divide(
        data["NBA_FANTASY_PTS"] * 36,
        data["MIN"],
    )
    data["PLUS_MINUS_PER_36"] = safe_divide(data["PLUS_MINUS"] * 36, data["MIN"])
    data["USAGE_PROXY_PER_36"] = safe_divide(data["USAGE_PROXY"] * 36, data["MIN"])

    grouped = data.groupby(["GAME_ID", "TEAM_ID"], group_keys=False)
    totals = grouped[["MIN", "NBA_FANTASY_PTS"]].sum().rename(
        columns={"MIN": "TEAM_MIN", "NBA_FANTASY_PTS": "TEAM_FANTASY_PTS"}
    )
    metrics = totals.reset_index()
    rotation_counts = (data["MIN"] >= 10).groupby(
        [data["GAME_ID"], data["TEAM_ID"]]
    ).sum()
    metrics["PLAYER_ROTATION_PLAYERS_10_MIN"] = rotation_counts.to_numpy()
    metrics["PLAYER_TOP3_MIN_SHARE"] = safe_divide(
        aligned_series(top_group_sum(data, "MIN", 3), metrics.index),
        metrics["TEAM_MIN"],
    )
    metrics["PLAYER_TOP5_MIN_SHARE"] = safe_divide(
        aligned_series(top_group_sum(data, "MIN", 5), metrics.index),
        metrics["TEAM_MIN"],
    )
    metrics["PLAYER_TOP3_FANTASY_SHARE"] = safe_divide(
        aligned_series(top_group_sum(data, "NBA_FANTASY_PTS", 3), metrics.index),
        metrics["TEAM_FANTASY_PTS"],
    )

    for column in [
        "FANTASY_PER_36",
        "PLUS_MINUS_PER_36",
        "USAGE_PROXY_PER_36",
    ]:
        weighted = (data[column] * data["MIN"]).groupby(
            [data["GAME_ID"], data["TEAM_ID"]]
        ).sum()
        metrics[f"PLAYER_MIN_WEIGHTED_{column}"] = safe_divide(
            aligned_series(weighted, metrics.index),
            metrics["TEAM_MIN"],
        )

    return metrics[["GAME_ID", "TEAM_ID", *PLAYER_GAME_METRIC_COLUMNS]]


AVAILABILITY_FEATURE_COLUMNS = [
    "INACTIVE_COUNT",
    "INACTIVE_PRIOR_MIN",
    "INACTIVE_PRIOR_FANTASY",
    "INACTIVE_PRIOR_GAMESCORE",
    "INACTIVE_PRIOR_PLUS_MINUS",
    "INACTIVE_RECENT_GAMESCORE",
    "INACTIVE_VALUE_OVER_REPLACEMENT",
    "INACTIVE_MAX_VALUE_OUT",
]
AVAILABILITY_VALUE_COLUMNS = [
    "CUM_AVG_MIN",
    "CUM_AVG_FANTASY",
    "CUM_AVG_GAMESCORE",
    "CUM_AVG_PLUS_MINUS",
    "RECENT10_AVG_GAMESCORE",
]


def build_player_availability_features(
    player_game_logs: pd.DataFrame,
    games: pd.DataFrame,
    inactives: pd.DataFrame,
) -> pd.DataFrame:
    """Per team-game strength lost to players declared inactive pre-tip.

    Each inactive player is valued by season-to-date (and last-10-game) averages over
    games played strictly before this game (no leakage): minutes, fantasy points,
    Hollinger Game Score, and plus-minus. Value-over-replacement floors each player's
    Game Score at a league-wide running replacement level (also computed from prior
    games only) before summing, so bench absences don't inflate the signal the way raw
    minutes/fantasy sums do. Teams with no inactives get zeros (full strength).
    """
    logs = prepare_player_game_logs(player_game_logs)
    logs["GAME_ID"] = logs["GAME_ID"].astype(str)
    for column in GAME_SCORE_INPUT_COLUMNS:
        logs[column] = pd.to_numeric(logs[column], errors="coerce").fillna(0.0).astype("float64")
    logs["GAME_SCORE"] = (
        logs["PTS"]
        + 0.4 * logs["FGM"]
        - 0.7 * logs["FGA"]
        - 0.4 * (logs["FTA"] - logs["FTM"])
        + 0.7 * logs["OREB"]
        + 0.3 * logs["DREB"]
        + logs["STL"]
        + 0.7 * logs["AST"]
        + 0.7 * logs["BLK"]
        - 0.4 * logs["PF"]
        - logs["TOV"]
    )

    league_sorted = logs.sort_values(["GAME_DATE", "GAME_ID"], ignore_index=True)
    league_sorted["RUNNING_REPLACEMENT_GAMESCORE"] = (
        league_sorted["GAME_SCORE"]
        .expanding(min_periods=REPLACEMENT_MIN_PERIODS)
        .quantile(REPLACEMENT_PERCENTILE)
    )
    daily_replacement = (
        league_sorted.groupby("GAME_DATE")["RUNNING_REPLACEMENT_GAMESCORE"]
        .last()
        .reset_index()
        .sort_values("GAME_DATE", ignore_index=True)
    )

    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"], ignore_index=True)
    by_player = logs.groupby("PLAYER_ID", group_keys=False)
    logs["CUM_AVG_MIN"] = by_player["MIN"].transform(lambda values: values.expanding().mean())
    logs["CUM_AVG_FANTASY"] = by_player["NBA_FANTASY_PTS"].transform(
        lambda values: values.expanding().mean()
    )
    logs["CUM_AVG_GAMESCORE"] = by_player["GAME_SCORE"].transform(
        lambda values: values.expanding().mean()
    )
    logs["CUM_AVG_PLUS_MINUS"] = by_player["PLUS_MINUS"].transform(
        lambda values: values.expanding().mean()
    )
    logs["RECENT10_AVG_GAMESCORE"] = by_player["GAME_SCORE"].transform(
        lambda values: values.rolling(10, min_periods=1).mean()
    )
    player_history = logs[
        ["PLAYER_ID", "GAME_DATE", *AVAILABILITY_VALUE_COLUMNS]
    ].sort_values("GAME_DATE", ignore_index=True)

    game_dates = games[["GAME_ID", "GAME_DATE"]].drop_duplicates().copy()
    game_dates["GAME_ID"] = game_dates["GAME_ID"].astype(str)
    game_dates["GAME_DATE"] = pd.to_datetime(game_dates["GAME_DATE"])

    inactive = inactives.copy()
    inactive["GAME_ID"] = inactive["GAME_ID"].astype(str)
    inactive = inactive.merge(game_dates, on="GAME_ID", how="inner").sort_values(
        "GAME_DATE", ignore_index=True
    )
    valued = pd.merge_asof(
        inactive,
        player_history,
        on="GAME_DATE",
        by="PLAYER_ID",
        direction="backward",
        allow_exact_matches=False,
    )
    valued = pd.merge_asof(
        valued,
        daily_replacement,
        on="GAME_DATE",
        direction="backward",
        allow_exact_matches=False,
    )
    valued[[*AVAILABILITY_VALUE_COLUMNS, "RUNNING_REPLACEMENT_GAMESCORE"]] = valued[
        [*AVAILABILITY_VALUE_COLUMNS, "RUNNING_REPLACEMENT_GAMESCORE"]
    ].fillna(0.0)
    valued["VALUE_OVER_REPLACEMENT"] = (
        valued["CUM_AVG_GAMESCORE"] - valued["RUNNING_REPLACEMENT_GAMESCORE"]
    ).clip(lower=0.0)

    aggregated = (
        valued.groupby(["GAME_ID", "TEAM_ID"])
        .agg(
            INACTIVE_COUNT=("PLAYER_ID", "size"),
            INACTIVE_PRIOR_MIN=("CUM_AVG_MIN", "sum"),
            INACTIVE_PRIOR_FANTASY=("CUM_AVG_FANTASY", "sum"),
            INACTIVE_PRIOR_GAMESCORE=("CUM_AVG_GAMESCORE", "sum"),
            INACTIVE_PRIOR_PLUS_MINUS=("CUM_AVG_PLUS_MINUS", "sum"),
            INACTIVE_RECENT_GAMESCORE=("RECENT10_AVG_GAMESCORE", "sum"),
            INACTIVE_VALUE_OVER_REPLACEMENT=("VALUE_OVER_REPLACEMENT", "sum"),
            INACTIVE_MAX_VALUE_OUT=("VALUE_OVER_REPLACEMENT", "max"),
        )
        .reset_index()
    )
    aggregated["GAME_ID"] = aggregated["GAME_ID"].astype(str)

    universe = logs[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    features = universe.merge(aggregated, on=["GAME_ID", "TEAM_ID"], how="left")
    for column in AVAILABILITY_FEATURE_COLUMNS:
        features[column] = features[column].fillna(0.0)
    return features[["GAME_ID", "TEAM_ID", *AVAILABILITY_FEATURE_COLUMNS]]


def build_player_team_features(
    player_game_logs: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    game_metrics = add_player_game_metrics(player_game_logs)
    game_dates = games[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    data = game_metrics.merge(game_dates, on="GAME_ID", validate="many_to_one")
    data = data.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"], ignore_index=True)

    by_team = data.groupby("TEAM_ID", group_keys=False)
    feature_columns = ["GAME_ID", "TEAM_ID"]
    for window in ROLLING_PLAYER_WINDOWS:
        rolling_values = by_team[PLAYER_GAME_METRIC_COLUMNS].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        for column in PLAYER_GAME_METRIC_COLUMNS:
            feature = f"ROLLING_{window}_{column}"
            data[feature] = rolling_values[column]
            feature_columns.append(feature)

    return data[feature_columns]
