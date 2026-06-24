"""Command entry point for processing ingested NBA data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEASON_TYPES = ("regular_season", "playoffs")
ROLLING_WINDOWS = (5, 10)


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


def load_team_game_logs(season: str) -> pd.DataFrame:
    frames = []

    for season_type in SEASON_TYPES:
        path = RAW_DATA_DIR / season / "team_game_logs" / f"{season_type}.parquet"
        if not path.exists():
            print(f"Missing raw input, skipping {path}", file=sys.stderr)
            continue

        data = pd.read_parquet(path)
        data["SEASON"] = season
        data["SEASON_TYPE"] = season_type
        frames.append(data)

    if not frames:
        raise FileNotFoundError(
            f"No team game logs found under {RAW_DATA_DIR / season / 'team_game_logs'}"
        )

    data = pd.concat(frames, ignore_index=True)
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    return data


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
    offensive_rating = 100 * points_for / possessions_for.where(possessions_for != 0)
    defensive_rating = 100 * points_against / possessions_against.where(
        possessions_against != 0
    )
    return offensive_rating - defensive_rating


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


def build_team_game_features(team_game_logs: pd.DataFrame) -> pd.DataFrame:
    data = add_estimated_possessions(add_matchup_sides(team_game_logs))
    data["WIN"] = (data["WL"] == "W").astype(int)
    data["POINT_DIFF"] = data["PLUS_MINUS"]
    data["PTS_AGAINST"] = data["PTS"] - data["PLUS_MINUS"]
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
        "SEASON_TO_DATE_WIN_PCT",
        "SEASON_TO_DATE_POINT_DIFF",
        "SEASON_TO_DATE_NET_RATING",
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
        columns.extend(
            [
                f"ROLLING_{window}_WIN_PCT",
                f"ROLLING_{window}_PTS_FOR",
                f"ROLLING_{window}_PTS_AGAINST",
                f"ROLLING_{window}_POINT_DIFF",
                f"ROLLING_{window}_NET_RATING",
            ]
        )

    return data[columns].sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"], ignore_index=True)


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


def write_or_load(
    path: Path,
    build: Callable[[], pd.DataFrame],
    overwrite: bool,
) -> pd.DataFrame:
    if path.exists() and not overwrite:
        print(f"Skipping existing {path}")
        return pd.read_parquet(path)

    print(f"Processing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build()
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    data.to_parquet(temp_path, index=False)
    temp_path.replace(path)
    print(f"Wrote {len(data):,} rows to {path}")
    return data


def process_season(season: str, overwrite: bool) -> None:
    team_game_logs = load_team_game_logs(season)
    season_dir = PROCESSED_DATA_DIR / season

    games = write_or_load(
        season_dir / "games.parquet",
        lambda: build_games(team_game_logs),
        overwrite,
    )
    team_features = write_or_load(
        season_dir / "team_game_features.parquet",
        lambda: build_team_game_features(team_game_logs),
        overwrite,
    )
    write_or_load(
        season_dir / "model_games.parquet",
        lambda: build_model_games(games, team_features),
        overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process ingested NBA data for one season.")
    parser.add_argument("season", help='NBA season, for example "2024-25".')
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet files instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the NBA data processing pipeline."""
    args = parse_args()
    try:
        process_season(args.season, args.overwrite)
    except FileNotFoundError as error:
        print(f"Unable to process {args.season}: {error}", file=sys.stderr)
        raise SystemExit(1) from None
