"""Command entry point for processing ingested NBA data."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEASON_TYPES = ("regular_season", "playoffs")
ROLLING_WINDOWS = (5, 10)
START_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0
ELO_CARRYOVER = 0.75


def as_int(value: Any) -> int:
    return int(value)


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
    return offensive_rating(points_for, possessions_for) - defensive_rating(
        points_against, possessions_against
    )


def offensive_rating(points_for: pd.Series, possessions_for: pd.Series) -> pd.Series:
    return 100 * points_for / possessions_for.where(possessions_for != 0)


def defensive_rating(points_against: pd.Series, possessions_against: pd.Series) -> pd.Series:
    return 100 * points_against / possessions_against.where(possessions_against != 0)


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


def previous_season(season: str) -> str:
    start_year = int(season[:4])
    previous_start = start_year - 1
    previous_end = str(start_year)[-2:]
    return f"{previous_start}-{previous_end}"


def expected_home_win_probability(home_elo: float, away_elo: float) -> float:
    rating_gap = (home_elo + ELO_HOME_ADVANTAGE) - away_elo
    return 1 / (1 + 10 ** (-rating_gap / 400))


def updated_elos(home_elo: float, away_elo: float, home_win: int) -> tuple[float, float]:
    expected_home_win = expected_home_win_probability(home_elo, away_elo)
    delta = ELO_K * (home_win - expected_home_win)
    return home_elo + delta, away_elo - delta


def final_flat_elos(games: pd.DataFrame) -> dict[int, float]:
    elos: defaultdict[int, float] = defaultdict(lambda: START_ELO)
    for game in games.itertuples(index=False):
        home_team_id = as_int(game.HOME_TEAM_ID)
        away_team_id = as_int(game.AWAY_TEAM_ID)
        elos[home_team_id], elos[away_team_id] = updated_elos(
            elos[home_team_id],
            elos[away_team_id],
            as_int(game.HOME_WIN),
        )
    return dict(elos)


def carryover_elos(previous_games: pd.DataFrame | None) -> dict[int, float]:
    if previous_games is None:
        return {}

    return {
        team_id: START_ELO + (ELO_CARRYOVER * (elo - START_ELO))
        for team_id, elo in final_flat_elos(previous_games).items()
    }


def build_elo_features(
    games: pd.DataFrame,
    previous_games: pd.DataFrame | None,
) -> pd.DataFrame:
    flat_elos: defaultdict[int, float] = defaultdict(lambda: START_ELO)
    carry_elos: defaultdict[int, float] = defaultdict(lambda: START_ELO)
    carry_elos.update(carryover_elos(previous_games))

    sos_sums: defaultdict[int, float] = defaultdict(float)
    sos_counts: defaultdict[int, int] = defaultdict(int)
    rows = []
    for game in games.sort_values(["GAME_DATE", "GAME_ID"]).itertuples(index=False):
        home_team_id = as_int(game.HOME_TEAM_ID)
        away_team_id = as_int(game.AWAY_TEAM_ID)
        home_flat_elo = flat_elos[home_team_id]
        away_flat_elo = flat_elos[away_team_id]
        home_carry_elo = carry_elos[home_team_id]
        away_carry_elo = carry_elos[away_team_id]
        home_sos = (
            sos_sums[home_team_id] / sos_counts[home_team_id]
            if sos_counts[home_team_id]
            else float("nan")
        )
        away_sos = (
            sos_sums[away_team_id] / sos_counts[away_team_id]
            if sos_counts[away_team_id]
            else float("nan")
        )

        rows.extend(
            [
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": home_team_id,
                    "ELO_FLAT_PRE_GAME": home_flat_elo,
                    "ELO_CARRYOVER_PRE_GAME": home_carry_elo,
                    "STRENGTH_OF_SCHEDULE_ELO_TO_DATE": home_sos,
                },
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": away_team_id,
                    "ELO_FLAT_PRE_GAME": away_flat_elo,
                    "ELO_CARRYOVER_PRE_GAME": away_carry_elo,
                    "STRENGTH_OF_SCHEDULE_ELO_TO_DATE": away_sos,
                },
            ]
        )

        sos_sums[home_team_id] += away_flat_elo
        sos_counts[home_team_id] += 1
        sos_sums[away_team_id] += home_flat_elo
        sos_counts[away_team_id] += 1

        flat_elos[home_team_id], flat_elos[away_team_id] = updated_elos(
            home_flat_elo,
            away_flat_elo,
            as_int(game.HOME_WIN),
        )
        carry_elos[home_team_id], carry_elos[away_team_id] = updated_elos(
            home_carry_elo,
            away_carry_elo,
            as_int(game.HOME_WIN),
        )

    return pd.DataFrame(rows)


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


def load_previous_games(season: str) -> pd.DataFrame | None:
    previous = previous_season(season)
    season_dir = RAW_DATA_DIR / previous / "team_game_logs"
    if not season_dir.exists():
        return None

    return build_games(load_team_game_logs(previous))


def process_season(season: str, overwrite: bool) -> None:
    team_game_logs = load_team_game_logs(season)
    previous_games = load_previous_games(season)
    season_dir = PROCESSED_DATA_DIR / season

    games = write_or_load(
        season_dir / "games.parquet",
        lambda: build_games(team_game_logs),
        overwrite,
    )
    team_features = write_or_load(
        season_dir / "team_game_features.parquet",
        lambda: build_team_game_features(team_game_logs, games, previous_games),
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
