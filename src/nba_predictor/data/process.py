"""Command entry point for processing ingested NBA data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from nba_predictor.data.elo import (
    ELO_CARRYOVER,
    ELO_HOME_ADVANTAGE,
    ELO_K,
    START_ELO,
    as_int,
    build_elo_features,
    carryover_elos,
    expected_home_win_probability,
    final_flat_elos,
    updated_elos,
)
from nba_predictor.data.games import add_matchup_sides, build_games
from nba_predictor.data.model_games import build_model_games
from nba_predictor.data.player_features import build_player_team_features
from nba_predictor.data.team_features import (
    ROLLING_WINDOWS,
    add_estimated_possessions,
    build_team_game_features,
    defensive_rating,
    net_rating,
    offensive_rating,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEASON_TYPES = ("regular_season", "playoffs")

__all__ = [
    "ELO_CARRYOVER",
    "ELO_HOME_ADVANTAGE",
    "ELO_K",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "ROLLING_WINDOWS",
    "SEASON_TYPES",
    "START_ELO",
    "add_estimated_possessions",
    "add_matchup_sides",
    "as_int",
    "build_elo_features",
    "build_games",
    "build_model_games",
    "build_player_team_features",
    "build_team_game_features",
    "carryover_elos",
    "defensive_rating",
    "expected_home_win_probability",
    "final_flat_elos",
    "load_previous_games",
    "load_player_game_logs",
    "load_team_game_logs",
    "main",
    "net_rating",
    "offensive_rating",
    "parse_args",
    "previous_season",
    "process_season",
    "updated_elos",
    "write_or_load",
]


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


def load_player_game_logs(season: str) -> pd.DataFrame | None:
    frames = []

    for season_type in SEASON_TYPES:
        path = RAW_DATA_DIR / season / "player_game_logs" / f"{season_type}.parquet"
        if not path.exists():
            print(f"Missing raw input, skipping {path}", file=sys.stderr)
            continue

        data = pd.read_parquet(path)
        data["SEASON"] = season
        data["SEASON_TYPE"] = season_type
        frames.append(data)

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    return data


def previous_season(season: str) -> str:
    start_year = int(season[:4])
    previous_start = start_year - 1
    previous_end = str(start_year)[-2:]
    return f"{previous_start}-{previous_end}"


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
    player_game_logs = load_player_game_logs(season)
    previous_games = load_previous_games(season)
    season_dir = PROCESSED_DATA_DIR / season

    games = write_or_load(
        season_dir / "games.parquet",
        lambda: build_games(team_game_logs),
        overwrite,
    )
    team_features = write_or_load(
        season_dir / "team_game_features.parquet",
        lambda: build_team_game_features(
            team_game_logs,
            games,
            previous_games,
            player_game_logs,
        ),
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
