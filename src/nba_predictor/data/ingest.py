"""Command entry point for ingesting NBA data."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguedashplayerstats,
    leaguedashteamstats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SEASON_TYPES = (
    ("regular_season", "Regular Season"),
    ("playoffs", "Playoffs"),
)
MEASURES = ("Base", "Advanced")


def fetch_team_game_logs(season: str, season_type: str) -> pd.DataFrame:
    return leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season,
        season_type_nullable=season_type,
    ).get_data_frames()[0]


def fetch_team_stats(season: str, season_type: str, measure: str) -> pd.DataFrame:
    return leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_player_stats(season: str, season_type: str, measure: str) -> pd.DataFrame:
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense=measure,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def write_if_needed(
    path: Path,
    fetch: Callable[[], pd.DataFrame],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        print(f"Skipping existing {path}")
        return

    print(f"Fetching {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch()
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    data.to_parquet(temp_path, index=False)
    temp_path.replace(path)
    print(f"Wrote {len(data):,} rows to {path}")


def ingest_season(season: str, overwrite: bool) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    season_dir = RAW_DATA_DIR / season

    for file_season_type, api_season_type in SEASON_TYPES:
        write_if_needed(
            season_dir / "team_game_logs" / f"{file_season_type}.parquet",
            partial(fetch_team_game_logs, season, api_season_type),
            overwrite,
        )

        for measure in MEASURES:
            file_measure = measure.lower()
            write_if_needed(
                season_dir
                / "team_stats"
                / file_measure
                / f"{file_season_type}.parquet",
                partial(fetch_team_stats, season, api_season_type, measure),
                overwrite,
            )
            write_if_needed(
                season_dir
                / "player_stats"
                / file_measure
                / f"{file_season_type}.parquet",
                partial(fetch_player_stats, season, api_season_type, measure),
                overwrite,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NBA API data for one season.")
    parser.add_argument("season", help='NBA season, for example "2024-25".')
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet files instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the NBA data ingestion pipeline."""
    args = parse_args()
    ingest_season(args.season, args.overwrite)
