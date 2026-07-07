"""Command entry point for ingesting NBA data."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (  # type: ignore[import-untyped]
    boxscoresummaryv2,
    boxscoresummaryv3,
    leaguegamefinder,
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelogs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SEASON_TYPES = (
    ("regular_season", "Regular Season"),
    ("playoffs", "Playoffs"),
)
MEASURES = ("Base", "Advanced")

# BoxScoreSummaryV2 has known InactivePlayers gaps for games on or after this date;
# use BoxScoreSummaryV3 for those, and V2 (which covers older seasons) before it.
INACTIVES_V2_CUTOFF = pd.Timestamp("2025-04-10")
INACTIVES_COLUMNS = ["GAME_ID", "TEAM_ID", "PLAYER_ID"]


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


def fetch_player_game_logs(season: str, season_type: str) -> pd.DataFrame:
    return playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
    ).get_data_frames()[0]


def _normalize_inactives_v2(game_id: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        # V2's known-gap warning is handled by routing recent games to V3.
        warnings.simplefilter("ignore", UserWarning)
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
    frame = summary.inactive_players.get_data_frame()
    return pd.DataFrame(
        {
            "GAME_ID": game_id,
            "TEAM_ID": frame["TEAM_ID"],
            "PLAYER_ID": frame["PLAYER_ID"],
        }
    )


def _normalize_inactives_v3(game_id: str) -> pd.DataFrame:
    frame = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id).inactive_players.get_data_frame()
    return pd.DataFrame(
        {
            "GAME_ID": game_id,
            "TEAM_ID": frame["teamId"],
            "PLAYER_ID": frame["personId"],
        }
    )


INACTIVES_REQUEST_DELAY = 0.5  # polite spacing to avoid stats.nba.com throttling


def _fetch_with_retry(
    fetch: Callable[[str], pd.DataFrame], game_id: str, attempts: int = 4
) -> pd.DataFrame:
    for attempt in range(attempts):
        try:
            return fetch(game_id)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (2**attempt))  # 2s, 4s, 8s backoff past throttle windows
    raise RuntimeError("unreachable")


def fetch_game_inactives(game_id: str, game_date: pd.Timestamp) -> pd.DataFrame:
    """Return the inactive (declared out pre-tip) players for one game.

    Chooses BoxScoreSummaryV3 for games in V2's known-gap window and V2 otherwise,
    retrying each endpoint on transient errors and falling back to the other one.
    """
    if game_date >= INACTIVES_V2_CUTOFF:
        order = (_normalize_inactives_v3, _normalize_inactives_v2)
    else:
        order = (_normalize_inactives_v2, _normalize_inactives_v3)
    last_error: Exception | None = None
    for fetch in order:
        try:
            return _fetch_with_retry(fetch, game_id)
        except Exception as error:
            last_error = error
    assert last_error is not None
    raise last_error


def ingest_inactives_season(season: str, overwrite: bool) -> None:
    """Fetch per-game inactive players for one season (resumable, one file per season type)."""
    season_dir = RAW_DATA_DIR / season

    for file_season_type, _ in SEASON_TYPES:
        source_path = season_dir / "team_game_logs" / f"{file_season_type}.parquet"
        if not source_path.exists():
            print(f"Skipping inactives, missing {source_path}", file=sys.stderr)
            continue

        out_path = season_dir / "inactives" / f"{file_season_type}.parquet"
        marker_path = out_path.with_name(f".{out_path.stem}.fetched.parquet")
        games = pd.read_parquet(source_path)[["GAME_ID", "GAME_DATE"]].drop_duplicates()
        games["GAME_ID"] = games["GAME_ID"].astype(str)
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

        rows: list[pd.DataFrame] = []
        fetched: list[str] = []
        if not overwrite and out_path.exists() and marker_path.exists():
            existing = pd.read_parquet(out_path)
            existing["GAME_ID"] = existing["GAME_ID"].astype(str)
            rows.append(existing)
            fetched = pd.read_parquet(marker_path)["GAME_ID"].astype(str).tolist()

        remaining = games[~games["GAME_ID"].isin(set(fetched))]
        if remaining.empty:
            print(f"Inactives complete, skipping {out_path}")
            continue

        print(f"Fetching inactives for {len(remaining):,} games -> {out_path}")
        failed: list[str] = []
        pairs = zip(remaining["GAME_ID"], remaining["GAME_DATE"], strict=True)
        for count, (game_id, game_date) in enumerate(pairs, start=1):
            try:
                rows.append(fetch_game_inactives(str(game_id), pd.Timestamp(game_date)))
                fetched.append(str(game_id))
            except Exception as error:
                # Leave unfetched so a later resume retries it, rather than aborting.
                failed.append(str(game_id))
                print(f"  WARN {game_id} failed: {error!r}"[:160], file=sys.stderr)
            time.sleep(INACTIVES_REQUEST_DELAY)
            if count % 100 == 0:
                print(f"  {count:,}/{len(remaining):,} games", flush=True)
                _write_inactives(out_path, marker_path, rows, fetched)
        _write_inactives(out_path, marker_path, rows, fetched)
        total = sum(len(frame) for frame in rows)
        print(f"Wrote {total:,} inactive rows for {len(fetched):,} games to {out_path}")
        if failed:
            print(f"  {len(failed)} games failed; re-run to retry them", file=sys.stderr)


def _write_inactives(
    path: Path, marker_path: Path, frames: list[pd.DataFrame], fetched: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        pd.concat(frames, ignore_index=True)[INACTIVES_COLUMNS].drop_duplicates()
        if frames
        else pd.DataFrame(columns=INACTIVES_COLUMNS)
    )
    _atomic_write_parquet(path, data)
    _atomic_write_parquet(marker_path, pd.DataFrame({"GAME_ID": sorted(set(fetched))}))


def _atomic_write_parquet(path: Path, data: pd.DataFrame) -> None:
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    data.to_parquet(temp_path, index=False)
    temp_path.replace(path)


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
        write_if_needed(
            season_dir / "player_game_logs" / f"{file_season_type}.parquet",
            partial(fetch_player_game_logs, season, api_season_type),
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


def inactives_main() -> None:
    """Fetch per-game inactive players for one season."""
    args = parse_args()
    ingest_inactives_season(args.season, args.overwrite)
