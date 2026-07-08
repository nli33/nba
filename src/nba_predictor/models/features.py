"""Feature column configuration shared by model implementations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    # Team strength and recent form.
    "DIFF_SEASON_TO_DATE_WIN_PCT",
    "DIFF_SEASON_TO_DATE_POINT_DIFF",
    "DIFF_SEASON_TO_DATE_NET_RATING",
    "DIFF_ROLLING_5_WIN_PCT",
    "DIFF_ROLLING_5_POINT_DIFF",
    "DIFF_ROLLING_5_NET_RATING",
    "DIFF_ROLLING_10_WIN_PCT",
    "DIFF_ROLLING_10_POINT_DIFF",
    "DIFF_ROLLING_10_NET_RATING",
    # Schedule / rest.
    "DIFF_DAYS_REST",
    "DIFF_IS_BACK_TO_BACK",
    "DIFF_IS_3_IN_4",
    # Cross-season strength carried over via Elo.
    "DIFF_ELO_CARRYOVER_PRE_GAME",
    # Pre-tip player availability (declared inactives). Available from 2005-06 on;
    # the strongest single lever over team-aggregate form (see docs/related-work).
    "DIFF_INACTIVE_COUNT",
    "DIFF_INACTIVE_PRIOR_MIN",
    "DIFF_INACTIVE_PRIOR_FANTASY",
    "DIFF_INACTIVE_PRIOR_GAMESCORE",
    "DIFF_INACTIVE_PRIOR_PLUS_MINUS",
    "DIFF_INACTIVE_RECENT_GAMESCORE",
    "DIFF_INACTIVE_VALUE_OVER_REPLACEMENT",
    "DIFF_INACTIVE_MAX_VALUE_OUT",
]


def read_feature_file(path: Path) -> list[str]:
    feature_columns = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            feature_columns.append(line)
    return feature_columns


def resolve_feature_columns(
    args: argparse.Namespace,
    default: list[str] = DEFAULT_FEATURE_COLUMNS,
) -> list[str]:
    if args.features and args.features_file:
        raise ValueError("Use either --features or --features-file, not both")
    if args.features:
        return args.features
    if args.features_file:
        return read_feature_file(args.features_file)
    return default


def validate_feature_columns(model_games: pd.DataFrame, feature_columns: list[str]) -> None:
    if not feature_columns:
        raise ValueError("At least one feature column is required")

    seen_columns = set()
    duplicate_columns = set()
    for column in feature_columns:
        if column in seen_columns:
            duplicate_columns.add(column)
        else:
            seen_columns.add(column)
    if duplicate_columns:
        columns = ", ".join(sorted(duplicate_columns))
        raise ValueError(f"Duplicate feature columns: {columns}")

    missing_columns = sorted(set(feature_columns) - set(model_games.columns))
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"Feature columns not found in processed data: {columns}")
