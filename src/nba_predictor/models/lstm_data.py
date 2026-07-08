"""Sequence data structures and builders for the LSTM game predictor.

Pure numpy/pandas (no torch): turns per-team game-feature rows into padded,
scaled prior-game windows the LSTM consumes. The sequence for game ``t`` always
ends at ``t-1`` (leakage-safe, mirroring the rest of the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nba_predictor.prediction import PROCESSED_DATA_DIR


# Team-centric counterparts of the default DIFF_* feature set, plus a per-timestep
# home/away indicator so the model knows the context of each past game.
DEFAULT_SEQUENCE_FEATURES = [
    "SEASON_TO_DATE_WIN_PCT",
    "SEASON_TO_DATE_POINT_DIFF",
    "SEASON_TO_DATE_NET_RATING",
    "ROLLING_5_WIN_PCT",
    "ROLLING_5_POINT_DIFF",
    "ROLLING_5_NET_RATING",
    "ROLLING_10_WIN_PCT",
    "ROLLING_10_POINT_DIFF",
    "ROLLING_10_NET_RATING",
    "DAYS_REST",
    "IS_BACK_TO_BACK",
    "IS_3_IN_4",
    "IS_HOME",
]

DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_MIN_HISTORY = 5


def as_int(value: Any) -> int:
    return int(value)


def load_team_game_features(season: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / season / "team_game_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed team game features not found at {path}")
    return pd.read_parquet(path)


def validate_sequence_features(
    team_features: pd.DataFrame, feature_columns: list[str]
) -> None:
    if not feature_columns:
        raise ValueError("At least one feature column is required")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for column in feature_columns:
        if column in seen:
            duplicates.add(column)
        seen.add(column)
    if duplicates:
        raise ValueError(f"Duplicate feature columns: {', '.join(sorted(duplicates))}")

    missing = sorted(set(feature_columns) - set(team_features.columns))
    if missing:
        raise ValueError(
            f"Feature columns not found in team game features: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class TeamSequenceStore:
    """Ordered per-team feature vectors for one season, keyed by team id."""

    feature_columns: list[str]
    sequences: dict[int, np.ndarray]
    positions: dict[int, dict[str, int]]

    def prior_window(
        self, team_id: int, game_id: str, sequence_length: int
    ) -> np.ndarray | None:
        team_positions = self.positions.get(int(team_id))
        if team_positions is None or game_id not in team_positions:
            return None
        end = team_positions[game_id]
        start = max(0, end - sequence_length)
        return self.sequences[int(team_id)][start:end]


def build_sequence_store(
    team_features: pd.DataFrame, feature_columns: list[str]
) -> TeamSequenceStore:
    validate_sequence_features(team_features, feature_columns)
    ordered = team_features.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"])

    sequences: dict[int, np.ndarray] = {}
    positions: dict[int, dict[str, int]] = {}
    for team_id, group in ordered.groupby("TEAM_ID", sort=False):
        values = group[feature_columns].to_numpy(dtype=np.float64)
        sequences[as_int(team_id)] = values
        positions[as_int(team_id)] = {
            str(game_id): index
            for index, game_id in enumerate(group["GAME_ID"].to_numpy())
        }
    return TeamSequenceStore(list(feature_columns), sequences, positions)


@dataclass(frozen=True)
class SequenceScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, window: np.ndarray) -> np.ndarray:
        scaled = (window - self.mean) / self.std
        return np.nan_to_num(scaled, nan=0.0)

    def padded(self, window: np.ndarray, sequence_length: int) -> np.ndarray:
        scaled = self.transform(window)
        pad_rows = sequence_length - len(scaled)
        if pad_rows <= 0:
            return scaled[-sequence_length:]
        padding = np.zeros((pad_rows, scaled.shape[1]), dtype=np.float64)
        return np.vstack([padding, scaled])


def fit_sequence_scaler(store: TeamSequenceStore) -> SequenceScaler:
    stacked = np.vstack(list(store.sequences.values()))
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return SequenceScaler(mean, std)


@dataclass(frozen=True)
class SequenceDataset:
    home: np.ndarray
    away: np.ndarray
    labels: np.ndarray
    games_available: int


def build_dataset(
    model_games: pd.DataFrame,
    store: TeamSequenceStore,
    scaler: SequenceScaler,
    sequence_length: int,
    min_history: int,
) -> SequenceDataset:
    home_samples: list[np.ndarray] = []
    away_samples: list[np.ndarray] = []
    labels: list[int] = []
    for game in model_games.itertuples(index=False):
        if pd.isna(game.HOME_WIN):
            continue
        home_window = store.prior_window(
            as_int(game.HOME_TEAM_ID), str(game.GAME_ID), sequence_length
        )
        away_window = store.prior_window(
            as_int(game.AWAY_TEAM_ID), str(game.GAME_ID), sequence_length
        )
        if home_window is None or away_window is None:
            continue
        if len(home_window) < min_history or len(away_window) < min_history:
            continue
        home_samples.append(scaler.padded(home_window, sequence_length))
        away_samples.append(scaler.padded(away_window, sequence_length))
        labels.append(as_int(game.HOME_WIN))

    if not labels:
        return SequenceDataset(
            np.empty((0, sequence_length, len(store.feature_columns))),
            np.empty((0, sequence_length, len(store.feature_columns))),
            np.empty((0,)),
            len(model_games),
        )
    return SequenceDataset(
        np.stack(home_samples),
        np.stack(away_samples),
        np.array(labels, dtype=np.float64),
        len(model_games),
    )
