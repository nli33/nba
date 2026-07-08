"""Train, evaluate, and persist an LSTM sequence game predictor.

The LSTM consumes, for each game, the chronological sequence of each team's own
prior pre-game feature vectors (drawn from ``team_game_features.parquet``). Because
every team-game feature row is already leakage-safe (built with ``shift(1)``) and
only games strictly before the predicted game are used, the sequence for game ``t``
ends at ``t-1``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss
from torch import nn

from nba_predictor.models.lstm_data import (
    DEFAULT_MIN_HISTORY,
    DEFAULT_SEQUENCE_FEATURES,
    DEFAULT_SEQUENCE_LENGTH,
    SequenceDataset,
    SequenceScaler,
    TeamSequenceStore,
    as_int,
    build_dataset,
    build_sequence_store,
    fit_sequence_scaler,
    load_team_game_features,
)
from nba_predictor.prediction import (
    GamePrediction,
    load_model_games,
    make_prediction,
)


DEFAULT_HIDDEN_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_PATIENCE = 5
RANDOM_SEED = 0


class GameSequenceLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequences)
        return hidden[-1]

    def forward(self, home: torch.Tensor, away: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.encode(home), self.encode(away)], dim=1)
        return self.head(combined).squeeze(1)


@dataclass(frozen=True)
class LSTMModel:
    train_season: str
    feature_columns: list[str]
    sequence_length: int
    min_history: int
    hidden_size: int
    state_dict: dict[str, Any]
    scaler: SequenceScaler
    games_available: int
    games_trained: int
    epochs_trained: int = DEFAULT_EPOCHS
    validation_games: int = 0
    validation_log_loss: float | None = None

    @property
    def games_dropped(self) -> int:
        return self.games_available - self.games_trained

    def build_module(self) -> GameSequenceLSTM:
        module = GameSequenceLSTM(len(self.feature_columns), self.hidden_size)
        module.load_state_dict(self.state_dict)
        module.eval()
        return module


@dataclass(frozen=True)
class LSTMEvaluation:
    train_seasons: list[str]
    eval_season: str
    feature_columns: list[str]
    games_evaluated: int
    predictions_made: int
    correct_predictions: int
    log_loss: float
    brier_score: float

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.games_evaluated

    @property
    def accuracy_when_predicted(self) -> float:
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made


def _home_win_probabilities(
    module: GameSequenceLSTM, home: np.ndarray, away: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        logits = module(
            torch.from_numpy(home).float(),
            torch.from_numpy(away).float(),
        )
        return torch.sigmoid(logits).numpy()


def _validation_loss(
    module: GameSequenceLSTM,
    home: torch.Tensor,
    away: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    module.eval()
    with torch.no_grad():
        loss = loss_fn(module(home, away), labels)
    module.train()
    return float(loss)


def fit_lstm_module(
    dataset: SequenceDataset,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation: SequenceDataset | None = None,
    patience: int = DEFAULT_PATIENCE,
) -> tuple[GameSequenceLSTM, int, float | None]:
    """Fit the LSTM, returning the module, the epoch kept, and its validation loss.

    With a ``validation`` set, training runs up to ``epochs`` and keeps the weights
    from the epoch with the lowest validation log loss (early stopping with
    ``patience``). Without one, it trains for exactly ``epochs`` (the prior behavior).
    """
    torch.manual_seed(RANDOM_SEED)
    module = GameSequenceLSTM(dataset.home.shape[2], hidden_size)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    home = torch.from_numpy(dataset.home).float()
    away = torch.from_numpy(dataset.away).float()
    labels = torch.from_numpy(dataset.labels).float()

    if validation is not None:
        val_home = torch.from_numpy(validation.home).float()
        val_away = torch.from_numpy(validation.away).float()
        val_labels = torch.from_numpy(validation.labels).float()

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = epochs
    epochs_without_improvement = 0

    module.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = module(home[batch], away[batch])
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            optimizer.step()

        if validation is None:
            continue

        val_loss = _validation_loss(module, val_home, val_away, val_labels, loss_fn)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in module.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if validation is not None and best_state is not None:
        module.load_state_dict(best_state)
        module.eval()
        return module, best_epoch, best_val_loss

    module.eval()
    return module, epochs, None


def _split_off_validation(
    dataset: SequenceDataset, validation_fraction: float
) -> tuple[SequenceDataset, SequenceDataset | None]:
    """Hold out the most recent ``validation_fraction`` of samples for validation.

    Samples arrive in chronological order, so the tail is the latest games — a
    leakage-free validation set that mirrors predicting a future stretch.
    """
    total = len(dataset.labels)
    validation_count = round(total * validation_fraction)
    if validation_fraction <= 0 or not 0 < validation_count < total:
        return dataset, None

    split = total - validation_count
    train = SequenceDataset(
        dataset.home[:split],
        dataset.away[:split],
        dataset.labels[:split],
        dataset.games_available,
    )
    validation = SequenceDataset(
        dataset.home[split:],
        dataset.away[split:],
        dataset.labels[split:],
        dataset.games_available,
    )
    return train, validation


def train_lstm_from_frames(
    model_games: pd.DataFrame,
    team_features: pd.DataFrame,
    train_label: str,
    feature_columns: list[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    min_history: int = DEFAULT_MIN_HISTORY,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    patience: int = DEFAULT_PATIENCE,
) -> LSTMModel:
    store = build_sequence_store(team_features, feature_columns)
    scaler = fit_sequence_scaler(store)
    ordered_games = model_games.sort_values(["GAME_DATE", "GAME_ID"])
    dataset = build_dataset(
        ordered_games, store, scaler, sequence_length, min_history
    )
    if len(dataset.labels) == 0:
        raise ValueError(f"No complete training sequences found for {train_label}")

    train_dataset, validation = _split_off_validation(dataset, validation_fraction)
    module, epochs_trained, validation_log_loss = fit_lstm_module(
        train_dataset,
        hidden_size,
        epochs,
        batch_size,
        learning_rate,
        validation=validation,
        patience=patience,
    )
    return LSTMModel(
        train_season=train_label,
        feature_columns=list(feature_columns),
        sequence_length=sequence_length,
        min_history=min_history,
        hidden_size=hidden_size,
        state_dict={key: value.clone() for key, value in module.state_dict().items()},
        scaler=scaler,
        games_available=dataset.games_available,
        games_trained=len(train_dataset.labels),
        epochs_trained=epochs_trained,
        validation_games=0 if validation is None else len(validation.labels),
        validation_log_loss=validation_log_loss,
    )


def train_lstm(
    season: str,
    feature_columns: list[str] | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    min_history: int = DEFAULT_MIN_HISTORY,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    patience: int = DEFAULT_PATIENCE,
) -> LSTMModel:
    active_features = (
        DEFAULT_SEQUENCE_FEATURES if feature_columns is None else feature_columns
    )
    return train_lstm_from_frames(
        load_model_games(season),
        load_team_game_features(season),
        season,
        active_features,
        sequence_length=sequence_length,
        min_history=min_history,
        hidden_size=hidden_size,
        epochs=epochs,
        validation_fraction=validation_fraction,
        patience=patience,
    )


def train_lstm_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    min_history: int = DEFAULT_MIN_HISTORY,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    patience: int = DEFAULT_PATIENCE,
) -> LSTMModel:
    if not seasons:
        raise ValueError("At least one training season is required")

    game_frames = []
    feature_frames = []
    for season in seasons:
        game_frames.append(load_model_games(season))
        feature_frames.append(load_team_game_features(season))

    return train_lstm_from_frames(
        pd.concat(game_frames, ignore_index=True),
        pd.concat(feature_frames, ignore_index=True),
        ", ".join(seasons),
        feature_columns,
        sequence_length=sequence_length,
        min_history=min_history,
        hidden_size=hidden_size,
        epochs=epochs,
        validation_fraction=validation_fraction,
        patience=patience,
    )


@dataclass
class LSTMPredictor:
    artifact: LSTMModel
    name: str = "lstm"

    def __post_init__(self) -> None:
        self._module = self.artifact.build_module()
        self._stores: dict[str, TeamSequenceStore] = {}

    def _store_for(self, season: str) -> TeamSequenceStore:
        if season not in self._stores:
            self._stores[season] = build_sequence_store(
                load_team_game_features(season), self.artifact.feature_columns
            )
        return self._stores[season]

    def predict(self, game: pd.Series) -> GamePrediction:
        store = self._store_for(str(game["SEASON"]))
        home_window = store.prior_window(
            as_int(game["HOME_TEAM_ID"]),
            str(game["GAME_ID"]),
            self.artifact.sequence_length,
        )
        away_window = store.prior_window(
            as_int(game["AWAY_TEAM_ID"]),
            str(game["GAME_ID"]),
            self.artifact.sequence_length,
        )
        if (
            home_window is None
            or away_window is None
            or len(home_window) < self.artifact.min_history
            or len(away_window) < self.artifact.min_history
        ):
            return make_prediction(
                game=game,
                predictor_name=self.name,
                predicted_team_id=None,
                predicted_team_abbreviation=None,
                reason="Insufficient game history for LSTM sequence",
            )

        home = self.artifact.scaler.padded(
            home_window, self.artifact.sequence_length
        )[np.newaxis]
        away = self.artifact.scaler.padded(
            away_window, self.artifact.sequence_length
        )[np.newaxis]
        home_win_probability = float(
            _home_win_probabilities(self._module, home, away)[0]
        )
        if home_win_probability >= 0.5:
            predicted_team_id = as_int(game["HOME_TEAM_ID"])
            predicted_team_abbreviation = str(game["HOME_TEAM_ABBREVIATION"])
        else:
            predicted_team_id = as_int(game["AWAY_TEAM_ID"])
            predicted_team_abbreviation = str(game["AWAY_TEAM_ABBREVIATION"])

        return make_prediction(
            game=game,
            predictor_name=self.name,
            predicted_team_id=predicted_team_id,
            predicted_team_abbreviation=predicted_team_abbreviation,
            reason="LSTM estimated home win probability",
            home_win_probability=home_win_probability,
        )


def evaluate_lstm_on_frames(
    artifact: LSTMModel,
    model_games: pd.DataFrame,
    team_features: pd.DataFrame,
    eval_label: str,
    train_seasons: list[str],
) -> LSTMEvaluation:
    store = build_sequence_store(team_features, artifact.feature_columns)
    dataset = build_dataset(
        model_games,
        store,
        artifact.scaler,
        artifact.sequence_length,
        artifact.min_history,
    )
    if len(dataset.labels) == 0:
        raise ValueError(f"No complete evaluation sequences found for {eval_label}")

    module = artifact.build_module()
    probabilities = _home_win_probabilities(module, dataset.home, dataset.away)
    predictions = probabilities >= 0.5
    labels = dataset.labels.astype(int)

    return LSTMEvaluation(
        train_seasons=train_seasons,
        eval_season=eval_label,
        feature_columns=artifact.feature_columns,
        games_evaluated=len(model_games),
        predictions_made=len(dataset.labels),
        correct_predictions=int((predictions == labels).sum()),
        log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
        brier_score=float(brier_score_loss(labels, probabilities)),
    )


def save_model(artifact: LSTMModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact, file)


def load_model(path: Path) -> LSTMModel:
    with path.open("rb") as file:
        artifact = pickle.load(file)
    if not isinstance(artifact, LSTMModel):
        raise ValueError(f"Expected LSTMModel in {path}")
    return artifact


def format_model_details(
    artifact: LSTMModel,
    model_path: Path,
    title: str,
) -> str:
    lines = [
        title,
        f"  Train season: {artifact.train_season}",
        f"  Games available: {artifact.games_available:,}",
        f"  Games trained: {artifact.games_trained:,}",
        f"  Games dropped: {artifact.games_dropped:,}",
        f"  Validation games: {artifact.validation_games:,}",
        f"  Epochs kept: {artifact.epochs_trained}"
        + (
            f" (val log loss {artifact.validation_log_loss:.4f})"
            if artifact.validation_log_loss is not None
            else " (no validation)"
        ),
        f"  Sequence length: {artifact.sequence_length}",
        f"  Min history: {artifact.min_history}",
        f"  Hidden size: {artifact.hidden_size}",
        f"  Model path: {model_path}",
        "",
        "Sequence features",
    ]
    lines.extend(f"  {name}" for name in artifact.feature_columns)
    return "\n".join(lines)
