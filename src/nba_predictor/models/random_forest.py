"""Train, evaluate, and persist random forest game predictors."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss

from nba_predictor.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    validate_feature_columns,
)
from nba_predictor.prediction import GamePrediction, load_model_games, make_prediction


@dataclass(frozen=True)
class RandomForestModel:
    train_season: str
    feature_columns: list[str]
    model: Any
    games_available: int
    games_trained: int

    @property
    def games_dropped(self) -> int:
        return self.games_available - self.games_trained


@dataclass(frozen=True)
class RandomForestEvaluation:
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


@dataclass(frozen=True)
class RandomForestPredictor:
    artifact: RandomForestModel
    name: str = "random_forest"

    def predict(self, game: pd.Series) -> GamePrediction:
        features = game[self.artifact.feature_columns]
        if features.isna().any():
            return make_prediction(
                game=game,
                predictor_name=self.name,
                predicted_team_id=None,
                predicted_team_abbreviation=None,
                reason="Missing random forest feature value",
            )

        feature_frame = pd.DataFrame(
            [features.to_dict()], columns=self.artifact.feature_columns
        )
        home_win_probability = float(self.artifact.model.predict_proba(feature_frame)[0][1])
        if home_win_probability >= 0.5:
            predicted_team_id = int(game["HOME_TEAM_ID"])
            predicted_team_abbreviation = str(game["HOME_TEAM_ABBREVIATION"])
        else:
            predicted_team_id = int(game["AWAY_TEAM_ID"])
            predicted_team_abbreviation = str(game["AWAY_TEAM_ABBREVIATION"])

        return make_prediction(
            game=game,
            predictor_name=self.name,
            predicted_team_id=predicted_team_id,
            predicted_team_abbreviation=predicted_team_abbreviation,
            reason="Random forest estimated home win probability",
            home_win_probability=home_win_probability,
        )


def fit_random_forest(train_data: pd.DataFrame, feature_columns: list[str]) -> Any:
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(train_data[feature_columns], train_data["HOME_WIN"].astype(int))
    return model


def train_random_forest_from_frame(
    model_games: pd.DataFrame,
    train_label: str,
    feature_columns: list[str],
) -> RandomForestModel:
    validate_feature_columns(model_games, feature_columns)
    train_data = model_games.dropna(subset=feature_columns + ["HOME_WIN"])
    if train_data.empty:
        raise ValueError(f"No complete training rows found for {train_label}")

    return RandomForestModel(
        train_season=train_label,
        feature_columns=list(feature_columns),
        model=fit_random_forest(train_data, feature_columns),
        games_available=len(model_games),
        games_trained=len(train_data),
    )


def train_random_forest(
    season: str,
    feature_columns: list[str] | None = None,
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> RandomForestModel:
    model_games = load_games(season)
    active_feature_columns = (
        DEFAULT_FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    return train_random_forest_from_frame(
        model_games,
        season,
        active_feature_columns,
    )


def train_random_forest_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> RandomForestModel:
    if not seasons:
        raise ValueError("At least one training season is required")

    frames = []
    for season in seasons:
        model_games = load_games(season)
        validate_feature_columns(model_games, feature_columns)
        frames.append(model_games)

    return train_random_forest_from_frame(
        pd.concat(frames, ignore_index=True),
        ", ".join(seasons),
        feature_columns,
    )


def evaluate_random_forest_on_frame(
    artifact: RandomForestModel,
    model_games: pd.DataFrame,
    eval_label: str,
    train_seasons: list[str],
) -> RandomForestEvaluation:
    validate_feature_columns(model_games, artifact.feature_columns)
    eval_data = model_games.dropna(subset=artifact.feature_columns + ["HOME_WIN"])
    if eval_data.empty:
        raise ValueError(f"No complete evaluation rows found for {eval_label}")

    y_true = eval_data["HOME_WIN"].astype(int)
    probabilities = artifact.model.predict_proba(eval_data[artifact.feature_columns])[:, 1]
    predictions = probabilities >= 0.5

    return RandomForestEvaluation(
        train_seasons=train_seasons,
        eval_season=eval_label,
        feature_columns=artifact.feature_columns,
        games_evaluated=len(model_games),
        predictions_made=len(eval_data),
        correct_predictions=int((predictions == y_true).sum()),
        log_loss=float(log_loss(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
    )


def evaluate_random_forest(
    artifact: RandomForestModel,
    eval_season: str,
    train_seasons: list[str],
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> RandomForestEvaluation:
    return evaluate_random_forest_on_frame(
        artifact,
        load_games(eval_season),
        eval_season,
        train_seasons,
    )


def save_model(artifact: RandomForestModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact, file)


def load_model(path: Path) -> RandomForestModel:
    with path.open("rb") as file:
        artifact = pickle.load(file)
    if not isinstance(artifact, RandomForestModel):
        raise ValueError(f"Expected RandomForestModel in {path}")
    return artifact


def feature_importances(artifact: RandomForestModel) -> list[tuple[str, float]]:
    return sorted(
        zip(artifact.feature_columns, artifact.model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )


def format_model_details(
    artifact: RandomForestModel,
    model_path: Path,
    title: str,
) -> str:
    lines = [
        title,
        f"  Train season: {artifact.train_season}",
        f"  Games available: {artifact.games_available:,}",
        f"  Games trained: {artifact.games_trained:,}",
        f"  Games dropped: {artifact.games_dropped:,}",
        f"  Trees: {artifact.model.n_estimators:,}",
        f"  Model path: {model_path}",
        "",
        "Feature importances",
    ]
    lines.extend(
        f"  {name}: {importance:.4f}"
        for name, importance in feature_importances(artifact)
    )
    return "\n".join(lines)
