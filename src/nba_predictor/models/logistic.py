"""Train, evaluate, and persist logistic regression game predictors."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_predictor.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    validate_feature_columns,
)
from nba_predictor.prediction import GamePrediction, load_model_games, make_prediction


@dataclass(frozen=True)
class LogisticRegressionModel:
    train_season: str
    feature_columns: list[str]
    pipeline: Any
    games_available: int
    games_trained: int

    @property
    def games_dropped(self) -> int:
        return self.games_available - self.games_trained


@dataclass(frozen=True)
class LogisticEvaluation:
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
class LogisticRegressionPredictor:
    artifact: LogisticRegressionModel
    name: str = "logistic_regression"

    def predict(self, game: pd.Series) -> GamePrediction:
        features = game[self.artifact.feature_columns]
        if features.isna().any():
            return make_prediction(
                game=game,
                predictor_name=self.name,
                predicted_team_id=None,
                predicted_team_abbreviation=None,
                reason="Missing logistic regression feature value",
            )

        feature_frame = pd.DataFrame(
            [features.to_dict()], columns=self.artifact.feature_columns
        )
        home_win_probability = float(
            self.artifact.pipeline.predict_proba(feature_frame)[0][1]
        )
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
            reason="Logistic regression estimated home win probability",
            home_win_probability=home_win_probability,
        )


def fit_logistic_pipeline(train_data: pd.DataFrame, feature_columns: list[str]) -> Any:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(train_data[feature_columns], train_data["HOME_WIN"].astype(int))
    return pipeline


def train_logistic_regression_from_frame(
    model_games: pd.DataFrame,
    train_label: str,
    feature_columns: list[str],
) -> LogisticRegressionModel:
    validate_feature_columns(model_games, feature_columns)
    train_data = model_games.dropna(subset=feature_columns + ["HOME_WIN"])
    if train_data.empty:
        raise ValueError(f"No complete training rows found for {train_label}")

    return LogisticRegressionModel(
        train_season=train_label,
        feature_columns=list(feature_columns),
        pipeline=fit_logistic_pipeline(train_data, feature_columns),
        games_available=len(model_games),
        games_trained=len(train_data),
    )


def train_logistic_regression(
    season: str,
    feature_columns: list[str] | None = None,
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> LogisticRegressionModel:
    model_games = load_games(season)
    active_feature_columns = (
        DEFAULT_FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    return train_logistic_regression_from_frame(
        model_games,
        season,
        active_feature_columns,
    )


def train_logistic_regression_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> LogisticRegressionModel:
    if not seasons:
        raise ValueError("At least one training season is required")

    frames = []
    games_available = 0
    for season in seasons:
        model_games = load_games(season)
        validate_feature_columns(model_games, feature_columns)
        games_available += len(model_games)
        frames.append(model_games)

    return train_logistic_regression_from_frame(
        pd.concat(frames, ignore_index=True),
        ", ".join(seasons),
        feature_columns,
    )


def evaluate_logistic_regression_on_frame(
    artifact: LogisticRegressionModel,
    model_games: pd.DataFrame,
    eval_label: str,
    train_seasons: list[str],
) -> LogisticEvaluation:
    validate_feature_columns(model_games, artifact.feature_columns)
    eval_data = model_games.dropna(subset=artifact.feature_columns + ["HOME_WIN"])
    if eval_data.empty:
        raise ValueError(f"No complete evaluation rows found for {eval_label}")

    y_true = eval_data["HOME_WIN"].astype(int)
    probabilities = artifact.pipeline.predict_proba(
        eval_data[artifact.feature_columns]
    )[:, 1]
    predictions = probabilities >= 0.5

    return LogisticEvaluation(
        train_seasons=train_seasons,
        eval_season=eval_label,
        feature_columns=artifact.feature_columns,
        games_evaluated=len(model_games),
        predictions_made=len(eval_data),
        correct_predictions=int((predictions == y_true).sum()),
        log_loss=float(log_loss(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
    )


def evaluate_logistic_regression(
    artifact: LogisticRegressionModel,
    eval_season: str,
    train_seasons: list[str],
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> LogisticEvaluation:
    return evaluate_logistic_regression_on_frame(
        artifact,
        load_games(eval_season),
        eval_season,
        train_seasons,
    )


def save_model(artifact: LogisticRegressionModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact, file)


def load_model(path: Path) -> LogisticRegressionModel:
    with path.open("rb") as file:
        artifact = pickle.load(file)
    if not isinstance(artifact, LogisticRegressionModel):
        raise ValueError(f"Expected LogisticRegressionModel in {path}")
    return artifact


def model_step(artifact: LogisticRegressionModel) -> LogisticRegression:
    return artifact.pipeline.named_steps["model"]


def format_model_details(
    artifact: LogisticRegressionModel,
    model_path: Path,
    title: str,
) -> str:
    model = model_step(artifact)
    coefficients = sorted(
        zip(artifact.feature_columns, model.coef_[0], strict=True),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    lines = [
        title,
        f"  Train season: {artifact.train_season}",
        f"  Games available: {artifact.games_available:,}",
        f"  Games trained: {artifact.games_trained:,}",
        f"  Games dropped: {artifact.games_dropped:,}",
        f"  Intercept: {model.intercept_[0]:.4f}",
        f"  Model path: {model_path}",
        "",
        "Learned coefficients",
    ]
    lines.extend(f"  {name}: {coefficient:.4f}" for name, coefficient in coefficients)
    return "\n".join(lines)
