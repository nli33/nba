"""Train and evaluate a logistic regression game predictor."""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_predictor.evaluation import evaluate_season, format_evaluation
from nba_predictor.prediction import (
    GamePrediction,
    find_game_season,
    format_prediction,
    load_model_games,
    make_prediction,
    predict_game_by_id,
)


DEFAULT_FEATURE_COLUMNS = [
    "DIFF_SEASON_TO_DATE_WIN_PCT",
    "DIFF_SEASON_TO_DATE_POINT_DIFF",
    "DIFF_SEASON_TO_DATE_NET_RATING",
    "DIFF_ROLLING_5_WIN_PCT",
    "DIFF_ROLLING_5_POINT_DIFF",
    "DIFF_ROLLING_5_NET_RATING",
    "DIFF_ROLLING_10_WIN_PCT",
    "DIFF_ROLLING_10_POINT_DIFF",
    "DIFF_ROLLING_10_NET_RATING",
    "DIFF_DAYS_REST",
    "DIFF_IS_BACK_TO_BACK",
    "DIFF_IS_3_IN_4",
]


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

        feature_frame = pd.DataFrame([features.to_dict()], columns=self.artifact.feature_columns)
        home_win_probability = float(self.artifact.pipeline.predict_proba(feature_frame)[0][1])
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


def read_feature_file(path: Path) -> list[str]:
    feature_columns = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            feature_columns.append(line)
    return feature_columns


def resolve_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.features and args.features_file:
        raise ValueError("Use either --features or --features-file, not both")
    if args.features:
        return args.features
    if args.features_file:
        return read_feature_file(args.features_file)
    return DEFAULT_FEATURE_COLUMNS


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


def train_logistic_regression(
    season: str,
    feature_columns: list[str] | None = None,
) -> LogisticRegressionModel:
    model_games = load_model_games(season)
    active_feature_columns = (
        DEFAULT_FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    validate_feature_columns(model_games, active_feature_columns)

    train_data = model_games.dropna(subset=active_feature_columns + ["HOME_WIN"])
    if train_data.empty:
        raise ValueError(f"No complete training rows found for season {season}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(train_data[active_feature_columns], train_data["HOME_WIN"].astype(int))

    return LogisticRegressionModel(
        train_season=season,
        feature_columns=list(active_feature_columns),
        pipeline=pipeline,
        games_available=len(model_games),
        games_trained=len(train_data),
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


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression NBA game predictor."
    )
    parser.add_argument("season", help='Training season, for example "2024-25".')
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/logistic_regression.pkl"),
        help="Path to write the trained model artifact.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="Feature column names to use instead of the default feature set.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        help="Text file with one feature column per line. Blank lines and # comments are ignored.",
    )
    return parser.parse_args()


def train_main() -> None:
    """Train and save a logistic regression predictor."""
    args = parse_train_args()
    try:
        feature_columns = resolve_feature_columns(args)
        artifact = train_logistic_regression(args.season, feature_columns)
        save_model(artifact, args.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to train logistic regression: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_model_details(artifact, args.output, "Logistic Regression Training"))


def parse_inspect_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a trained logistic regression NBA game predictor."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    return parser.parse_args()


def inspect_main() -> None:
    """Print details for a saved logistic regression predictor."""
    args = parse_inspect_args()
    try:
        artifact = load_model(args.model)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to inspect logistic regression model: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_model_details(artifact, args.model, "Logistic Regression Model"))


def parse_evaluate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained logistic regression NBA game predictor."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    parser.add_argument("season", help='Evaluation season, for example "2025-26".')
    return parser.parse_args()


def evaluate_main() -> None:
    """Evaluate a saved logistic regression predictor."""
    args = parse_evaluate_args()
    try:
        artifact = load_model(args.model)
        evaluation = evaluate_season(args.season, LogisticRegressionPredictor(artifact))
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate logistic regression: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_evaluation(evaluation))


def parse_predict_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one NBA game with a trained logistic regression model."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    parser.add_argument("game_id", help="NBA game ID to predict.")
    return parser.parse_args()


def predict_main() -> None:
    """Predict one game with a saved logistic regression predictor."""
    args = parse_predict_args()
    try:
        artifact = load_model(args.model)
        season = find_game_season(args.game_id)
        prediction = predict_game_by_id(
            season,
            args.game_id,
            LogisticRegressionPredictor(artifact),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to predict with logistic regression: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_prediction(prediction))
