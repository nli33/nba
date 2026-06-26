"""Train and evaluate a logistic regression game predictor."""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
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
class AblationResult:
    removed_feature: str | None
    evaluations: list[LogisticEvaluation]

    @property
    def label(self) -> str:
        if self.removed_feature is None:
            return "(baseline)"
        return self.removed_feature

    @property
    def avg_accuracy(self) -> float:
        return sum(evaluation.accuracy for evaluation in self.evaluations) / len(
            self.evaluations
        )

    @property
    def avg_accuracy_when_predicted(self) -> float:
        return sum(
            evaluation.accuracy_when_predicted for evaluation in self.evaluations
        ) / len(self.evaluations)

    @property
    def avg_correct_predictions(self) -> float:
        return sum(evaluation.correct_predictions for evaluation in self.evaluations) / len(
            self.evaluations
        )

    @property
    def avg_log_loss(self) -> float:
        return sum(evaluation.log_loss for evaluation in self.evaluations) / len(
            self.evaluations
        )

    @property
    def avg_brier_score(self) -> float:
        return sum(evaluation.brier_score for evaluation in self.evaluations) / len(
            self.evaluations
        )


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


def fit_logistic_pipeline(train_data: pd.DataFrame, feature_columns: list[str]) -> Any:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(train_data[feature_columns], train_data["HOME_WIN"].astype(int))
    return pipeline


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

    pipeline = fit_logistic_pipeline(train_data, active_feature_columns)

    return LogisticRegressionModel(
        train_season=season,
        feature_columns=list(active_feature_columns),
        pipeline=pipeline,
        games_available=len(model_games),
        games_trained=len(train_data),
    )


def train_logistic_regression_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
) -> LogisticRegressionModel:
    if not seasons:
        raise ValueError("At least one training season is required")

    frames = []
    games_available = 0
    for season in seasons:
        model_games = load_model_games(season)
        validate_feature_columns(model_games, feature_columns)
        games_available += len(model_games)
        frames.append(model_games)

    combined = pd.concat(frames, ignore_index=True)
    train_data = combined.dropna(subset=feature_columns + ["HOME_WIN"])
    if train_data.empty:
        season_list = ", ".join(seasons)
        raise ValueError(f"No complete training rows found for seasons: {season_list}")

    return LogisticRegressionModel(
        train_season=", ".join(seasons),
        feature_columns=list(feature_columns),
        pipeline=fit_logistic_pipeline(train_data, feature_columns),
        games_available=games_available,
        games_trained=len(train_data),
    )


def evaluate_logistic_regression(
    artifact: LogisticRegressionModel,
    eval_season: str,
    train_seasons: list[str],
) -> LogisticEvaluation:
    model_games = load_model_games(eval_season)
    validate_feature_columns(model_games, artifact.feature_columns)
    eval_data = model_games.dropna(subset=artifact.feature_columns + ["HOME_WIN"])
    if eval_data.empty:
        raise ValueError(f"No complete evaluation rows found for season {eval_season}")

    y_true = eval_data["HOME_WIN"].astype(int)
    probabilities = artifact.pipeline.predict_proba(eval_data[artifact.feature_columns])[:, 1]
    predictions = probabilities >= 0.5

    return LogisticEvaluation(
        train_seasons=train_seasons,
        eval_season=eval_season,
        feature_columns=artifact.feature_columns,
        games_evaluated=len(model_games),
        predictions_made=len(eval_data),
        correct_predictions=int((predictions == y_true).sum()),
        log_loss=float(log_loss(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
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


def rolling_splits(seasons: list[str], min_train_seasons: int) -> list[tuple[list[str], str]]:
    if min_train_seasons < 1:
        raise ValueError("--min-train-seasons must be at least 1")
    if len(seasons) <= min_train_seasons:
        raise ValueError("Need more seasons than --min-train-seasons to ablate")

    return [
        (seasons[:index], seasons[index])
        for index in range(min_train_seasons, len(seasons))
    ]


def run_ablation(
    seasons: list[str],
    feature_columns: list[str],
    min_train_seasons: int,
) -> tuple[AblationResult, list[AblationResult]]:
    if len(feature_columns) < 2:
        raise ValueError("At least two feature columns are required for ablation")

    splits = rolling_splits(seasons, min_train_seasons)
    candidates: list[tuple[str | None, list[str]]] = [(None, feature_columns)]
    candidates.extend(
        (feature, [column for column in feature_columns if column != feature])
        for feature in feature_columns
    )

    results = []
    total_runs = len(candidates) * len(splits)
    run_number = 0
    for removed_feature, active_features in candidates:
        evaluations = []
        label = "(baseline)" if removed_feature is None else f"remove {removed_feature}"
        for train_seasons, eval_season in splits:
            run_number += 1
            print(
                f"[{run_number}/{total_runs}] {label}; "
                f"train {', '.join(train_seasons)} -> eval {eval_season}",
                file=sys.stderr,
                flush=True,
            )
            artifact = train_logistic_regression_for_seasons(
                train_seasons,
                active_features,
            )
            evaluations.append(
                evaluate_logistic_regression(artifact, eval_season, train_seasons)
            )
        results.append(AblationResult(removed_feature, evaluations))

    baseline = results[0]
    return baseline, results[1:]


def format_ablation_report(
    seasons: list[str],
    feature_columns: list[str],
    baseline: AblationResult,
    ablations: list[AblationResult],
) -> str:
    sorted_ablations = sorted(
        ablations,
        key=lambda result: result.avg_correct_predictions
        - baseline.avg_correct_predictions,
    )
    lines = [
        "Logistic Regression Feature Ablation",
        f"  Seasons: {', '.join(seasons)}",
        f"  Active features: {len(feature_columns)}",
        f"  Rolling splits: {len(baseline.evaluations)}",
        "",
        "Baseline",
        (
            f"  Accuracy: {100 * baseline.avg_accuracy:.2f}% | "
            f"When predicted: {100 * baseline.avg_accuracy_when_predicted:.2f}% | "
            f"Avg correct: {baseline.avg_correct_predictions:.1f} | "
            f"Log loss: {baseline.avg_log_loss:.4f} | "
            f"Brier: {baseline.avg_brier_score:.4f}"
        ),
        "",
        (
            f"{'Removed feature':<40} {'Acc':>8} {'Correct':>9} {'Delta':>8} "
            f"{'LogLoss':>8} {'Brier':>8}"
        ),
        "-" * 91,
    ]
    for result in sorted_ablations:
        correct_delta = result.avg_correct_predictions - baseline.avg_correct_predictions
        lines.append(
            f"{result.label:<40} "
            f"{100 * result.avg_accuracy:>7.2f}% "
            f"{result.avg_correct_predictions:>9.1f} "
            f"{correct_delta:>+8.1f} "
            f"{result.avg_log_loss:>8.4f} "
            f"{result.avg_brier_score:>8.4f}"
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "  Negative delta means removing the feature reduced average correct picks.",
            "  Positive delta means removing the feature improved average correct picks.",
            "  Small deltas can be noise; compare with log loss and Brier score.",
        ]
    )
    return "\n".join(lines)


def parse_ablate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run drop-one-feature ablation for logistic regression."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Chronological seasons to use for rolling train/eval splits.",
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=1,
        help="Number of initial seasons before the first evaluation split.",
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


def ablate_main() -> None:
    """Run logistic regression feature ablation."""
    args = parse_ablate_args()
    try:
        feature_columns = resolve_feature_columns(args)
        baseline, ablations = run_ablation(
            args.seasons,
            feature_columns,
            args.min_train_seasons,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to run logistic regression ablation: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_ablation_report(args.seasons, feature_columns, baseline, ablations))
