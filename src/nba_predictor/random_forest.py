"""Command entry points for random forest game predictors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_predictor.evaluation import (
    evaluate_season,
    format_evaluation,
    format_game_predictions,
)
from nba_predictor.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    read_feature_file,
    resolve_feature_columns,
    validate_feature_columns,
)
from nba_predictor.models.random_forest import (
    RandomForestEvaluation,
    RandomForestModel,
    RandomForestPredictor,
    evaluate_random_forest as _evaluate_random_forest,
    feature_importances,
    fit_random_forest,
    format_model_details,
    load_model,
    save_model,
    train_random_forest as _train_random_forest,
    train_random_forest_for_seasons as _train_random_forest_for_seasons,
)
from nba_predictor.prediction import (
    find_game_season,
    format_prediction,
    load_model_games,
    predict_game_by_id,
)

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "RandomForestEvaluation",
    "RandomForestModel",
    "RandomForestPredictor",
    "evaluate_main",
    "evaluate_random_forest",
    "feature_importances",
    "fit_random_forest",
    "format_model_details",
    "inspect_main",
    "load_model",
    "load_model_games",
    "parse_evaluate_args",
    "parse_inspect_args",
    "parse_predict_args",
    "parse_train_args",
    "predict_main",
    "read_feature_file",
    "resolve_feature_columns",
    "save_model",
    "train_main",
    "train_random_forest",
    "train_random_forest_for_seasons",
    "validate_feature_columns",
]


def train_random_forest(
    season: str,
    feature_columns: list[str] | None = None,
) -> RandomForestModel:
    return _train_random_forest(
        season,
        feature_columns,
        load_games=load_model_games,
    )


def train_random_forest_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
) -> RandomForestModel:
    return _train_random_forest_for_seasons(
        seasons,
        feature_columns,
        load_games=load_model_games,
    )


def evaluate_random_forest(
    artifact: RandomForestModel,
    eval_season: str,
    train_seasons: list[str],
) -> RandomForestEvaluation:
    return _evaluate_random_forest(
        artifact,
        eval_season,
        train_seasons,
        load_games=load_model_games,
    )


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random forest NBA game predictor."
    )
    parser.add_argument("season", help='Training season, for example "2024-25".')
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/random_forest.pkl"),
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
    """Train and save a random forest predictor."""
    args = parse_train_args()
    try:
        feature_columns = resolve_feature_columns(args)
        artifact = train_random_forest(args.season, feature_columns)
        save_model(artifact, args.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to train random forest: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_model_details(artifact, args.output, "Random Forest Training"))


def parse_inspect_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a trained random forest NBA game predictor."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    return parser.parse_args()


def inspect_main() -> None:
    """Print details for a saved random forest predictor."""
    args = parse_inspect_args()
    try:
        artifact = load_model(args.model)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to inspect random forest model: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_model_details(artifact, args.model, "Random Forest Model"))


def parse_evaluate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained random forest NBA game predictor."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    parser.add_argument("season", help='Evaluation season, for example "2025-26".')
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print game-by-game prediction results instead of summary metrics.",
    )
    return parser.parse_args()


def evaluate_main() -> None:
    """Evaluate a saved random forest predictor."""
    args = parse_evaluate_args()
    try:
        artifact = load_model(args.model)
        predictor = RandomForestPredictor(artifact)
        if args.details:
            report = format_game_predictions(args.season, [predictor])
        else:
            report = format_evaluation(evaluate_season(args.season, predictor))
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate random forest: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(report)


def parse_predict_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one NBA game with a trained random forest model."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    parser.add_argument("game_id", help="NBA game ID to predict.")
    return parser.parse_args()


def predict_main() -> None:
    """Predict one game with a saved random forest predictor."""
    args = parse_predict_args()
    try:
        artifact = load_model(args.model)
        season = find_game_season(args.game_id)
        prediction = predict_game_by_id(
            season,
            args.game_id,
            RandomForestPredictor(artifact),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to predict with random forest: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_prediction(prediction))
