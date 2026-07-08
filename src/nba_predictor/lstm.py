"""Command entry points for the LSTM sequence game predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_predictor.evaluation import (
    evaluate_season,
    format_evaluation,
    format_game_predictions,
)
from nba_predictor.models.features import read_feature_file
from nba_predictor.models.lstm import (
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MIN_HISTORY,
    DEFAULT_SEQUENCE_FEATURES,
    DEFAULT_SEQUENCE_LENGTH,
    LSTMEvaluation,
    LSTMModel,
    LSTMPredictor,
    evaluate_lstm as _evaluate_lstm,
    format_model_details,
    load_model,
    save_model,
    train_lstm as _train_lstm,
    train_lstm_for_seasons as _train_lstm_for_seasons,
)
from nba_predictor.prediction import (
    find_game_season,
    format_prediction,
    predict_game_by_id,
)

__all__ = [
    "DEFAULT_SEQUENCE_FEATURES",
    "LSTMEvaluation",
    "LSTMModel",
    "LSTMPredictor",
    "evaluate_lstm",
    "evaluate_main",
    "format_model_details",
    "inspect_main",
    "load_model",
    "parse_evaluate_args",
    "parse_inspect_args",
    "parse_predict_args",
    "parse_train_args",
    "predict_main",
    "read_feature_file",
    "save_model",
    "train_lstm",
    "train_lstm_for_seasons",
    "train_main",
]


def resolve_sequence_features(args: argparse.Namespace) -> list[str]:
    if args.features and args.features_file:
        raise ValueError("Use either --features or --features-file, not both")
    if args.features:
        return args.features
    if args.features_file:
        return read_feature_file(args.features_file)
    return DEFAULT_SEQUENCE_FEATURES


def train_lstm(
    season: str,
    feature_columns: list[str] | None,
    sequence_length: int,
    min_history: int,
    hidden_size: int,
    epochs: int,
) -> LSTMModel:
    return _train_lstm(
        season,
        feature_columns,
        sequence_length=sequence_length,
        min_history=min_history,
        hidden_size=hidden_size,
        epochs=epochs,
    )


def train_lstm_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
    sequence_length: int,
    min_history: int,
    hidden_size: int,
    epochs: int,
) -> LSTMModel:
    return _train_lstm_for_seasons(
        seasons,
        feature_columns,
        sequence_length=sequence_length,
        min_history=min_history,
        hidden_size=hidden_size,
        epochs=epochs,
    )


def evaluate_lstm(
    artifact: LSTMModel,
    eval_season: str,
    train_seasons: list[str],
) -> LSTMEvaluation:
    return _evaluate_lstm(artifact, eval_season, train_seasons)


def add_model_hyperparameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Number of prior games per team fed to the LSTM.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_MIN_HISTORY,
        help="Minimum prior games required for both teams to predict a game.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="LSTM hidden state size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM NBA game predictor.")
    parser.add_argument(
        "seasons",
        nargs="+",
        help='One or more training seasons, for example "2023-24 2024-25".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/lstm.pkl"),
        help="Path to write the trained model artifact.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="Team feature column names to use instead of the default sequence set.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        help="Text file with one feature column per line. Blank lines and # comments are ignored.",
    )
    add_model_hyperparameters(parser)
    return parser.parse_args()


def train_main() -> None:
    """Train and save an LSTM predictor."""
    args = parse_train_args()
    try:
        feature_columns = resolve_sequence_features(args)
        if len(args.seasons) == 1:
            artifact = train_lstm(
                args.seasons[0],
                feature_columns,
                args.sequence_length,
                args.min_history,
                args.hidden_size,
                args.epochs,
            )
        else:
            artifact = train_lstm_for_seasons(
                args.seasons,
                feature_columns,
                args.sequence_length,
                args.min_history,
                args.hidden_size,
                args.epochs,
            )
        save_model(artifact, args.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to train LSTM: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_model_details(artifact, args.output, "LSTM Training"))


def parse_inspect_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a trained LSTM NBA game predictor."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    return parser.parse_args()


def inspect_main() -> None:
    """Print details for a saved LSTM predictor."""
    args = parse_inspect_args()
    try:
        artifact = load_model(args.model)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to inspect LSTM model: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_model_details(artifact, args.model, "LSTM Model"))


def parse_evaluate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LSTM NBA game predictor."
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
    """Evaluate a saved LSTM predictor."""
    args = parse_evaluate_args()
    try:
        artifact = load_model(args.model)
        predictor = LSTMPredictor(artifact)
        if args.details:
            report = format_game_predictions(args.season, [predictor])
        else:
            report = format_evaluation(evaluate_season(args.season, predictor))
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate LSTM: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(report)


def parse_predict_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one NBA game with a trained LSTM model."
    )
    parser.add_argument("model", type=Path, help="Path to a trained model artifact.")
    parser.add_argument("game_id", help="NBA game ID to predict.")
    return parser.parse_args()


def predict_main() -> None:
    """Predict one game with a saved LSTM predictor."""
    args = parse_predict_args()
    try:
        artifact = load_model(args.model)
        season = find_game_season(args.game_id)
        prediction = predict_game_by_id(
            season,
            args.game_id,
            LSTMPredictor(artifact),
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to predict with LSTM: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_prediction(prediction))
