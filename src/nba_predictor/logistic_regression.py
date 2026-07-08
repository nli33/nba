"""Command entry points for logistic regression game predictors."""

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
from nba_predictor.models.logistic import (
    LogisticEvaluation,
    LogisticRegressionModel,
    LogisticRegressionPredictor,
    evaluate_logistic_regression as _evaluate_logistic_regression,
    fit_logistic_pipeline,
    format_model_details,
    load_model,
    model_step,
    save_model,
    train_logistic_regression as _train_logistic_regression,
    train_logistic_regression_for_seasons as _train_logistic_regression_for_seasons,
)
from nba_predictor.models.logistic_ablation import (
    AblationResult,
    SPLIT_STRATEGIES,
    EvaluationSplit,
    evaluation_splits,
    format_ablation_report,
    randomized_evaluation_splits,
    rolling_splits,
    run_ablation as _run_ablation,
)
from nba_predictor.models.logistic_diagnostics import (
    LogisticDiagnostics,
    format_diagnostics_report,
    run_diagnostics as _run_diagnostics,
    save_diagnostic_plots,
)
from nba_predictor.prediction import (
    find_game_season,
    format_prediction,
    load_model_games,
    predict_game_by_id,
)

__all__ = [
    "AblationResult",
    "DEFAULT_FEATURE_COLUMNS",
    "LogisticEvaluation",
    "LogisticRegressionModel",
    "LogisticRegressionPredictor",
    "SPLIT_STRATEGIES",
    "LogisticDiagnostics",
    "ablate_main",
    "diagnose_main",
    "evaluate_logistic_regression",
    "evaluate_main",
    "EvaluationSplit",
    "evaluation_splits",
    "fit_logistic_pipeline",
    "format_ablation_report",
    "format_diagnostics_report",
    "format_model_details",
    "inspect_main",
    "load_model",
    "load_model_games",
    "model_step",
    "parse_ablate_args",
    "parse_diagnose_args",
    "parse_evaluate_args",
    "parse_inspect_args",
    "parse_predict_args",
    "parse_train_args",
    "predict_main",
    "read_feature_file",
    "randomized_evaluation_splits",
    "resolve_feature_columns",
    "rolling_splits",
    "run_ablation",
    "run_diagnostics",
    "save_diagnostic_plots",
    "save_model",
    "train_logistic_regression",
    "train_logistic_regression_for_seasons",
    "train_main",
    "validate_feature_columns",
]


def train_logistic_regression(
    season: str,
    feature_columns: list[str] | None = None,
) -> LogisticRegressionModel:
    return _train_logistic_regression(
        season,
        feature_columns,
        load_games=load_model_games,
    )


def train_logistic_regression_for_seasons(
    seasons: list[str],
    feature_columns: list[str],
) -> LogisticRegressionModel:
    return _train_logistic_regression_for_seasons(
        seasons,
        feature_columns,
        load_games=load_model_games,
    )


def evaluate_logistic_regression(
    artifact: LogisticRegressionModel,
    eval_season: str,
    train_seasons: list[str],
) -> LogisticEvaluation:
    return _evaluate_logistic_regression(
        artifact,
        eval_season,
        train_seasons,
        load_games=load_model_games,
    )


def run_ablation(
    seasons: list[str],
    feature_columns: list[str],
    min_train_seasons: int,
    split_strategy: str = "chronological",
    test_size: float = 0.2,
    random_repeats: int = 1,
    random_seed: int = 0,
) -> tuple[AblationResult, list[AblationResult]]:
    return _run_ablation(
        seasons,
        feature_columns,
        min_train_seasons,
        split_strategy=split_strategy,
        test_size=test_size,
        random_repeats=random_repeats,
        random_seed=random_seed,
        load_games=load_model_games,
    )


def run_diagnostics(
    seasons: list[str],
    feature_columns: list[str],
    min_train_seasons: int = 1,
) -> LogisticDiagnostics:
    return _run_diagnostics(
        seasons,
        feature_columns,
        min_train_seasons,
        load_games=load_model_games,
    )


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
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print game-by-game prediction results instead of summary metrics.",
    )
    return parser.parse_args()


def evaluate_main() -> None:
    """Evaluate a saved logistic regression predictor."""
    args = parse_evaluate_args()
    try:
        artifact = load_model(args.model)
        predictor = LogisticRegressionPredictor(artifact)
        if args.details:
            report = format_game_predictions(args.season, [predictor])
        else:
            report = format_evaluation(evaluate_season(args.season, predictor))
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate logistic regression: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(report)


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
        help="Number of initial seasons before the first chronological evaluation split.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=SPLIT_STRATEGIES,
        default="chronological",
        help="Train/eval split strategy. Chronological preserves the existing rolling season split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows assigned to eval for randomized splits.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=1,
        help="Number of randomized train/eval splits to run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Base random seed for randomized splits.",
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
            split_strategy=args.split_strategy,
            test_size=args.test_size,
            random_repeats=args.random_repeats,
            random_seed=args.random_seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to run logistic regression ablation: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(
        format_ablation_report(
            args.seasons,
            feature_columns,
            baseline,
            ablations,
            split_strategy=args.split_strategy,
        )
    )


def parse_diagnose_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report walk-forward probability diagnostics for logistic regression."
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
        help="Number of initial seasons before the first chronological evaluation split.",
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
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory to write calibration, selective-prediction, and upset PNG charts "
        "(requires the optional viz extra).",
    )
    return parser.parse_args()


def diagnose_main() -> None:
    """Report logistic regression walk-forward probability diagnostics."""
    args = parse_diagnose_args()
    try:
        feature_columns = resolve_feature_columns(args)
        diagnostics = run_diagnostics(
            args.seasons,
            feature_columns,
            args.min_train_seasons,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to run logistic regression diagnostics: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_diagnostics_report(diagnostics))

    if args.plots_dir is not None:
        try:
            paths = save_diagnostic_plots(diagnostics, args.plots_dir)
        except ModuleNotFoundError as error:
            print(f"Unable to write diagnostic plots: {error}", file=sys.stderr)
            raise SystemExit(1) from None
        print("")
        print("Saved plots:")
        for path in paths:
            print(f"  {path}")
