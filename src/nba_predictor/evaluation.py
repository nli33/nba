"""Evaluate game predictors against processed season results."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from nba_predictor.prediction import (
    GamePredictor,
    HomeTeamPredictor,
    PREDICTORS,
    Rolling10NetRatingPredictor,
    SeasonToDateNetRatingPredictor,
    SeasonToDateWinPctPredictor,
    load_model_games,
)


BASELINE_PREDICTORS = (
    HomeTeamPredictor,
    SeasonToDateNetRatingPredictor,
    SeasonToDateWinPctPredictor,
    Rolling10NetRatingPredictor,
)


@dataclass(frozen=True)
class SeasonEvaluation:
    season: str
    predictor_name: str
    games_evaluated: int
    predictions_made: int
    correct_predictions: int

    @property
    def null_predictions(self) -> int:
        return self.games_evaluated - self.predictions_made

    @property
    def accuracy(self) -> float:
        if self.games_evaluated == 0:
            return 0.0
        return self.correct_predictions / self.games_evaluated

    @property
    def accuracy_when_predicted(self) -> float:
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made


def actual_winner_team_id(game: pd.Series) -> int:
    if int(game["HOME_WIN"]) == 1:
        return int(game["HOME_TEAM_ID"])
    return int(game["AWAY_TEAM_ID"])


def evaluate_season(
    season: str,
    predictor: GamePredictor,
) -> SeasonEvaluation:
    model_games = load_model_games(season)
    if model_games.empty:
        raise ValueError(f"No processed games found for season {season}")

    predictions_made = 0
    correct_predictions = 0
    for _, game in model_games.iterrows():
        prediction = predictor.predict(game)
        if prediction.is_null:
            continue

        predictions_made += 1
        if prediction.is_correct(actual_winner_team_id(game)):
            correct_predictions += 1

    return SeasonEvaluation(
        season=season,
        predictor_name=predictor.name,
        games_evaluated=len(model_games),
        predictions_made=predictions_made,
        correct_predictions=correct_predictions,
    )


def actual_winner_abbreviation(game: pd.Series) -> str:
    if int(game["HOME_WIN"]) == 1:
        return str(game["HOME_TEAM_ABBREVIATION"])
    return str(game["AWAY_TEAM_ABBREVIATION"])


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def format_evaluation(evaluation: SeasonEvaluation) -> str:
    return "\n".join(
        [
            "Season Prediction Evaluation",
            f"  Predictor: {evaluation.predictor_name}",
            f"  Season: {evaluation.season}",
            f"  Games evaluated: {evaluation.games_evaluated:,}",
            f"  Predictions made: {evaluation.predictions_made:,}",
            f"  No prediction: {evaluation.null_predictions:,}",
            (
                "  Correct predictions: "
                f"{evaluation.correct_predictions:,}/{evaluation.games_evaluated:,}"
            ),
            f"  Accuracy: {format_percent(evaluation.accuracy)}",
            (
                "  Accuracy when predicted: "
                f"{format_percent(evaluation.accuracy_when_predicted)}"
            ),
        ]
    )


def format_baseline_evaluations(evaluations: list[SeasonEvaluation]) -> str:
    sorted_evaluations = sorted(
        evaluations,
        key=lambda evaluation: evaluation.accuracy,
        reverse=True,
    )
    lines = [
        "Baseline Prediction Evaluation",
        f"  Season: {evaluations[0].season}",
        "",
        (
            f"{'Predictor':<30} {'Accuracy':>9} {'When picked':>12} "
            f"{'Correct':>13} {'Picked':>8} {'No pick':>8}"
        ),
        "-" * 86,
    ]
    for evaluation in sorted_evaluations:
        lines.append(
            f"{evaluation.predictor_name:<30} "
            f"{format_percent(evaluation.accuracy):>9} "
            f"{format_percent(evaluation.accuracy_when_predicted):>12} "
            f"{evaluation.correct_predictions:>6,}/{evaluation.games_evaluated:<6,} "
            f"{evaluation.predictions_made:>8,} "
            f"{evaluation.null_predictions:>8,}"
        )
    return "\n".join(lines)


def format_prediction_cell(game: pd.Series, predictor: GamePredictor) -> str:
    prediction = predictor.predict(game)
    if prediction.predicted_team_abbreviation is None:
        return "no pick"

    actual_team_id = actual_winner_team_id(game)
    outcome = "ok" if prediction.is_correct(actual_team_id) else "miss"
    if prediction.home_win_probability is None:
        return f"{prediction.predicted_team_abbreviation} {outcome}"
    predicted_probability = prediction.home_win_probability
    if prediction.predicted_team_id == prediction.away_team_id:
        predicted_probability = 1 - prediction.home_win_probability
    return (
        f"{prediction.predicted_team_abbreviation} {outcome} "
        f"({100 * predicted_probability:.1f}%)"
    )


def format_game_predictions(season: str, predictors: Sequence[GamePredictor]) -> str:
    model_games = load_model_games(season)
    if model_games.empty:
        raise ValueError(f"No processed games found for season {season}")

    rows = []
    for _, game in model_games.iterrows():
        rows.append(
            [
                str(pd.Timestamp(game["GAME_DATE"]).date()),
                str(game["GAME_ID"]),
                f"{game['AWAY_TEAM_ABBREVIATION']} at {game['HOME_TEAM_ABBREVIATION']}",
                actual_winner_abbreviation(game),
                *[format_prediction_cell(game, predictor) for predictor in predictors],
            ]
        )

    headers = ["Date", "Game ID", "Matchup", "Actual", *[p.name for p in predictors]]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [
        "Detailed Game Predictions",
        f"  Season: {season}",
        "",
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one predictor for one processed NBA season."
    )
    parser.add_argument(
        "predictor",
        choices=sorted(PREDICTORS),
        help="Predictor to use.",
    )
    parser.add_argument("season", help='NBA season, for example "2025-26".')
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print game-by-game prediction results instead of summary metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one season prediction evaluation."""
    args = parse_args()
    predictor = PREDICTORS[args.predictor]()
    try:
        if args.details:
            report = format_game_predictions(args.season, [predictor])
        else:
            report = format_evaluation(evaluate_season(args.season, predictor))
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate season: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(report)


def parse_baselines_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all baseline predictors for one processed NBA season."
    )
    parser.add_argument("season", help='NBA season, for example "2025-26".')
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print game-by-game prediction results instead of summary metrics.",
    )
    return parser.parse_args()


def baselines_main() -> None:
    """Run all baseline season evaluations."""
    args = parse_baselines_args()
    try:
        predictors = [predictor_class() for predictor_class in BASELINE_PREDICTORS]
        if args.details:
            report = format_game_predictions(args.season, predictors)
        else:
            evaluations = [
                evaluate_season(args.season, predictor) for predictor in predictors
            ]
            report = format_baseline_evaluations(evaluations)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate baselines: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(report)
