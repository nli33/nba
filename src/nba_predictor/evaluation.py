"""Evaluate game predictors against processed season results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import pandas as pd

from nba_predictor.prediction import (
    GamePredictor,
    PREDICTORS,
    load_model_games,
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
    return parser.parse_args()


def main() -> None:
    """Run one season prediction evaluation."""
    args = parse_args()
    predictor = PREDICTORS[args.predictor]()
    try:
        evaluation = evaluate_season(args.season, predictor)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to evaluate season: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_evaluation(evaluation))
