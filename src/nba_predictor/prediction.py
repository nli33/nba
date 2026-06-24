"""Interfaces for making game predictions from processed NBA data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class GamePrediction:
    season: str
    game_id: str
    predictor_name: str
    home_team_id: int
    home_team_abbreviation: str
    away_team_id: int
    away_team_abbreviation: str
    predicted_team_id: int | None
    predicted_team_abbreviation: str | None
    reason: str
    home_win_probability: float | None = None

    @property
    def is_null(self) -> bool:
        return self.predicted_team_id is None

    def is_correct(self, actual_winner_team_id: int) -> bool:
        return self.predicted_team_id == actual_winner_team_id


class GamePredictor(Protocol):
    @property
    def name(self) -> str:
        """Predictor display name."""
        ...

    def predict(self, game: pd.Series) -> GamePrediction:
        """Predict one game from a processed model_games row."""


def make_prediction(
    game: pd.Series,
    predictor_name: str,
    predicted_team_id: int | None,
    predicted_team_abbreviation: str | None,
    reason: str,
    home_win_probability: float | None = None,
) -> GamePrediction:
    return GamePrediction(
        season=str(game["SEASON"]),
        game_id=str(game["GAME_ID"]),
        predictor_name=predictor_name,
        home_team_id=int(game["HOME_TEAM_ID"]),
        home_team_abbreviation=str(game["HOME_TEAM_ABBREVIATION"]),
        away_team_id=int(game["AWAY_TEAM_ID"]),
        away_team_abbreviation=str(game["AWAY_TEAM_ABBREVIATION"]),
        predicted_team_id=predicted_team_id,
        predicted_team_abbreviation=predicted_team_abbreviation,
        reason=reason,
        home_win_probability=home_win_probability,
    )


@dataclass(frozen=True)
class HomeTeamPredictor:
    name: str = "home_team"

    def predict(self, game: pd.Series) -> GamePrediction:
        return make_prediction(
            game=game,
            predictor_name=self.name,
            predicted_team_id=int(game["HOME_TEAM_ID"]),
            predicted_team_abbreviation=str(game["HOME_TEAM_ABBREVIATION"]),
            reason="Always predicts the home team",
        )


@dataclass(frozen=True)
class SeasonToDateNetRatingPredictor:
    name: str = "season_to_date_net_rating"

    def predict(self, game: pd.Series) -> GamePrediction:
        home_net_rating = game["HOME_SEASON_TO_DATE_NET_RATING"]
        away_net_rating = game["AWAY_SEASON_TO_DATE_NET_RATING"]

        predicted_team_id = None
        predicted_team_abbreviation = None
        if pd.isna(home_net_rating) or pd.isna(away_net_rating):
            reason = "Missing season-to-date net rating"
        elif home_net_rating == away_net_rating:
            reason = "Season-to-date net ratings are tied"
        elif home_net_rating > away_net_rating:
            predicted_team_id = int(game["HOME_TEAM_ID"])
            predicted_team_abbreviation = str(game["HOME_TEAM_ABBREVIATION"])
            reason = "Home team has higher season-to-date net rating"
        else:
            predicted_team_id = int(game["AWAY_TEAM_ID"])
            predicted_team_abbreviation = str(game["AWAY_TEAM_ABBREVIATION"])
            reason = "Away team has higher season-to-date net rating"

        return make_prediction(
            game=game,
            predictor_name=self.name,
            predicted_team_id=predicted_team_id,
            predicted_team_abbreviation=predicted_team_abbreviation,
            reason=reason,
        )


@dataclass(frozen=True)
class SeasonToDateWinPctPredictor:
    name: str = "season_to_date_win_pct"

    def predict(self, game: pd.Series) -> GamePrediction:
        home_win_pct = game["HOME_SEASON_TO_DATE_WIN_PCT"]
        away_win_pct = game["AWAY_SEASON_TO_DATE_WIN_PCT"]

        predicted_team_id = None
        predicted_team_abbreviation = None
        if pd.isna(home_win_pct) or pd.isna(away_win_pct):
            reason = "Missing season-to-date win percentage"
        elif home_win_pct == away_win_pct:
            reason = "Season-to-date win percentages are tied"
        elif home_win_pct > away_win_pct:
            predicted_team_id = int(game["HOME_TEAM_ID"])
            predicted_team_abbreviation = str(game["HOME_TEAM_ABBREVIATION"])
            reason = "Home team has higher season-to-date win percentage"
        else:
            predicted_team_id = int(game["AWAY_TEAM_ID"])
            predicted_team_abbreviation = str(game["AWAY_TEAM_ABBREVIATION"])
            reason = "Away team has higher season-to-date win percentage"

        return make_prediction(
            game=game,
            predictor_name=self.name,
            predicted_team_id=predicted_team_id,
            predicted_team_abbreviation=predicted_team_abbreviation,
            reason=reason,
        )


@dataclass(frozen=True)
class Rolling10NetRatingPredictor:
    name: str = "rolling_10_net_rating"

    def predict(self, game: pd.Series) -> GamePrediction:
        home_net_rating = game["HOME_ROLLING_10_NET_RATING"]
        away_net_rating = game["AWAY_ROLLING_10_NET_RATING"]

        predicted_team_id = None
        predicted_team_abbreviation = None
        if pd.isna(home_net_rating) or pd.isna(away_net_rating):
            reason = "Missing rolling 10-game net rating"
        elif home_net_rating == away_net_rating:
            reason = "Rolling 10-game net ratings are tied"
        elif home_net_rating > away_net_rating:
            predicted_team_id = int(game["HOME_TEAM_ID"])
            predicted_team_abbreviation = str(game["HOME_TEAM_ABBREVIATION"])
            reason = "Home team has higher rolling 10-game net rating"
        else:
            predicted_team_id = int(game["AWAY_TEAM_ID"])
            predicted_team_abbreviation = str(game["AWAY_TEAM_ABBREVIATION"])
            reason = "Away team has higher rolling 10-game net rating"

        return make_prediction(
            game=game,
            predictor_name=self.name,
            predicted_team_id=predicted_team_id,
            predicted_team_abbreviation=predicted_team_abbreviation,
            reason=reason,
        )


PREDICTORS: dict[str, type[GamePredictor]] = {
    HomeTeamPredictor.name: HomeTeamPredictor,
    Rolling10NetRatingPredictor.name: Rolling10NetRatingPredictor,
    SeasonToDateNetRatingPredictor.name: SeasonToDateNetRatingPredictor,
    SeasonToDateWinPctPredictor.name: SeasonToDateWinPctPredictor,
}


def load_model_games(season: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / season / "model_games.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed model games not found at {path}")

    return pd.read_parquet(path)


def predict_game_by_id(
    season: str,
    game_id: str,
    predictor: GamePredictor | None = None,
) -> GamePrediction:
    active_predictor = predictor or SeasonToDateNetRatingPredictor()
    model_games = load_model_games(season)
    matches = model_games.loc[model_games["GAME_ID"] == game_id]
    if matches.empty:
        raise ValueError(f"Game {game_id} not found in processed season {season}")
    if len(matches) > 1:
        raise ValueError(f"Expected one processed row for game {game_id}, found {len(matches)}")

    return active_predictor.predict(matches.iloc[0])


def find_game_season(game_id: str) -> str:
    matches = []
    for path in sorted(PROCESSED_DATA_DIR.glob("*/model_games.parquet")):
        model_games = pd.read_parquet(path, columns=["GAME_ID"])
        if model_games["GAME_ID"].eq(game_id).any():
            matches.append(path.parent.name)

    if not matches:
        raise ValueError(f"Game {game_id} not found in processed data")
    if len(matches) > 1:
        seasons = ", ".join(matches)
        raise ValueError(f"Game {game_id} found in multiple processed seasons: {seasons}")

    return matches[0]


def format_prediction(prediction: GamePrediction) -> str:
    if prediction.predicted_team_abbreviation is None:
        predicted = "No prediction"
    else:
        predicted = (
            f"{prediction.predicted_team_abbreviation} "
            f"(team ID {prediction.predicted_team_id})"
        )

    lines = [
        "Game Prediction",
        f"  Predictor: {prediction.predictor_name}",
        f"  Season: {prediction.season}",
        f"  Game ID: {prediction.game_id}",
        (
            "  Matchup: "
            f"{prediction.away_team_abbreviation} at {prediction.home_team_abbreviation}"
        ),
        f"  Predicted winner: {predicted}",
    ]
    if prediction.home_win_probability is not None:
        lines.append(f"  P(home wins): {100 * prediction.home_win_probability:.2f}%")
    lines.append(f"  Reason: {prediction.reason}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the winner of one processed NBA game.")
    parser.add_argument(
        "predictor",
        choices=sorted(PREDICTORS),
        help="Predictor to use.",
    )
    parser.add_argument("game_id", help="NBA game ID to predict.")
    return parser.parse_args()


def main() -> None:
    """Run one game prediction from processed data."""
    args = parse_args()
    predictor = PREDICTORS[args.predictor]()
    try:
        season = find_game_season(args.game_id)
        prediction = predict_game_by_id(season, args.game_id, predictor)
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to predict game: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(format_prediction(prediction))
