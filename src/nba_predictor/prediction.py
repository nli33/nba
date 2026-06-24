"""Interfaces for making game predictions from processed NBA data."""

from __future__ import annotations

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

    @property
    def is_null(self) -> bool:
        return self.predicted_team_id is None

    def is_correct(self, actual_winner_team_id: int) -> bool:
        return self.predicted_team_id == actual_winner_team_id


class GamePredictor(Protocol):
    name: str

    def predict(self, game: pd.Series) -> GamePrediction:
        """Predict one game from a processed model_games row."""


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
            predicted_team_abbreviation = game["HOME_TEAM_ABBREVIATION"]
            reason = "Home team has higher season-to-date net rating"
        else:
            predicted_team_id = int(game["AWAY_TEAM_ID"])
            predicted_team_abbreviation = game["AWAY_TEAM_ABBREVIATION"]
            reason = "Away team has higher season-to-date net rating"

        return GamePrediction(
            season=str(game["SEASON"]),
            game_id=str(game["GAME_ID"]),
            predictor_name=self.name,
            home_team_id=int(game["HOME_TEAM_ID"]),
            home_team_abbreviation=str(game["HOME_TEAM_ABBREVIATION"]),
            away_team_id=int(game["AWAY_TEAM_ID"]),
            away_team_abbreviation=str(game["AWAY_TEAM_ABBREVIATION"]),
            predicted_team_id=predicted_team_id,
            predicted_team_abbreviation=predicted_team_abbreviation,
            reason=reason,
        )


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
