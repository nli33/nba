from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_predictor import evaluation
from nba_predictor.models import random_forest as rf


def model_games(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(
        GAME_DATE=lambda data: pd.to_datetime(data["GAME_DATE"])
    )


def train_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset in range(20):
        home_win = int(offset % 2 == 0)
        strength = 8.0 if home_win else -8.0
        rows.append(
            {
                "GAME_ID": f"T{offset}",
                "GAME_DATE": f"2025-01-{offset + 1:02d}",
                "SEASON": "2024-25",
                "HOME_TEAM_ID": offset,
                "HOME_TEAM_ABBREVIATION": f"H{offset}",
                "AWAY_TEAM_ID": offset + 100,
                "AWAY_TEAM_ABBREVIATION": f"A{offset}",
                "HOME_WIN": home_win,
                "DIFF_POINT": strength,
                "DIFF_REST": 0.0,
            }
        )
    rows.append(
        {
            "GAME_ID": "T_MISSING",
            "GAME_DATE": "2025-01-21",
            "SEASON": "2024-25",
            "HOME_TEAM_ID": 21,
            "HOME_TEAM_ABBREVIATION": "HM",
            "AWAY_TEAM_ID": 121,
            "AWAY_TEAM_ABBREVIATION": "AM",
            "HOME_WIN": 1,
            "DIFF_POINT": None,
            "DIFF_REST": 0.0,
        }
    )
    return model_games(rows)


def eval_frame() -> pd.DataFrame:
    return model_games(
        [
            {
                "GAME_ID": "E1",
                "GAME_DATE": "2026-01-01",
                "SEASON": "2025-26",
                "HOME_TEAM_ID": 11,
                "HOME_TEAM_ABBREVIATION": "KKK",
                "AWAY_TEAM_ID": 12,
                "AWAY_TEAM_ABBREVIATION": "LLL",
                "HOME_WIN": 1,
                "DIFF_POINT": 8.0,
                "DIFF_REST": 0.0,
            },
            {
                "GAME_ID": "E2",
                "GAME_DATE": "2026-01-02",
                "SEASON": "2025-26",
                "HOME_TEAM_ID": 13,
                "HOME_TEAM_ABBREVIATION": "MMM",
                "AWAY_TEAM_ID": 14,
                "AWAY_TEAM_ABBREVIATION": "NNN",
                "HOME_WIN": 0,
                "DIFF_POINT": -8.0,
                "DIFF_REST": 0.0,
            },
            {
                "GAME_ID": "E3",
                "GAME_DATE": "2026-01-03",
                "SEASON": "2025-26",
                "HOME_TEAM_ID": 15,
                "HOME_TEAM_ABBREVIATION": "OOO",
                "AWAY_TEAM_ID": 16,
                "AWAY_TEAM_ABBREVIATION": "PPP",
                "HOME_WIN": 1,
                "DIFF_POINT": None,
                "DIFF_REST": 0.0,
            },
        ]
    )


def test_train_evaluate_predict_and_save_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = {
        "2024-25": train_frame(),
        "2025-26": eval_frame(),
    }
    monkeypatch.setattr(rf, "load_model_games", lambda season: frames[season])
    feature_columns = ["DIFF_POINT", "DIFF_REST"]

    artifact = rf.train_random_forest("2024-25", feature_columns)

    assert artifact.train_season == "2024-25"
    assert artifact.feature_columns == feature_columns
    assert artifact.games_available == 21
    assert artifact.games_trained == 20
    assert artifact.games_dropped == 1

    evaluation_result = rf.evaluate_random_forest(
        artifact,
        "2025-26",
        train_seasons=["2024-25"],
    )

    assert evaluation_result.games_evaluated == 3
    assert evaluation_result.predictions_made == 2
    assert evaluation_result.correct_predictions == 2
    assert evaluation_result.accuracy == pytest.approx(2 / 3)
    assert evaluation_result.accuracy_when_predicted == 1.0
    assert 0.0 < evaluation_result.log_loss < 1.0
    assert 0.0 <= evaluation_result.brier_score < 1.0

    predictor = rf.RandomForestPredictor(artifact)
    home_prediction = predictor.predict(frames["2025-26"].iloc[0])
    away_prediction = predictor.predict(frames["2025-26"].iloc[1])
    null_prediction = predictor.predict(frames["2025-26"].iloc[2])

    assert home_prediction.predicted_team_abbreviation == "KKK"
    assert away_prediction.predicted_team_abbreviation == "NNN"
    assert null_prediction.is_null

    model_path = tmp_path / "model.pkl"
    rf.save_model(artifact, model_path)
    loaded_artifact = rf.load_model(model_path)

    assert loaded_artifact.feature_columns == artifact.feature_columns
    assert loaded_artifact.games_trained == artifact.games_trained


def test_format_model_details_sorts_feature_importances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rf, "load_model_games", lambda season: train_frame())
    artifact = rf.train_random_forest("2024-25", ["DIFF_POINT", "DIFF_REST"])

    report = rf.format_model_details(
        artifact,
        Path("models/test.pkl"),
        "Random Forest Model",
    )

    assert "Random Forest Model" in report
    assert "  Train season: 2024-25" in report
    assert "  Games available: 21" in report
    assert "  Games trained: 20" in report
    assert "  Games dropped: 1" in report
    assert "Feature importances" in report
    assert report.index("DIFF_POINT") < report.index("DIFF_REST")


def test_evaluate_details_report_uses_random_forest_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        "2024-25": train_frame(),
        "2025-26": eval_frame(),
    }
    monkeypatch.setattr(rf, "load_model_games", lambda season: frames[season])
    monkeypatch.setattr(evaluation, "load_model_games", lambda season: frames[season])
    artifact = rf.train_random_forest("2024-25", ["DIFF_POINT", "DIFF_REST"])

    predictor = rf.RandomForestPredictor(artifact)
    away_prediction = predictor.predict(frames["2025-26"].iloc[1])
    assert away_prediction.home_win_probability is not None
    away_pick_probability = 1 - away_prediction.home_win_probability

    report = evaluation.format_game_predictions("2025-26", [predictor])

    assert "Detailed Game Predictions" in report
    assert "LLL at KKK" in report
    assert "KKK ok" in report
    assert f"NNN ok ({100 * away_pick_probability:.1f}%)" in report
    assert "no pick" in report
