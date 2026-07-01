from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from nba_predictor import evaluation
from nba_predictor import logistic_regression as lr


def model_games(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(
        GAME_DATE=lambda data: pd.to_datetime(data["GAME_DATE"])
    )


def train_frame() -> pd.DataFrame:
    return model_games(
        [
            {
                "GAME_ID": "T1",
                "GAME_DATE": "2025-01-01",
                "SEASON": "2024-25",
                "HOME_TEAM_ID": 1,
                "HOME_TEAM_ABBREVIATION": "AAA",
                "AWAY_TEAM_ID": 2,
                "AWAY_TEAM_ABBREVIATION": "BBB",
                "HOME_WIN": 1,
                "DIFF_POINT": 4.0,
                "DIFF_REST": 1.0,
            },
            {
                "GAME_ID": "T2",
                "GAME_DATE": "2025-01-02",
                "SEASON": "2024-25",
                "HOME_TEAM_ID": 3,
                "HOME_TEAM_ABBREVIATION": "CCC",
                "AWAY_TEAM_ID": 4,
                "AWAY_TEAM_ABBREVIATION": "DDD",
                "HOME_WIN": 1,
                "DIFF_POINT": 3.0,
                "DIFF_REST": 0.0,
            },
            {
                "GAME_ID": "T3",
                "GAME_DATE": "2025-01-03",
                "SEASON": "2024-25",
                "HOME_TEAM_ID": 5,
                "HOME_TEAM_ABBREVIATION": "EEE",
                "AWAY_TEAM_ID": 6,
                "AWAY_TEAM_ABBREVIATION": "FFF",
                "HOME_WIN": 0,
                "DIFF_POINT": -4.0,
                "DIFF_REST": -1.0,
            },
            {
                "GAME_ID": "T4",
                "GAME_DATE": "2025-01-04",
                "SEASON": "2024-25",
                "HOME_TEAM_ID": 7,
                "HOME_TEAM_ABBREVIATION": "GGG",
                "AWAY_TEAM_ID": 8,
                "AWAY_TEAM_ABBREVIATION": "HHH",
                "HOME_WIN": 0,
                "DIFF_POINT": -3.0,
                "DIFF_REST": 0.0,
            },
            {
                "GAME_ID": "T5",
                "GAME_DATE": "2025-01-05",
                "SEASON": "2024-25",
                "HOME_TEAM_ID": 9,
                "HOME_TEAM_ABBREVIATION": "III",
                "AWAY_TEAM_ID": 10,
                "AWAY_TEAM_ABBREVIATION": "JJJ",
                "HOME_WIN": 1,
                "DIFF_POINT": None,
                "DIFF_REST": 0.0,
            },
        ]
    )


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
                "DIFF_POINT": 5.0,
                "DIFF_REST": 1.0,
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
                "DIFF_POINT": -5.0,
                "DIFF_REST": -1.0,
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


def split_frame(season: str, start: int = 0) -> pd.DataFrame:
    rows = []
    for offset in range(12):
        home_win = int(offset % 2 == 0)
        strength = 6.0 if home_win else -6.0
        rows.append(
            {
                "GAME_ID": f"{season}-{offset}",
                "GAME_DATE": f"2025-02-{offset + 1:02d}",
                "SEASON": season,
                "HOME_TEAM_ID": start + offset,
                "HOME_TEAM_ABBREVIATION": f"H{offset}",
                "AWAY_TEAM_ID": start + offset + 100,
                "AWAY_TEAM_ABBREVIATION": f"A{offset}",
                "HOME_WIN": home_win,
                "DIFF_POINT": strength,
                "DIFF_REST": float(offset % 3),
            }
        )
    return model_games(rows)


def test_read_feature_file_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "features.txt"
    path.write_text("\n# ignored\nDIFF_POINT\n\nDIFF_REST\n")

    assert lr.read_feature_file(path) == ["DIFF_POINT", "DIFF_REST"]


def test_resolve_feature_columns_rejects_ambiguous_sources(tmp_path: Path) -> None:
    args = Namespace(
        features=["DIFF_POINT"],
        features_file=tmp_path / "features.txt",
    )

    with pytest.raises(ValueError, match="Use either --features or --features-file"):
        lr.resolve_feature_columns(args)


def test_validate_feature_columns_rejects_empty_duplicates_and_missing() -> None:
    frame = train_frame()

    with pytest.raises(ValueError, match="At least one feature"):
        lr.validate_feature_columns(frame, [])
    with pytest.raises(ValueError, match="Duplicate feature"):
        lr.validate_feature_columns(frame, ["DIFF_POINT", "DIFF_POINT"])
    with pytest.raises(ValueError, match="Feature columns not found"):
        lr.validate_feature_columns(frame, ["DOES_NOT_EXIST"])


def test_train_evaluate_predict_and_save_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = {
        "2024-25": train_frame(),
        "2025-26": eval_frame(),
    }
    monkeypatch.setattr(lr, "load_model_games", lambda season: frames[season])
    feature_columns = ["DIFF_POINT", "DIFF_REST"]

    artifact = lr.train_logistic_regression("2024-25", feature_columns)

    assert artifact.train_season == "2024-25"
    assert artifact.feature_columns == feature_columns
    assert artifact.games_available == 5
    assert artifact.games_trained == 4
    assert artifact.games_dropped == 1

    evaluation = lr.evaluate_logistic_regression(
        artifact,
        "2025-26",
        train_seasons=["2024-25"],
    )

    assert evaluation.games_evaluated == 3
    assert evaluation.predictions_made == 2
    assert evaluation.correct_predictions == 2
    assert evaluation.accuracy == pytest.approx(2 / 3)
    assert evaluation.accuracy_when_predicted == 1.0
    assert 0.0 < evaluation.log_loss < 1.0
    assert 0.0 < evaluation.brier_score < 1.0

    predictor = lr.LogisticRegressionPredictor(artifact)
    home_prediction = predictor.predict(frames["2025-26"].iloc[0])
    away_prediction = predictor.predict(frames["2025-26"].iloc[1])
    null_prediction = predictor.predict(frames["2025-26"].iloc[2])

    assert home_prediction.predicted_team_abbreviation == "KKK"
    assert away_prediction.predicted_team_abbreviation == "NNN"
    assert null_prediction.is_null

    model_path = tmp_path / "model.pkl"
    lr.save_model(artifact, model_path)
    loaded_artifact = lr.load_model(model_path)

    assert loaded_artifact.feature_columns == artifact.feature_columns
    assert loaded_artifact.games_trained == artifact.games_trained


def test_format_model_details_sorts_coefficients_by_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lr, "load_model_games", lambda season: train_frame())
    artifact = lr.train_logistic_regression("2024-25", ["DIFF_POINT", "DIFF_REST"])

    report = lr.format_model_details(
        artifact,
        Path("models/test.pkl"),
        "Logistic Regression Model",
    )

    assert "Logistic Regression Model" in report
    assert "  Train season: 2024-25" in report
    assert "  Games available: 5" in report
    assert "  Games trained: 4" in report
    assert "  Games dropped: 1" in report
    assert report.index("DIFF_POINT") < report.index("DIFF_REST")


def test_evaluate_details_report_uses_logistic_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        "2024-25": train_frame(),
        "2025-26": eval_frame(),
    }
    monkeypatch.setattr(lr, "load_model_games", lambda season: frames[season])
    monkeypatch.setattr(evaluation, "load_model_games", lambda season: frames[season])
    artifact = lr.train_logistic_regression("2024-25", ["DIFF_POINT", "DIFF_REST"])

    predictor = lr.LogisticRegressionPredictor(artifact)
    away_prediction = predictor.predict(frames["2025-26"].iloc[1])
    assert away_prediction.home_win_probability is not None
    away_pick_probability = 1 - away_prediction.home_win_probability

    report = lr.format_game_predictions("2025-26", [predictor])

    assert "Detailed Game Predictions" in report
    assert "KKK at LLL" not in report
    assert "LLL at KKK" in report
    assert "KKK ok" in report
    assert f"NNN ok ({100 * away_pick_probability:.1f}%)" in report
    assert "no pick" in report


def test_randomized_evaluation_splits_are_disjoint_and_repeatable() -> None:
    frames = {
        "2024-25": split_frame("2024-25"),
        "2025-26": split_frame("2025-26", start=1000),
    }

    splits = lr.randomized_evaluation_splits(
        ["2024-25", "2025-26"],
        test_size=0.25,
        repeats=2,
        seed=11,
        load_games=lambda season: frames[season],
    )

    assert [split.label for split in splits] == [
        "random split 1/2",
        "random split 2/2",
    ]
    assert [len(split.eval_data) for split in splits] == [6, 6]
    for split in splits:
        train_ids = set(split.train_data["SPLIT_ROW_ID"])
        eval_ids = set(split.eval_data["SPLIT_ROW_ID"])
        assert not train_ids.intersection(eval_ids)
        assert len(train_ids | eval_ids) == 24

    repeated = lr.randomized_evaluation_splits(
        ["2024-25", "2025-26"],
        test_size=0.25,
        repeats=2,
        seed=11,
        load_games=lambda season: frames[season],
    )
    assert (
        splits[0].eval_data["SPLIT_ROW_ID"].tolist()
        == repeated[0].eval_data["SPLIT_ROW_ID"].tolist()
    )


def test_run_ablation_supports_randomized_split_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        "2024-25": split_frame("2024-25"),
        "2025-26": split_frame("2025-26", start=1000),
    }
    monkeypatch.setattr(lr, "load_model_games", lambda season: frames[season])

    baseline, ablations = lr.run_ablation(
        ["2024-25", "2025-26"],
        ["DIFF_POINT", "DIFF_REST"],
        min_train_seasons=1,
        split_strategy="randomized",
        test_size=0.25,
        random_repeats=2,
        random_seed=7,
    )

    assert len(baseline.evaluations) == 2
    assert len(ablations) == 2
    assert {evaluation.eval_season for evaluation in baseline.evaluations} == {
        "random eval 1/2",
        "random eval 2/2",
    }

    report = lr.format_ablation_report(
        ["2024-25", "2025-26"],
        ["DIFF_POINT", "DIFF_REST"],
        baseline,
        ablations,
        split_strategy="randomized",
    )
    assert "Split strategy: randomized" in report
    assert "Splits: 2" in report
