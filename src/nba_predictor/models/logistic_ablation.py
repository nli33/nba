"""Drop-one-feature ablation for logistic regression models."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nba_predictor.models.logistic import (
    LogisticEvaluation,
    evaluate_logistic_regression,
    train_logistic_regression_for_seasons,
)
from nba_predictor.prediction import load_model_games


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
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
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
                load_games=load_games,
            )
            evaluations.append(
                evaluate_logistic_regression(
                    artifact,
                    eval_season,
                    train_seasons,
                    load_games=load_games,
                )
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
