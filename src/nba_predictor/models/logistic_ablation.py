"""Drop-one-feature ablation for logistic regression models."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nba_predictor.models.logistic import (
    LogisticEvaluation,
    evaluate_logistic_regression_on_frame,
    train_logistic_regression_from_frame,
)
from nba_predictor.prediction import load_model_games


SPLIT_STRATEGIES = ("chronological", "randomized")


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
        return sum(
            evaluation.correct_predictions for evaluation in self.evaluations
        ) / len(self.evaluations)

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


@dataclass(frozen=True)
class EvaluationSplit:
    label: str
    train_label: str
    eval_label: str
    train_data: pd.DataFrame
    eval_data: pd.DataFrame


def rolling_splits(
    seasons: list[str], min_train_seasons: int
) -> list[tuple[list[str], str]]:
    if min_train_seasons < 1:
        raise ValueError("--min-train-seasons must be at least 1")
    if len(seasons) <= min_train_seasons:
        raise ValueError("Need more seasons than --min-train-seasons to ablate")

    return [
        (seasons[:index], seasons[index])
        for index in range(min_train_seasons, len(seasons))
    ]


def load_combined_games(
    seasons: list[str],
    load_games: Callable[[str], pd.DataFrame],
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        frame = load_games(season).copy()
        frame["SPLIT_SOURCE_SEASON"] = season
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["SPLIT_ROW_ID"] = range(len(combined))
    return combined


def chronological_evaluation_splits(
    seasons: list[str],
    min_train_seasons: int,
    load_games: Callable[[str], pd.DataFrame],
) -> list[EvaluationSplit]:
    return [
        EvaluationSplit(
            label=f"train {', '.join(train_seasons)} -> eval {eval_season}",
            train_label=", ".join(train_seasons),
            eval_label=eval_season,
            train_data=pd.concat(
                [load_games(season) for season in train_seasons],
                ignore_index=True,
            ),
            eval_data=load_games(eval_season),
        )
        for train_seasons, eval_season in rolling_splits(seasons, min_train_seasons)
    ]


def randomized_evaluation_splits(
    seasons: list[str],
    test_size: float,
    repeats: int,
    seed: int,
    load_games: Callable[[str], pd.DataFrame],
) -> list[EvaluationSplit]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1")
    if repeats < 1:
        raise ValueError("--random-repeats must be at least 1")

    combined = load_combined_games(seasons, load_games)
    if len(combined) < 2:
        raise ValueError("Need at least two rows for randomized train/eval splits")

    test_count = round(len(combined) * test_size)
    test_count = min(max(1, test_count), len(combined) - 1)
    splits = []
    for repeat_index in range(repeats):
        shuffled = combined.sample(
            frac=1,
            random_state=seed + repeat_index,
        )
        eval_data = shuffled.iloc[:test_count].sort_index()
        train_data = shuffled.iloc[test_count:].sort_index()
        if set(train_data.index).intersection(eval_data.index):
            raise ValueError("Randomized train/eval split produced overlapping rows")
        split_number = repeat_index + 1
        splits.append(
            EvaluationSplit(
                label=f"random split {split_number}/{repeats}",
                train_label=f"random train {split_number}/{repeats}",
                eval_label=f"random eval {split_number}/{repeats}",
                train_data=train_data.reset_index(drop=True),
                eval_data=eval_data.reset_index(drop=True),
            )
        )
    return splits


def evaluation_splits(
    seasons: list[str],
    min_train_seasons: int,
    split_strategy: str,
    test_size: float,
    random_repeats: int,
    random_seed: int,
    load_games: Callable[[str], pd.DataFrame],
) -> list[EvaluationSplit]:
    if split_strategy == "chronological":
        return chronological_evaluation_splits(seasons, min_train_seasons, load_games)
    if split_strategy == "randomized":
        return randomized_evaluation_splits(
            seasons,
            test_size,
            random_repeats,
            random_seed,
            load_games,
        )
    raise ValueError(f"Unknown split strategy: {split_strategy}")


def run_ablation(
    seasons: list[str],
    feature_columns: list[str],
    min_train_seasons: int,
    split_strategy: str = "chronological",
    test_size: float = 0.2,
    random_repeats: int = 1,
    random_seed: int = 0,
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> tuple[AblationResult, list[AblationResult]]:
    if len(feature_columns) < 2:
        raise ValueError("At least two feature columns are required for ablation")

    splits = evaluation_splits(
        seasons,
        min_train_seasons,
        split_strategy,
        test_size,
        random_repeats,
        random_seed,
        load_games,
    )
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
        for split in splits:
            run_number += 1
            print(
                f"[{run_number}/{total_runs}] {label}; {split.label}",
                file=sys.stderr,
                flush=True,
            )
            artifact = train_logistic_regression_from_frame(
                split.train_data,
                split.train_label,
                active_features,
            )
            evaluations.append(
                evaluate_logistic_regression_on_frame(
                    artifact,
                    split.eval_data,
                    split.eval_label,
                    [split.train_label],
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
    split_strategy: str = "chronological",
) -> str:
    sorted_ablations = sorted(
        ablations,
        key=lambda result: (
            result.avg_correct_predictions - baseline.avg_correct_predictions
        ),
    )
    lines = [
        "Logistic Regression Feature Ablation",
        f"  Seasons: {', '.join(seasons)}",
        f"  Split strategy: {split_strategy}",
        f"  Active features: {len(feature_columns)}",
        f"  Splits: {len(baseline.evaluations)}",
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
        correct_delta = (
            result.avg_correct_predictions - baseline.avg_correct_predictions
        )
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
