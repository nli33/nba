"""Walk-forward probability diagnostics for logistic regression models.

Collects out-of-fold home-win probabilities across the same rolling season splits
used by the ablation harness, then reports how trustworthy and how decisive those
probabilities are: calibration (are the stated probabilities honest?), selective
prediction (accuracy when only the most-confident games are graded), and upset
structure (where the model's wrong picks fall on the confidence axis).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from nba_predictor.models.logistic import fit_logistic_pipeline
from nba_predictor.models.logistic_ablation import chronological_evaluation_splits
from nba_predictor.prediction import load_model_games


CALIBRATION_BINS = 10
CONFIDENCE_DECILES = 10
UPSET_CONFIDENCE_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)


@dataclass(frozen=True)
class CalibrationBin:
    predicted: float
    observed: float
    count: int


@dataclass(frozen=True)
class ConfidenceDecile:
    index: int
    conf_low: float
    conf_high: float
    count: int
    accuracy: float


@dataclass(frozen=True)
class UpsetThreshold:
    confidence: float
    share_of_games: float
    share_of_upsets: float
    local_upset_rate: float


@dataclass(frozen=True)
class LogisticDiagnostics:
    seasons: list[str]
    n_games: int
    accuracy: float
    home_base_rate: float
    brier_score: float
    log_loss: float
    # Calibration (Murphy decomposition: brier = reliability - resolution + uncertainty).
    ece: float
    reliability: float
    resolution: float
    uncertainty: float
    calibration_bins: list[CalibrationBin]
    # Selective prediction.
    confidence_deciles: list[ConfidenceDecile]
    # Upset structure.
    upset_rate: float
    upset_thresholds: list[UpsetThreshold]
    # Raw per-game material (home-win probability and outcome), kept for plotting.
    probabilities: np.ndarray
    outcomes: np.ndarray


def _collect_walk_forward_probabilities(
    seasons: list[str],
    feature_columns: list[str],
    min_train_seasons: int,
    load_games: Callable[[str], pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    splits = chronological_evaluation_splits(seasons, min_train_seasons, load_games)
    probabilities: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    for index, split in enumerate(splits, start=1):
        print(
            f"[{index}/{len(splits)}] {split.label}",
            file=sys.stderr,
            flush=True,
        )
        train_data = split.train_data.dropna(subset=feature_columns + ["HOME_WIN"])
        eval_data = split.eval_data.dropna(subset=feature_columns + ["HOME_WIN"])
        if train_data.empty or eval_data.empty:
            continue
        pipeline = fit_logistic_pipeline(train_data, feature_columns)
        probabilities.append(
            pipeline.predict_proba(eval_data[feature_columns])[:, 1]
        )
        outcomes.append(eval_data["HOME_WIN"].astype(int).to_numpy())

    if not probabilities:
        raise ValueError("No complete evaluation rows across the walk-forward splits")
    return np.concatenate(probabilities), np.concatenate(outcomes)


def _calibration(
    probabilities: np.ndarray, outcomes: np.ndarray
) -> tuple[list[CalibrationBin], float, float, float, float]:
    n = len(outcomes)
    base_rate = float(outcomes.mean())
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    bin_index = np.clip(np.digitize(probabilities, edges) - 1, 0, CALIBRATION_BINS - 1)

    bins: list[CalibrationBin] = []
    reliability = resolution = ece = 0.0
    for k in range(CALIBRATION_BINS):
        mask = bin_index == k
        count = int(mask.sum())
        if count == 0:
            continue
        predicted = float(probabilities[mask].mean())
        observed = float(outcomes[mask].mean())
        bins.append(CalibrationBin(predicted, observed, count))
        weight = count / n
        reliability += weight * (predicted - observed) ** 2
        resolution += weight * (observed - base_rate) ** 2
        ece += weight * abs(predicted - observed)
    uncertainty = base_rate * (1.0 - base_rate)
    return bins, ece, reliability, resolution, uncertainty


def _confidence_deciles(
    probabilities: np.ndarray, outcomes: np.ndarray
) -> list[ConfidenceDecile]:
    confidence = np.abs(probabilities - 0.5) * 2.0
    correct = ((probabilities >= 0.5).astype(int) == outcomes).astype(int)
    order = np.argsort(confidence, kind="stable")
    groups = np.array_split(order, CONFIDENCE_DECILES)

    deciles: list[ConfidenceDecile] = []
    for index, group in enumerate(groups):
        if len(group) == 0:
            continue
        group_conf = confidence[group]
        deciles.append(
            ConfidenceDecile(
                index=index,
                conf_low=float(group_conf.min()),
                conf_high=float(group_conf.max()),
                count=len(group),
                accuracy=float(correct[group].mean()),
            )
        )
    return deciles


def _upset_structure(
    probabilities: np.ndarray, outcomes: np.ndarray
) -> tuple[float, list[UpsetThreshold]]:
    favorite_confidence = np.maximum(probabilities, 1.0 - probabilities)
    upset = (probabilities >= 0.5).astype(int) != outcomes
    total_upsets = int(upset.sum())

    thresholds: list[UpsetThreshold] = []
    for threshold in UPSET_CONFIDENCE_THRESHOLDS:
        mask = favorite_confidence < threshold
        selected = int(mask.sum())
        thresholds.append(
            UpsetThreshold(
                confidence=threshold,
                share_of_games=float(mask.mean()),
                share_of_upsets=(
                    float(upset[mask].sum() / total_upsets) if total_upsets else 0.0
                ),
                local_upset_rate=float(upset[mask].mean()) if selected else 0.0,
            )
        )
    return float(upset.mean()), thresholds


def run_diagnostics(
    seasons: list[str],
    feature_columns: list[str],
    min_train_seasons: int = 1,
    load_games: Callable[[str], pd.DataFrame] = load_model_games,
) -> LogisticDiagnostics:
    if len(feature_columns) < 1:
        raise ValueError("At least one feature column is required")

    probabilities, outcomes = _collect_walk_forward_probabilities(
        seasons, feature_columns, min_train_seasons, load_games
    )
    predictions = (probabilities >= 0.5).astype(int)
    bins, ece, reliability, resolution, uncertainty = _calibration(
        probabilities, outcomes
    )
    upset_rate, upset_thresholds = _upset_structure(probabilities, outcomes)

    return LogisticDiagnostics(
        seasons=list(seasons),
        n_games=len(outcomes),
        accuracy=float((predictions == outcomes).mean()),
        home_base_rate=float(outcomes.mean()),
        brier_score=float(brier_score_loss(outcomes, probabilities)),
        log_loss=float(log_loss(outcomes, probabilities)),
        ece=ece,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        calibration_bins=bins,
        confidence_deciles=_confidence_deciles(probabilities, outcomes),
        upset_rate=upset_rate,
        upset_thresholds=upset_thresholds,
        probabilities=probabilities,
        outcomes=outcomes,
    )


def format_diagnostics_report(diagnostics: LogisticDiagnostics) -> str:
    lines = [
        "Logistic Regression Walk-Forward Diagnostics",
        f"  Seasons: {', '.join(diagnostics.seasons)}",
        f"  Held-out games: {diagnostics.n_games:,}",
        f"  Accuracy: {100 * diagnostics.accuracy:.2f}% | "
        f"Home base rate: {100 * diagnostics.home_base_rate:.2f}%",
        f"  Log loss: {diagnostics.log_loss:.4f} | Brier: {diagnostics.brier_score:.4f}",
        "",
        "Calibration (are the probabilities honest?)",
        f"  Expected calibration error: {100 * diagnostics.ece:.2f}%",
        f"  Brier = reliability - resolution + uncertainty = "
        f"{diagnostics.reliability:.4f} - {diagnostics.resolution:.4f} + "
        f"{diagnostics.uncertainty:.4f}",
        "  Lower reliability is better (miscalibration); higher resolution is better "
        "(separating power).",
        "",
        "Selective prediction (accuracy by confidence decile, 0 = least, 9 = most)",
    ]
    for decile in diagnostics.confidence_deciles:
        lines.append(
            f"  decile {decile.index}: n={decile.count:>6,} "
            f"conf[{decile.conf_low:.2f}, {decile.conf_high:.2f}] "
            f"acc={100 * decile.accuracy:.2f}%"
        )
    lines.extend(
        [
            "",
            "Upset structure (model's pick loses)",
            f"  Overall upset rate: {100 * diagnostics.upset_rate:.2f}%",
        ]
    )
    for threshold in diagnostics.upset_thresholds:
        lines.append(
            f"  confidence < {threshold.confidence:.2f}: "
            f"{100 * threshold.share_of_games:.1f}% of games, "
            f"{100 * threshold.share_of_upsets:.1f}% of all upsets, "
            f"local upset rate {100 * threshold.local_upset_rate:.1f}%"
        )
    return "\n".join(lines)


def save_diagnostic_plots(diagnostics: LogisticDiagnostics, output_dir: Path) -> list[Path]:
    """Write calibration, selective-prediction, and upset PNG charts.

    Requires the optional ``viz`` extra (matplotlib).
    """
    try:
        import matplotlib  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install the optional extra: "
            "uv pip install -e '.[viz]'"
        ) from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = diagnostics.probabilities
    outcomes = diagnostics.outcomes
    confidence = np.abs(probabilities - 0.5) * 2.0
    correct = ((probabilities >= 0.5).astype(int) == outcomes).astype(int)
    favorite_confidence = np.maximum(probabilities, 1.0 - probabilities)
    upset = correct == 0
    accent, good, bad = "#2563eb", "#059669", "#dc2626"
    paths: list[Path] = []

    # Calibration reliability diagram with a prediction histogram.
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    top.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    top.plot(
        [b.predicted for b in diagnostics.calibration_bins],
        [b.observed for b in diagnostics.calibration_bins],
        "o-",
        color=accent,
        label="model",
    )
    top.set_ylabel("observed home-win rate")
    top.set_title(f"Calibration (ECE={diagnostics.ece:.3f}, N={diagnostics.n_games:,})")
    top.legend()
    top.grid(alpha=0.3)
    bottom.hist(probabilities, bins=20, color="#93c5fd")
    bottom.set_ylabel("games")
    bottom.set_xlabel("predicted P(home win)")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    calibration_path = output_dir / "calibration.png"
    fig.savefig(calibration_path, dpi=120)
    plt.close(fig)
    paths.append(calibration_path)

    # Selective prediction: accuracy versus coverage (most-confident first).
    order = np.argsort(-confidence, kind="stable")
    coverage = np.arange(1, diagnostics.n_games + 1) / diagnostics.n_games
    selective_accuracy = np.cumsum(correct[order]) / np.arange(1, diagnostics.n_games + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverage * 100, selective_accuracy * 100, color=good)
    ax.axhline(
        diagnostics.accuracy * 100,
        ls="--",
        color="gray",
        label=f"all-games acc={100 * diagnostics.accuracy:.1f}%",
    )
    ax.set_xlabel("coverage: % of games predicted (most-confident first)")
    ax.set_ylabel("accuracy on covered games (%)")
    ax.set_title("Selective prediction: accuracy vs coverage")
    ax.set_xlim(5, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    selective_path = output_dir / "selective_prediction.png"
    fig.savefig(selective_path, dpi=120)
    plt.close(fig)
    paths.append(selective_path)

    # Upset structure: confidence density by outcome, and cumulative upset share.
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.hist(
        favorite_confidence[~upset],
        bins=25,
        alpha=0.6,
        color=good,
        density=True,
        label="favorite won",
    )
    left.hist(
        favorite_confidence[upset],
        bins=25,
        alpha=0.6,
        color=bad,
        density=True,
        label="upset (favorite lost)",
    )
    left.set_xlabel("model confidence in its pick  P(favorite)")
    left.set_ylabel("density")
    left.set_title("Where upsets live on the confidence axis")
    left.legend()
    left.grid(alpha=0.3)

    ascending = np.argsort(favorite_confidence, kind="stable")
    total_upsets = int(upset.sum())
    cumulative_upset_share = (
        np.cumsum(upset[ascending]) / total_upsets if total_upsets else np.zeros(diagnostics.n_games)
    )
    cumulative_game_share = np.arange(1, diagnostics.n_games + 1) / diagnostics.n_games
    right.plot(favorite_confidence[ascending], cumulative_upset_share * 100, color=bad)
    right.plot(
        favorite_confidence[ascending],
        cumulative_game_share * 100,
        ":",
        color=accent,
        label="cumulative % of games",
    )
    right.plot([0.5, 1], [0, 100], "--", color="gray", lw=1, label="upsets spread evenly")
    right.set_xlabel("model confidence threshold  P(favorite)")
    right.set_ylabel("cumulative % of all upsets below threshold")
    right.set_title("Upsets concentrate in low-confidence games")
    right.legend()
    right.grid(alpha=0.3)
    fig.tight_layout()
    upset_path = output_dir / "upset_structure.png"
    fig.savefig(upset_path, dpi=120)
    plt.close(fig)
    paths.append(upset_path)

    return paths
