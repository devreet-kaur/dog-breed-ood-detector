"""Compare entropy and binary-CNN OOD detection strategies."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

COMPARISON_METRICS = (
    "auroc",
    "aupr_in",
    "aupr_out",
    "fpr95",
)

HIGHER_IS_BETTER = {
    "auroc": True,
    "aupr_in": True,
    "aupr_out": True,
    "fpr95": False,
}

DISPLAY_NAMES = {
    "auroc": "AUROC",
    "aupr_in": "AUPR-IN",
    "aupr_out": "AUPR-OUT",
    "fpr95": "FPR@95TPR",
}


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a metrics JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise TypeError("Metrics file must contain a JSON object")

    return metrics


def validate_strategy_metrics(
    metrics: dict[str, Any],
    strategy_name: str,
) -> None:
    """Validate required metrics for one OOD strategy."""
    missing_metrics = set(COMPARISON_METRICS) - set(metrics)

    if missing_metrics:
        missing = ", ".join(sorted(missing_metrics))
        raise KeyError(
            f"{strategy_name} metrics are missing: {missing}"
        )

    for metric_name in COMPARISON_METRICS:
        value = metrics[metric_name]

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{strategy_name} metric '{metric_name}' "
                "must be numeric"
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{strategy_name} metric '{metric_name}' "
                "must be between 0 and 1"
            )


def choose_better_strategy(
    metric_name: str,
    strategy_a_value: float,
    strategy_b_value: float,
) -> str:
    """Return the better strategy for one metric."""
    if metric_name not in HIGHER_IS_BETTER:
        raise KeyError(f"Unsupported comparison metric: {metric_name}")

    if strategy_a_value == strategy_b_value:
        return "Tie"

    if HIGHER_IS_BETTER[metric_name]:
        return (
            "Strategy A"
            if strategy_a_value > strategy_b_value
            else "Strategy B"
        )

    return (
        "Strategy A"
        if strategy_a_value < strategy_b_value
        else "Strategy B"
    )


def build_comparison(
    strategy_a_metrics: dict[str, Any],
    strategy_b_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured Strategy A versus Strategy B comparison."""
    validate_strategy_metrics(
        strategy_a_metrics,
        strategy_name="Strategy A",
    )
    validate_strategy_metrics(
        strategy_b_metrics,
        strategy_name="Strategy B",
    )

    metric_comparison: dict[str, dict[str, float | str]] = {}
    strategy_a_wins = 0
    strategy_b_wins = 0

    for metric_name in COMPARISON_METRICS:
        strategy_a_value = float(strategy_a_metrics[metric_name])
        strategy_b_value = float(strategy_b_metrics[metric_name])

        better_strategy = choose_better_strategy(
            metric_name=metric_name,
            strategy_a_value=strategy_a_value,
            strategy_b_value=strategy_b_value,
        )

        if better_strategy == "Strategy A":
            strategy_a_wins += 1
        elif better_strategy == "Strategy B":
            strategy_b_wins += 1

        metric_comparison[metric_name] = {
            "display_name": DISPLAY_NAMES[metric_name],
            "strategy_a": strategy_a_value,
            "strategy_b": strategy_b_value,
            "better_strategy": better_strategy,
        }

    if strategy_a_wins > strategy_b_wins:
        overall_winner = "Strategy A"
    elif strategy_b_wins > strategy_a_wins:
        overall_winner = "Strategy B"
    else:
        overall_winner = "Tie"

    return {
        "strategy_a_name": "Predictive Entropy",
        "strategy_b_name": "Binary CNN",
        "metrics": metric_comparison,
        "strategy_a_metric_wins": strategy_a_wins,
        "strategy_b_metric_wins": strategy_b_wins,
        "overall_winner": overall_winner,
    }


def save_comparison(
    comparison: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the strategy comparison as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
  
    
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for strategy comparison."""
    parser = argparse.ArgumentParser(
        description="Compare entropy and binary-CNN OOD strategies."
    )

    parser.add_argument(
        "--strategy-a-metrics",
        type=Path,
        default=Path("reports/ood/strategy_a_metrics.json"),
        help="Path to Strategy A metrics JSON.",
    )

    parser.add_argument(
        "--strategy-b-metrics",
        type=Path,
        default=Path(
            "reports/ood/strategy_b_test_metrics.json"
        ),
        help="Path to Strategy B metrics JSON.",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/ood/strategy_comparison.json"),
        help="Path for comparison JSON.",
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/ood/strategy_comparison.csv"),
        help="Path for comparison CSV.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/ood/comparison_summary.txt"),
        help="Path for written comparison summary.",
    )

    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("reports/ood/strategy_comparison.png"),
        help="Path for comparison chart.",
    )
    
    parser.add_argument(
        "--strategy-a-scores",
        type=Path,
        default=Path("reports/ood/strategy_a_test_scores.npz"),
        help="Path to Strategy A labels and normalized OOD scores.",
    )

    parser.add_argument(
        "--strategy-b-scores",
        type=Path,
        default=Path("reports/ood/strategy_b_test_scores.npz"),
        help="Path to Strategy B labels and OOD probabilities.",
    )

    parser.add_argument(
        "--reliability-output",
        type=Path,
        default=Path("reports/ood/reliability_comparison.png"),
        help="Path for the reliability-curve comparison plot.",
    )

    parser.add_argument(
        "--reliability-bins",
        type=int,
        default=10,
        help="Number of bins used for reliability curves.",
    )

    return parser.parse_args()


def save_comparison_csv(
    comparison: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the strategy comparison as a CSV table."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Metric",
                "Strategy A",
                "Strategy B",
                "Better Strategy",
            ]
        )

        for metric_name in COMPARISON_METRICS:
            result = comparison["metrics"][metric_name]

            writer.writerow(
                [
                    result["display_name"],
                    result["strategy_a"],
                    result["strategy_b"],
                    result["better_strategy"],
                ]
            )
            
            
def save_summary_text(
    comparison: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save a human-readable strategy comparison summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "OOD Detection Strategy Comparison",
        "=" * 40,
        "",
    ]

    for metric_name in COMPARISON_METRICS:
        result = comparison["metrics"][metric_name]

        lines.extend(
            [
                result["display_name"],
                f"  Strategy A: {result['strategy_a']:.4f}",
                f"  Strategy B: {result['strategy_b']:.4f}",
                f"  Better:     {result['better_strategy']}",
                "",
            ]
        )

    lines.extend(
        [
            (
                "Strategy A metric wins: "
                f"{comparison['strategy_a_metric_wins']}"
            ),
            (
                "Strategy B metric wins: "
                f"{comparison['strategy_b_metric_wins']}"
            ),
            f"Overall winner: {comparison['overall_winner']}",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    
    
def plot_metric_comparison(
    comparison: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Create a grouped bar chart comparing both OOD strategies."""
    labels: list[str] = []
    strategy_a_values: list[float] = []
    strategy_b_values: list[float] = []

    for metric_name in COMPARISON_METRICS:
        result = comparison["metrics"][metric_name]

        labels.append(str(result["display_name"]))
        strategy_a_values.append(float(result["strategy_a"]))
        strategy_b_values.append(float(result["strategy_b"]))

    positions = np.arange(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        positions - width / 2,
        strategy_a_values,
        width,
        label="Strategy A — Predictive Entropy",
    )
    axis.bar(
        positions + width / 2,
        strategy_b_values,
        width,
        label="Strategy B — Binary CNN",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Metric Value")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("OOD Detection Strategy Comparison")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    
    figure.text(
        0.5,
        0.01,
        "Higher is better except FPR@95TPR, where lower is better.",
        ha="center",
    )

    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
    

    
def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Load, compare, and save Strategy A and Strategy B results."""
    strategy_a_metrics = load_metrics(args.strategy_a_metrics)
    strategy_b_metrics = load_metrics(args.strategy_b_metrics)

    strategy_a_labels, strategy_a_scores = load_score_file(
        args.strategy_a_scores
    )

    strategy_b_labels, strategy_b_scores = load_score_file(
        args.strategy_b_scores
    )

    comparison = build_comparison(
        strategy_a_metrics=strategy_a_metrics,
        strategy_b_metrics=strategy_b_metrics,
    )

    save_comparison(
        comparison=comparison,
        output_path=args.json_output,
    )

    save_comparison_csv(
        comparison=comparison,
        output_path=args.csv_output,
    )

    save_summary_text(
        comparison=comparison,
        output_path=args.summary_output,
    )

    plot_metric_comparison(
        comparison=comparison,
        output_path=args.plot_output,
    )
    
    calibration_metrics = plot_reliability_comparison(
        strategy_a_labels=strategy_a_labels,
        strategy_a_scores=strategy_a_scores,
        strategy_b_labels=strategy_b_labels,
        strategy_b_scores=strategy_b_scores,
        output_path=args.reliability_output,
        number_of_bins=args.reliability_bins,
    )

    comparison["calibration"] = calibration_metrics

    save_comparison(
        comparison=comparison,
        output_path=args.json_output,
    )

    print("OOD strategy comparison complete")
    print(f"Overall winner: {comparison['overall_winner']}")
    print(f"JSON:    {args.json_output}")
    print(f"CSV:     {args.csv_output}")
    print(f"Summary: {args.summary_output}")
    print(f"Plot:    {args.plot_output}")
    print(
        "Reliability: "
        f"{args.reliability_output}"
    )
    print(
        "Strategy A ECE: "
        f"{calibration_metrics['strategy_a_ece']:.4f}"
    )
    print(
        "Strategy B ECE: "
        f"{calibration_metrics['strategy_b_ece']:.4f}"
    )

    return comparison


def load_score_file(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load binary labels and OOD confidence scores from an NPZ file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Score file not found: {path}")

    data = np.load(path)

    if "labels" not in data or "scores" not in data:
        raise KeyError(
            "Score file must contain 'labels' and 'scores' arrays"
        )

    labels = np.asarray(data["labels"])
    scores = np.asarray(data["scores"], dtype=float)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")

    if labels.size != scores.size:
        raise ValueError("labels and scores must have equal length")

    if labels.size == 0:
        raise ValueError("score file must not be empty")

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values")

    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("scores must be between 0 and 1")

    return labels.astype(int), scores


def compute_reliability_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    number_of_bins: int = 10,
) -> dict[str, list[float] | float]:
    """Compute reliability bins and expected calibration error."""
    if number_of_bins <= 1:
        raise ValueError("number_of_bins must be greater than one")

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")

    if labels.size != scores.size:
        raise ValueError("labels and scores must have equal length")

    if labels.size == 0:
        raise ValueError("labels and scores must not be empty")

    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)

    mean_confidences: list[float] = []
    observed_frequencies: list[float] = []
    sample_counts: list[int] = []

    expected_calibration_error = 0.0

    for bin_index in range(number_of_bins):
        lower_bound = bin_edges[bin_index]
        upper_bound = bin_edges[bin_index + 1]

        if bin_index == number_of_bins - 1:
            in_bin = (
                (scores >= lower_bound)
                & (scores <= upper_bound)
            )
        else:
            in_bin = (
                (scores >= lower_bound)
                & (scores < upper_bound)
            )

        count = int(in_bin.sum())

        if count == 0:
            continue

        mean_confidence = float(scores[in_bin].mean())
        observed_frequency = float(labels[in_bin].mean())

        mean_confidences.append(mean_confidence)
        observed_frequencies.append(observed_frequency)
        sample_counts.append(count)

        expected_calibration_error += (
            count / labels.size
        ) * abs(mean_confidence - observed_frequency)

    return {
        "mean_confidences": mean_confidences,
        "observed_frequencies": observed_frequencies,
        "sample_counts": sample_counts,
        "ece": float(expected_calibration_error),
    }


def plot_reliability_comparison(
    strategy_a_labels: np.ndarray,
    strategy_a_scores: np.ndarray,
    strategy_b_labels: np.ndarray,
    strategy_b_scores: np.ndarray,
    output_path: str | Path,
    number_of_bins: int = 10,
) -> dict[str, float]:
    """Plot Strategy A and Strategy B reliability curves."""
    strategy_a_curve = compute_reliability_curve(
        labels=strategy_a_labels,
        scores=strategy_a_scores,
        number_of_bins=number_of_bins,
    )

    strategy_b_curve = compute_reliability_curve(
        labels=strategy_b_labels,
        scores=strategy_b_scores,
        number_of_bins=number_of_bins,
    )

    figure, axis = plt.subplots(figsize=(8, 7))

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.plot(
        strategy_a_curve["mean_confidences"],
        strategy_a_curve["observed_frequencies"],
        marker="o",
        label=(
            "Strategy A — Predictive Entropy "
            f"(ECE={strategy_a_curve['ece']:.4f})"
        ),
    )

    axis.plot(
        strategy_b_curve["mean_confidences"],
        strategy_b_curve["observed_frequencies"],
        marker="s",
        label=(
            "Strategy B — Binary CNN "
            f"(ECE={strategy_b_curve['ece']:.4f})"
        ),
    )

    axis.set_xlabel("Mean predicted OOD confidence")
    axis.set_ylabel("Observed OOD frequency")
    axis.set_title("OOD Reliability Curves")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return {
        "strategy_a_ece": float(strategy_a_curve["ece"]),
        "strategy_b_ece": float(strategy_b_curve["ece"]),
    }


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()