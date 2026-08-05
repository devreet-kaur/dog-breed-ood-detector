"""Tests for OOD strategy comparison utilities."""

import json

import numpy as np
import pytest

from src.ood_compare import (
    build_comparison,
    choose_better_strategy,
    compute_reliability_curve,
    load_metrics,
    load_score_file,
    plot_reliability_comparison,
    save_comparison,
    validate_strategy_metrics,
)


def make_strategy_a_metrics() -> dict[str, float]:
    return {
        "auroc": 0.9772,
        "aupr_in": 0.9922,
        "aupr_out": 0.9446,
        "fpr95": 0.1200,
    }


def make_strategy_b_metrics() -> dict[str, float]:
    return {
        "auroc": 0.8145,
        "aupr_in": 0.9955,
        "aupr_out": 0.2901,
        "fpr95": 0.6813,
    }


def test_load_metrics_reads_json(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    expected = make_strategy_a_metrics()

    metrics_path.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    loaded = load_metrics(metrics_path)

    assert loaded == expected


def test_load_metrics_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_metrics(tmp_path / "missing.json")


def test_validate_strategy_metrics_accepts_required_metrics() -> None:
    validate_strategy_metrics(
        make_strategy_a_metrics(),
        strategy_name="Strategy A",
    )


def test_validate_strategy_metrics_rejects_missing_metric() -> None:
    metrics = make_strategy_a_metrics()
    del metrics["fpr95"]

    with pytest.raises(KeyError, match="fpr95"):
        validate_strategy_metrics(
            metrics,
            strategy_name="Strategy A",
        )


@pytest.mark.parametrize(
    ("metric_name", "strategy_a_value", "strategy_b_value", "expected"),
    [
        ("auroc", 0.9, 0.8, "Strategy A"),
        ("aupr_in", 0.8, 0.9, "Strategy B"),
        ("fpr95", 0.1, 0.5, "Strategy A"),
        ("auroc", 0.8, 0.8, "Tie"),
    ],
)
def test_choose_better_strategy(
    metric_name: str,
    strategy_a_value: float,
    strategy_b_value: float,
    expected: str,
) -> None:
    result = choose_better_strategy(
        metric_name,
        strategy_a_value,
        strategy_b_value,
    )

    assert result == expected


def test_build_comparison_selects_strategy_a_as_winner() -> None:
    comparison = build_comparison(
        strategy_a_metrics=make_strategy_a_metrics(),
        strategy_b_metrics=make_strategy_b_metrics(),
    )

    assert comparison["overall_winner"] == "Strategy A"
    assert comparison["strategy_a_metric_wins"] == 3
    assert comparison["strategy_b_metric_wins"] == 1
    assert (
        comparison["metrics"]["fpr95"]["better_strategy"]
        == "Strategy A"
    )


def test_save_comparison_writes_json(tmp_path) -> None:
    output_path = tmp_path / "reports" / "comparison.json"

    comparison = build_comparison(
        strategy_a_metrics=make_strategy_a_metrics(),
        strategy_b_metrics=make_strategy_b_metrics(),
    )

    save_comparison(comparison, output_path)

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved == comparison
    
    
def test_load_score_file_reads_npz(tmp_path) -> None:
    output_path = tmp_path / "scores.npz"

    np.savez_compressed(
        output_path,
        labels=np.array([0, 1, 0, 1]),
        scores=np.array([0.1, 0.8, 0.2, 0.9]),
    )

    labels, scores = load_score_file(output_path)

    assert labels.tolist() == [0, 1, 0, 1]
    assert np.allclose(scores, [0.1, 0.8, 0.2, 0.9])


def test_compute_reliability_curve_perfect_calibration() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])

    result = compute_reliability_curve(
        labels=labels,
        scores=scores,
        number_of_bins=5,
    )

    assert result["ece"] == pytest.approx(0.0)


def test_compute_reliability_curve_returns_counts() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = compute_reliability_curve(
        labels=labels,
        scores=scores,
        number_of_bins=5,
    )

    assert sum(result["sample_counts"]) == 4


def test_plot_reliability_comparison_creates_file(
    tmp_path,
) -> None:
    output_path = tmp_path / "plots" / "reliability.png"

    result = plot_reliability_comparison(
        strategy_a_labels=np.array([0, 0, 1, 1]),
        strategy_a_scores=np.array([0.1, 0.2, 0.8, 0.9]),
        strategy_b_labels=np.array([0, 0, 1, 1]),
        strategy_b_scores=np.array([0.2, 0.3, 0.7, 0.8]),
        output_path=output_path,
        number_of_bins=5,
    )

    assert output_path.exists()
    assert "strategy_a_ece" in result
    assert "strategy_b_ece" in result