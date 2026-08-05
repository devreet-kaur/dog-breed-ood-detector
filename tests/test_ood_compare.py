"""Tests for OOD strategy comparison utilities."""

import json

import pytest

from src.ood_compare import (
    build_comparison,
    choose_better_strategy,
    load_metrics,
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