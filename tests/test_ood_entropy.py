"""Tests for entropy-based OOD utilities."""

import json
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ood_entropy import (
    calibrate_threshold,
    classify_ood,
    collect_entropy_scores,
    compute_entropy,
    compute_fpr_at_tpr,
    compute_ood_metrics,
    load_config,
    plot_entropy_distribution,
    save_entropy_scores,
    save_metrics,
    save_threshold,
    validate_ood_scores,
)


class TinyClassifier(nn.Module):
    """Small deterministic classifier used for entropy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 3, bias=False)

        with torch.no_grad():
            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


def test_uniform_logits_have_maximum_entropy() -> None:
    logits = torch.zeros((2, 4))

    entropy = compute_entropy(logits)

    expected = torch.full((2,), math.log(4))
    assert torch.allclose(entropy, expected, atol=1e-6)


def test_confident_logits_have_low_entropy() -> None:
    logits = torch.tensor([[20.0, -10.0, -10.0]])

    entropy = compute_entropy(logits)

    assert entropy.shape == (1,)
    assert entropy.item() < 1e-4


def test_higher_temperature_increases_entropy() -> None:
    logits = torch.tensor([[5.0, 1.0, -2.0]])

    low_temperature = compute_entropy(logits, temperature=0.5)
    high_temperature = compute_entropy(logits, temperature=2.0)

    assert high_temperature.item() > low_temperature.item()


def test_entropy_returns_one_score_per_sample() -> None:
    logits = torch.randn(8, 120)

    entropy = compute_entropy(logits)

    assert entropy.shape == (8,)
    assert torch.isfinite(entropy).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_invalid_temperature_raises_error(temperature: float) -> None:
    logits = torch.randn(2, 3)

    with pytest.raises(ValueError, match="greater than zero"):
        compute_entropy(logits, temperature=temperature)


def test_invalid_logits_shape_raises_error() -> None:
    logits = torch.randn(3)

    with pytest.raises(ValueError, match="shape"):
        compute_entropy(logits)


def test_classify_ood_uses_strict_threshold() -> None:
    scores = torch.tensor([0.2, 0.5, 0.8])

    predictions = classify_ood(scores, threshold=0.5)

    expected = torch.tensor([False, False, True])
    assert torch.equal(predictions, expected)
    

def test_calibrate_threshold_uses_target_quantile() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    threshold = calibrate_threshold(scores, target_tpr=0.8)

    expected = torch.quantile(scores, 0.8).item()
    assert threshold == pytest.approx(expected)


def test_calibrated_threshold_accepts_expected_fraction_of_id_samples() -> None:
    scores = torch.linspace(0.0, 1.0, steps=101)

    threshold = calibrate_threshold(scores, target_tpr=0.95)
    accepted_fraction = (scores <= threshold).float().mean().item()

    assert accepted_fraction == pytest.approx(0.95, abs=0.02)


@pytest.mark.parametrize("target_tpr", [0.0, 1.0, -0.1, 1.1])
def test_invalid_target_tpr_raises_error(target_tpr: float) -> None:
    scores = torch.tensor([0.1, 0.2])

    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_threshold(scores, target_tpr=target_tpr)


def test_empty_entropy_scores_raise_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_threshold(torch.tensor([]))


def test_nonfinite_entropy_scores_raise_error() -> None:
    scores = torch.tensor([0.1, float("nan")])

    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold(scores)


def test_save_threshold_writes_expected_json(tmp_path) -> None:
    output_path = tmp_path / "entropy_threshold.json"

    save_threshold(
        threshold=1.234,
        output_path=output_path,
        target_tpr=0.95,
        num_validation_samples=3084,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == {
        "threshold": 1.234,
        "target_tpr": 0.95,
        "num_validation_samples": 3084,
    }    
    

def test_collect_entropy_scores_returns_one_score_per_sample() -> None:
    features = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    labels = torch.zeros(5, dtype=torch.long)

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=2,
        shuffle=False,
    )

    model = TinyClassifier()

    scores = collect_entropy_scores(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        temperature=1.0,
    )

    assert scores.shape == (5,)
    assert scores.device.type == "cpu"
    assert torch.isfinite(scores).all()


def test_collect_entropy_scores_matches_direct_computation() -> None:
    features = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )

    dataloader = DataLoader(features, batch_size=2)
    model = TinyClassifier()

    collected = collect_entropy_scores(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    expected_logits = model(features)
    expected = compute_entropy(expected_logits)

    assert torch.allclose(collected, expected)


def test_collect_entropy_scores_sets_model_to_evaluation_mode() -> None:
    features = torch.randn(3, 4)
    dataloader = DataLoader(features, batch_size=2)

    model = TinyClassifier()
    model.train()

    collect_entropy_scores(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert model.training is False


def test_collect_entropy_scores_rejects_empty_dataloader() -> None:
    features = torch.empty((0, 4))
    dataloader = DataLoader(features, batch_size=2)

    with pytest.raises(ValueError, match="did not provide any samples"):
        collect_entropy_scores(
            model=TinyClassifier(),
            dataloader=dataloader,
            device=torch.device("cpu"),
        )


def test_collect_entropy_scores_rejects_unsupported_batch() -> None:
    invalid_dataloader = [{"images": torch.randn(2, 4)}]

    with pytest.raises(TypeError, match="batches must be tensors"):
        collect_entropy_scores(
            model=TinyClassifier(),
            dataloader=invalid_dataloader,
            device=torch.device("cpu"),
        )
        

def test_perfectly_separated_scores_have_perfect_metrics() -> None:
    id_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ood_scores = torch.tensor([0.8, 0.9, 1.0, 1.1])

    metrics = compute_ood_metrics(
        id_scores=id_scores,
        ood_scores=ood_scores,
        target_tpr=0.95,
    )

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["aupr_in"] == pytest.approx(1.0)
    assert metrics["aupr_out"] == pytest.approx(1.0)
    assert metrics["fpr95"] == pytest.approx(0.0)
    assert metrics["num_id_samples"] == 4
    assert metrics["num_ood_samples"] == 4


def test_reversed_scores_have_poor_auroc() -> None:
    id_scores = torch.tensor([0.8, 0.9, 1.0])
    ood_scores = torch.tensor([0.1, 0.2, 0.3])

    metrics = compute_ood_metrics(id_scores, ood_scores)

    assert metrics["auroc"] == pytest.approx(0.0)


def test_identical_score_distributions_have_chance_auroc() -> None:
    id_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ood_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])

    metrics = compute_ood_metrics(id_scores, ood_scores)

    assert metrics["auroc"] == pytest.approx(0.5)


def test_validate_ood_scores_returns_numpy_arrays() -> None:
    id_scores = torch.tensor([0.1, 0.2])
    ood_scores = torch.tensor([0.8, 0.9])

    id_array, ood_array = validate_ood_scores(id_scores, ood_scores)

    assert isinstance(id_array, np.ndarray)
    assert isinstance(ood_array, np.ndarray)
    assert id_array.dtype == np.float64
    assert ood_array.dtype == np.float64


@pytest.mark.parametrize(
    ("id_scores", "ood_scores", "message"),
    [
        (torch.tensor([]), torch.tensor([0.5]), "id_scores must not be empty"),
        (torch.tensor([0.1]), torch.tensor([]), "ood_scores must not be empty"),
        (
            torch.tensor([[0.1, 0.2]]),
            torch.tensor([0.5]),
            "id_scores must be one-dimensional",
        ),
        (
            torch.tensor([0.1]),
            torch.tensor([float("inf")]),
            "ood_scores must contain only finite values",
        ),
    ],
)
def test_validate_ood_scores_rejects_invalid_inputs(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_ood_scores(id_scores, ood_scores)


def test_compute_fpr_at_tpr_for_perfect_separation() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])

    fpr = compute_fpr_at_tpr(
        labels=labels,
        scores=scores,
        target_tpr=0.95,
    )

    assert fpr == pytest.approx(0.0)


@pytest.mark.parametrize("target_tpr", [0.0, 1.0, -0.2, 1.2])
def test_compute_fpr_rejects_invalid_target_tpr(
    target_tpr: float,
) -> None:
    labels = np.array([0, 1])
    scores = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_fpr_at_tpr(labels, scores, target_tpr)


def test_save_metrics_writes_json(tmp_path) -> None:
    output_path = tmp_path / "strategy_a_metrics.json"

    metrics = {
        "auroc": 0.91,
        "fpr95": 0.22,
        "aupr_in": 0.89,
        "aupr_out": 0.93,
    }

    save_metrics(metrics, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == metrics
    
    
def test_load_config_reads_entropy_parameters(tmp_path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        """
ood_entropy:
  temperature: 1.5
  target_tpr: 0.95
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["ood_entropy"]["temperature"] == 1.5
    assert config["ood_entropy"]["target_tpr"] == 0.95


def test_load_config_rejects_missing_file(tmp_path) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(config_path)


def test_load_config_rejects_missing_entropy_section(tmp_path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        "data:\n  img_size: 224\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="ood_entropy"):
        load_config(config_path)


def test_load_config_rejects_missing_entropy_key(tmp_path) -> None:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        """
ood_entropy:
  temperature: 1.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="target_tpr"):
        load_config(config_path)
        
        
def test_plot_entropy_distribution_creates_file(tmp_path) -> None:
    id_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ood_scores = torch.tensor([0.8, 0.9, 1.0, 1.1])
    output_path = tmp_path / "entropy_distribution.png"

    plot_entropy_distribution(
        id_scores=id_scores,
        ood_scores=ood_scores,
        threshold=0.6,
        output_path=output_path,
        bins=10,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_entropy_distribution_creates_parent_directory(
    tmp_path,
) -> None:
    output_path = tmp_path / "nested" / "plots" / "entropy.png"

    plot_entropy_distribution(
        id_scores=torch.tensor([0.1, 0.2]),
        ood_scores=torch.tensor([0.8, 0.9]),
        threshold=0.5,
        output_path=output_path,
    )

    assert output_path.exists()


def test_plot_entropy_distribution_rejects_invalid_bins(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="bins must be greater than zero"):
        plot_entropy_distribution(
            id_scores=torch.tensor([0.1]),
            ood_scores=torch.tensor([0.9]),
            threshold=0.5,
            output_path=tmp_path / "plot.png",
            bins=0,
        )


def test_plot_entropy_distribution_rejects_nonfinite_threshold(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        plot_entropy_distribution(
            id_scores=torch.tensor([0.1]),
            ood_scores=torch.tensor([0.9]),
            threshold=float("nan"),
            output_path=tmp_path / "plot.png",
        )
        
        
def test_save_entropy_scores_writes_npz(tmp_path) -> None:
    output_path = tmp_path / "scores" / "entropy_scores.npz"

    id_scores = torch.tensor([0.2, 0.4])
    ood_scores = torch.tensor([1.2, 1.4])

    save_entropy_scores(
        id_scores=id_scores,
        ood_scores=ood_scores,
        number_of_classes=4,
        output_path=output_path,
    )

    saved = np.load(output_path)

    assert saved["labels"].tolist() == [0, 0, 1, 1]
    assert len(saved["scores"]) == 4
    assert np.all(saved["scores"] >= 0.0)
    assert np.all(saved["scores"] <= 1.0)


def test_save_entropy_scores_rejects_invalid_class_count(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than one",
    ):
        save_entropy_scores(
            id_scores=torch.tensor([0.2]),
            ood_scores=torch.tensor([0.8]),
            number_of_classes=1,
            output_path=tmp_path / "scores.npz",
        )