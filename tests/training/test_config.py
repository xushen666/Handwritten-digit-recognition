import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from digit_recognizer.training.config import TrainingConfig, split_indices
from digit_recognizer.training.runner import benchmark, confusion_matrix, run_epoch


def test_split_is_deterministic_and_disjoint() -> None:
    first_train, first_validation = split_indices(60_000, 5_000, 42)
    second_train, second_validation = split_indices(60_000, 5_000, 42)
    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 55_000
    assert len(first_validation) == 5_000
    assert set(first_train).isdisjoint(first_validation)


def test_default_training_contract() -> None:
    config = TrainingConfig()
    assert config.epochs == 15
    assert config.seed == 42
    assert config.validation_size == 5_000


@pytest.mark.parametrize(
    ("size", "validation_size"),
    [(0, 0), (10, 0), (10, 10), (10, 11), (-1, 1)],
)
def test_split_rejects_empty_partitions(size: int, validation_size: int) -> None:
    with pytest.raises(ValueError):
        split_indices(size, validation_size, 42)


class FixedClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.bias


def test_run_epoch_reports_mean_loss_and_accuracy() -> None:
    inputs = torch.tensor(
        [[8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=1)

    loss, accuracy = run_epoch(
        FixedClassifier(), loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )

    assert loss >= 0.0
    assert accuracy == 1.0


def test_confusion_matrix_has_fixed_shape_and_counts() -> None:
    inputs = torch.zeros((2, 10))
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    matrix = confusion_matrix(FixedClassifier(), loader, torch.device("cpu"))

    assert len(matrix) == 10
    assert all(len(row) == 10 for row in matrix)
    assert sum(sum(row) for row in matrix) == 2
    assert matrix[0][0] == 1
    assert matrix[1][0] == 1


def test_benchmark_returns_nonnegative_latency_percentiles() -> None:
    result = benchmark(FixedClassifier(), torch.zeros((1, 10)), runs=3)

    assert set(result) == {"median", "p95"}
    assert 0.0 <= result["median"] <= result["p95"]
