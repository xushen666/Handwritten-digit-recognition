import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from digit_recognizer.core import preprocessing
from digit_recognizer.training import runner
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


def test_build_loaders_reuses_shared_preprocessing_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeMNIST:
        def __init__(
            self, root: Path, train: bool, download: bool, transform: transforms.Compose
        ) -> None:
            self.root = root
            self.train = train
            self.download = download
            self.transform = transform
            datasets_created.append(self)

        def __len__(self) -> int:
            return 10

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros((1, 28, 28)), index % 10

    datasets_created: list[FakeMNIST] = []
    monkeypatch.setattr(runner.datasets, "MNIST", FakeMNIST)

    runner.build_loaders(
        TrainingConfig(validation_size=2, batch_size=2),
        tmp_path,
    )

    assert len(datasets_created) == 3
    training_steps = datasets_created[0].transform.transforms
    validation_steps = datasets_created[1].transform.transforms
    test_steps = datasets_created[2].transform.transforms
    assert any(isinstance(step, transforms.RandomAffine) for step in training_steps)
    assert not any(isinstance(step, transforms.RandomAffine) for step in validation_steps)
    assert not any(isinstance(step, transforms.RandomAffine) for step in test_steps)
    for steps in (training_steps, validation_steps, test_steps):
        resize = next(step for step in steps if isinstance(step, transforms.Resize))
        normalize = next(step for step in steps if isinstance(step, transforms.Normalize))
        assert tuple(resize.size) == preprocessing.IMAGE_SIZE
        assert tuple(normalize.mean) == preprocessing.NORMALIZATION_MEAN
        assert tuple(normalize.std) == preprocessing.NORMALIZATION_STD


class TrackingLoader:
    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.batches = batches
        self.iterations = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iterations += 1
        return iter(self.batches)


def test_train_writes_complete_artifacts_and_iterates_test_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_batch = torch.zeros((1, 1, 28, 28))
    label_batch = torch.zeros(1, dtype=torch.long)
    test_loader = TrackingLoader([(input_batch, label_batch)])
    loaders = (TrackingLoader([]), TrackingLoader([]), test_loader)
    epoch_results = iter([(0.4, 0.8), (0.3, 0.9), (0.2, 0.85), (0.35, 0.8)])
    training_states = iter([1.0, 2.0])
    evaluated_state: dict[str, float] = {}

    def fake_run_epoch(
        model: nn.Module,
        loader: TrackingLoader,
        loss_function: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[float, float]:
        if optimizer is not None:
            with torch.no_grad():
                next(model.parameters()).fill_(next(training_states))
        return next(epoch_results)

    def fake_final_evaluation(
        model: nn.Module,
        loader: TrackingLoader,
        loss_function: nn.Module,
        device: torch.device,
    ) -> tuple[float, float, list[list[int]], torch.Tensor]:
        evaluated_state["parameter"] = float(next(model.parameters()).detach().flatten()[0])
        list(loader)
        return 0.02, 0.995, [[0] * 10 for _ in range(10)], input_batch

    monkeypatch.setattr(runner, "build_loaders", lambda config, data_dir: loaders)
    monkeypatch.setattr(runner, "run_epoch", fake_run_epoch)
    monkeypatch.setattr(runner, "_evaluate_final", fake_final_evaluation)
    monkeypatch.setattr(runner, "benchmark", lambda *args, **kwargs: {"median": 1.0, "p95": 2.0})
    monkeypatch.setattr(
        runner,
        "_plot_training_curves",
        lambda history, destination: destination.write_bytes(b"training-curve"),
    )
    monkeypatch.setattr(
        runner,
        "_plot_confusion_matrix",
        lambda matrix, destination: destination.write_bytes(b"confusion-matrix"),
    )
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    config = TrainingConfig(epochs=2)
    models_dir.mkdir()
    model_path = models_dir / "mnist_cnn.pth"
    model_path.write_bytes(b"previous-model")
    replace_targets: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def track_replace(source: Path, target: Path) -> Path:
        replace_targets.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", track_replace)

    metrics = runner.train(
        config,
        tmp_path / "data",
        models_dir,
        reports_dir,
        "cpu",
    )

    assert test_loader.iterations == 1
    assert evaluated_state["parameter"] == 1.0
    assert json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8")) == metrics
    for filename in (
        "history.json",
        "metrics.json",
        "training_curves.png",
        "confusion_matrix.png",
    ):
        assert (reports_dir / filename).is_file()
    assert len(replace_targets) == 1
    assert replace_targets[0][0] != model_path
    assert replace_targets[0][1] == model_path
    assert model_path.read_bytes() != b"previous-model"
    metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["version"] == "0.1.0"
    assert metadata["input_shape"] == [1, 28, 28]
    assert metadata["classes"] == list(range(10))
    assert metadata["parameter_count"] == 585_578
    assert metadata["configuration"] == config.to_dict()
    assert metadata["test_accuracy"] == 0.995
    assert metadata["latency_ms"] == {"median": 1.0, "p95": 2.0}
    assert metadata["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()


def test_train_rejects_non_finite_validation_without_touching_existing_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaders = (TrackingLoader([]), TrackingLoader([]), TrackingLoader([]))
    epoch_results = iter([(0.4, 0.8), (float("nan"), float("nan"))])
    monkeypatch.setattr(runner, "build_loaders", lambda config, data_dir: loaders)
    monkeypatch.setattr(runner, "run_epoch", lambda *args, **kwargs: next(epoch_results))
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "mnist_cnn.pth"
    previous_model = b"previous-model-must-remain"
    model_path.write_bytes(previous_model)

    with pytest.raises(RuntimeError, match="non-finite validation metrics"):
        runner.train(
            TrainingConfig(epochs=1),
            tmp_path / "data",
            models_dir,
            tmp_path / "reports",
            "cpu",
        )

    assert model_path.read_bytes() == previous_model
    assert list(models_dir.glob(".mnist_cnn-*.tmp")) == []


def test_main_exits_nonzero_below_minimum_accuracy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "train", lambda *args, **kwargs: {"test_accuracy": 0.98})

    with pytest.raises(SystemExit) as raised:
        runner.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--models-dir",
                str(tmp_path / "models"),
                "--reports-dir",
                str(tmp_path / "reports"),
            ]
        )

    assert raised.value.code != 0
