import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from digit_recognizer import __version__
from digit_recognizer.core.model import ImprovedMNISTNet, parameter_count
from digit_recognizer.core.preprocessing import (
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)

from .config import TrainingConfig, seed_everything, split_indices

MINIMUM_TEST_ACCURACY = 0.99


def build_loaders(
    config: TrainingConfig, data_dir: Path
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    training_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, antialias=True),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )
    augmented_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=training_transform
    )
    validation_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=evaluation_transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=evaluation_transform
    )
    training_indices, validation_indices = split_indices(
        len(augmented_dataset), config.validation_size, config.seed
    )
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    training_loader = DataLoader(
        Subset(augmented_dataset, training_indices), shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), shuffle=False, **common
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return training_loader, validation_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("loader must contain at least one sample")
    return total_loss / total_samples, total_correct / total_samples


def confusion_matrix(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> list[list[int]]:
    model.eval()
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    with torch.inference_mode():
        for inputs, labels in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            for actual, predicted in zip(labels.view(-1), predictions, strict=True):
                matrix[int(actual), int(predicted)] += 1
    return matrix.tolist()


def benchmark(
    model: nn.Module, input_tensor: torch.Tensor, runs: int = 100
) -> dict[str, float]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = input_tensor.device
    sample = input_tensor.to(device)
    model.eval()
    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((perf_counter() - started) * 1_000.0)
    return {
        "median": float(median(timings)),
        "p95": float(np.percentile(timings, 95)),
    }


def _evaluate_final(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[list[int]], torch.Tensor]:
    model.eval()
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    benchmark_sample: torch.Tensor | None = None
    with torch.inference_mode():
        for inputs, labels in loader:
            if benchmark_sample is None:
                benchmark_sample = inputs[:1].clone()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            for actual, predicted in zip(
                labels.cpu().view(-1), predictions.cpu().view(-1), strict=True
            ):
                matrix[int(actual), int(predicted)] += 1
    if total_samples == 0 or benchmark_sample is None:
        raise ValueError("test loader must contain at least one sample")
    return (
        total_loss / total_samples,
        total_correct / total_samples,
        matrix.tolist(),
        benchmark_sample,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_training_curves(history: dict[str, list[float]], destination: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["validation_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["validation_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title="MNIST test confusion matrix", xlabel="Predicted", ylabel="Actual")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train(
    config: TrainingConfig,
    data_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    device_name: str = "auto",
) -> dict[str, object]:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    seed_everything(config.seed)
    device = _resolve_device(device_name)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_loader, validation_loader, test_loader = build_loaders(config, data_dir)
    model = ImprovedMNISTNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    model_path = models_dir / "mnist_cnn.pth"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_validation_accuracy = -1.0

    for epoch in range(config.epochs):
        train_loss, train_accuracy = run_epoch(
            model, training_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function, device
        )
        scheduler.step(validation_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch + 1:02d}/{config.epochs}: "
            f"train_acc={train_accuracy:.4f} val_acc={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_path)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_loss, test_accuracy, matrix, sample = _evaluate_final(
        model, test_loader, loss_function, device
    )

    cpu_model = model.cpu()
    latency = benchmark(cpu_model, sample.cpu())
    model_hash = _sha256(model_path)
    _write_json(reports_dir / "history.json", history)
    _plot_training_curves(history, reports_dir / "training_curves.png")
    _plot_confusion_matrix(matrix, reports_dir / "confusion_matrix.png")

    metrics: dict[str, object] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_validation_accuracy": best_validation_accuracy,
        "latency_ms": latency,
        "confusion_matrix": matrix,
    }
    _write_json(reports_dir / "metrics.json", metrics)
    metadata: dict[str, object] = {
        "version": __version__,
        "input_shape": [1, 28, 28],
        "classes": list(range(10)),
        "parameter_count": parameter_count(cpu_model),
        "configuration": config.to_dict(),
        "test_accuracy": test_accuracy,
        "latency_ms": latency,
        "sha256": model_hash,
    }
    _write_json(models_dir / "model_metadata.json", metadata)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the MNIST recognizer")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = TrainingConfig(epochs=args.epochs, seed=args.seed)
    metrics = train(config, args.data_dir, args.models_dir, args.reports_dir, args.device)
    accuracy = float(metrics["test_accuracy"])
    if accuracy < MINIMUM_TEST_ACCURACY:
        raise SystemExit(
            f"Final test accuracy {accuracy:.4%} is below the required "
            f"{MINIMUM_TEST_ACCURACY:.2%}"
        )


if __name__ == "__main__":
    main()
