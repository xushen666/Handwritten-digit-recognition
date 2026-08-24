# Handwritten Digit Recognition One-Day Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible MNIST inference core, FastAPI/Web service, offline PyQt client, automated tests, Docker image, Windows release workflow, and truthful portfolio documentation in one day.

**Architecture:** A single Python package owns the CNN definition, preprocessing, blank-input validation, model loading, and inference. FastAPI and PyQt are thin adapters over that core; the browser calls FastAPI, while the desktop app calls the core directly. Training writes versioned model metadata and reports that documentation consumes as the only source of quantitative claims.

**Tech Stack:** Python 3.11, PyTorch, torchvision, Pillow, NumPy, FastAPI, Uvicorn, PyQt5, pytest, Ruff, matplotlib, Docker, PyInstaller, GitHub Actions.

---

## File map

- `pyproject.toml`: package metadata, dependencies, command entry points, pytest and Ruff configuration.
- `.gitignore`: datasets, environments, caches, reports under construction, and build products.
- `src/digit_recognizer/core/model.py`: the single CNN definition.
- `src/digit_recognizer/core/preprocessing.py`: image conversion, blank detection, and tensor preparation.
- `src/digit_recognizer/core/predictor.py`: model loading, inference, confidence, and latency.
- `src/digit_recognizer/core/paths.py`: development and PyInstaller model path resolution.
- `src/digit_recognizer/training/config.py`: deterministic training configuration and split generation.
- `src/digit_recognizer/training/runner.py`: train, validate, final-test, benchmark, and artifact generation.
- `src/digit_recognizer/api/app.py`: FastAPI application factory and error mapping.
- `src/digit_recognizer/api/schemas.py`: public API response models.
- `src/digit_recognizer/web/index.html`: browser drawing surface.
- `src/digit_recognizer/web/app.js`: canvas events and API call.
- `src/digit_recognizer/web/style.css`: minimal presentation.
- `src/digit_recognizer/desktop/app.py`: PyQt canvas, window, and entry point.
- `scripts/train.py`: stable training CLI.
- `scripts/sync_docs.py`: reads generated metrics and updates README/resume claims deterministically.
- `tests/`: fast core, configuration, API, and desktop conversion tests.
- `Dockerfile`, `.dockerignore`: API/Web CPU image.
- `packaging/windows.spec`: onedir PyInstaller build.
- `.github/workflows/quality.yml`: lint, tests, API import, and Docker build.
- `.github/workflows/release.yml`: tagged Windows artifact and GitHub Release.
- `README.md`: English summary plus Chinese documentation.
- `docs/resume-project.md`: verified resume wording.
- `LICENSE`: MIT license.

### Task 1: Create the installable project skeleton

**Files:**
- Modify: `.gitignore`
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `README.md`
- Create: `src/digit_recognizer/__init__.py`
- Create: `src/digit_recognizer/core/__init__.py`
- Create: `src/digit_recognizer/api/__init__.py`
- Create: `src/digit_recognizer/desktop/__init__.py`
- Create: `src/digit_recognizer/training/__init__.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write the failing package test**

```python
# tests/test_package.py
from digit_recognizer import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and verify the package does not exist**

Run: `python -m pytest tests/test_package.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'digit_recognizer'`.

- [ ] **Step 3: Add packaging, dependency groups, and package initializers**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "handwritten-digit-recognition"
version = "0.1.0"
description = "Reproducible MNIST inference with FastAPI, Web, and PyQt clients"
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = { text = "MIT" }
authors = [{ name = "xushen666" }]
dependencies = [
  "numpy>=1.26,<3",
  "pillow>=10,<12",
  "torch>=2.3,<3",
  "torchvision>=0.18,<1",
]

[project.optional-dependencies]
api = ["fastapi>=0.115,<1", "python-multipart>=0.0.9,<1", "uvicorn[standard]>=0.30,<1"]
desktop = ["PyQt5>=5.15.10,<6"]
train = ["matplotlib>=3.9,<4"]
dev = ["build>=1.2,<2", "httpx>=0.27,<1", "pytest>=8,<9", "ruff>=0.6,<1"]
all = [
  "fastapi>=0.115,<1", "python-multipart>=0.0.9,<1", "uvicorn[standard]>=0.30,<1",
  "PyQt5>=5.15.10,<6", "matplotlib>=3.9,<4",
]

[project.scripts]
digit-api = "digit_recognizer.api.app:main"
digit-desktop = "digit_recognizer.desktop.app:main"
digit-train = "digit_recognizer.training.runner:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
digit_recognizer = ["web/*.html", "web/*.js", "web/*.css"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

```python
# src/digit_recognizer/__init__.py
__version__ = "0.1.0"
```

Create the four subpackage `__init__.py` files as empty files. Extend `.gitignore` with:

```gitignore
.worktrees/
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.idea/
data/
build/
dist/
*.spec.bak
*.zip
*.exe
```

Add the standard MIT License text with copyright line `Copyright (c) 2026 xushen666`.

Create the initial README so editable installation has a valid readme target:

```markdown
# Handwritten Digit Recognition

Portfolio-oriented reconstruction of an MNIST recognizer with a shared PyTorch core,
FastAPI/Web delivery, and an offline PyQt client. Verified metrics and full Chinese
usage documentation are generated after reproducible retraining.
```

- [ ] **Step 4: Install the editable development environment and rerun the test**

Run: `python -m pip install -e ".[api,desktop,train,dev]"`

Expected: installation completes without dependency resolution errors.

Run: `python -m pytest tests/test_package.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit the skeleton**

```bash
git add .gitignore pyproject.toml LICENSE README.md src tests/test_package.py
git commit -m "chore: scaffold installable Python project"
```

Before later opening a pull request, publish the already-approved design-only `main` branch from the primary worktree:

```bash
git -C "E:\csdiy\项目重构\Handwritten-digit-recognition" push -u origin main
```

### Task 2: Implement deterministic preprocessing and blank detection

**Files:**
- Create: `src/digit_recognizer/core/errors.py`
- Create: `src/digit_recognizer/core/preprocessing.py`
- Test: `tests/core/test_preprocessing.py`

- [ ] **Step 1: Write preprocessing tests**

```python
# tests/core/test_preprocessing.py
import numpy as np
import pytest
from PIL import Image, ImageDraw

from digit_recognizer.core.errors import BlankImageError, InvalidImageError
from digit_recognizer.core.preprocessing import is_blank, prepare_image


def test_prepare_image_returns_normalized_mnist_tensor() -> None:
    image = Image.new("RGB", (280, 280), "white")
    ImageDraw.Draw(image).line((30, 30, 250, 250), fill="black", width=20)
    tensor = prepare_image(image)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype.is_floating_point
    assert -1.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_blank_white_image_is_rejected() -> None:
    image = Image.new("L", (280, 280), 255)
    assert is_blank(image)
    with pytest.raises(BlankImageError):
        prepare_image(image)


def test_invalid_array_shape_is_rejected() -> None:
    with pytest.raises(InvalidImageError):
        prepare_image(np.zeros((2, 2, 2, 2), dtype=np.uint8))
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest tests/core/test_preprocessing.py -v`

Expected: FAIL with `ModuleNotFoundError` for `digit_recognizer.core.errors`.

- [ ] **Step 3: Implement preprocessing**

```python
# src/digit_recognizer/core/errors.py
class DigitRecognizerError(Exception):
    """Base error for expected application failures."""


class InvalidImageError(DigitRecognizerError):
    pass


class BlankImageError(DigitRecognizerError):
    pass


class ModelLoadError(DigitRecognizerError):
    pass
```

```python
# src/digit_recognizer/core/preprocessing.py
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2

from .errors import BlankImageError, InvalidImageError

ImageInput: TypeAlias = Image.Image | np.ndarray
INK_RATIO_THRESHOLD = 0.002

_transform = v2.Compose(
    [
        v2.Resize((28, 28), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def to_grayscale(image: ImageInput) -> Image.Image:
    try:
        if isinstance(image, np.ndarray):
            if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
                raise InvalidImageError("Unsupported NumPy image shape")
            image = Image.fromarray(image.astype(np.uint8, copy=False))
        if not isinstance(image, Image.Image):
            raise InvalidImageError("Expected a PIL image or NumPy array")
        return image.convert("L")
    except (TypeError, ValueError, UnidentifiedImageError) as exc:
        raise InvalidImageError("Image could not be decoded") from exc


def is_blank(image: ImageInput) -> bool:
    gray = np.asarray(to_grayscale(image), dtype=np.float32)
    ink_ratio = float(np.mean((255.0 - gray) / 255.0))
    return ink_ratio < INK_RATIO_THRESHOLD


def prepare_image(image: ImageInput, *, reject_blank: bool = True) -> torch.Tensor:
    gray = to_grayscale(image)
    if reject_blank and is_blank(gray):
        raise BlankImageError("The image does not contain visible handwriting")
    inverted = Image.eval(gray, lambda value: 255 - value)
    return _transform(inverted).unsqueeze(0)
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/core/test_preprocessing.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit preprocessing**

```bash
git add src/digit_recognizer/core tests/core/test_preprocessing.py
git commit -m "feat: add shared image preprocessing"
```

### Task 3: Implement the CNN, model path resolution, and Predictor

**Files:**
- Create: `src/digit_recognizer/core/model.py`
- Create: `src/digit_recognizer/core/paths.py`
- Create: `src/digit_recognizer/core/predictor.py`
- Create: `models/.gitkeep`
- Test: `tests/core/test_predictor.py`

- [ ] **Step 1: Write predictor tests**

```python
# tests/core/test_predictor.py
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from digit_recognizer.core.model import ImprovedMNISTNet, parameter_count
from digit_recognizer.core.predictor import Predictor


def test_model_parameter_count_is_stable() -> None:
    assert parameter_count(ImprovedMNISTNet()) == 585_578


def test_predictor_loads_state_dict_and_returns_prediction(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    torch.save(ImprovedMNISTNet().state_dict(), model_path)
    image = Image.new("L", (280, 280), 255)
    ImageDraw.Draw(image).line((140, 40, 140, 240), fill=0, width=24)
    result = Predictor.load(model_path).predict(image)
    assert 0 <= result.digit <= 9
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms >= 0.0
```

- [ ] **Step 2: Run tests and verify missing model failure**

Run: `python -m pytest tests/core/test_predictor.py -v`

Expected: FAIL with missing `digit_recognizer.core.model`.

- [ ] **Step 3: Implement model, paths, and prediction**

Create the model with the complete architecture below:

```python
# src/digit_recognizer/core/model.py
import torch
from torch import nn


class ImprovedMNISTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
```

```python
# src/digit_recognizer/core/paths.py
import os
import sys
from pathlib import Path


def default_model_path() -> Path:
    if configured := os.getenv("DIGIT_MODEL_PATH"):
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "models" / "mnist_cnn.pth"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[3] / "models" / "mnist_cnn.pth"
```

```python
# src/digit_recognizer/core/predictor.py
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from .errors import ModelLoadError
from .model import ImprovedMNISTNet
from .preprocessing import ImageInput, prepare_image


@dataclass(frozen=True, slots=True)
class Prediction:
    digit: int
    confidence: float
    latency_ms: float


class Predictor:
    def __init__(self, model: ImprovedMNISTNet) -> None:
        self.model = model.eval().cpu()

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        model_path = Path(path)
        if not model_path.is_file():
            raise ModelLoadError(f"Model file not found: {model_path}")
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model = ImprovedMNISTNet()
            model.load_state_dict(state)
            return cls(model)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelLoadError("Model file is invalid or incompatible") from exc

    def predict(self, image: ImageInput) -> Prediction:
        tensor = prepare_image(image)
        started = perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
            confidence, prediction = probabilities.max(dim=1)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return Prediction(int(prediction.item()), float(confidence.item()), elapsed_ms)
```

- [ ] **Step 4: Run core tests**

Run: `python -m pytest tests/core -v`

Expected: `5 passed`.

- [ ] **Step 5: Commit inference core**

```bash
git add src/digit_recognizer/core models tests/core
git commit -m "feat: add reusable MNIST predictor"
```

### Task 4: Make training and evaluation reproducible

**Files:**
- Create: `src/digit_recognizer/training/config.py`
- Create: `src/digit_recognizer/training/runner.py`
- Create: `scripts/train.py`
- Test: `tests/training/test_config.py`

- [ ] **Step 1: Write deterministic split tests**

```python
# tests/training/test_config.py
from digit_recognizer.training.config import TrainingConfig, split_indices


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
```

- [ ] **Step 2: Run tests and verify missing training config**

Run: `python -m pytest tests/training/test_config.py -v`

Expected: FAIL with missing `digit_recognizer.training.config`.

- [ ] **Step 3: Implement the fixed configuration and split**

```python
# src/digit_recognizer/training/config.py
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    validation_size: int = 5_000
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 0.001
    num_workers: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(size: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    return order[validation_size:], order[:validation_size]
```

- [ ] **Step 4: Implement the runner with one final test evaluation**

`runner.py` must define six typed units, each independently callable by tests or the CLI:

- `build_loaders` accepts `TrainingConfig` and a `Path`, returning the training, validation, and test `DataLoader` objects.
- `run_epoch` accepts a model, loader, loss function, device, and optional optimizer, returning mean loss and accuracy as two floats.
- `confusion_matrix` accepts a model, loader, and device, returning a 10-by-10 nested integer list.
- `benchmark` accepts a model, one input tensor, and an integer run count defaulting to 100, returning float values under `median` and `p95`.
- `train` accepts the configuration, data/models/reports paths, and device name, returning the exact dictionary persisted to `reports/metrics.json`.
- `main` parses `--data-dir`, `--models-dir`, `--reports-dir`, `--device`, `--epochs`, and `--seed`, constructs the configuration, and calls `train`.

Implementation requirements:

- Create separate MNIST dataset objects for augmented training and non-augmented validation.
- Use `split_indices` for both datasets, `CrossEntropyLoss`, Adam, and `ReduceLROnPlateau`.
- Save `models/mnist_cnn.pth` only when validation accuracy improves.
- Load that file after all epochs, then evaluate the test loader exactly once.
- Save `reports/history.json`, `reports/metrics.json`, `reports/training_curves.png`, and `reports/confusion_matrix.png`.
- Save `models/model_metadata.json` containing version `0.1.0`, input shape `[1, 28, 28]`, parameter count `585578`, configuration, test accuracy, latency median/P95, and SHA-256 of `mnist_cnn.pth`.
- Exit nonzero when final test accuracy is below `0.99`.

```python
# scripts/train.py
from digit_recognizer.training.runner import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run fast tests, then start the real GPU training**

Run: `python -m pytest tests/training tests/core -v`

Expected: all tests PASS without downloading MNIST.

Run: `python scripts/train.py --device auto --epochs 15`

Expected: CUDA is selected when available; the command creates the model, metadata, metrics, two PNG reports, and exits successfully only when test accuracy is at least 99.0%.

- [ ] **Step 6: Commit training code separately from generated results**

```bash
git add src/digit_recognizer/training scripts/train.py tests/training
git commit -m "feat: add reproducible training and evaluation"
git add models/mnist_cnn.pth models/model_metadata.json reports
git commit -m "data: publish verified MNIST model metrics"
```

### Task 5: Add FastAPI and the minimal Web client

**Files:**
- Create: `src/digit_recognizer/api/schemas.py`
- Create: `src/digit_recognizer/api/app.py`
- Create: `src/digit_recognizer/web/index.html`
- Create: `src/digit_recognizer/web/app.js`
- Create: `src/digit_recognizer/web/style.css`
- Test: `tests/api/test_app.py`

- [ ] **Step 1: Write API tests with a stub predictor**

```python
# tests/api/test_app.py
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from digit_recognizer.api.app import create_app
from digit_recognizer.core.predictor import Prediction


class StubPredictor:
    def predict(self, image: Image.Image) -> Prediction:
        return Prediction(digit=7, confidence=0.98, latency_ms=1.25)


def png_bytes(color: str = "black") -> bytes:
    image = Image.new("RGB", (28, 28), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_health_and_prediction() -> None:
    client = TestClient(create_app(predictor=StubPredictor()))
    assert client.get("/health").json()["model_loaded"] is True
    response = client.post("/api/v1/predict", files={"file": ("digit.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"digit": 7, "confidence": 0.98, "latency_ms": 1.25}


def test_non_image_is_rejected() -> None:
    client = TestClient(create_app(predictor=StubPredictor()))
    response = client.post("/api/v1/predict", files={"file": ("bad.txt", b"bad", "text/plain")})
    assert response.status_code == 415
```

- [ ] **Step 2: Run tests and verify the API module is missing**

Run: `python -m pytest tests/api/test_app.py -v`

Expected: FAIL importing `digit_recognizer.api.app`.

- [ ] **Step 3: Implement schemas and application factory**

```python
# src/digit_recognizer/api/schemas.py
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    latency_ms: float
```

`app.py` must use `create_app(predictor: Predictor | None = None)`, load the default model during lifespan when no predictor is injected, mount the packaged Web directory, and enforce:

- content type in `image/png` or `image/jpeg`;
- maximum body size of 2 MiB;
- malformed image -> HTTP 400;
- blank image -> HTTP 422;
- missing model -> HTTP 503;
- no filesystem path or traceback in responses.

The response is built only from `Prediction.digit`, `confidence`, and `latency_ms`. `main()` runs `uvicorn.run("digit_recognizer.api.app:create_app", factory=True, host="0.0.0.0", port=8000)`.

- [ ] **Step 4: Implement the three-file Web page**

Use a 280x280 white `<canvas>`, pointer events with a 15-pixel round black stroke, and two buttons. On recognition, convert the canvas to a PNG `Blob`, submit `FormData` to `/api/v1/predict`, and display the digit, confidence percentage, and latency. On non-2xx responses, display the API `detail` message. No framework or build step is allowed.

- [ ] **Step 5: Run API tests and a local smoke test**

Run: `python -m pytest tests/api/test_app.py -v`

Expected: both tests PASS.

Run: `python -m uvicorn digit_recognizer.api.app:create_app --factory --host 127.0.0.1 --port 8000`

Expected: `GET http://127.0.0.1:8000/health` returns HTTP 200 with `model_loaded: true`, and `/` displays the drawing page.

- [ ] **Step 6: Commit API and Web**

```bash
git add src/digit_recognizer/api src/digit_recognizer/web tests/api
git commit -m "feat: add FastAPI and browser demo"
```

### Task 6: Refactor the offline PyQt desktop client

**Files:**
- Create: `src/digit_recognizer/desktop/app.py`
- Test: `tests/desktop/test_canvas.py`

- [ ] **Step 1: Write the offscreen canvas conversion test**

```python
# tests/desktop/test_canvas.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from digit_recognizer.desktop.app import DrawingCanvas


def test_canvas_starts_blank_and_converts_to_pil() -> None:
    app = QApplication.instance() or QApplication([])
    canvas = DrawingCanvas()
    image = canvas.to_pil()
    assert image.size == (280, 280)
    assert image.convert("L").getextrema() == (255, 255)
    assert app is not None
```

- [ ] **Step 2: Run test and verify desktop module is missing**

Run: `python -m pytest tests/desktop/test_canvas.py -v`

Expected: FAIL importing `digit_recognizer.desktop.app`.

- [ ] **Step 3: Implement the desktop application**

`DrawingCanvas(QLabel)` owns a 280x280 `QImage`, handles mouse events in its own coordinate system, paints round black 15-pixel lines, exposes `clear()` and converts to PIL through an in-memory PNG `QBuffer`.

`DigitRecognizerWindow(QWidget)` receives a `Predictor`, owns the canvas, clear/recognize buttons, result label, and confidence/latency label. `recognize()` catches `BlankImageError` and shows an informational message; unexpected inference errors show a critical message without a traceback.

`main()` creates `QApplication`, loads `Predictor` from `default_model_path()`, shows a critical startup dialog and exits with code 1 on `ModelLoadError`, otherwise shows the window and returns the Qt event-loop code.

- [ ] **Step 4: Run the desktop test and manually smoke-test drawing**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/desktop/test_canvas.py -v`

Expected: PASS.

Run: `digit-desktop`

Expected: the window opens, drawing follows the pointer without coordinate offset, blank recognition is rejected, and a drawn digit returns a prediction.

- [ ] **Step 5: Commit desktop client**

```bash
git add src/digit_recognizer/desktop tests/desktop
git commit -m "feat: add offline PyQt desktop client"
```

### Task 7: Add Docker, CI, and Windows Release automation

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `packaging/windows.spec`
- Create: `.github/workflows/quality.yml`
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Add the CPU API image**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY models ./models
RUN python -m pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir ".[api]"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "digit_recognizer.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore` excludes `.git`, `.github`, `.worktrees`, `.venv`, caches, `data`, `build`, `dist`, tests, docs, and `reports`.

- [ ] **Step 2: Add the onedir Windows build**

`packaging/windows.spec` analyzes `src/digit_recognizer/desktop/app.py` with `pathex=["src"]`, includes `models/mnist_cnn.pth` under `models`, sets `console=False`, and produces a directory named `HandwrittenDigitRecognizer`. Do not use onefile mode and do not include MNIST or training dependencies.

- [ ] **Step 3: Add quality workflow**

`.github/workflows/quality.yml` runs on pushes and pull requests using Python 3.11. It installs CPU torch/torchvision, then `.[api,desktop,dev]`, runs `ruff check .`, runs pytest with `QT_QPA_PLATFORM=offscreen`, imports `create_app`, and runs `docker build -t handwritten-digit-recognition:test .` in a separate job.

- [ ] **Step 4: Add tagged release workflow**

`.github/workflows/release.yml` runs on tags matching `v*`, uses `windows-latest`, installs Python 3.11, `.[desktop]`, and PyInstaller, runs `pyinstaller packaging/windows.spec --clean --noconfirm`, compresses `dist/HandwrittenDigitRecognizer` to `HandwrittenDigitRecognizer-windows-x64.zip`, and publishes it with `softprops/action-gh-release@v2` using `GITHUB_TOKEN`.

- [ ] **Step 5: Verify locally**

Run: `docker build -t handwritten-digit-recognition:local .`

Expected: build succeeds.

Run: `docker run --rm -d --name digit-api-test -p 8000:8000 handwritten-digit-recognition:local`

Expected: container becomes healthy and `/health` reports the model loaded. Stop it with `docker stop digit-api-test`.

Run: `python -m pip install pyinstaller && pyinstaller packaging/windows.spec --clean --noconfirm`

Expected: `dist/HandwrittenDigitRecognizer/HandwrittenDigitRecognizer.exe` exists and starts.

- [ ] **Step 6: Commit delivery automation**

```bash
git add Dockerfile .dockerignore packaging .github
git commit -m "ci: add container and Windows release pipelines"
```

### Task 8: Generate truthful README and resume material

**Files:**
- Modify: `README.md`
- Create: `docs/resume-project.md`
- Create: `scripts/sync_docs.py`
- Create: `docs/images/desktop.png`
- Create: `docs/images/web.png`
- Test: `tests/test_documentation.py`

- [ ] **Step 1: Write documentation consistency test**

```python
# tests/test_documentation.py
import json
from pathlib import Path


def test_documented_metrics_match_generated_metrics() -> None:
    metrics = json.loads(Path("reports/metrics.json").read_text(encoding="utf-8"))
    readme = Path("README.md").read_text(encoding="utf-8")
    resume = Path("docs/resume-project.md").read_text(encoding="utf-8")
    accuracy = f"{metrics['test_accuracy'] * 100:.2f}%"
    median = f"{metrics['latency_ms']['median']:.2f} ms"
    p95 = f"{metrics['latency_ms']['p95']:.2f} ms"
    for document in (readme, resume):
        assert accuracy in document
        assert median in document
        assert p95 in document
```

- [ ] **Step 2: Implement deterministic documentation synchronization**

`scripts/sync_docs.py` reads `reports/metrics.json` and writes both documents from fixed UTF-8 templates. It formats test accuracy to two percentage decimals and median/P95 to two milliseconds. The README template must contain:

- a five-line English overview;
- Chinese project positioning and verified metric table;
- architecture and shared-core data flow;
- local API, Docker, Web, and desktop quick starts;
- curl request example and exact response keys;
- training, testing, and lint commands;
- screenshots and links to the two report PNGs;
- truthful course-project origin and personal contribution;
- limitations: one digit, MNIST, no production OCR claim;
- MIT License.

The resume template must contain project name, technology stack, GitHub URL, and exactly three bullets covering shared-core architecture, reproducible evaluation, and FastAPI/Docker/PyQt delivery. Quantitative claims are inserted only from the metrics JSON.

- [ ] **Step 3: Capture the two screenshots and generate docs**

Run the API/Web and desktop clients, capture one clean screenshot of each, and save them as the exact PNG paths listed above. Do not include desktop paths, usernames, terminals, or personal data in either image.

Run: `python scripts/sync_docs.py`

Expected: the script prints `Documentation synchronized from reports/metrics.json` and writes both documents with actual measured values.

- [ ] **Step 4: Run documentation test**

Run: `python -m pytest tests/test_documentation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit portfolio documentation**

```bash
git add README.md docs/resume-project.md docs/images scripts/sync_docs.py tests/test_documentation.py
git commit -m "docs: add verified project and resume documentation"
```

### Task 9: Perform final verification and prepare the first release

**Files:**
- Modify only files implicated by verification failures.

- [ ] **Step 1: Run all local quality gates**

Run: `python -m ruff check .`

Expected: `All checks passed!`.

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest -v`

Expected: all tests PASS with zero skipped tests required by this plan.

Run: `python -m build`

Expected: source distribution and wheel build successfully.

- [ ] **Step 2: Validate quantitative artifact consistency**

Run a Python check that recomputes the SHA-256 of `models/mnist_cnn.pth`, compares it with `models/model_metadata.json`, verifies parameter count `585578`, verifies test accuracy at least `0.99`, and verifies the README/resume values equal `reports/metrics.json`.

Expected: command exits 0 and prints `artifact consistency verified`.

- [ ] **Step 3: Validate both delivery paths**

Run the Docker health/prediction smoke test with a generated nonblank PNG and verify HTTP 200 with digit 0-9, confidence 0-1, and nonnegative latency.

Launch the packaged Windows executable and manually verify draw, clear, blank rejection, prediction, and clean shutdown.

- [ ] **Step 4: Review repository hygiene and secrets**

Run: `git status --short`

Expected: clean after committing fixes.

Run: `git ls-files`

Expected: no `.idea`, cache, MNIST data, original reports, original PPT, ZIP, EXE, `build`, `dist`, or `.worktrees` entries.

Search tracked text for the original student number, teacher name, absolute `E:\` paths, tokens, passwords, and private keys.

Expected: no matches.

- [ ] **Step 5: Make the verification commit**

```bash
git add -A
git commit -m "chore: complete release verification"
```

If no files changed after verification, do not create an empty commit.

- [ ] **Step 6: Push only after all gates pass**

```bash
git push -u origin feature/one-day-rebuild
```

Open a pull request into `main`, verify GitHub Actions, and merge it:

```bash
gh pr create --base main --head feature/one-day-rebuild --title "Rebuild handwritten digit recognition portfolio project" --body "Implements the approved one-day rebuild specification."
gh pr checks --watch
gh pr merge --squash --delete-branch
```

After the merge, create the release tag from the updated primary worktree, where `main` is already checked out:

```bash
git -C "E:\csdiy\项目重构\Handwritten-digit-recognition" pull --ff-only origin main
git -C "E:\csdiy\项目重构\Handwritten-digit-recognition" tag -a v0.1.0 -m "First portfolio-ready release"
git -C "E:\csdiy\项目重构\Handwritten-digit-recognition" push origin v0.1.0
```

Expected: the Release workflow attaches `HandwrittenDigitRecognizer-windows-x64.zip`. If the workflow fails, keep the tag history intact, fix the workflow on a new commit, and publish a new patch tag rather than claiming the Release succeeded.
