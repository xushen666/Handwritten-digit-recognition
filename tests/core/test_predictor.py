from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from digit_recognizer.core.errors import ModelLoadError
from digit_recognizer.core.model import ImprovedMNISTNet, parameter_count
from digit_recognizer.core.paths import default_model_path
from digit_recognizer.core.predictor import Predictor


def _line_image() -> Image.Image:
    image = Image.new("L", (280, 280), 255)
    ImageDraw.Draw(image).line((140, 40, 140, 240), fill=0, width=24)
    return image


def test_model_parameter_count_is_stable() -> None:
    assert parameter_count(ImprovedMNISTNet()) == 585_578


def test_predictor_loads_state_dict_and_returns_prediction(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    torch.save(ImprovedMNISTNet().state_dict(), model_path)

    result = Predictor.load(model_path).predict(_line_image())

    assert 0 <= result.digit <= 9
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms >= 0.0


def test_predictor_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        Predictor.load(tmp_path / "missing.pth")


def test_predictor_rejects_corrupt_model(tmp_path: Path) -> None:
    model_path = tmp_path / "corrupt.pth"
    model_path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(ModelLoadError):
        Predictor.load(model_path)


def test_predictor_rejects_incompatible_state_dict(tmp_path: Path) -> None:
    model_path = tmp_path / "incompatible.pth"
    torch.save({"unexpected": torch.tensor(1)}, model_path)

    with pytest.raises(ModelLoadError):
        Predictor.load(model_path)


def test_frozen_default_model_path_uses_meipass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DIGIT_MODEL_PATH", raising=False)
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

    assert default_model_path() == tmp_path / "models" / "mnist_cnn.pth"


def test_environment_model_path_wins_over_frozen_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = tmp_path / "configured.pth"
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("DIGIT_MODEL_PATH", str(configured))

    assert default_model_path() == configured


def test_development_default_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIGIT_MODEL_PATH", raising=False)
    monkeypatch.setattr("sys.frozen", False, raising=False)
    monkeypatch.delattr("sys._MEIPASS", raising=False)

    expected = Path(__file__).resolve().parents[2] / "models" / "mnist_cnn.pth"

    assert default_model_path() == expected
