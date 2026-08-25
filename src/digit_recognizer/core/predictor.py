from __future__ import annotations

import pickle
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
        self.model = model.cpu().eval()

    @classmethod
    def load(cls, path: str | Path) -> Predictor:
        model_path = Path(path)
        if not model_path.is_file():
            raise ModelLoadError(f"Model file not found: {model_path}")
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model = ImprovedMNISTNet()
            model.load_state_dict(state)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            EOFError,
            pickle.UnpicklingError,
        ) as exc:
            raise ModelLoadError("Model file is invalid or incompatible") from exc
        return cls(model)

    def predict(self, image: ImageInput) -> Prediction:
        tensor = prepare_image(image)
        started = perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
            confidence, prediction = probabilities.max(dim=1)
        return Prediction(
            digit=int(prediction.item()),
            confidence=float(confidence.item()),
            latency_ms=(perf_counter() - started) * 1000.0,
        )
