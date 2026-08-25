from __future__ import annotations

import os
import sys
from pathlib import Path


def default_model_path() -> Path:
    if configured := os.getenv("DIGIT_MODEL_PATH"):
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "models" / "mnist_cnn.pth"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[3] / "models" / "mnist_cnn.pth"
