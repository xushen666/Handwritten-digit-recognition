from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from PIL import Image

from .errors import BlankImageError, InvalidImageError

ImageInput: TypeAlias = Image.Image | np.ndarray
INK_RATIO_THRESHOLD = 0.002
_IMAGE_SIZE = (28, 28)


def _as_pil_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("image must be a PIL image or NumPy array")
    if image.ndim not in (2, 3):
        raise InvalidImageError("NumPy image must have 2 or 3 dimensions")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise InvalidImageError("NumPy image must have 3 or 4 channels")
    if image.size == 0:
        raise InvalidImageError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise InvalidImageError("NumPy image must contain numeric values")

    values = np.asarray(image)
    if np.issubdtype(values.dtype, np.floating):
        values = values * 255 if np.nanmax(values) <= 1 else values
    values = np.clip(values, 0, 255).astype(np.uint8)
    try:
        return Image.fromarray(values)
    except (TypeError, ValueError) as exc:
        raise InvalidImageError("could not decode NumPy image") from exc


def to_grayscale(image: ImageInput) -> Image.Image:
    """Convert a supported PIL or NumPy image to an 8-bit grayscale PIL image."""
    return _as_pil_image(image).convert("L")


def is_blank(image: ImageInput) -> bool:
    """Return whether normalized mean darkness is below the blank threshold."""
    grayscale = np.asarray(to_grayscale(image), dtype=np.float32) / 255.0
    ink_ratio = float(np.mean(1.0 - grayscale))
    return ink_ratio < INK_RATIO_THRESHOLD


def prepare_image(
    image: ImageInput,
    *,
    reject_blank: bool = True,
) -> torch.Tensor:
    """Prepare an image for inference as a normalized ``(1, 1, 28, 28)`` tensor."""
    grayscale = to_grayscale(image)
    if reject_blank and is_blank(grayscale):
        raise BlankImageError("image is blank")

    resized = grayscale.resize(_IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(1.0 - pixels).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5
