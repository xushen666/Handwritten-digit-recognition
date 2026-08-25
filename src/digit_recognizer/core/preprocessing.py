from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2

from .errors import BlankImageError, InvalidImageError

ImageInput: TypeAlias = Image.Image | np.ndarray
INK_RATIO_THRESHOLD = 0.002
_IMAGE_SIZE = (28, 28)

_transform = v2.Compose(
    [
        v2.Resize(_IMAGE_SIZE, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ]
)


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
    if np.issubdtype(image.dtype, np.complexfloating) or not np.issubdtype(
        image.dtype, np.number
    ):
        raise InvalidImageError("NumPy image must contain numeric values")

    values = np.asarray(image)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 255):
        raise InvalidImageError("NumPy image values must be finite and within [0, 255]")
    if np.issubdtype(values.dtype, np.floating):
        if np.all(values <= 1):
            values = values * 255
        values = np.rint(values)
    values = values.astype(np.uint8)
    try:
        return Image.fromarray(values)
    except (TypeError, ValueError, UnidentifiedImageError) as exc:
        raise InvalidImageError("could not decode NumPy image") from exc


def to_grayscale(image: ImageInput) -> Image.Image:
    """Convert a supported PIL or NumPy image to an 8-bit grayscale PIL image."""
    try:
        return _as_pil_image(image).convert("L")
    except (TypeError, ValueError, UnidentifiedImageError) as exc:
        raise InvalidImageError("image could not be decoded") from exc


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

    inverted = Image.eval(grayscale, lambda value: 255 - value)
    return _transform(inverted).unsqueeze(0)
