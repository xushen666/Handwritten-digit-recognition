from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2

from .errors import BlankImageError, InvalidImageError

ImageInput: TypeAlias = Image.Image | np.ndarray
INK_RATIO_THRESHOLD = 0.002
MAX_IMAGE_PIXELS = 4_000_000
IMAGE_SIZE = (28, 28)
NORMALIZATION_MEAN = (0.5,)
NORMALIZATION_STD = (0.5,)

_transform = v2.Compose(
    [
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
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


def _validate_dimensions(image: Image.Image) -> None:
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise InvalidImageError("image dimensions are unsupported")


def to_grayscale(image: ImageInput) -> Image.Image:
    """Convert an image to grayscale, compositing transparency onto white.

    Float NumPy arrays entirely within [0, 1] are normalized floats; otherwise
    float values must be within [0, 255].
    """
    try:
        source = _as_pil_image(image)
        _validate_dimensions(source)
        source.load()
        if source.mode in ("P", "LA", "RGBA"):
            rgba = source.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("L")
        return source.convert("L")
    except InvalidImageError:
        raise
    except (OSError, TypeError, ValueError, UnidentifiedImageError) as exc:
        raise InvalidImageError("image could not be decoded") from exc


def _ink_ratio(grayscale: Image.Image) -> float:
    gray = np.asarray(grayscale, dtype=np.float32)
    return float(np.mean((255.0 - gray) / 255.0))


def is_blank(image: ImageInput) -> bool:
    """Return whether normalized mean darkness is below the blank threshold."""
    return _ink_ratio(to_grayscale(image)) < INK_RATIO_THRESHOLD


def prepare_image(
    image: ImageInput,
    *,
    reject_blank: bool = True,
) -> torch.Tensor:
    """Prepare an image for inference as a normalized ``(1, 1, 28, 28)`` tensor."""
    grayscale = to_grayscale(image)
    if reject_blank and _ink_ratio(grayscale) < INK_RATIO_THRESHOLD:
        raise BlankImageError("image is blank")

    inverted = Image.eval(grayscale, lambda value: 255 - value)
    return _transform(inverted).unsqueeze(0)
