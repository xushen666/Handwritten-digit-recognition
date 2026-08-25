import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from digit_recognizer.core.errors import BlankImageError, InvalidImageError
from digit_recognizer.core.preprocessing import is_blank, prepare_image


def test_prepare_image_returns_normalized_batched_tensor_for_rgb_drawing() -> None:
    image = Image.new("RGB", (56, 56), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 35, 35), fill="black")

    result = prepare_image(image)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1, 28, 28)
    assert result.dtype == torch.float32
    assert float(result.min()) >= -1.0
    assert float(result.max()) <= 1.0


def test_all_white_image_is_blank_and_rejected() -> None:
    image = Image.new("RGB", (28, 28), "white")

    assert is_blank(image)
    with pytest.raises(BlankImageError):
        prepare_image(image)


def test_four_dimensional_numpy_array_is_rejected() -> None:
    image = np.zeros((1, 28, 28, 1), dtype=np.uint8)

    with pytest.raises(InvalidImageError):
        prepare_image(image)
