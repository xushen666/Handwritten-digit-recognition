from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import v2

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


@pytest.mark.parametrize(
    "values",
    [
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-1.0]]),
        np.array([[256.0]]),
        np.array([[1 + 2j]]),
    ],
)
def test_invalid_numpy_values_are_rejected(values: np.ndarray) -> None:
    with pytest.raises(InvalidImageError):
        prepare_image(values)


def test_float_numpy_values_in_unit_interval_are_supported() -> None:
    image = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    result = prepare_image(image, reject_blank=False)

    assert result.shape == (1, 1, 28, 28)


def test_closed_pil_image_is_rejected() -> None:
    image = Image.new("L", (28, 28), 255)
    image.close()

    with pytest.raises(InvalidImageError):
        prepare_image(image)


def test_prepare_image_matches_torchvision_v2_pipeline() -> None:
    image = Image.new("RGB", (17, 23), "white")
    ImageDraw.Draw(image).ellipse((2, 4, 14, 19), fill="black")

    expected_transform = v2.Compose(
        [
            v2.Resize((28, 28), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    inverted = Image.eval(image.convert("L"), lambda value: 255 - value)
    expected = expected_transform(inverted).unsqueeze(0)

    torch.testing.assert_close(prepare_image(image), expected)


def test_zero_size_pil_image_is_rejected() -> None:
    image = Image.new("L", (0, 10))

    with pytest.raises(InvalidImageError):
        prepare_image(image)


def test_truncated_lazy_pil_image_is_rejected() -> None:
    source = Image.new("L", (28, 28), 255)
    output = BytesIO()
    source.save(output, format="PNG")
    image = Image.open(BytesIO(output.getvalue()[:-40]))

    with pytest.raises(InvalidImageError):
        prepare_image(image)


def test_fully_transparent_black_rgba_is_blank() -> None:
    image = Image.new("RGBA", (28, 28), (0, 0, 0, 0))
    array = np.zeros((28, 28, 4), dtype=np.uint8)

    assert is_blank(image)
    assert is_blank(array)


def test_semi_transparent_black_stroke_composites_on_white() -> None:
    rgba = Image.new("RGBA", (28, 28), (255, 255, 255, 255))
    ImageDraw.Draw(rgba).line((5, 5, 22, 22), fill=(0, 0, 0, 128), width=3)
    expected = Image.new("RGB", (28, 28), "white")
    expected.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))

    torch.testing.assert_close(prepare_image(rgba), prepare_image(expected))


def test_image_exceeding_pixel_cap_is_rejected() -> None:
    image = Image.new("L", (2001, 2000), 255)

    with pytest.raises(InvalidImageError):
        prepare_image(image)


def test_equivalent_float_and_uint8_inputs_match() -> None:
    uint8_image = np.full((28, 28), 255, dtype=np.uint8)
    uint8_image[8:20, 10:18] = 0

    torch.testing.assert_close(prepare_image(uint8_image), prepare_image(uint8_image / 255.0))
