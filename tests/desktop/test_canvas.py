from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

from digit_recognizer.core.errors import BlankImageError, ModelLoadError
from digit_recognizer.core.predictor import Prediction
from digit_recognizer.desktop import app as desktop_app
from digit_recognizer.desktop.app import DigitRecognizerWindow, DrawingCanvas


def test_desktop_entry_loads_pytorch_before_pyqt() -> None:
    source = Path(desktop_app.__file__).read_text(encoding="utf-8")

    assert source.index("from digit_recognizer.core.predictor import") < source.index(
        "from PyQt5"
    )


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    return application


def _mouse_event(event_type: int, x: int, y: int, button: Qt.MouseButton) -> QMouseEvent:
    return QMouseEvent(event_type, QPoint(x, y), button, button, Qt.NoModifier)


def test_canvas_starts_white_and_converts_to_memory_png(qt_app: QApplication) -> None:
    canvas = DrawingCanvas()

    image = canvas.to_pil_image()

    assert isinstance(image, Image.Image)
    assert image.size == (280, 280)
    assert image.convert("RGB").getpixel((140, 140)) == (255, 255, 255)


def test_canvas_draws_in_its_own_coordinates_and_clears(qt_app: QApplication) -> None:
    canvas = DrawingCanvas()
    canvas.mousePressEvent(_mouse_event(QMouseEvent.MouseButtonPress, 20, 30, Qt.LeftButton))
    canvas.mouseMoveEvent(_mouse_event(QMouseEvent.MouseMove, 80, 30, Qt.NoButton))
    canvas.mouseReleaseEvent(_mouse_event(QMouseEvent.MouseButtonRelease, 80, 30, Qt.LeftButton))

    drawn = canvas.to_pil_image().convert("RGB")
    assert drawn.getpixel((50, 30)) == (0, 0, 0)
    assert drawn.getpixel((50, 80)) == (255, 255, 255)

    canvas.clear()

    assert canvas.to_pil_image().convert("RGB").getbbox() == (0, 0, 280, 280)
    assert canvas.to_pil_image().convert("RGB").getpixel((50, 30)) == (255, 255, 255)


@dataclass
class StubPredictor:
    result: Prediction | None = None
    error: Exception | None = None
    received: Image.Image | None = None

    def predict(self, image: Image.Image) -> Prediction:
        self.received = image
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_window_displays_prediction_confidence_and_latency(qt_app: QApplication) -> None:
    predictor = StubPredictor(result=Prediction(digit=7, confidence=0.9876, latency_ms=3.21))
    window = DigitRecognizerWindow(predictor)

    window.recognize_digit()

    assert predictor.received is not None
    assert predictor.received.size == (280, 280)
    assert "7" in window.result_label.text()
    assert "98.76%" in window.confidence_label.text()
    assert "3.21 ms" in window.latency_label.text()


def test_window_maps_blank_image_to_actionable_message(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, message: shown.append((title, message)),
    )
    window = DigitRecognizerWindow(StubPredictor(error=BlankImageError("internal details")))

    window.recognize_digit()

    assert shown == [("无法识别", "请先在画板上书写数字。")]
    assert "internal details" not in str(shown)


def test_window_maps_unexpected_inference_error_without_leaking_details(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: shown.append((title, message)),
    )
    secret = r"C:\\private\\model.pth traceback"
    window = DigitRecognizerWindow(StubPredictor(error=RuntimeError(secret)))

    window.recognize_digit()

    assert shown == [("识别失败", "暂时无法完成识别，请重试。")]
    assert secret not in str(shown)


def test_main_uses_fusion_and_returns_event_loop_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeApplication:
        def __init__(self, arguments: list[str]) -> None:
            calls["arguments"] = arguments

        def setStyle(self, style: str) -> None:
            calls["style"] = style

        def exec_(self) -> int:
            return 23

    class FakeWindow:
        def __init__(self, predictor: object) -> None:
            calls["predictor"] = predictor

        def show(self) -> None:
            calls["shown"] = True

    expected_predictor = object()
    monkeypatch.setattr(desktop_app, "QApplication", FakeApplication)
    monkeypatch.setattr(desktop_app, "DigitRecognizerWindow", FakeWindow)
    monkeypatch.setattr(desktop_app, "default_model_path", lambda: "model-path")
    monkeypatch.setattr(
        desktop_app.Predictor,
        "load",
        lambda path: expected_predictor if path == "model-path" else None,
    )

    assert desktop_app.main() == 23
    assert calls["style"] == "Fusion"
    assert calls["predictor"] is expected_predictor
    assert calls["shown"] is True


def test_main_maps_model_load_error_to_safe_message(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[tuple[str, str]] = []

    class FakeApplication:
        def __init__(self, _arguments: list[str]) -> None:
            pass

        def setStyle(self, _style: str) -> None:
            pass

    def fail_load(_path: object) -> None:
        raise ModelLoadError(r"C:\\private\\missing.pth")

    monkeypatch.setattr(desktop_app, "QApplication", FakeApplication)
    monkeypatch.setattr(desktop_app.Predictor, "load", fail_load)
    monkeypatch.setattr(
        desktop_app.QMessageBox,
        "critical",
        lambda _parent, title, message: shown.append((title, message)),
    )

    assert desktop_app.main() == 1
    assert shown == [("模型加载失败", "无法加载识别模型，请重新安装或检查模型文件。")]
    assert "private" not in str(shown)
