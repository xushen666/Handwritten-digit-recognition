from __future__ import annotations

import sys
from io import BytesIO
from typing import Protocol

from PIL import Image

# PyTorch must load before PyQt on Windows to avoid c10.dll initialization failures.
# isort: off
from digit_recognizer.core.predictor import Prediction, Predictor

from PyQt5.QtCore import QBuffer, QIODevice, QPoint, Qt
from PyQt5.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
# isort: on

from digit_recognizer.core.errors import BlankImageError, ModelLoadError
from digit_recognizer.core.paths import default_model_path


class PredictorLike(Protocol):
    def predict(self, image: Image.Image) -> Prediction: ...


class DrawingCanvas(QWidget):
    SIZE = 280
    PEN_WIDTH = 15

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._image = QImage(self.SIZE, self.SIZE, QImage.Format_RGB32)
        self._drawing = False
        self._last_point = QPoint()
        self.clear()

    def clear(self) -> None:
        self._image.fill(Qt.white)
        self.update()

    def to_pil_image(self) -> Image.Image:
        buffer = QBuffer()
        if not buffer.open(QIODevice.WriteOnly):
            raise RuntimeError("Unable to create the in-memory canvas buffer")
        if not self._image.save(buffer, "PNG"):
            raise RuntimeError("Unable to encode the canvas image")
        with Image.open(BytesIO(bytes(buffer.data()))) as image:
            return image.copy()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._last_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._drawing:
            return
        painter = QPainter(self._image)
        painter.setPen(
            QPen(Qt.black, self.PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        painter.drawLine(self._last_point, event.pos())
        self._last_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drawing = False


class DigitRecognizerWindow(QWidget):
    def __init__(self, predictor: PredictorLike) -> None:
        super().__init__()
        self._predictor = predictor
        self.setWindowTitle("手写数字识别")

        self.canvas = DrawingCanvas()
        self.result_label = QLabel("识别结果：--")
        self.confidence_label = QLabel("置信度：--")
        self.latency_label = QLabel("推理耗时：--")
        self.recognize_button = QPushButton("识别")
        self.clear_button = QPushButton("清除")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.recognize_button)
        button_layout.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.confidence_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.latency_label, alignment=Qt.AlignCenter)

        self.recognize_button.clicked.connect(self.recognize_digit)
        self.clear_button.clicked.connect(self.clear_canvas)

    def clear_canvas(self) -> None:
        self.canvas.clear()
        self.result_label.setText("识别结果：--")
        self.confidence_label.setText("置信度：--")
        self.latency_label.setText("推理耗时：--")

    def recognize_digit(self) -> None:
        try:
            prediction = self._predictor.predict(self.canvas.to_pil_image())
        except BlankImageError:
            QMessageBox.information(self, "无法识别", "请先在画板上书写数字。")
            return
        except Exception:
            QMessageBox.critical(self, "识别失败", "暂时无法完成识别，请重试。")
            return

        self.result_label.setText(f"识别结果：{prediction.digit}")
        self.confidence_label.setText(f"置信度：{prediction.confidence:.2%}")
        self.latency_label.setText(f"推理耗时：{prediction.latency_ms:.2f} ms")


def main() -> int:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    try:
        predictor = Predictor.load(default_model_path())
    except ModelLoadError:
        QMessageBox.critical(
            None,
            "模型加载失败",
            "无法加载识别模型，请重新安装或检查模型文件。",
        )
        return 1

    window = DigitRecognizerWindow(predictor)
    window.show()
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
