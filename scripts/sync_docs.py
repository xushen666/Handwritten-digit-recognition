from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
README_PATH = ROOT / "README.md"
RESUME_PATH = ROOT / "docs" / "resume-project.md"
REPOSITORY_URL = "https://github.com/xushen666/Handwritten-digit-recognition.git"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def _real_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _accuracy(value: object, name: str) -> float:
    result = _real_number(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _latency(value: object) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise ValueError("latency_ms must be a JSON object")
    median = _real_number(value.get("median"), "latency median")
    p95 = _real_number(value.get("p95"), "latency p95")
    if median < 0:
        raise ValueError("latency median must be non-negative")
    if p95 < 0:
        raise ValueError("latency p95 must be non-negative")
    if p95 < median:
        raise ValueError("latency p95 must be greater than or equal to median")
    return median, p95


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _format_published_values(
    metrics: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, str]:
    test_accuracy = _accuracy(metrics.get("test_accuracy"), "test_accuracy")
    metadata_accuracy = _accuracy(metadata.get("test_accuracy"), "test_accuracy")
    validation_accuracy = _accuracy(
        metrics.get("best_validation_accuracy"), "best_validation_accuracy"
    )
    latency = _latency(metrics.get("latency_ms"))
    metadata_latency = _latency(metadata.get("latency_ms"))
    if test_accuracy != metadata_accuracy:
        raise ValueError("Published accuracy differs between metrics and model metadata")
    if latency != metadata_latency:
        raise ValueError("Published latency differs between metrics and model metadata")

    parameter_count = _positive_integer(metadata.get("parameter_count"), "parameter_count")
    configuration = metadata.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be a JSON object")
    seed_value = configuration.get("seed")
    if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    epochs = _positive_integer(configuration.get("epochs"), "epochs")
    return {
        "accuracy": f"{test_accuracy:.2%}",
        "validation_accuracy": f"{validation_accuracy:.2%}",
        "median": f"{latency[0]:.2f} ms",
        "p95": f"{latency[1]:.2f} ms",
        "parameters": f"{parameter_count:,}",
        "seed": str(seed_value),
        "epochs": str(epochs),
    }


def _published_values() -> dict[str, str]:
    return _format_published_values(_read_json(METRICS_PATH), _read_json(METADATA_PATH))


def _render_readme(values: dict[str, str]) -> str:
    return f"""# Handwritten Digit Recognition

An offline MNIST desktop application with a shared PyTorch inference core,
reproducible evaluation, and a PyQt5 drawing canvas.

## 中文说明

一个可复现评测、可打包发布的手写数字识别离线桌面应用。
该版本不包含后端服务，聚焦从模型训练到 Windows 客户端交付的完整闭环。

![PyQt 手写数字识别桌面端](docs/images/desktop.png)

## 技术栈

- Python 3.11、PyTorch、torchvision
- PyQt5、Pillow、NumPy
- pytest、Ruff、GitHub Actions、PyInstaller

## 架构与数据流

```text
MNIST -> 固定训练/验证划分 -> CNN 训练 -> 最佳验证权重 -> 最终测试与评测报告
PyQt 画板 -> 共享图像预处理 -> Predictor -> 数字、置信度、推理耗时
```

`src/digit_recognizer/core` 统一维护模型、预处理、路径解析和推理；
训练与 PyQt 客户端复用同一核心，避免算法逻辑漂移。

## 源码安装与启动

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -e ".[desktop]"
digit-desktop
```

应用完全离线运行；在 280×280 画板中书写一个数字，点击“识别”即可查看预测、置信度和耗时。

## 训练与质量检查

```powershell
python -m pip install -e ".[desktop,train,dev]"
python scripts/train.py --device auto --epochs {values["epochs"]}
python -m pytest -v
python -m ruff check .
```

训练固定随机种子 `{values["seed"]}`，从 MNIST 训练集划出独立验证集，
只按验证集选择最佳权重，测试集仅用于最终评测。

## 真实评测结果

| 指标 | 结果 |
| --- | ---: |
| MNIST 测试准确率 | {values["accuracy"]} |
| 最佳验证准确率 | {values["validation_accuracy"]} |
| CPU 单样本延迟 median | {values["median"]} |
| CPU 单样本延迟 P95 | {values["p95"]} |
| 模型参数量 | {values["parameters"]} |

评测数字由 `reports/metrics.json` 与 `models/model_metadata.json` 自动同步。
完整图表见[训练曲线](reports/training_curves.png)和
[混淆矩阵](reports/confusion_matrix.png)。

## Windows Release

版本标签会触发 GitHub Actions 构建 Windows onedir 压缩包；
发布后可在仓库的
[Releases](https://github.com/xushen666/Handwritten-digit-recognition/releases) 页面下载。

## 项目来源与职责

项目源于课程设计小组作业，公开版本由本人主导完成核心实现、工程化重构、
模型复训评测和交付整理。

## 局限

当前模型只面向 MNIST 风格的单数字分类，是教学与工程展示项目，
并非生产级 OCR，不适用于多字符、复杂背景或通用文档识别。

## License

本项目采用 [MIT License](LICENSE)。
"""


def _render_resume(values: dict[str, str]) -> str:
    shared_core = (
        "重构共享模型、图像预处理与 Predictor 推理核心，驱动 PyQt5 离线手写画板展示"
        "数字、置信度与延迟，消除训练和桌面推理逻辑重复。"
    )
    evaluation = (
        "建立无泄漏的可复现训练评测流程，以固定训练/验证划分选择最佳权重，"
        f'测试集仅最终评测；达到 {values["accuracy"]} 测试准确率，'
        f'CPU 单样本延迟 median {values["median"]}、P95 {values["p95"]}。'
    )
    delivery = (
        "使用 pytest 与 Ruff 覆盖核心、训练和桌面关键路径，通过 GitHub Actions 执行质量门，"
        "并以 PyInstaller onedir 交付 Windows 可执行程序。"
    )
    return f"""# 手写数字识别桌面应用

**技术栈：** Python 3.11 / PyTorch / torchvision / PyQt5 / pytest / GitHub Actions / PyInstaller

**GitHub：** {REPOSITORY_URL}

**项目背景：** 课程设计项目，核心实现与公开版本工程化重构由本人主导完成。

- {shared_core}
- {evaluation}
- {delivery}

适用边界：MNIST 风格单数字分类教学项目，并非生产级 OCR。
"""


def _render_documents() -> tuple[str, str]:
    values = _published_values()
    return _render_readme(values), _render_resume(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synchronize generated portfolio documentation")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated documents without changing files",
    )
    args = parser.parse_args(argv)
    readme, resume = _render_documents()

    if args.check:
        stale_paths = [
            path.relative_to(ROOT).as_posix()
            for path, expected in ((README_PATH, readme), (RESUME_PATH, resume))
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if stale_paths:
            print(f"Documentation is out of date: {', '.join(stale_paths)}", file=sys.stderr)
            return 1
        return 0

    RESUME_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_bytes(readme.encode("utf-8"))
    RESUME_PATH.write_bytes(resume.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
