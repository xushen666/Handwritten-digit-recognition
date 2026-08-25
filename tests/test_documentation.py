from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts import sync_docs

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
RESUME_PATH = ROOT / "docs" / "resume-project.md"
SCREENSHOT_PATH = ROOT / "docs" / "images" / "desktop.png"
SYNC_SCRIPT_PATH = ROOT / "scripts" / "sync_docs.py"
METRICS_PATH = ROOT / "reports" / "metrics.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing documentation artifact: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def _formatted_metrics() -> tuple[str, str, str]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return (
        f'{float(metrics["test_accuracy"]):.2%}',
        f'{float(metrics["latency_ms"]["median"]):.2f} ms',
        f'{float(metrics["latency_ms"]["p95"]):.2f} ms',
    )


def _run_sync(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SYNC_SCRIPT_PATH), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_sync_check_accepts_current_documents_without_writing() -> None:
    assert SYNC_SCRIPT_PATH.is_file()
    before = (README_PATH.read_bytes(), RESUME_PATH.read_bytes())

    result = _run_sync("--check")

    assert result.returncode == 0, result.stderr
    assert (README_PATH.read_bytes(), RESUME_PATH.read_bytes()) == before


def test_sync_check_rejects_stale_documents_without_writing() -> None:
    original_readme = README_PATH.read_bytes()
    original_resume = RESUME_PATH.read_bytes()
    published_accuracy = _formatted_metrics()[0].encode()
    stale_accuracy = b"0.00%" if published_accuracy != b"0.00%" else b"100.00%"
    stale_readme = original_readme.replace(published_accuracy, stale_accuracy, 1)
    assert stale_readme != original_readme
    try:
        README_PATH.write_bytes(stale_readme)
        before_check = (README_PATH.read_bytes(), RESUME_PATH.read_bytes())

        result = _run_sync("--check")

        assert result.returncode != 0
        assert "out of date" in result.stderr.lower()
        assert (README_PATH.read_bytes(), RESUME_PATH.read_bytes()) == before_check
    finally:
        README_PATH.write_bytes(original_readme)
        RESUME_PATH.write_bytes(original_resume)


def test_default_sync_writes_both_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readme_path = tmp_path / "README.md"
    resume_path = tmp_path / "docs" / "resume-project.md"
    monkeypatch.setattr(sync_docs, "README_PATH", readme_path)
    monkeypatch.setattr(sync_docs, "RESUME_PATH", resume_path)

    assert sync_docs.main([]) == 0
    published_accuracy = _formatted_metrics()[0]
    assert published_accuracy in readme_path.read_text(encoding="utf-8")
    assert published_accuracy in resume_path.read_text(encoding="utf-8")


def _valid_published_data() -> tuple[dict[str, object], dict[str, object]]:
    latency = {"median": 2.5, "p95": 3.5}
    metrics: dict[str, object] = {
        "test_accuracy": 0.9957,
        "best_validation_accuracy": 0.9966,
        "latency_ms": latency.copy(),
    }
    metadata: dict[str, object] = {
        "test_accuracy": 0.9957,
        "latency_ms": latency.copy(),
        "parameter_count": 585_578,
        "configuration": {"seed": 42, "epochs": 15},
    }
    return metrics, metadata


@pytest.mark.parametrize(
    ("document", "path", "invalid_value", "message"),
    [
        ("metrics", ("test_accuracy",), True, "test_accuracy must be a real number"),
        ("metadata", ("test_accuracy",), "0.9957", "test_accuracy must be a real number"),
        (
            "metrics",
            ("best_validation_accuracy",),
            1.01,
            "best_validation_accuracy must be between 0 and 1",
        ),
        ("metrics", ("latency_ms", "median"), -0.1, "latency median must be non-negative"),
        (
            "metrics",
            ("latency_ms", "p95"),
            2.0,
            "latency p95 must be greater than or equal to median",
        ),
        ("metadata", ("parameter_count",), True, "parameter_count must be a positive integer"),
        ("metadata", ("parameter_count",), 0, "parameter_count must be a positive integer"),
        ("metadata", ("configuration", "seed"), -1, "seed must be a non-negative integer"),
        ("metadata", ("configuration", "seed"), 1.5, "seed must be a non-negative integer"),
        ("metadata", ("configuration", "epochs"), 0, "epochs must be a positive integer"),
        ("metadata", ("configuration", "epochs"), "15", "epochs must be a positive integer"),
    ],
)
def test_published_values_reject_invalid_quantitative_data(
    document: str,
    path: tuple[str, ...],
    invalid_value: object,
    message: str,
) -> None:
    metrics, metadata = _valid_published_data()
    target = metrics if document == "metrics" else metadata
    current: object = target
    for key in path[:-1]:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict)
    current[path[-1]] = invalid_value

    with pytest.raises(ValueError, match=message):
        sync_docs._format_published_values(metrics, metadata)


def test_documents_use_the_published_json_metrics() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metadata["test_accuracy"] == metrics["test_accuracy"]
    assert metadata["latency_ms"] == metrics["latency_ms"]

    expected_values = _formatted_metrics()
    for document_path in (README_PATH, RESUME_PATH):
        document = _read(document_path)
        for value in expected_values:
            assert value in document


def test_resume_has_exactly_three_project_bullets_and_required_facts() -> None:
    resume = _read(RESUME_PATH)
    bullets = re.findall(r"^- .+$", resume, flags=re.MULTILINE)

    assert len(bullets) == 3
    assert "PyQt" in bullets[0] and "共享" in bullets[0]
    assert "无泄漏" in bullets[1] and "可复现" in bullets[1]
    assert "pytest" in bullets[2] and "GitHub Actions" in bullets[2] and "PyInstaller" in bullets[2]
    assert "https://github.com/xushen666/Handwritten-digit-recognition.git" in resume
    assert "课程" in resume and "主导" in resume


def test_readme_describes_the_desktop_delivery_and_real_reports() -> None:
    readme = _read(README_PATH)
    english_intro = readme.split("## 中文说明", maxsplit=1)[0]
    assert len([line for line in english_intro.splitlines() if line.strip()]) <= 5

    for required in (
        "docs/images/desktop.png",
        "PyTorch",
        "PyQt5",
        "digit-desktop",
        "scripts/train.py",
        "pytest",
        "ruff",
        "reports/training_curves.png",
        "reports/confusion_matrix.png",
        "Windows Release",
        "课程",
        "主导",
        "MNIST",
        "单数字",
        "非生产级 OCR",
        "MIT",
    ):
        assert required in readme
    assert "离线桌面应用" in readme
    assert "后端服务" in readme
    assert 'python -m pip install -e ".[desktop,train,dev]"' in readme


def test_public_documents_do_not_advertise_removed_delivery_surfaces() -> None:
    documents = "\n".join(_read(path).lower() for path in (README_PATH, RESUME_PATH))
    for forbidden in ("fastapi", "uvicorn", "docker", "web", "curl"):
        assert forbidden not in documents
    assert not re.search(r"[a-z]:\\", documents)


def test_documents_do_not_replace_exact_metrics_with_ambiguous_ranges() -> None:
    documents = "\n".join(_read(path).lower() for path in (README_PATH, RESUME_PATH))
    ambiguous_patterns = (
        r"\d+(?:\.\d+)?%\s*\+",
        r">=|以上",
        r"\d+(?:\.\d+)?%\s*[-–—~～至到]\s*\d+(?:\.\d+)?%",
        r"(?:约|大约|左右|≈|~)\s*\d+(?:\.\d+)?\s*ms",
        r"\d+(?:\.\d+)?\s*ms\s*(?:左右|上下)",
    )
    for pattern in ambiguous_patterns:
        assert not re.search(pattern, documents)


def test_desktop_screenshot_exists() -> None:
    assert SCREENSHOT_PATH.is_file()
    with Image.open(SCREENSHOT_PATH) as screenshot:
        assert screenshot.format == "PNG"
        assert screenshot.width >= 280
        assert screenshot.height >= 400
        screenshot.verify()
