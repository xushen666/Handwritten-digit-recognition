from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

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


def test_sync_script_deterministically_generates_both_documents() -> None:
    assert SYNC_SCRIPT_PATH.is_file()
    committed_documents = (README_PATH.read_bytes(), RESUME_PATH.read_bytes())
    first = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT_PATH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    generated_once = (README_PATH.read_bytes(), RESUME_PATH.read_bytes())
    assert generated_once == committed_documents

    second = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT_PATH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    assert (README_PATH.read_bytes(), RESUME_PATH.read_bytes()) == generated_once


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
    assert SCREENSHOT_PATH.stat().st_size > 0
