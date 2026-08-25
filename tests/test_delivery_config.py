from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "packaging" / "windows.spec"
QUALITY_PATH = ROOT / ".github" / "workflows" / "quality.yml"
RELEASE_PATH = ROOT / ".github" / "workflows" / "release.yml"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing delivery configuration: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_windows_spec_packages_only_the_desktop_application() -> None:
    spec = _read(SPEC_PATH)
    compile(spec, str(SPEC_PATH), "exec")
    normalized = spec.replace("\\", "/").lower()

    assert 'name="handwrittendigitrecognizer"' in normalized
    assert "console=false" in normalized
    assert "collect(" in normalized  # COLLECT marks an onedir build.
    assert '"src" / "digit_recognizer" / "desktop" / "app.py"' in normalized
    assert '"src"' in normalized and "pathex=" in normalized
    assert '"models" / "mnist_cnn.pth"' in normalized
    assert '(str(model_path), "models")' in normalized
    assert '"digit_recognizer.training"' in normalized

    for forbidden_payload in ('"data"', '"reports"', '"tests"'):
        assert forbidden_payload not in normalized


def test_quality_workflow_runs_the_desktop_quality_gates() -> None:
    workflow = _read(QUALITY_PATH)
    normalized = workflow.lower()

    assert "contents: read" in normalized
    assert "python-version: \"3.11\"" in normalized
    assert "https://download.pytorch.org/whl/cpu" in normalized
    assert '"torch>=2.10,<2.12"' in normalized
    assert '"torchvision>=0.25,<0.27"' in normalized
    assert '.[desktop,dev]' in normalized
    assert "qt_qpa_platform: offscreen" in normalized
    for quality_gate in (
        "python -m ruff check .",
        "python -m pytest -v",
        "python -m build",
        "import digit_recognizer.desktop.app",
    ):
        assert quality_gate in normalized


def test_release_workflow_builds_a_windows_zip_for_version_tags() -> None:
    workflow = _read(RELEASE_PATH)
    normalized = workflow.replace("\\", "/").lower()

    assert "contents: write" in normalized
    assert "windows-latest" in normalized
    assert "tags:" in normalized and "v*" in normalized
    assert "python-version: \"3.11\"" in normalized
    assert "https://download.pytorch.org/whl/cpu" in normalized
    assert '.[desktop]' in normalized
    assert "pyinstaller" in normalized
    assert "packaging/windows.spec --clean --noconfirm" in normalized
    assert "compress-archive" in normalized
    assert "handwrittendigitrecognizer-windows-x64.zip" in normalized
    assert "softprops/action-gh-release@v2" in normalized


def test_delivery_configuration_has_no_backend_or_web_surface() -> None:
    delivery_config = "\n".join(
        _read(path).lower() for path in (SPEC_PATH, QUALITY_PATH, RELEASE_PATH)
    )
    for forbidden in ("docker", "fastapi", "uvicorn", "digit-api", "digit-web"):
        assert forbidden not in delivery_config
