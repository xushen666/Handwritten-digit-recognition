# Windows Release Startup Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the packaged Windows desktop application loads PyTorch before PyQt and make Release smoke tests reject frozen startup error dialogs.

**Architecture:** A custom PyInstaller runtime hook imports PyTorch before PyInstaller's standard PyQt5 hook. The desktop entry point exposes an internal `--smoke-test` path that assembles the real application and exits deterministically, allowing the Release workflow to validate timeout and exit status.

**Tech Stack:** Python 3.11, PyTorch, PyQt5, PyInstaller 6, pytest, Ruff, GitHub Actions PowerShell

---

### Task 1: Force PyTorch to load before PyQt in the frozen application

**Files:**
- Create: `packaging/pyi_rth_torch_first.py`
- Modify: `packaging/windows.spec`
- Test: `tests/test_delivery_config.py`

- [ ] **Step 1: Write the failing delivery test**

Add assertions that `packaging/pyi_rth_torch_first.py` exists, contains an `import torch`, and that
`windows.spec` defines its path and passes it through `runtime_hooks`:

```python
RUNTIME_HOOK_PATH = ROOT / "packaging" / "pyi_rth_torch_first.py"


def test_windows_spec_loads_torch_before_standard_pyqt_runtime_hook() -> None:
    hook = _read(RUNTIME_HOOK_PATH)
    spec = _read(SPEC_PATH).replace("\\", "/").lower()

    assert "import torch" in hook
    assert 'root / "packaging" / "pyi_rth_torch_first.py"' in spec
    assert "runtime_hooks=[str(torch_runtime_hook)]" in spec
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/test_delivery_config.py::test_windows_spec_loads_torch_before_standard_pyqt_runtime_hook -v
```

Expected: FAIL because the runtime hook file and spec registration do not exist.

- [ ] **Step 3: Add the minimal runtime hook and spec registration**

Create the hook:

```python
"""Load PyTorch before PyInstaller's standard PyQt5 runtime hook on Windows."""

import torch  # noqa: F401
```

Add to `windows.spec`:

```python
TORCH_RUNTIME_HOOK = ROOT / "packaging" / "pyi_rth_torch_first.py"
```

and change the `Analysis` argument to:

```python
runtime_hooks=[str(TORCH_RUNTIME_HOOK)],
```

- [ ] **Step 4: Run the focused test and Ruff**

Run:

```powershell
python -m pytest tests/test_delivery_config.py::test_windows_spec_loads_torch_before_standard_pyqt_runtime_hook -v
python -m ruff check packaging tests/test_delivery_config.py
```

Expected: PASS and `All checks passed!`.

### Task 2: Add a deterministic frozen-application smoke mode

**Files:**
- Modify: `src/digit_recognizer/desktop/app.py`
- Test: `tests/desktop/test_canvas.py`

- [ ] **Step 1: Write the failing smoke-mode test**

Add a test that supplies explicit arguments and verifies the real assembly path without showing the window or
entering the event loop:

```python
def test_main_smoke_test_loads_window_without_starting_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeApplication:
        def __init__(self, arguments: list[str]) -> None:
            calls["arguments"] = arguments

        def setStyle(self, style: str) -> None:
            calls["style"] = style

        def exec_(self) -> int:
            raise AssertionError("smoke test must not enter the event loop")

    class FakeWindow:
        def __init__(self, predictor: object) -> None:
            calls["predictor"] = predictor

        def show(self) -> None:
            raise AssertionError("smoke test must not show the window")

    expected_predictor = object()
    monkeypatch.setattr(desktop_app, "QApplication", FakeApplication)
    monkeypatch.setattr(desktop_app, "DigitRecognizerWindow", FakeWindow)
    monkeypatch.setattr(desktop_app, "default_model_path", lambda: "model-path")
    monkeypatch.setattr(desktop_app.Predictor, "load", lambda _path: expected_predictor)

    arguments = ["HandwrittenDigitRecognizer.exe", "--smoke-test"]
    assert desktop_app.main(arguments) == 0
    assert calls["arguments"] == arguments
    assert calls["style"] == "Fusion"
    assert calls["predictor"] is expected_predictor
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/desktop/test_canvas.py::test_main_smoke_test_loads_window_without_starting_event_loop -v
```

Expected: FAIL because `main()` does not accept arguments or implement smoke mode.

- [ ] **Step 3: Implement the minimal smoke mode**

Change the entry point to accept optional arguments, assemble the window, and return before `show()`:

```python
def main(arguments: list[str] | None = None) -> int:
    application_arguments = sys.argv if arguments is None else arguments
    application = QApplication(application_arguments)
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
    if "--smoke-test" in application_arguments[1:]:
        return 0
    window.show()
    return int(application.exec_())
```

- [ ] **Step 4: Run desktop tests and Ruff**

Run:

```powershell
python -m pytest tests/desktop/test_canvas.py -v
python -m ruff check src/digit_recognizer/desktop/app.py tests/desktop/test_canvas.py
```

Expected: all desktop tests pass and Ruff reports no errors.

### Task 3: Make the Release smoke gate deterministic

**Files:**
- Modify: `.github/workflows/release.yml`
- Test: `tests/test_delivery_config.py`

- [ ] **Step 1: Tighten the failing workflow contract test**

Require `--smoke-test`, `WaitForExit(30000)`, and `ExitCode`, and reject the old fixed sleep:

```python
assert '"--smoke-test"' in normalized
assert ".waitforexit(30000)" in normalized
assert ".exitcode" in normalized
assert "start-sleep" not in normalized
```

Update the ordered gates so the smoke-test arguments, timeout, exit-code check, cleanup, compression and Release
publication appear in that order.

- [ ] **Step 2: Run the workflow contract test and verify RED**

Run:

```powershell
python -m pytest tests/test_delivery_config.py::test_release_workflow_builds_a_windows_zip_for_version_tags -v
```

Expected: FAIL because the workflow still uses `Start-Sleep` and process liveness.

- [ ] **Step 3: Replace the smoke-test script**

Use the frozen smoke mode with a timeout and explicit status:

```powershell
$process = $null
try {
  $process = Start-Process -FilePath $executablePath -ArgumentList "--smoke-test" -PassThru -WindowStyle Hidden
  if (-not $process.WaitForExit(30000)) {
    throw "Packaged application did not finish its smoke test within 30 seconds."
  }
  $process.Refresh()
  if ($process.ExitCode -ne 0) {
    throw "Packaged application smoke test failed with exit code $($process.ExitCode)."
  }
}
finally {
  if ($null -ne $process -and -not $process.HasExited) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
  }
}
```

- [ ] **Step 4: Run delivery tests and Ruff**

Run:

```powershell
python -m pytest tests/test_delivery_config.py -v
python -m ruff check tests/test_delivery_config.py
```

Expected: all delivery tests pass and Ruff reports no errors.

### Task 4: Verify the complete release path

**Files:**
- Verify only; no additional files expected

- [ ] **Step 1: Run repository quality gates**

Run:

```powershell
python -m ruff check .
python -m pytest -v
python scripts/sync_docs.py --check
```

Expected: Ruff passes, all tests pass, and documentation is current.

- [ ] **Step 2: Build the Windows onedir package**

Run:

```powershell
pyinstaller packaging/windows.spec --clean --noconfirm
```

Expected: `dist/HandwrittenDigitRecognizer/HandwrittenDigitRecognizer.exe` and
`dist/HandwrittenDigitRecognizer/_internal/models/mnist_cnn.pth` exist.

- [ ] **Step 3: Run the packaged smoke mode**

Run the same timeout and exit-code script used by `.github/workflows/release.yml` against the local EXE.

Expected: the process exits within 30 seconds with code 0 and no error dialog remains.

- [ ] **Step 4: Inspect and commit the implementation**

Run:

```powershell
git diff --check
git status --short
```

Expected: only the planned runtime hook, spec, desktop entry, workflow, tests and planning documents changed.
Commit the implementation with:

```powershell
git add packaging/pyi_rth_torch_first.py packaging/windows.spec src/digit_recognizer/desktop/app.py tests/desktop/test_canvas.py tests/test_delivery_config.py .github/workflows/release.yml
git commit -m "fix: validate frozen Windows startup"
```
