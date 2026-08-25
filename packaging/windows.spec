# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(SPECPATH).resolve().parent
SOURCE_ROOT = ROOT / "src"
ENTRY_POINT = ROOT / "src" / "digit_recognizer" / "desktop" / "app.py"
MODEL_PATH = ROOT / "models" / "mnist_cnn.pth"
TORCH_RUNTIME_HOOK = ROOT / "pyi_rth_torch_first.py"

hidden_imports = [
    "PyQt5.sip",
    "numpy",
    "PIL",
    "torch",
    "torchvision",
    *collect_submodules("torchvision.transforms.v2"),
]

analysis = Analysis(
    [str(ENTRY_POINT)],
    pathex=[str(SOURCE_ROOT)],
    binaries=[],
    datas=[(str(MODEL_PATH), "models")],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(TORCH_RUNTIME_HOOK)],
    excludes=["digit_recognizer.training", "matplotlib", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="HandwrittenDigitRecognizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HandwrittenDigitRecognizer",
)
