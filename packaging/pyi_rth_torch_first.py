"""Load PyTorch before PyInstaller's standard PyQt5 runtime hook on Windows."""

import torch  # noqa: F401
