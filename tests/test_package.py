import tomllib
from pathlib import Path

from setuptools import find_packages

from digit_recognizer import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_package_scope_is_desktop_only() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    optional_dependencies = project["optional-dependencies"]
    scripts = project["scripts"]
    serialized = str(project).lower()

    assert "api" not in optional_dependencies
    assert "digit-api" not in scripts
    for backend_dependency in ("fastapi", "uvicorn", "python-multipart", "httpx"):
        assert backend_dependency not in serialized


def test_package_discovery_excludes_server_interfaces() -> None:
    discovered_packages = set(find_packages(where="src"))

    assert "digit_recognizer.api" not in discovered_packages
    assert "digit_recognizer.web" not in discovered_packages
