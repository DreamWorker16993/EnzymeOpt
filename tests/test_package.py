from importlib.metadata import version
from pathlib import Path

import enzymeopt


def test_package_exposes_installed_version() -> None:
    assert enzymeopt.__version__ == version("enzymeopt")


def test_package_exports_only_public_version() -> None:
    assert enzymeopt.__all__ == ["__version__"]


def test_readme_documents_substrate_inhibition_limitation() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "not valid for assays with substrate inhibition" in readme
    assert "PROJECT_PLAN.md" not in readme
