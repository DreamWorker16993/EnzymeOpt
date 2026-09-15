from importlib.metadata import version

import enzymeopt


def test_package_exposes_installed_version() -> None:
    assert enzymeopt.__version__ == version("enzymeopt")


def test_package_exports_only_public_version() -> None:
    assert enzymeopt.__all__ == ["__version__"]

