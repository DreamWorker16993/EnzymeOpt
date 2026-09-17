"""Keep test scratch files independent of shared Windows temp permissions."""

from pathlib import Path
import shutil
from tempfile import mkdtemp

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Respect an explicitly supplied --basetemp. Otherwise allocate a fresh
    # parent per invocation, so different accounts never reuse owned folders.
    if config.option.basetemp is None:
        run_directory = Path(mkdtemp(prefix=".pytest-run-", dir=config.rootpath))
        config.option.basetemp = str(run_directory / "tmp")
        config._enzymeopt_run_directory = run_directory


def pytest_unconfigure(config):
    """Remove the invocation's private scratch directory after pytest exits."""

    run_directory = getattr(config, "_enzymeopt_run_directory", None)
    if run_directory is not None:
        shutil.rmtree(run_directory, ignore_errors=True)
