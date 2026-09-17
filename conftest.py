"""Keep test scratch files independent of shared Windows temp permissions."""

from pathlib import Path
from tempfile import mkdtemp

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Respect an explicitly supplied --basetemp. Otherwise allocate a fresh
    # parent per invocation, so different accounts never reuse owned folders.
    if config.option.basetemp is None:
        run_directory = Path(mkdtemp(prefix=".pytest-run-", dir=config.rootpath))
        config.option.basetemp = str(run_directory / "tmp")
