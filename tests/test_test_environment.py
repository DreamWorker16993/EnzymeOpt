"""Regression coverage for inaccessible shared pytest directories."""

import subprocess
import sys
from pathlib import Path


def test_suite_works_when_shared_pytest_directories_are_inaccessible():
    root = Path(__file__).resolve().parents[1]
    script = '''
import os
import pytest
original = os.scandir
def guarded(path):
    if "pytest-of-" in str(path) or ".pytest_cache" in str(path):
        raise PermissionError("simulated shared-directory permission denial")
    return original(path)
os.scandir = guarded
raise SystemExit(pytest.main([
    "tests/test_cli.py::test_invalid_toml_shape_is_rejected", "-W", "error"
]))
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_temporary_directory_is_writable(tmp_path):
    path = tmp_path / "probe.txt"
    path.write_text("writable", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "writable"
