"""Test fixtures that do not depend on a shared operating-system temp folder."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide a unique per-test directory without pytest's shared temp root.

    Pytest's built-in ``tmp_path`` scans and reuses a directory such as
    ``%TEMP%/pytest-of-<user>``. On Windows that shared directory can retain an
    ACL from another Python/IDE process and make every test fail during setup.
    A UUID directory directly below this project has no shared parent to scan or
    delete. Stale directories from interrupted runs never block later runs.
    """

    case_directory = PROJECT_ROOT / f".pytest_case_{uuid4().hex}"
    case_directory.mkdir()
    try:
        yield case_directory
    finally:
        # Test files are disposable. A stale directory is harmless and ignored
        # by Git if Windows temporarily holds a file handle during cleanup.
        shutil.rmtree(case_directory, ignore_errors=True)
