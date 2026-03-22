from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def run_cli(project_root: Path):
    def _run(
        script_relpath: str,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        script = project_root / script_relpath
        full_env = {**os.environ, **(env or {})}
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=project_root,
            env=full_env,
            capture_output=True,
            text=True,
        )

    return _run
