"""Run every standalone CPU check script in tests/ in its own interpreter."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SCRIPTS = sorted(p for p in HERE.glob("test_*.py") if p.name != Path(__file__).name)
NEEDS = {"test_pair_carriers": ("torch",)}  # the other checks stub torch themselves


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_check_script(script: Path) -> None:
    for module in NEEDS.get(script.stem, ()):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"needs {module}")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=HERE.parent,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
