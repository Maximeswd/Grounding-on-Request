"""The CPU checks in this folder are standalone scripts that stub torch on import.

pytest does not import them; test_checks.py runs each one in its own interpreter instead.
"""

from pathlib import Path

collect_ignore = [p.name for p in Path(__file__).parent.glob("test_*.py") if p.name != "test_checks.py"]
