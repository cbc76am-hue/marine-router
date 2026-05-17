"""Module entry-point so the CLI works as `python3 -m tolly_router.build_nogo`.
The real script lives in scripts/build_nogo.py; this just delegates."""

import runpy
import sys
from pathlib import Path

_script = Path(__file__).resolve().parent.parent / "scripts" / "build_nogo.py"
sys.argv[0] = str(_script)
runpy.run_path(str(_script), run_name="__main__")
