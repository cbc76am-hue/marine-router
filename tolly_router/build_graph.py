"""Module entry-point so the CLI works as `python3 -m tolly_router.build_graph`.
The real script lives in scripts/build_graph.py; this just delegates."""

import runpy
import sys
from pathlib import Path

_script = Path(__file__).resolve().parent.parent / "scripts" / "build_graph.py"
sys.argv[0] = str(_script)
runpy.run_path(str(_script), run_name="__main__")
