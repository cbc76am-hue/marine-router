#!/usr/bin/env python3
"""Run the HTTP service.

Equivalent to ``python3 -m tolly_router.service``; provided as a
convenience entry point matching the ``scripts/`` pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router.service import main  # noqa: E402

if __name__ == "__main__":
    main()
