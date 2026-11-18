#!/usr/bin/env python3
"""Print the environment record as JSON. Thin wrapper over ``passk.envinfo``.

The implementation lives in the package, so an installed wheel carries it too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from passk.envinfo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
