"""Upstream diffct tests, run against the vendored engine ``diffct_mlx.engine``.

Some tests import upstream's example helpers (``examples._block_cgls`` and
friends). Those live in ``_upstream/examples`` and are put on ``sys.path`` here.
"""

import sys
from pathlib import Path

_UPSTREAM = Path(__file__).resolve().parent / "_upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
