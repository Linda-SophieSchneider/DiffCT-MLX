"""Backend selection for the unified ``diffct_mlx`` package.

At import time exactly one compute backend is chosen:

* ``mlx``   — Apple Silicon (Metal kernels), used when ``mlx.core`` imports.
* ``torch`` — Torch / numba-CUDA, used otherwise.

Override with the ``DIFFCT_BACKEND`` environment variable (``"mlx"`` / ``"torch"``).
The rest of the package imports :data:`active` (the chosen backend module),
:data:`xp` (its array namespace) and :data:`NAME`.
"""

import os


def _detect():
    forced = os.environ.get("DIFFCT_BACKEND", "").strip().lower()
    if forced in ("mlx", "torch"):
        return forced
    if forced:
        raise ValueError(
            f"DIFFCT_BACKEND must be 'mlx' or 'torch', got {forced!r}"
        )
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        return "torch"
    except Exception:
        pass
    raise ImportError(
        "No usable backend found. Install 'mlx' on Apple Silicon or 'torch' "
        "(with a CUDA build + numba) elsewhere."
    )


NAME = _detect()

if NAME == "mlx":
    from . import _mlx as active
else:
    from . import _torch as active

xp = active.xp
