"""Backend selection for the unified ``diffct_mlx`` package.

At import time exactly one compute backend is chosen:

* ``mlx``   — Apple Silicon (Metal kernels), used when ``mlx.core`` imports.
* ``torch`` — Torch / numba-CUDA, used otherwise.

Override with the ``DIFFCT_BACKEND`` environment variable (``"mlx"`` / ``"torch"``).
The rest of the package imports :data:`active` (the chosen backend module),
:data:`xp` (its array namespace) and :data:`NAME`.
"""

import os


def _mlx_usable():
    """True when ``mlx`` imports AND its Metal (Apple GPU) backend is available.

    MLX also ships Linux/CPU wheels; on such systems the Metal kernels this
    package uses cannot run, so auto-detection must not pick MLX just because
    the module imports (an explicit ``DIFFCT_BACKEND=mlx`` still forces it).
    """
    try:
        import mlx.core as mx
    except Exception:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return True  # very old mlx without mx.metal — Apple-only anyway


def _detect():
    forced = os.environ.get("DIFFCT_BACKEND", "").strip().lower()
    if forced == "mlx":
        try:
            import mlx.core  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "DIFFCT_BACKEND=mlx is set but 'mlx' is not importable. "
                "Install it with: pip install mlx"
            ) from exc
        return forced
    if forced == "torch":
        return forced
    if forced:
        raise ValueError(
            f"DIFFCT_BACKEND must be 'mlx' or 'torch', got {forced!r}"
        )
    if _mlx_usable():
        return "mlx"
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
