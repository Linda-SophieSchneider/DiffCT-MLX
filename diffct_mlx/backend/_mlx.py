"""MLX / Metal backend (Apple Silicon).

Wiring of this backend against the MLX Metal kernels is completed in a later
phase (see ATTRIBUTION.md / the project roadmap). On Apple Silicon the MLX port
lives on the ``main`` branch of DiffCT-MLX; this module will bridge to it so the
unified ``diffct_mlx`` API dispatches to Metal kernels.

Until then, selecting this backend and calling an operator raises a clear error.
It is only ever selected when ``mlx.core`` is importable, so the Torch/CUDA path
is unaffected.
"""

NAME = "mlx"

try:  # pragma: no cover - only meaningful on Apple Silicon
    import mlx.core as _mx
    from types import SimpleNamespace as _NS

    float32 = _mx.float32

    #: MLX-core-shaped array namespace (same interface as the torch backend's).
    xp = _NS(
        array=lambda value, dtype=None: (_mx.array(value) if dtype is None else _mx.array(value, dtype=dtype)),
        zeros=lambda shape, dtype=None: _mx.zeros(tuple(shape), dtype=dtype or _mx.float32),
        ones=lambda shape, dtype=None: _mx.ones(tuple(shape), dtype=dtype or _mx.float32),
        zeros_like=_mx.zeros_like,
        maximum=_mx.maximum,
        minimum=_mx.minimum,
        where=_mx.where,
        eval=_mx.eval,
        norm=lambda x: float(_mx.linalg.norm(x)),
        float32=_mx.float32,
    )
except Exception:  # pragma: no cover
    xp = None
    float32 = None


def as_array(data, dtype=None, device=None):  # pragma: no cover - Apple only
    a = data if xp is not None and isinstance(data, xp.array) else xp.array(data)
    if dtype is not None:
        a = a.astype(dtype)
    return a


def to_numpy(x):  # pragma: no cover - Apple only
    import numpy as np
    return np.array(x)


def device_of(x):  # pragma: no cover - Apple only (unified memory)
    return None


def clamp_min(x, value):  # pragma: no cover - Apple only
    return xp.maximum(x, float(value))


def _pending(*_args, **_kwargs):
    raise NotImplementedError(
        "The MLX backend is not wired up on this branch yet. "
        "Use the DiffCT-MLX 'main' branch on Apple Silicon, or set "
        "DIFFCT_BACKEND=torch to force the CUDA backend."
    )


# Placeholder operators (filled in when the MLX kernels are bridged).
parallel_forward = parallel_backward = _pending
fan_forward = fan_backward = _pending
cone_forward = cone_backward = _pending
geometry = None
analytical = None
