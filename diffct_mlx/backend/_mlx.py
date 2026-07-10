"""MLX / Metal backend (Apple Silicon).

Adapter over the vendored Metal implementation (:mod:`diffct_mlx.backend.metal`):
re-exports the ``mx.custom_function`` projectors under the unified functional
operator API and provides :data:`xp` — an ``mlx.core``-shaped array namespace
mirroring the torch backend's, so the higher-level reconstruction layer is
written once for both backends.

This module is only imported when the backend selector picked MLX (i.e.
``mlx.core`` is importable); the Metal kernels require an Apple-Silicon GPU at
call time.
"""

from types import SimpleNamespace as _NS

import mlx.core as _mx

NAME = "mlx"

float32 = _mx.float32


def _mlx_clip(x, lo=None, hi=None):
    if lo is not None:
        x = _mx.maximum(x, lo)
    if hi is not None:
        x = _mx.minimum(x, hi)
    return x


def _mlx_fftfreq(n):
    n = int(n)
    k = _mx.arange(n, dtype=_mx.float32)
    n_half = (n + 1) // 2
    return _mx.where(k < n_half, k, k - n) / float(n)


def _mlx_take(a, indices):
    """Gather from flattened ``a``; integer index dtypes are preserved (int64
    stays int64 — float32 index arithmetic rounds above 2^24 elements)."""
    if "int" not in str(indices.dtype):
        indices = indices.astype(_mx.int32)
    return _mx.take(a, indices)


# FFT sub-namespace (mirrors the torch backend's ``xp.fft``).
_mlx_fft = _NS(
    fft=lambda x, axis=-1: _mx.fft.fft(x, axis=axis),
    ifft=lambda x, axis=-1: _mx.fft.ifft(x, axis=axis),
    fftfreq=_mlx_fftfreq,
)

#: MLX-core-shaped array namespace (same interface as the torch backend's).
xp = _NS(
    array=lambda value, dtype=None: (_mx.array(value) if dtype is None else _mx.array(value, dtype=dtype)),
    zeros=lambda shape, dtype=None: _mx.zeros(tuple(shape), dtype=dtype or _mx.float32),
    ones=lambda shape, dtype=None: _mx.ones(tuple(shape), dtype=dtype or _mx.float32),
    zeros_like=_mx.zeros_like,
    ones_like=lambda x: _mx.ones_like(x),
    maximum=_mx.maximum,
    minimum=_mx.minimum,
    where=_mx.where,
    eval=_mx.eval,
    norm=lambda x: float(_mx.linalg.norm(x)),
    grad=_mx.grad,
    sum=lambda x, axis=None: (_mx.sum(x) if axis is None else _mx.sum(x, axis=axis)),
    mean=_mx.mean,
    min=lambda x, axis=None: (_mx.min(x) if axis is None else _mx.min(x, axis=axis)),
    max=lambda x, axis=None: (_mx.max(x) if axis is None else _mx.max(x, axis=axis)),
    sqrt=_mx.sqrt,
    exp=_mx.exp,
    log=lambda x: _mx.log(x),
    log1p=_mx.log1p,
    sign=lambda x: _mx.sign(x),
    clip=_mlx_clip,
    reshape=lambda x, shape: _mx.reshape(x, tuple(int(s) for s in shape)),
    concatenate=lambda arrays, axis=0: _mx.concatenate(list(arrays), axis=axis),
    stack=lambda arrays, axis=0: _mx.stack(list(arrays), axis=axis),
    sort=lambda x, axis=-1: _mx.sort(x, axis=axis),
    floor=lambda x: _mx.floor(x),
    take=_mlx_take,
    real=lambda x: _mx.real(x),
    square=_mx.square,
    abs=_mx.abs,
    # torch-parity semantics: float32 unless a dtype is given (mx.arange would
    # return int32 for integer arguments).
    arange=lambda n, dtype=None: _mx.arange(int(n), dtype=dtype or _mx.float32),
    astype=lambda x, dtype: x.astype(dtype),
    moveaxis=_mx.moveaxis,
    int64=_mx.int64,
    cos=_mx.cos,
    sin=_mx.sin,
    arctan=_mx.arctan,
    fft=_mlx_fft,
    float32=_mx.float32,
)


def as_array(data, dtype=None, device=None):
    """Create a backend array from array-like ``data`` (``device`` ignored —
    MLX uses unified memory)."""
    a = data if isinstance(data, _mx.array) else _mx.array(data)
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype)
    return a


def to_numpy(x):
    import numpy as np
    return np.array(x)


def device_of(x):
    return None  # unified memory


def clamp_min(x, value):
    return _mx.maximum(x, float(value))


# --- Metal projectors under the unified functional API ----------------------

from .metal.projectors import (  # noqa: E402
    parallel_forward,
    parallel_backward,
    parallel_forward_footprint,
    parallel_backward_footprint,
    fan_forward,
    fan_backward,
    fan_forward_footprint,
    fan_backward_footprint,
    cone_forward,
    cone_backward,
    cone_forward_footprint,
    cone_backward_footprint,
)
from .metal import geometry  # noqa: E402,F401

#: Torch-only analytical helpers (voxel-driven weighted backprojection etc.)
#: have no Metal port yet; :mod:`diffct_mlx.analytical` degrades gracefully.
analytical = None
