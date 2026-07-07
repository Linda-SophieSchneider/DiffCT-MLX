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

    # FFT sub-namespace (mirrors the torch backend's ``xp.fft``). Lambdas defer
    # attribute resolution so a missing symbol never breaks import-time backend
    # selection.
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
        sqrt=_mx.sqrt,
        exp=_mx.exp,
        log=lambda x: _mx.log(x),
        log1p=lambda x: _mx.log(1.0 + x),
        sign=lambda x: _mx.sign(x),
        clip=_mlx_clip,
        reshape=lambda x, shape: _mx.reshape(x, tuple(int(s) for s in shape)),
        concatenate=lambda arrays, axis=0: _mx.concatenate(list(arrays), axis=axis),
        stack=lambda arrays, axis=0: _mx.stack(list(arrays), axis=axis),
        sort=lambda x, axis=-1: _mx.sort(x, axis=axis),
        floor=lambda x: _mx.floor(x),
        take=lambda a, indices: _mx.take(a, indices.astype(_mx.int32)),
        real=lambda x: _mx.real(x),
        square=_mx.square,
        max=_mx.max,
        abs=_mx.abs,
        arange=_mx.arange,
        cos=_mx.cos,
        sin=_mx.sin,
        arctan=_mx.arctan,
        fft=_mlx_fft,
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
