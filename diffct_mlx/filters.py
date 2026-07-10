"""Edge-preserving denoisers and a Plug-and-Play regularizer sequence.

These mirror the LEAP *filter sequence*: each denoiser is a proximable
:class:`~diffct_mlx.functionals.Functional` whose ``prox`` applies the filter, so
they drop straight into the proximal step of ``rwls`` / ``rdls`` (Plug-and-Play
FISTA) or any POCS-style loop::

    from diffct_mlx.filters import Bilateral, Median, RegularizerSequence
    denoiser = RegularizerSequence([Bilateral(radius=1), Median(radius=1)])
    x = dct.rwls(A, b, constraint=denoiser, iterations=60)   # PnP-FISTA

The core filters (box / bilateral / guided / median / histogram sparsity) are
backend-neutral (built from ``xp`` slicing + arithmetic). The two that need
image rotation / a patch transform (azimuthal, dictionary/DCT) are implemented
on the torch backend; on other backends they raise a clear error until the
matching Metal kernels land.
"""

from __future__ import annotations

import itertools
from typing import Sequence

from .backend import NAME as _BACKEND
from .backend import active as _b
from .functionals import Functional

xp = _b.xp

__all__ = [
    "box_filter",
    "bilateral_filter",
    "guided_filter",
    "median_filter",
    "histogram_sparsity",
    "azimuthal_smooth",
    "dictionary_denoise",
    "Bilateral",
    "Guided",
    "Median",
    "HistogramSparsity",
    "Azimuthal",
    "DictionarySparsity",
    "RegularizerSequence",
]


# ---------------------------------------------------------------------------
# Backend-neutral neighbourhood helpers
# ---------------------------------------------------------------------------
def _slice_along(x, axis, sl):
    idx = [slice(None)] * x.ndim
    idx[axis] = sl
    return x[tuple(idx)]


def _pad_replicate(x, r):
    """Edge-replicate pad every axis by ``r`` (built from slices + concat)."""
    for axis in range(x.ndim):
        n = int(x.shape[axis])
        first = _slice_along(x, axis, slice(0, 1))
        last = _slice_along(x, axis, slice(n - 1, n))
        x = xp.concatenate([first] * r + [x] + [last] * r, axis=axis)
    return x


def _window_offsets(ndim, r):
    return list(itertools.product(range(-r, r + 1), repeat=ndim))


def _neighbor(x_pad, offset, r, shape):
    idx = tuple(slice(o + r, o + r + int(shape[axis])) for axis, o in enumerate(offset))
    return x_pad[idx]


def _value_range(x):
    lo = float(xp.min(x))
    hi = float(xp.max(x))
    return lo, hi, max(hi - lo, 1e-6)


# ---------------------------------------------------------------------------
# Core filters (backend-neutral)
# ---------------------------------------------------------------------------
def _box_filter_1d(x, r, axis):
    """Mean filter of width ``2r+1`` along ``axis`` with edge replication."""
    n = int(x.shape[axis])
    first = _slice_along(x, axis, slice(0, 1))
    last = _slice_along(x, axis, slice(n - 1, n))
    padded = xp.concatenate([first] * r + [x] + [last] * r, axis=axis)
    acc = None
    for off in range(2 * r + 1):
        nb = _slice_along(padded, axis, slice(off, off + n))
        acc = nb if acc is None else acc + nb
    return acc / float(2 * r + 1)


def box_filter(x, radius: int = 1, axes=None):
    """Uniform (mean) filter over a ``(2r+1)^k`` window with edge replication.

    ``axes`` selects which axes are filtered (default: all). The box window is
    separable, so the filter runs one axis at a time — ``k*(2r+1)`` shifted adds
    instead of ``(2r+1)^k`` — with identical results.
    """
    r = int(radius)
    if r <= 0:
        return x
    x = xp.array(x, dtype=_b.float32)
    axes = tuple(range(x.ndim)) if axes is None else tuple(a % x.ndim for a in axes)
    for axis in axes:
        x = _box_filter_1d(x, r, axis)
    return x


def bilateral_filter(x, radius: int = 1, sigma_spatial: float = 1.0, sigma_range: float | None = None):
    """Edge-preserving bilateral filter (spatial x range Gaussian weights)."""
    r = int(radius)
    x = xp.array(x, dtype=_b.float32)
    if r <= 0:
        return x
    if sigma_range is None:
        _, _, rng = _value_range(x)
        sigma_range = 0.1 * rng
    inv_s2 = 1.0 / (2.0 * float(sigma_spatial) ** 2)
    inv_r2 = 1.0 / (2.0 * float(sigma_range) ** 2)
    x_pad = _pad_replicate(x, r)
    shape = x.shape
    num = None
    den = None
    for off in _window_offsets(x.ndim, r):
        nb = _neighbor(x_pad, off, r, shape)
        dist2 = float(sum(o * o for o in off))
        w_spatial = float(pow(2.718281828459045, -dist2 * inv_s2))
        diff = nb - x
        w = w_spatial * xp.exp(-(diff * diff) * inv_r2)
        num = w * nb if num is None else num + w * nb
        den = w if den is None else den + w
    return num / xp.maximum(den, 1e-12)


def guided_filter(x, radius: int = 2, eps: float | None = None):
    """Self-guided filter (He et al.): edge-preserving linear-model smoothing."""
    x = xp.array(x, dtype=_b.float32)
    if eps is None:
        _, _, rng = _value_range(x)
        eps = (0.02 * rng) ** 2
    mean_i = box_filter(x, radius)
    mean_ii = box_filter(x * x, radius)
    var_i = mean_ii - mean_i * mean_i
    a = var_i / (var_i + float(eps))
    b = mean_i - a * mean_i
    return box_filter(a, radius) * x + box_filter(b, radius)


def median_filter(x, radius: int = 1):
    """Median filter over a ``(2r+1)^ndim`` window (via a sorted neighbour stack)."""
    r = int(radius)
    x = xp.array(x, dtype=_b.float32)
    if r <= 0:
        return x
    x_pad = _pad_replicate(x, r)
    shape = x.shape
    neigh = [_neighbor(x_pad, off, r, shape) for off in _window_offsets(x.ndim, r)]
    stack = xp.stack(neigh, axis=0)
    ordered = xp.sort(stack, axis=0)
    return ordered[stack.shape[0] // 2]


def histogram_sparsity(x, levels: Sequence[float] | None = None, num_levels: int = 8, strength: float = 0.5):
    """Attract voxels toward the nearest of a few intensity levels (piecewise-constant prior).

    A simple, GPU-friendly stand-in for LEAP's histogram-sparsity filter: it
    softly pulls each voxel toward its nearest target level. ``levels`` defaults
    to ``num_levels`` uniform bins over the data range.
    """
    x = xp.array(x, dtype=_b.float32)
    if levels is None:
        lo, hi, _ = _value_range(x)
        k = max(2, int(num_levels))
        levels = [lo + (i + 0.5) / k * (hi - lo) for i in range(k)]
    levels = [float(v) for v in levels]
    nearest = xp.zeros_like(x) + levels[0]
    best = xp.abs(x - levels[0])
    for lvl in levels[1:]:
        dist = xp.abs(x - lvl)
        closer = dist < best
        nearest = xp.where(closer, xp.zeros_like(x) + lvl, nearest)
        best = xp.where(closer, dist, best)
    s = float(strength)
    return x + s * (nearest - x)


# ---------------------------------------------------------------------------
# torch-backed filters (rotation / patch transform)
# ---------------------------------------------------------------------------
def _require_torch(name):
    if _BACKEND != "torch":
        raise NotImplementedError(
            f"{name} is currently implemented on the torch backend only; a Metal "
            f"version lands with the MLX backend wiring."
        )
    import torch  # noqa: F401

    return __import__("torch")


def azimuthal_smooth(x, n_angles: int = 5, max_angle_deg: float = 2.0):
    """Azimuthal (angular) smoothing: average small rotations about the image centre.

    Suppresses angular streak artefacts. 2D images are smoothed directly; 3D
    volumes are smoothed slice-wise over the last two axes. torch backend only.
    """
    torch = _require_torch("azimuthal_smooth")
    import torch.nn.functional as F
    import math

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(_b.to_numpy(x))
    t = t.to(dtype=torch.float32)
    was_2d = t.ndim == 2
    if was_2d:
        t = t[None]
    D, H, W = t.shape
    batch = t[:, None]  # (D, 1, H, W)
    angles = [(-max_angle_deg + 2.0 * max_angle_deg * i / max(1, n_angles - 1)) * math.pi / 180.0
              for i in range(n_angles)] if n_angles > 1 else [0.0]
    acc = torch.zeros_like(batch)
    for a in angles:
        ca, sa = math.cos(a), math.sin(a)
        theta = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0]], dtype=torch.float32, device=t.device)
        theta = theta[None].expand(D, 2, 3)
        grid = F.affine_grid(theta, batch.shape, align_corners=False)
        acc = acc + F.grid_sample(batch, grid, align_corners=False, padding_mode="reflection")
    out = (acc / float(len(angles)))[:, 0]
    return out[0] if was_2d else out


def _dct_matrix(n, device, torch):
    import math

    k = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    m = torch.arange(n, device=device, dtype=torch.float32)[None, :]
    d = torch.cos(math.pi / n * (m + 0.5) * k) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d  # orthonormal DCT-II, so inverse == d.T


def _dictionary_denoise_2d(img, patch, stride, threshold, torch):
    import torch.nn.functional as F

    H, W = img.shape
    x = img[None, None]  # (1,1,H,W)
    cols = F.unfold(x, kernel_size=patch, stride=stride)  # (1, patch*patch, L)
    L = cols.shape[-1]
    p = cols.reshape(patch, patch, L).permute(2, 0, 1)  # (L, patch, patch)
    D = _dct_matrix(patch, img.device, torch)
    coeff = D @ p @ D.transpose(0, 1)  # 2D DCT of each patch
    coeff = torch.sign(coeff) * torch.clamp(torch.abs(coeff) - float(threshold), min=0.0)  # soft-threshold
    rec = D.transpose(0, 1) @ coeff @ D  # inverse DCT
    rec_cols = rec.permute(1, 2, 0).reshape(1, patch * patch, L)
    num = F.fold(rec_cols, output_size=(H, W), kernel_size=patch, stride=stride)
    den = F.fold(torch.ones_like(rec_cols), output_size=(H, W), kernel_size=patch, stride=stride)
    return (num / torch.clamp(den, min=1e-6))[0, 0]


def dictionary_denoise(x, patch: int = 8, stride: int = 4, threshold: float | None = None):
    """Transform-domain (DCT patch) sparsity denoiser.

    Overlapping patches are DCT-transformed, soft-thresholded (sparse coding in a
    fixed dictionary), inverse-transformed and averaged. 2D directly, 3D
    slice-wise. torch backend only. ``threshold`` defaults to ``0.03 * range``.
    """
    torch = _require_torch("dictionary_denoise")
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(_b.to_numpy(x))
    t = t.to(dtype=torch.float32)
    if threshold is None:
        rng = float(t.max() - t.min())
        threshold = 0.03 * max(rng, 1e-6)
    if t.ndim == 2:
        return _dictionary_denoise_2d(t, patch, stride, threshold, torch)
    if t.ndim == 3:
        return torch.stack([_dictionary_denoise_2d(t[z], patch, stride, threshold, torch) for z in range(t.shape[0])], 0)
    raise ValueError(f"dictionary_denoise expects a 2D or 3D array, got ndim={t.ndim}.")


# ---------------------------------------------------------------------------
# Proximable Functional wrappers + the sequence
# ---------------------------------------------------------------------------
class _DenoiserFunctional(Functional):
    """A denoiser exposed as a proximable functional (its prox is the filter)."""

    is_proximable = True

    def value(self, x):
        return xp.sum(x) * 0.0  # no closed-form energy; used purely as a prox step


class Bilateral(_DenoiserFunctional):
    def __init__(self, radius: int = 1, sigma_spatial: float = 1.0, sigma_range: float | None = None):
        self.radius, self.sigma_spatial, self.sigma_range = int(radius), float(sigma_spatial), sigma_range

    def prox(self, x, tau: float):
        return bilateral_filter(x, self.radius, self.sigma_spatial, self.sigma_range)


class Guided(_DenoiserFunctional):
    def __init__(self, radius: int = 2, eps: float | None = None):
        self.radius, self.eps = int(radius), eps

    def prox(self, x, tau: float):
        return guided_filter(x, self.radius, self.eps)


class Median(_DenoiserFunctional):
    def __init__(self, radius: int = 1):
        self.radius = int(radius)

    def prox(self, x, tau: float):
        return median_filter(x, self.radius)


class HistogramSparsity(_DenoiserFunctional):
    def __init__(self, levels: Sequence[float] | None = None, num_levels: int = 8, strength: float = 0.5):
        self.levels, self.num_levels, self.strength = levels, int(num_levels), float(strength)

    def prox(self, x, tau: float):
        return histogram_sparsity(x, self.levels, self.num_levels, self.strength)


class Azimuthal(_DenoiserFunctional):
    def __init__(self, n_angles: int = 5, max_angle_deg: float = 2.0):
        self.n_angles, self.max_angle_deg = int(n_angles), float(max_angle_deg)

    def prox(self, x, tau: float):
        return azimuthal_smooth(x, self.n_angles, self.max_angle_deg)


class DictionarySparsity(_DenoiserFunctional):
    def __init__(self, patch: int = 8, stride: int = 4, threshold: float | None = None):
        self.patch, self.stride, self.threshold = int(patch), int(stride), threshold

    def prox(self, x, tau: float):
        return dictionary_denoise(x, self.patch, self.stride, self.threshold)


class RegularizerSequence(Functional):
    """Apply a list of regularizer/denoiser prox steps in sequence (LEAP filter sequence).

    Used as the proximal / Plug-and-Play step of ``rwls`` / ``rdls`` or a POCS
    loop. ``repeats`` runs the whole sequence more than once per call.
    """

    is_proximable = True

    def __init__(self, steps: Sequence[Functional], repeats: int = 1):
        self.steps = list(steps)
        self.repeats = int(repeats)

    def value(self, x):
        total = xp.sum(x) * 0.0
        for step in self.steps:
            total = total + step.value(x)
        return total

    def prox(self, x, tau: float):
        for _ in range(self.repeats):
            for step in self.steps:
                x = step.prox(x, tau)
        return x
