"""Out-of-core + multi-GPU cone-beam projection/backprojection.

Mirrors LEAP's mechanism (multi-GPU and out-of-core are the *same* thing):
host-resident data is partitioned into memory-budgeted chunks, each chunk runs
independently on one GPU, and partials are stitched on the host.

- **Backprojection / FDK**: split the output volume into **z-slabs**; each slab
  needs only the detector-row band in its cone "shadow".
- **Forward**: split the output projections into **detector-row (v) bands**; each
  band needs only the z-slab of the volume that projects into it.

Only the separable-footprint cone projectors are orchestrated (the matched,
atomic-free pair). Inference-only: the single-GPU autograd Functions are used
per chunk, but no graph is kept across chunk boundaries.

Sub-geometry conventions match ``diffct/kernels/cone_footprint.py`` exactly:
volume voxel world position ``p = (idx + 0.5 - n/2) * voxel_spacing`` per axis
(D↔world-z), detector row ``iv`` at ``(iv - (n_v-1)/2)*dv`` along ``det_v_vec``
from ``det_center``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from diffct.footprint import (
    ConeFootprintProjectorFunction,
    ConeFootprintBackprojectorFunction,
)

_EPS = 1e-8
#: Extra detector rows / volume slices added to every shadow range so the finite
#: voxel/pixel footprint never straddles a chunk boundary unaccounted for.
_SHADOW_PAD = 4


def _np32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


@dataclass
class ConeGeom:
    """Per-view cone-beam geometry (host numpy, shape (V, 3) each)."""

    src_pos: np.ndarray
    det_center: np.ndarray
    det_u_vec: np.ndarray
    det_v_vec: np.ndarray

    @staticmethod
    def from_arrays(src_pos, det_center, det_u_vec, det_v_vec) -> "ConeGeom":
        return ConeGeom(_np32(src_pos), _np32(det_center), _np32(det_u_vec), _np32(det_v_vec))

    @property
    def n_views(self) -> int:
        return int(self.src_pos.shape[0])


# --------------------------------------------------------------------------- #
# Shadow geometry (host numpy) — which detector rows a z-slab lights up, and
# which z-slices a detector-row band needs.
# --------------------------------------------------------------------------- #

def _v_bins_of_points(P, geom: ConeGeom, dv, n_v):
    """v-detector bin for every (point, view). P:(K,3) -> (K,V) float, nan if invalid."""
    S, C, Vv = geom.src_pos, geom.det_center, geom.det_v_vec
    n = np.cross(geom.det_u_vec, Vv)                 # (V,3) detector normal
    PS = P[:, None, :] - S[None, :, :]               # (K,V,3)
    denom = np.einsum("kvc,vc->kv", PS, n)           # (K,V)
    num = np.einsum("vc,vc->v", C - S, n)            # (V,)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num[None, :] / denom
    Q = S[None, :, :] + t[..., None] * PS            # (K,V,3)
    v0 = np.einsum("kvc,vc->kv", Q - C[None, :, :], Vv)   # signed distance along v
    v_bin = v0 / dv + (n_v - 1) * 0.5
    v_bin[np.abs(denom) < _EPS] = np.nan
    return v_bin


def _plane_v_extents(z_planes_world, geom: ConeGeom, W, H, dv, n_v, vs):
    """For each world-z plane, the [min,max] v-bin over the x/y volume extent × views."""
    xh, yh = 0.5 * W * vs, 0.5 * H * vs
    lo = np.full(z_planes_world.shape, np.inf, dtype=np.float64)
    hi = np.full(z_planes_world.shape, -np.inf, dtype=np.float64)
    for sx in (-xh, xh):
        for sy in (-yh, yh):
            P = np.stack([np.full_like(z_planes_world, sx),
                          np.full_like(z_planes_world, sy),
                          z_planes_world], axis=1).astype(np.float32)   # (K,3)
            vb = _v_bins_of_points(P, geom, dv, n_v)                    # (K,V)
            lo = np.fmin(lo, np.nanmin(vb, axis=1))
            hi = np.fmax(hi, np.nanmax(vb, axis=1))
    return lo, hi


def row_range_for_zslab(z0, z1, geom: ConeGeom, D, H, W, dv, n_v, vs, pad=_SHADOW_PAD):
    """Detector rows [v0,v1) needed to backproject volume z-slab [z0, z1)."""
    z_world = (np.array([z0, z1], dtype=np.float32) - 0.5 * D) * vs
    lo, hi = _plane_v_extents(z_world, geom, W, H, dv, n_v, vs)
    vmin, vmax = float(np.min(lo)), float(np.max(hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0, 0
    v0 = int(max(0, np.floor(vmin) - pad))
    v1 = int(min(n_v, np.ceil(vmax) + 1 + pad))
    return v0, max(v0, v1)


def slice_range_for_vband(v0, v1, geom: ConeGeom, D, H, W, dv, n_v, vs, pad=_SHADOW_PAD):
    """Volume z-slices [z0,z1) needed to forward-project into detector rows [v0, v1)."""
    planes = (np.arange(D + 1, dtype=np.float32) - 0.5 * D) * vs
    lo, hi = _plane_v_extents(planes, geom, W, H, dv, n_v, vs)          # per plane
    slab_lo = np.minimum(lo[:-1], lo[1:])                              # per slice iz
    slab_hi = np.maximum(hi[:-1], hi[1:])
    band_lo, band_hi = v0 - pad, v1 + pad
    hit = np.where((slab_hi >= band_lo) & (slab_lo <= band_hi))[0]
    if hit.size == 0:
        return 0, 0
    return int(hit[0]), int(hit[-1] + 1)


# --------------------------------------------------------------------------- #
# Sub-geometry construction (translate acquisition so a sub-volume centred at
# its own D_slab/2 and a sub-detector of `band` rows line up with the originals)
# --------------------------------------------------------------------------- #

def _zslab_shift(geom: ConeGeom, z0, z1, D, vs) -> ConeGeom:
    """Shift src+det_center along world-z so sub-volume [z0,z1) sits correctly."""
    z_c = ((z0 + z1) * 0.5 - 0.5 * D) * vs
    src = geom.src_pos.copy()
    det_c = geom.det_center.copy()
    src[:, 2] -= z_c
    det_c[:, 2] -= z_c
    return ConeGeom(src, det_c, geom.det_u_vec, geom.det_v_vec)


def _vband_shift(geom: ConeGeom, v0, band, n_v, dv) -> ConeGeom:
    """Shift det_center along det_v_vec so sub-detector row j maps to row v0+j."""
    v_off = (v0 + (band - 1) * 0.5 - (n_v - 1) * 0.5) * dv
    det_c = geom.det_center + v_off * geom.det_v_vec
    return ConeGeom(geom.src_pos, det_c, geom.det_u_vec, geom.det_v_vec)


def _to_dev(arr, dev):
    return torch.as_tensor(arr, dtype=torch.float32, device=dev)


# --------------------------------------------------------------------------- #
# Chunk-size budgeting
# --------------------------------------------------------------------------- #

def _auto_max_slices(D, H, W, n_views, n_u, n_v, gpus, requested):
    if requested is not None:
        return int(requested)
    free = min(torch.cuda.mem_get_info(g)[0] for g in gpus)
    # per z-slab: sub-volume (slab·H·W) + its detector-row band (~n_views·n_u·n_v,
    # worst case the full detector) — 4 bytes, ×2 safety for temporaries.
    per_slice = H * W * 4
    sino_worst = n_views * n_u * n_v * 4
    budget = 0.5 * free - sino_worst
    slabs = max(8, int(budget / (2.0 * per_slice))) if budget > 0 else 8
    return int(min(128, max(1, min(slabs, D))))


def _chunks(n, step):
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def _resolve_gpus(gpus):
    if gpus is None:
        gpus = list(range(torch.cuda.device_count())) or [torch.cuda.current_device()]
    return list(gpus)


def _run_on_gpus(tasks, gpus):
    """Run per-chunk callables, one active chunk per GPU (round-robin threads)."""
    if len(gpus) <= 1:
        dev = gpus[0]
        for fn in tasks:
            fn(dev)
        return
    lock_by_gpu = {g: threading.Lock() for g in gpus}

    def worker(idx_fn):
        idx, fn = idx_fn
        dev = gpus[idx % len(gpus)]
        with lock_by_gpu[dev]:
            fn(dev)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(worker, enumerate(tasks)))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def chunked_cone_backward(sinogram, geom: ConeGeom, D, H, W,
                          du=1.0, dv=1.0, voxel_spacing=1.0,
                          gpus=None, max_slices=None):
    """Out-of-core + multi-GPU cone footprint backprojection → volume (D,H,W).

    ``sinogram`` is host-resident (numpy or CPU tensor), shape (n_views, n_u, n_v).
    The output volume is split into z-slabs; each slab pulls only its shadow rows.
    """
    gpus = _resolve_gpus(gpus)
    sino_np = _np32(sinogram)
    n_views, n_u, n_v = sino_np.shape
    out = np.zeros((D, H, W), dtype=np.float32)
    step = _auto_max_slices(D, H, W, n_views, n_u, n_v, gpus, max_slices)

    def make_task(z0, z1):
        def task(dev):
            torch.cuda.set_device(dev)
            from numba import cuda as _cuda
            _cuda.select_device(dev)
            v0, v1 = row_range_for_zslab(z0, z1, geom, D, H, W, dv, n_v, voxel_spacing)
            if v1 <= v0:
                return  # slab projects outside the detector → stays zero
            g = _vband_shift(_zslab_shift(geom, z0, z1, D, voxel_spacing),
                             v0, v1 - v0, n_v, dv)
            sub_sino = _to_dev(sino_np[:, :, v0:v1], dev)
            sub_vol = ConeFootprintBackprojectorFunction.apply(
                sub_sino, _to_dev(g.src_pos, dev), _to_dev(g.det_center, dev),
                _to_dev(g.det_u_vec, dev), _to_dev(g.det_v_vec, dev),
                z1 - z0, H, W, du, dv, voxel_spacing)
            out[z0:z1] = sub_vol.detach().cpu().numpy()
        return task

    _run_on_gpus([make_task(z0, z1) for z0, z1 in _chunks(D, step)], gpus)
    return out


def chunked_cone_forward(volume, geom: ConeGeom, det_u, det_v,
                         du=1.0, dv=1.0, voxel_spacing=1.0,
                         gpus=None, max_rows=None):
    """Out-of-core + multi-GPU cone footprint forward projection → (n_views, det_u, det_v).

    ``volume`` is host-resident (numpy or CPU tensor), shape (D, H, W). The output
    projections are split into detector-row (v) bands; each band pulls only the
    z-slab of the volume in its shadow.
    """
    gpus = _resolve_gpus(gpus)
    vol_np = _np32(volume)
    D, H, W = vol_np.shape
    n_views = geom.n_views
    out = np.zeros((n_views, det_u, det_v), dtype=np.float32)
    if max_rows is None:
        max_rows = max(8, _auto_max_slices(D, H, W, n_views, det_u, det_v, gpus, None))

    def make_task(v0, v1):
        def task(dev):
            torch.cuda.set_device(dev)
            from numba import cuda as _cuda
            _cuda.select_device(dev)
            z0, z1 = slice_range_for_vband(v0, v1, geom, D, H, W, dv, det_v, voxel_spacing)
            if z1 <= z0:
                return  # no volume slice projects into this band → stays zero
            g = _vband_shift(_zslab_shift(geom, z0, z1, D, voxel_spacing),
                             v0, v1 - v0, det_v, dv)
            sub_vol = _to_dev(vol_np[z0:z1], dev)
            sub_sino = ConeFootprintProjectorFunction.apply(
                sub_vol, _to_dev(g.src_pos, dev), _to_dev(g.det_center, dev),
                _to_dev(g.det_u_vec, dev), _to_dev(g.det_v_vec, dev),
                det_u, v1 - v0, du, dv, voxel_spacing)
            out[:, :, v0:v1] = sub_sino.detach().cpu().numpy()
        return task

    _run_on_gpus([make_task(v0, v1) for v0, v1 in _chunks(det_v, max_rows)], gpus)
    return out


def chunked_cone_fdk(sinogram, geom: ConeGeom, D, H, W,
                     du=1.0, dv=1.0, voxel_spacing=1.0,
                     fdk_weights=None, normalization_scale=None,
                     enforce_positivity=True, filter_axis=1,
                     gpus=None, max_slices=None):
    """Out-of-core + multi-GPU FDK reconstruction → volume (D, H, W).

    Applies optional per-detector FDK weighting and a ramp filter along the
    detector-u axis (numpy FFT, same ``2|f|`` convention as
    :mod:`diffct_mlx.reconstruction_algorithms._analytic`), then backprojects
    with the chunked multi-GPU footprint backprojector and applies positivity +
    normalization. ``fdk_weights`` broadcasts over views, shape (1, n_u, n_v).
    """
    sino = _np32(sinogram)
    if fdk_weights is not None:
        sino = sino * _np32(fdk_weights)
    n = sino.shape[filter_axis]
    ramp = (2.0 * np.abs(np.fft.fftfreq(n))).astype(np.float32)
    shape = [1] * sino.ndim
    shape[filter_axis] = n
    filtered = np.real(
        np.fft.ifft(np.fft.fft(sino, axis=filter_axis) * ramp.reshape(shape), axis=filter_axis)
    ).astype(np.float32)
    vol = chunked_cone_backward(filtered, geom, D, H, W, du, dv, voxel_spacing,
                                gpus=gpus, max_slices=max_slices)
    if enforce_positivity:
        vol = np.maximum(vol, 0.0)
    if normalization_scale is not None:
        vol = vol * float(normalization_scale)
    return vol
