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

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from diffct.footprint import (
    ConeFootprintProjectorFunction,
    ConeFootprintBackprojectorFunction,
)
from diffct.projectors import ConeBackprojectorFunction

# Backward autograd Functions selectable for chunked backprojection. Both share
# the (sino, src, dc, u, v, D, H, W, du, dv, vs) -> (D,H,W) signature and the
# same volume centering (cz=D/2), so the z-slab / v-band sub-geometry is shared.
_BACKWARD_FUNCS = {
    "footprint": ConeFootprintBackprojectorFunction,
    "siddon": ConeBackprojectorFunction,
}
from diffct.kernels import _cone_3d_fdk_backproject_kernel
from diffct.utils import _grid_3d, TorchCUDABridge, _get_numba_external_stream_for
from diffct.constants import _DTYPE
from diffct.analytical import _cone_mean_sid_sdd

_EPS = 1e-8
#: Extra detector rows / volume slices added to every shadow range so the finite
#: voxel/pixel footprint never straddles a chunk boundary unaccounted for.
_SHADOW_PAD = 4


def _np32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _lazy_input(x):
    """Return an array that supports lazy slicing without materializing it.

    memmap and zarr arrays are passed through (only touched blocks load on
    slicing); torch tensors and plain array-likes are converted eagerly.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.memmap):
        return x
    if type(x).__module__.split(".")[0] == "zarr":
        return x
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
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=dev)


def open_memmap(path, shape, dtype=np.float32, mode="w+"):
    """Create/open a disk-backed float32 .npy array (numpy memmap).

    Use for volumes/sinograms too large for host RAM (e.g. TB-scale
    reconstructions): the chunked/streaming routines read and write only the
    slabs/bands they touch, so peak RAM stays bounded by the chunk size, not the
    full array. ``mode='w+'`` creates, ``'r+'`` opens existing read/write,
    ``'r'`` read-only.
    """
    from numpy.lib.format import open_memmap as _om
    if mode == "w+":
        return _om(path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
    return _om(path, mode=mode)


def _alloc_out(out, shape):
    """Return the caller's output array (disk memmap or in-RAM) or a fresh RAM array."""
    if out is None:
        return np.zeros(shape, dtype=np.float32)
    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"out has shape {tuple(out.shape)!r}, expected {tuple(shape)!r}")
    return out


# --------------------------------------------------------------------------- #
# Automatic RAM-vs-disk decision. Arrays that fit a host-RAM budget stay in RAM;
# larger ones are transparently backed by a disk memmap (path to TB-scale). Set
# the storage location once with set_out_of_core_dir("/mnt/bigdisk").
# --------------------------------------------------------------------------- #

_OOC_DIR = None
_OOC_COUNTER = 0
_OOC_BACKEND = "memmap"   # "memmap" (raw .npy) or "zarr" (chunked + compressed)


def set_out_of_core_dir(path):
    """Set the directory where auto out-of-core arrays are written (e.g. a big HDD)."""
    global _OOC_DIR
    _OOC_DIR = path


def get_out_of_core_dir():
    return _OOC_DIR


def set_out_of_core_backend(name):
    """Choose the disk backend for auto out-of-core arrays: 'memmap' or 'zarr'.

    'zarr' stores chunked + compressed arrays — less HDD traffic for TB-scale
    volumes (at some CPU cost); 'memmap' is a raw flat .npy. Both are read/written
    with numpy-style slicing, so the reconstruction code is identical.
    """
    global _OOC_BACKEND
    if name not in ("memmap", "zarr"):
        raise ValueError("backend must be 'memmap' or 'zarr'")
    _OOC_BACKEND = name


def open_zarr(path, shape, dtype=np.float32, chunks=None, mode="w"):
    """Create/open a chunked, compressed zarr array (numpy-style slicing)."""
    import zarr
    if chunks is None:
        chunks = (min(64, int(shape[0])),) + tuple(int(s) for s in shape[1:])
    return zarr.open_array(store=str(path), mode=mode, shape=tuple(int(s) for s in shape),
                           chunks=chunks, dtype=np.dtype(dtype))


def _available_ram():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 8 * (1024 ** 3)


def _ooc_path(tag, work_dir):
    global _OOC_COUNTER
    import os
    import warnings
    d = work_dir or _OOC_DIR
    if d is None:
        import tempfile
        d = tempfile.gettempdir()
        warnings.warn(
            f"diffct out-of-core: array too large for the RAM budget but no "
            f"out-of-core dir set — using {d!r}. Call "
            f"set_out_of_core_dir('/path/on/big/disk') to control this.",
            RuntimeWarning, stacklevel=3)
    os.makedirs(d, exist_ok=True)
    _OOC_COUNTER += 1
    ext = ".zarr" if _OOC_BACKEND == "zarr" else ".npy"
    return os.path.join(d, f"diffct_ooc_{os.getpid()}_{_OOC_COUNTER}_{tag}{ext}")


def _resolve_output(out, shape, tag, work_dir=None, ram_budget=None):
    """Pick storage for an output array: explicit ``out``, else RAM if it fits the
    budget, else a disk array (memmap or zarr per set_out_of_core_backend). This
    is how the disk-backed path is chosen *automatically*."""
    if out is not None:
        return _alloc_out(out, shape)
    nbytes = int(np.prod(shape)) * 4
    budget = int(ram_budget) if ram_budget is not None else int(0.5 * _available_ram())
    if work_dir is None and nbytes <= budget:
        return np.zeros(shape, dtype=np.float32)
    path = _ooc_path(tag, work_dir)
    if _OOC_BACKEND == "zarr":
        return open_zarr(path, shape)
    return open_memmap(path, shape, mode="w+")


def ramp_filter_memmap(sinogram, out=None, filter_axis=1, fdk_weights=None,
                       view_chunk=64, work_dir=None, ram_budget=None, device=None):
    """Ramp-filter a (possibly >RAM) sinogram, streaming over views.

    The ramp acts along the detector-u axis and is independent per view, so views
    are processed in chunks — a disk-memmap/zarr sinogram is filtered into another
    (auto disk-backed) array without ever loading the whole thing. ``fdk_weights``
    (broadcast over views) are applied before the ramp.

    ``device``: ``"cuda"`` runs the FFT on the GPU (torch.fft — much faster for
    large detectors), ``"cpu"`` uses numpy FFT; ``None`` picks CUDA when
    available. Both use the identical ``2|f|`` convention so FDK results match.
    Returns the filtered array.
    """
    sino = _lazy_input(sinogram)
    n_views = sino.shape[0]
    out = _resolve_output(out, sino.shape, "filtered", work_dir, ram_budget)
    n = sino.shape[filter_axis]
    ramp_np = (2.0 * np.abs(np.fft.fftfreq(n))).astype(np.float32)
    shape = [1] * sino.ndim
    shape[filter_axis] = n
    ramp_np = ramp_np.reshape(shape)
    w = _np32(fdk_weights) if fdk_weights is not None else None

    use_gpu = torch.cuda.is_available() if device is None else (str(device) != "cpu")
    if use_gpu:
        dev = torch.device("cuda" if device in (None, "cuda") else device)
        ramp_t = torch.as_tensor(ramp_np, device=dev)
        w_t = torch.as_tensor(w, device=dev) if w is not None else None
        for a, b in _chunks(n_views, view_chunk):
            bt = torch.from_numpy(np.ascontiguousarray(sino[a:b], dtype=np.float32)).to(dev)
            if w_t is not None:
                bt = bt * w_t
            filt = torch.fft.ifft(torch.fft.fft(bt, dim=filter_axis) * ramp_t, dim=filter_axis).real
            out[a:b] = filt.to(torch.float32).cpu().numpy()
        return out

    for a, b in _chunks(n_views, view_chunk):
        band = np.asarray(sino[a:b], dtype=np.float32)
        if w is not None:
            band = band * w
        out[a:b] = np.real(
            np.fft.ifft(np.fft.fft(band, axis=filter_axis) * ramp_np, axis=filter_axis)
        ).astype(np.float32)
    return out


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


def _pin_to_dev(arr, dev):
    """Host array → device via a pinned staging buffer + async copy."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    try:
        t = t.pin_memory()
    except Exception:
        pass
    return t.to(torch.device("cuda", dev), non_blocking=True)


def _conveyor_run(chunk_ids, read_fn, compute_fn, write_fn, gpus, prefetch=None, timings=None):
    """Asynchronous read→compute→write conveyor (TomocuPy-style overlap).

    One reader thread streams chunk inputs from disk (sequential — HDD-friendly),
    ``len(gpus)`` GPU workers compute (each pinned to its device), one writer
    thread persists results — coupled by bounded queues so disk read, GPU compute
    and disk write overlap *across* chunks while peak RAM stays ~prefetch chunks.
    ``read_fn(cid)->inp``, ``compute_fn(cid, inp, dev)->out``, ``write_fn(cid, out)``.
    If ``timings`` (dict) is given, per-stage active seconds are recorded.
    """
    ids = list(chunk_ids)
    if not ids:
        return
    if len(gpus) == 0:
        gpus = [torch.cuda.current_device()]
    prefetch = prefetch if prefetch is not None else max(2, 2 * len(gpus))
    read_q = queue.Queue(maxsize=prefetch)
    write_q = queue.Queue(maxsize=prefetch)
    tt = {"read": 0.0, "compute": 0.0, "write": 0.0, "wall": 0.0}
    lock = threading.Lock()

    def acc(k, dt):
        with lock:
            tt[k] += dt

    def reader():
        for cid in ids:
            t = time.perf_counter()
            inp = read_fn(cid)
            acc("read", time.perf_counter() - t)
            read_q.put((cid, inp))
        for _ in gpus:
            read_q.put(None)

    def gpu_worker(dev):
        torch.cuda.set_device(dev)
        from numba import cuda as _cuda
        _cuda.select_device(dev)
        while True:
            item = read_q.get()
            if item is None:
                write_q.put(None)
                return
            cid, inp = item
            t = time.perf_counter()
            out = compute_fn(cid, inp, dev)
            acc("compute", time.perf_counter() - t)
            write_q.put((cid, out))

    def writer():
        done = 0
        while done < len(gpus):
            item = write_q.get()
            if item is None:
                done += 1
                continue
            cid, out = item
            t = time.perf_counter()
            write_fn(cid, out)
            acc("write", time.perf_counter() - t)

    t0 = time.perf_counter()
    rt = threading.Thread(target=reader)
    gws = [threading.Thread(target=gpu_worker, args=(g,)) for g in gpus]
    wt = threading.Thread(target=writer)
    rt.start(); [g.start() for g in gws]; wt.start()
    rt.join(); [g.join() for g in gws]; wt.join()
    tt["wall"] = time.perf_counter() - t0
    if timings is not None:
        timings.update(tt)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def chunked_cone_backward(sinogram, geom: ConeGeom, D, H, W,
                          du=1.0, dv=1.0, voxel_spacing=1.0,
                          projector="footprint", gpus=None, max_slices=None,
                          out=None, post_fn=None, work_dir=None, ram_budget=None,
                          timings=None):
    """Out-of-core + multi-GPU cone backprojection → volume (D,H,W).

    ``sinogram`` is host-resident (numpy / CPU tensor / disk memmap), shape
    (n_views, n_u, n_v). The output volume is split into z-slabs; each slab pulls
    only its shadow rows. The output storage is chosen automatically: RAM if it
    fits the budget, otherwise a disk memmap (pass ``out=`` to force a specific
    array, or ``work_dir``/``ram_budget`` to steer the auto decision).
    ``post_fn(slab)`` is applied to each slab before writing (e.g. positivity +
    FDK scale) to keep the pipeline streaming. ``projector``: ``"footprint"``
    (matched, atomic-free) or ``"siddon"`` (line-integral adjoint).
    """
    gpus = _resolve_gpus(gpus)
    backward_fn = _BACKWARD_FUNCS[projector]
    sino = _lazy_input(sinogram)
    n_views, n_u, n_v = sino.shape
    out = _resolve_output(out, (D, H, W), "backward_vol", work_dir, ram_budget)
    step = _auto_max_slices(D, H, W, n_views, n_u, n_v, gpus, max_slices)

    def read_fn(zz):
        z0, z1 = zz
        v0, v1 = row_range_for_zslab(z0, z1, geom, D, H, W, dv, n_v, voxel_spacing)
        if v1 <= v0:
            return (v0, v1, None)  # slab projects off the detector
        return (v0, v1, np.asarray(sino[:, :, v0:v1], dtype=np.float32))

    def compute_fn(zz, inp, dev):
        z0, z1 = zz
        v0, v1, band = inp
        if band is None:
            slab = np.zeros((z1 - z0, H, W), dtype=np.float32)
        else:
            g = _vband_shift(_zslab_shift(geom, z0, z1, D, voxel_spacing),
                             v0, v1 - v0, n_v, dv)
            sub_vol = backward_fn.apply(
                _pin_to_dev(band, dev), _to_dev(g.src_pos, dev), _to_dev(g.det_center, dev),
                _to_dev(g.det_u_vec, dev), _to_dev(g.det_v_vec, dev),
                z1 - z0, H, W, du, dv, voxel_spacing)
            slab = sub_vol.detach().cpu().numpy()
        return post_fn(slab) if post_fn is not None else slab

    def write_fn(zz, slab):
        z0, z1 = zz
        out[z0:z1] = slab

    _conveyor_run(_chunks(D, step), read_fn, compute_fn, write_fn, gpus, timings=timings)
    return out


def chunked_cone_forward(volume, geom: ConeGeom, det_u, det_v,
                         du=1.0, dv=1.0, voxel_spacing=1.0,
                         gpus=None, max_rows=None, out=None,
                         work_dir=None, ram_budget=None, timings=None):
    """Out-of-core + multi-GPU cone footprint forward projection → (n_views, det_u, det_v).

    ``volume`` is host-resident (numpy / CPU tensor / disk memmap), shape
    (D, H, W). The output projections are split into detector-row (v) bands; each
    band pulls only the z-slab of the volume in its shadow. Output storage is
    chosen automatically (RAM if it fits the budget, else a disk memmap); pass
    ``out=`` to force it, or ``work_dir``/``ram_budget`` to steer the decision.
    """
    gpus = _resolve_gpus(gpus)
    vol = _lazy_input(volume)
    D, H, W = vol.shape
    n_views = geom.n_views
    out = _resolve_output(out, (n_views, det_u, det_v), "forward_sino", work_dir, ram_budget)
    if max_rows is None:
        max_rows = max(8, _auto_max_slices(D, H, W, n_views, det_u, det_v, gpus, None))

    def read_fn(vv):
        v0, v1 = vv
        z0, z1 = slice_range_for_vband(v0, v1, geom, D, H, W, dv, det_v, voxel_spacing)
        if z1 <= z0:
            return (z0, z1, None)  # no volume slice projects into this band
        return (z0, z1, np.asarray(vol[z0:z1], dtype=np.float32))

    def compute_fn(vv, inp, dev):
        v0, v1 = vv
        z0, z1, slab = inp
        if slab is None:
            return np.zeros((n_views, det_u, v1 - v0), dtype=np.float32)
        g = _vband_shift(_zslab_shift(geom, z0, z1, D, voxel_spacing),
                         v0, v1 - v0, det_v, dv)
        sub_sino = ConeFootprintProjectorFunction.apply(
            _pin_to_dev(slab, dev), _to_dev(g.src_pos, dev), _to_dev(g.det_center, dev),
            _to_dev(g.det_u_vec, dev), _to_dev(g.det_v_vec, dev),
            det_u, v1 - v0, du, dv, voxel_spacing)
        return sub_sino.detach().cpu().numpy()

    def write_fn(vv, band):
        v0, v1 = vv
        out[:, :, v0:v1] = band

    _conveyor_run(_chunks(det_v, max_rows), read_fn, compute_fn, write_fn, gpus, timings=timings)
    return out


def _fdk_gather_slab(filtered_dev, geom_tensors, D, z0, z1, H, W, du, dv, vs):
    """Run the distance-weighted FDK gather kernel for one z-slab on the current GPU.

    Uses ``cz = D/2 - z0`` (no geometry shift) so sub-voxels land at their true
    world-z; distance weighting and scale therefore stay consistent across slabs.
    ``geom_tensors`` = (src, det_center, det_u_vec, det_v_vec) torch tensors on
    this device (kept alive by the caller). Returns an *unscaled* (z1-z0, H, W).
    """
    src_t, dc_t, du_t, dv_t = geom_tensors
    n_views, n_u, n_v = filtered_dev.shape
    Nx, Ny, Nz = W, H, z1 - z0
    reco_perm = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=filtered_dev.device)
    grid, tpb = _grid_3d(Nz, Ny, Nx)
    stream = _get_numba_external_stream_for(torch.cuda.current_stream())
    _cone_3d_fdk_backproject_kernel[grid, tpb, stream](
        TorchCUDABridge.tensor_to_cuda_array(filtered_dev), n_views, n_u, n_v,
        TorchCUDABridge.tensor_to_cuda_array(reco_perm), Nx, Ny, Nz,
        _DTYPE(du), _DTYPE(dv),
        TorchCUDABridge.tensor_to_cuda_array(src_t),
        TorchCUDABridge.tensor_to_cuda_array(dc_t),
        TorchCUDABridge.tensor_to_cuda_array(du_t),
        TorchCUDABridge.tensor_to_cuda_array(dv_t),
        _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5 - z0), _DTYPE(vs),
    )
    return reco_perm.permute(2, 1, 0).contiguous()


def chunked_cone_fdk(sinogram, geom: ConeGeom, D, H, W,
                     du=1.0, dv=1.0, voxel_spacing=1.0,
                     fdk_weights=None, normalization_scale=None,
                     enforce_positivity=True, filter_axis=1,
                     backprojector="siddon", gpus=None, max_slices=None, out=None,
                     work_dir=None, ram_budget=None, timings=None, ramp_device=None):
    """Out-of-core + multi-GPU FDK reconstruction → volume (D, H, W).

    Applies optional per-detector FDK weighting and a ramp filter along the
    detector-u axis (numpy FFT, ``2|f|`` convention), then backprojects across
    z-slabs / GPUs and applies positivity + normalization. ``fdk_weights``
    broadcasts over views, shape (1, n_u, n_v). Provide ``normalization_scale``
    (e.g. ``pi*sid/(2*sdd*num_views)``) for a calibrated amplitude with the
    ``siddon`` / ``footprint`` backprojectors.

    Pass ``out=`` (a disk memmap, see :func:`open_memmap`) to stream the output
    volume to disk — positivity/scale are applied per slab so the full volume
    never materializes in RAM (TB-scale reconstructions). Note: the sinogram is
    ramp-filtered in RAM here; for a sinogram too large for RAM, pre-filter it
    into a memmap and pass it as ``sinogram`` with ``fdk_weights=None`` and a
    pre-multiplied ramp (see docs / follow-up).

    ``backprojector``: ``"siddon"`` (default, highest fidelity, matches
    ``reconstruct_fdk``), ``"footprint"`` (matched atomic-free), or ``"gather"``
    (distance-weighted voxel-driven FDK kernel; applies ``sdd/(2*pi*sid)``).
    """
    # Ramp-filter the sinogram, streaming over views (auto RAM/disk) so a
    # sinogram larger than RAM can be filtered without a full-array load.
    filtered = ramp_filter_memmap(sinogram, filter_axis=filter_axis,
                                  fdk_weights=fdk_weights,
                                  work_dir=work_dir, ram_budget=ram_budget,
                                  device=ramp_device)

    ns = float(normalization_scale) if normalization_scale is not None else 1.0

    def _post(slab, extra=1.0):
        s = slab * (float(extra) * ns) if (extra != 1.0 or ns != 1.0) else slab
        return np.maximum(s, 0.0) if enforce_positivity else s

    if backprojector in ("siddon", "footprint"):
        return chunked_cone_backward(
            filtered, geom, D, H, W, du, dv, voxel_spacing,
            projector=backprojector, gpus=gpus, max_slices=max_slices,
            out=out, post_fn=_post, work_dir=work_dir, ram_budget=ram_budget,
            timings=timings)
    elif backprojector == "gather":
        gpus_ = _resolve_gpus(gpus)
        n_views, n_u, n_v = filtered.shape
        step = _auto_max_slices(D, H, W, n_views, n_u, n_v, gpus_, max_slices)
        st, sd = _cone_mean_sid_sdd(
            torch.as_tensor(geom.src_pos), torch.as_tensor(geom.det_center),
            torch.as_tensor(geom.det_u_vec), torch.as_tensor(geom.det_v_vec))
        fdk_scale = sd / (2.0 * np.pi * st)
        out = _resolve_output(out, (D, H, W), "fdk_vol", work_dir, ram_budget)

        def make_task(z0, z1):
            def task(dev):
                torch.cuda.set_device(dev)
                from numba import cuda as _cuda
                _cuda.select_device(dev)
                filt_dev = _to_dev(filtered, dev)
                geom_tensors = (
                    _to_dev(geom.src_pos, dev), _to_dev(geom.det_center, dev),
                    _to_dev(geom.det_u_vec, dev), _to_dev(geom.det_v_vec, dev))
                slab = _fdk_gather_slab(filt_dev, geom_tensors, D, z0, z1, H, W,
                                        du, dv, voxel_spacing)
                out[z0:z1] = _post(slab.detach().cpu().numpy(), extra=fdk_scale)
            return task

        _run_on_gpus([make_task(z0, z1) for z0, z1 in _chunks(D, step)], gpus_)
        return out
    else:
        raise ValueError(
            f"backprojector must be 'siddon', 'footprint' or 'gather', got {backprojector!r}")


def chunked_sirt(measured, geom: ConeGeom, D, H, W, det_u, det_v,
                 du=1.0, dv=1.0, voxel_spacing=1.0,
                 n_iter=20, relaxation=1.0, projector="footprint",
                 enforce_positivity=True, gpus=None, max_slices=None,
                 work_dir=None, out=None, eps=1e-6, ram_budget=None):
    """Out-of-core + multi-GPU SIRT — works for volumes larger than RAM.

    Every large array (volume iterate, sinograms, row/column sums) is a disk
    memmap when ``work_dir`` is given; all projections stream slab/band-wise
    through the GPUs and every elementwise step is chunked, so peak RAM stays
    bounded by the chunk size rather than the full volume/sinogram. This is the
    path intended for TB-scale reconstruction (correct + feasible; disk-bound).

    Set ``work_dir`` to a directory on fast storage for TB-scale runs; leave it
    ``None`` to keep the scratch arrays in RAM (small problems). ``out`` may be a
    preallocated memmap for the result volume.
    """
    import os
    gpus = _resolve_gpus(gpus)
    m = _np32(measured)
    n_views = geom.n_views
    sino_shape = (n_views, det_u, det_v)
    vol_shape = (D, H, W)

    def alloc(shape, name):
        return _resolve_output(None, shape, "sirt_" + name, work_dir, ram_budget)

    fwd = lambda vol, o: chunked_cone_forward(vol, geom, det_u, det_v, du, dv,
                                              voxel_spacing, gpus=gpus, out=o)
    bwd = lambda sino, o: chunked_cone_backward(sino, geom, D, H, W, du, dv,
                                                voxel_spacing, projector=projector,
                                                gpus=gpus, max_slices=max_slices, out=o)

    ones_vol = alloc(vol_shape, "ones_vol"); ones_vol[:] = 1.0
    ones_sino = alloc(sino_shape, "ones_sino"); ones_sino[:] = 1.0
    rowsum = fwd(ones_vol, alloc(sino_shape, "rowsum"))      # A·1
    colsum = bwd(ones_sino, alloc(vol_shape, "colsum"))      # Aᵀ·1
    x = _alloc_out(out, vol_shape) if out is not None else alloc(vol_shape, "x")
    x[:] = 0.0
    ax = alloc(sino_shape, "ax")
    resid = alloc(sino_shape, "resid")
    bp = alloc(vol_shape, "bp")                               # reused each iteration

    vstep = max(1, min(n_views, 64))   # host-side elementwise chunking
    zstep = max(1, min(D, 64))

    for _ in range(int(n_iter)):
        fwd(x, ax)                                            # ax = A x
        for a, b in _chunks(n_views, vstep):                  # resid = R (m - ax)
            rs = rowsum[a:b]
            resid[a:b] = np.where(rs > eps, (m[a:b] - ax[a:b]) / np.where(rs > eps, rs, 1.0), 0.0)
        bwd(resid, bp)                                        # bp = Aᵀ resid
        for z0, z1 in _chunks(D, zstep):                      # x += relax C bp
            cs = colsum[z0:z1]
            upd = np.where(cs > eps, bp[z0:z1] / np.where(cs > eps, cs, 1.0), 0.0)
            xs = x[z0:z1] + float(relaxation) * upd
            x[z0:z1] = np.maximum(xs, 0.0) if enforce_positivity else xs
    return x


def _subset_geom(geom: ConeGeom, idx) -> ConeGeom:
    return ConeGeom(geom.src_pos[idx], geom.det_center[idx],
                    geom.det_u_vec[idx], geom.det_v_vec[idx])


def chunked_os_sart(measured, geom: ConeGeom, D, H, W, det_u, det_v,
                    du=1.0, dv=1.0, voxel_spacing=1.0,
                    n_iter=10, n_subsets=8, relaxation=1.0, projector="footprint",
                    enforce_positivity=True, gpus=None, max_slices=None,
                    work_dir=None, out=None, eps=1e-6, ram_budget=None):
    """Out-of-core + multi-GPU ordered-subset SART (OS-SART).

    The chunk-friendly SART variant: views are split into ``n_subsets``
    interleaved subsets and the volume is updated once per subset
    (``n_subsets=1`` reduces to SIRT). Reuses the conveyor-backed chunked
    forward/backprojection on the subset geometry, so it works for volumes
    larger than RAM. Global row/column sums are used as the SIRT-style
    normalizers. (Classic per-view SART is intentionally not offered out-of-core
    — its sequential per-view volume update cannot be chunked/parallelized; use
    ``diffct_mlx.run_sart`` for the in-VRAM case.)
    """
    gpus = _resolve_gpus(gpus)
    m = _lazy_input(measured)
    nv = geom.n_views
    vol_shape = (D, H, W)
    sino_shape = (nv, det_u, det_v)

    def alloc(shape, name):
        return _resolve_output(None, shape, "ossart_" + name, work_dir, ram_budget)

    ones_vol = alloc(vol_shape, "ones_vol"); ones_vol[:] = 1.0
    ones_sino = alloc(sino_shape, "ones_sino"); ones_sino[:] = 1.0
    rowsum = chunked_cone_forward(ones_vol, geom, det_u, det_v, du, dv, voxel_spacing,
                                  gpus=gpus, out=alloc(sino_shape, "rowsum"))
    colsum = chunked_cone_backward(ones_sino, geom, D, H, W, du, dv, voxel_spacing,
                                   projector=projector, gpus=gpus, max_slices=max_slices,
                                   out=alloc(vol_shape, "colsum"))
    x = _alloc_out(out, vol_shape) if out is not None else alloc(vol_shape, "x")
    x[:] = 0.0
    bp = alloc(vol_shape, "bp")
    subsets = [np.arange(j, nv, n_subsets) for j in range(int(n_subsets))]
    zstep = max(1, min(D, 64))

    for _ in range(int(n_iter)):
        for idx in subsets:
            gs = _subset_geom(geom, idx)
            ax_s = np.asarray(chunked_cone_forward(x, gs, det_u, det_v, du, dv,
                                                   voxel_spacing, gpus=gpus))
            rs = np.asarray(rowsum[idx], dtype=np.float32)
            m_s = np.asarray(m[idx], dtype=np.float32)
            resid = np.where(rs > eps, (m_s - ax_s) / np.where(rs > eps, rs, 1.0), 0.0).astype(np.float32)
            chunked_cone_backward(resid, gs, D, H, W, du, dv, voxel_spacing,
                                  projector=projector, gpus=gpus, max_slices=max_slices, out=bp)
            for z0, z1 in _chunks(D, zstep):
                cs = np.asarray(colsum[z0:z1])
                upd = np.where(cs > eps, np.asarray(bp[z0:z1]) / np.where(cs > eps, cs, 1.0), 0.0)
                xs = np.asarray(x[z0:z1]) + float(relaxation) * upd
                x[z0:z1] = np.maximum(xs, 0.0) if enforce_positivity else xs
    return x


# --------------------------------------------------------------------------- #
# View-parallel multi-GPU (for iterative reconstruction). Splitting by views
# needs NO shadow geometry — the per-view arrays are simply sliced — so this is
# the natural multi-GPU path when the volume fits one GPU (throughput scaling).
# --------------------------------------------------------------------------- #

def _view_chunks(n_views, gpus, per_gpu=2):
    n_bands = max(len(gpus), min(n_views, len(gpus) * per_gpu))
    step = (n_views + n_bands - 1) // n_bands
    return _chunks(n_views, step)


def _slice_geom(geom: ConeGeom, a, b) -> ConeGeom:
    return ConeGeom(geom.src_pos[a:b], geom.det_center[a:b],
                    geom.det_u_vec[a:b], geom.det_v_vec[a:b])


def mgpu_cone_forward(volume, geom: ConeGeom, det_u, det_v,
                      du=1.0, dv=1.0, voxel_spacing=1.0, gpus=None):
    """View-parallel multi-GPU cone footprint forward → (n_views, det_u, det_v).

    The volume is replicated to each GPU; views are split across GPUs. Use when
    the volume fits a single GPU (throughput scaling); use ``chunked_cone_*`` for
    volumes larger than VRAM.
    """
    gpus = _resolve_gpus(gpus)
    vol = _np32(volume)
    nv = geom.n_views
    out = np.zeros((nv, det_u, det_v), dtype=np.float32)

    def make(a, b):
        def task(dev):
            torch.cuda.set_device(dev)
            from numba import cuda as _cuda
            _cuda.select_device(dev)
            g = _slice_geom(geom, a, b)
            s = ConeFootprintProjectorFunction.apply(
                _to_dev(vol, dev), _to_dev(g.src_pos, dev), _to_dev(g.det_center, dev),
                _to_dev(g.det_u_vec, dev), _to_dev(g.det_v_vec, dev),
                det_u, det_v, du, dv, voxel_spacing)
            out[a:b] = s.detach().cpu().numpy()
        return task

    _run_on_gpus([make(a, b) for a, b in _view_chunks(nv, gpus)], gpus)
    return out


def mgpu_cone_backproject(sinogram, geom: ConeGeom, D, H, W,
                          du=1.0, dv=1.0, voxel_spacing=1.0,
                          projector="footprint", gpus=None):
    """View-parallel multi-GPU cone backprojection → volume (D, H, W).

    Backprojection is a sum over views, so each GPU backprojects its view band
    into a full-volume partial and the partials are summed on the host.
    """
    gpus = _resolve_gpus(gpus)
    backward_fn = _BACKWARD_FUNCS[projector]
    s = _np32(sinogram)
    nv = geom.n_views
    out = np.zeros((D, H, W), dtype=np.float32)
    lock = threading.Lock()

    def make(a, b):
        def task(dev):
            torch.cuda.set_device(dev)
            from numba import cuda as _cuda
            _cuda.select_device(dev)
            g = _slice_geom(geom, a, b)
            vol = backward_fn.apply(
                _to_dev(s[a:b], dev), _to_dev(g.src_pos, dev), _to_dev(g.det_center, dev),
                _to_dev(g.det_u_vec, dev), _to_dev(g.det_v_vec, dev),
                D, H, W, du, dv, voxel_spacing)
            part = vol.detach().cpu().numpy()
            with lock:
                out[:] += part
        return task

    _run_on_gpus([make(a, b) for a, b in _view_chunks(nv, gpus)], gpus)
    return out


def _bands_for(n_views, gpus):
    return [(a, b, gpus[i % len(gpus)]) for i, (a, b) in enumerate(_view_chunks(n_views, gpus))]


def _geom_band_dev(geom, a, b, dev):
    d = torch.device("cuda", dev)
    return (torch.as_tensor(geom.src_pos[a:b], device=d),
            torch.as_tensor(geom.det_center[a:b], device=d),
            torch.as_tensor(geom.det_u_vec[a:b], device=d),
            torch.as_tensor(geom.det_v_vec[a:b], device=d))


def _forward_resident(x_d0, geom, det_u, det_v, du, dv, vs, bands, d0):
    """A x with x resident on GPU d0; view bands fan out, result gathered on d0."""
    dev0 = torch.device("cuda", d0)
    res = {}

    def task(item):
        i, (a, b, dev) = item
        torch.cuda.set_device(dev)
        from numba import cuda as _cuda
        _cuda.select_device(dev)
        xg = x_d0 if dev == d0 else x_d0.to(torch.device("cuda", dev))
        s = ConeFootprintProjectorFunction.apply(xg, *_geom_band_dev(geom, a, b, dev),
                                                 det_u, det_v, du, dv, vs)
        res[i] = (a, b, s)

    with ThreadPoolExecutor(max_workers=len(set(d for _, _, d in bands))) as ex:
        list(ex.map(task, enumerate(bands)))
    ax = torch.empty((geom.n_views, det_u, det_v), dtype=torch.float32, device=dev0)
    for i in sorted(res):
        a, b, s = res[i]
        ax[a:b] = s.to(dev0)
    return ax


def _backproject_resident(sino_d0, geom, D, H, W, du, dv, vs, projector, bands, d0):
    """Aᵀ sino with sino resident on GPU d0; view bands fan out, summed on d0."""
    dev0 = torch.device("cuda", d0)
    backward_fn = _BACKWARD_FUNCS[projector]
    res = {}

    def task(item):
        i, (a, b, dev) = item
        torch.cuda.set_device(dev)
        from numba import cuda as _cuda
        _cuda.select_device(dev)
        sg = sino_d0[a:b] if dev == d0 else sino_d0[a:b].to(torch.device("cuda", dev))
        vol = backward_fn.apply(sg, *_geom_band_dev(geom, a, b, dev),
                                D, H, W, du, dv, vs)
        res[i] = vol

    with ThreadPoolExecutor(max_workers=len(set(d for _, _, d in bands))) as ex:
        list(ex.map(task, enumerate(bands)))
    acc = torch.zeros((D, H, W), dtype=torch.float32, device=dev0)
    for i in sorted(res):
        acc += res[i].to(dev0)
    return acc


def mgpu_sirt(measured, geom: ConeGeom, D, H, W, det_u, det_v,
              du=1.0, dv=1.0, voxel_spacing=1.0,
              n_iter=20, relaxation=1.0, projector="footprint",
              enforce_positivity=True, gpus=None, eps=1e-6, resident=True):
    """View-parallel multi-GPU SIRT cone-beam reconstruction → volume (D, H, W).

    Standard SIRT: ``x <- x + relax * C Aᵀ R (measured - A x)`` with
    ``R = 1/(A·1)``, ``C = 1/(Aᵀ·1)``. Every forward/backprojection fans out
    across GPUs by views (matched footprint pair by default).

    ``resident=True`` (default) keeps the iterate and sinogram intermediates on
    GPU0 as torch tensors and reduces partials device-to-device, avoiding the
    per-iteration host round-trips — better throughput scaling for volumes that
    fit one GPU. ``resident=False`` uses host-resident numpy intermediates (works
    the same, simpler; kept for reference). For volumes larger than VRAM/RAM use
    :func:`chunked_sirt` instead.
    """
    gpus = _resolve_gpus(gpus)
    m = _np32(measured)

    if resident:
        d0 = gpus[0]
        dev0 = torch.device("cuda", d0)
        bands = _bands_for(geom.n_views, gpus)
        torch.cuda.set_device(d0)
        m_t = torch.as_tensor(m, device=dev0)
        fwd = lambda x: _forward_resident(x, geom, det_u, det_v, du, dv, voxel_spacing, bands, d0)
        bwd = lambda s: _backproject_resident(s, geom, D, H, W, du, dv, voxel_spacing, projector, bands, d0)
        rowsum = fwd(torch.ones((D, H, W), dtype=torch.float32, device=dev0))
        colsum = bwd(torch.ones_like(m_t))
        row_ok, col_ok = rowsum > eps, colsum > eps
        rowsafe = torch.where(row_ok, rowsum, torch.ones_like(rowsum))
        colsafe = torch.where(col_ok, colsum, torch.ones_like(colsum))
        x = torch.zeros((D, H, W), dtype=torch.float32, device=dev0)
        for _ in range(int(n_iter)):
            residual = torch.where(row_ok, (m_t - fwd(x)) / rowsafe, torch.zeros_like(m_t))
            update = torch.where(col_ok, bwd(residual) / colsafe, torch.zeros_like(colsum))
            x = x + float(relaxation) * update
            if enforce_positivity:
                x = torch.clamp(x, min=0.0)
        return x.detach().cpu().numpy()

    fwd = lambda vol: mgpu_cone_forward(vol, geom, det_u, det_v, du, dv, voxel_spacing, gpus=gpus)
    bwd = lambda sino: mgpu_cone_backproject(sino, geom, D, H, W, du, dv, voxel_spacing,
                                             projector=projector, gpus=gpus)
    rowsum = fwd(np.ones((D, H, W), dtype=np.float32))
    colsum = bwd(np.ones_like(m))
    row_ok, col_ok = rowsum > eps, colsum > eps
    rowsafe = np.where(row_ok, rowsum, 1.0)
    colsafe = np.where(col_ok, colsum, 1.0)
    x = np.zeros((D, H, W), dtype=np.float32)
    for _ in range(int(n_iter)):
        residual = np.where(row_ok, (m - fwd(x)) / rowsafe, 0.0).astype(np.float32)
        update = np.where(col_ok, bwd(residual) / colsafe, 0.0).astype(np.float32)
        x = x + float(relaxation) * update
        if enforce_positivity:
            x = np.maximum(x, 0.0)
    return x
