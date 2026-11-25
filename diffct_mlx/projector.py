"""Unified :class:`Projector`: one object for projection and backprojection.

``dct.Projector`` mirrors the constructor of the engine's
:class:`diffct_mlx.engine.Projector` (upstream diffct 2.5) and offers the same
features on both backends, apart from multi-device execution:

* precision ``"fast"`` (native kernels), ``"float32"`` and ``"float64"``,
* per-axis ``voxel_spacing`` in tensor order ``(dy, dx)`` / ``(dz, dy, dx)``,
* beams ``"parallel"``, ``"fan"``, ``"cone"`` and ``"parallel3d"``,
* curved detectors through ``detector_surface`` (``ParameterizedSurface`` or
  a two-argument callback),
* chunking by views, volume blocks and detector blocks, leading batch
  dimensions, and ``project_into`` / ``backproject_into`` for outputs that do
  not fit in memory (NumPy memmaps, zarr arrays, engine stores on Torch),
* gradients with respect to the data, the trajectory and surface parameters.

**Torch.** The Siddon model runs on the engine, with all engine options.
The footprint model runs on the DiffCT-MLX footprint kernels (CUDA tensors).

**MLX.** Everything runs on Metal in float32: flat parallel, fan and cone
beams with scalar spacing on the frame-based kernels (``path="kernel"``),
per-axis spacing, ``parallel3d``, detector surfaces and ``"float32"`` on the
ray kernels of :mod:`diffct_mlx.backend.metal.rays` (``path="rays"``), which
take every ray explicitly. Metal has no float64; ``precision="float64"`` runs
in float32 with a warning, so scripts written for the Torch backend still
run.
Multi-device options (``devices``, ``distributed``, ``partition`` …) raise
:class:`NotImplementedError` on MLX.

The footprint model supports flat detectors, scalar spacing and float32 on
both backends.

Both backends follow the 3.0 conventions (detector pixel ``i`` at
``det_center + (i - (N-1)/2)·du``, physical line integrals);
``detector_convention="legacy"`` and ``line_integrals="voxel"`` reproduce
2.x.

Example::

    P = dct.Projector(dct.circular_trajectory_3d(360, 600, 900), (D, H, W), (nu, nv),
                      beam="cone", detector_spacing=(du, dv), voxel_spacing=vs)
    y = P(volume)                 # (360, nu, nv)
    x = P.backproject(y)          # (D, H, W)
    A = P.as_operator()           # LinearOperator for the solver registry
"""

from __future__ import annotations

import itertools
import warnings
import numbers
from typing import Any

import numpy as np

from .backend import NAME as _BACKEND, active as _b, xp
from .conventions import (
    DETECTOR_CONVENTIONS,
    LINE_INTEGRAL_UNITS,
    _check,
    kernel_detector_center,
)
from .operators import ProjectionOperator
from . import projectors as _p

if _BACKEND == "mlx":
    import mlx.core as mx
    from .backend.metal import rays as _rays
    from .backend.metal.detector import ParameterizedSurface
else:  # pragma: no cover - exercised on the Torch backend
    mx = None
    _rays = None
    from .engine import ParameterizedSurface

__all__ = ["Projector", "ParameterizedSurface", "BEAMS", "PROJECTOR_MODES", "PRECISIONS"]

#: Beam geometries.
BEAMS = ("parallel", "fan", "cone", "parallel3d")
#: Projector models: thin-ray Siddon or separable footprint.
PROJECTOR_MODES = ("siddon", "footprint")
#: Precision modes, as in the engine.
PRECISIONS = ("fast", "float32", "float64")

_NDIM = {"parallel": 2, "fan": 2, "cone": 3, "parallel3d": 3}
_N_FRAMES = {"parallel": 3, "fan": 3, "cone": 4, "parallel3d": 4}
#: Trajectory frames that are positions (shifted for volume blocks).
_POSITIONS = {"parallel": (1,), "fan": (0, 1), "cone": (0, 1), "parallel3d": (1,)}
#: Engine options that only concern multi-device execution; their defaults
#: are accepted everywhere.
_DEVICE_OPTIONS = {"devices": None, "distributed": False, "process_group": None,
                   "partition": "views", "partition_grid": None, "schedule": "auto",
                   "native_volume_layout": False}


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    """Broadcast a scalar or validate an ``n``-tuple of positive floats."""
    if isinstance(value, numbers.Real):
        out = (float(value),) * n
    else:
        out = tuple(float(v) for v in value)
        if len(out) != n:
            raise ValueError(f"{name} must be a scalar or a {n}-tuple, got {value!r}")
    if any(not v > 0 for v in out):
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def _requires_grad(arrays) -> bool:
    return any(bool(getattr(a, "requires_grad", False)) for a in arrays)


def _blocks(total: int, chunk: int | None) -> list[slice]:
    if chunk is None or int(chunk) >= total:
        return [slice(0, total)]
    step = int(chunk)
    if step < 1:
        raise ValueError("chunk sizes must be positive")
    return [slice(s, min(s + step, total)) for s in range(0, total, step)]


def _chunk_shape(value, n, name):
    if value is None:
        return (None,) * n
    if isinstance(value, numbers.Integral):
        return (int(value),) * n
    out = tuple(None if v is None else int(v) for v in value)
    if len(out) != n:
        raise ValueError(f"{name} must have {n} entries, got {value!r}")
    return out


def _concat(parts, axis):
    return parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=axis)


def _assemble(parts: dict, axis_blocks: list[list[slice]], first_axis: int):
    """Concatenate a grid of blocks keyed by their slice tuples."""
    def rec(prefix, axis):
        if axis == len(axis_blocks):
            return parts[prefix]
        return _concat([rec(prefix + (s,), axis + 1) for s in axis_blocks[axis]],
                       first_axis + axis)
    return rec((), 0)


def _length(s: slice) -> int:
    return s.stop - s.start


def _is_backend_array(a) -> bool:
    if _BACKEND == "mlx":
        return isinstance(a, mx.array)
    import torch

    return isinstance(a, torch.Tensor)


class Projector:
    """Matched projector/backprojector pair on the active backend.

    Parameters
    ----------
    trajectory : tuple of arrays
        Per-view geometry: ``(ray_dir, det_origin, det_u_vec)`` for
        ``"parallel"``, ``(src_pos, det_center, det_u_vec)`` for ``"fan"``,
        ``(src_pos, det_center, det_u_vec, det_v_vec)`` for ``"cone"`` and
        ``(ray_dir, det_origin, det_u_vec, det_v_vec)`` for ``"parallel3d"``,
        each of shape ``(n_views, 2 or 3)``. The trajectory generators return
        these tuples.
    volume_shape : tuple of int
        ``(H, W)`` for 2D beams, ``(D, H, W)`` for 3D beams.
    detector_shape : int or tuple of int
        ``n_det`` (or ``(n_det,)``) for 2D beams, ``(n_u, n_v)`` for 3D beams.
    beam : str
        One of :data:`BEAMS`.
    detector_spacing : float or tuple
        Detector pitch; ``(du, dv)`` for 3D beams (a scalar is broadcast).
    voxel_spacing : float or tuple
        Voxel pitch, or per-axis pitches in tensor order.
    projector_mode : str
        ``"siddon"`` (default) or ``"footprint"``.
    precision : str
        ``"fast"`` (default), ``"float32"`` or ``"float64"``. Torch: float64
        needs float64 data and trajectories. MLX: float64 runs in float32
        (with a warning), since Metal has no float64.
    detector_surface : ParameterizedSurface or callable, optional
        Curved detector (Siddon only), with the engine's local frame:
        ``center + o_u·det_u + o_v·det_v + o_n·cross(det_u, det_v)``.
    view_chunk_size : int, optional
        Process at most this many views at a time.
    volume_chunk_shape, detector_chunk_shape : int or tuple, optional
        Volume and detector blocking; ``None`` entries leave an axis whole.
    backend : str
        Torch: the engine's ``"auto"``, ``"numba"`` or ``"torch"``. MLX:
        ``"auto"`` or ``"metal"``.
    detector_convention : str
        ``"centered"`` (default) or ``"legacy"`` (2.x pixel grid).
    line_integrals : str
        ``"physical"`` (default) or ``"voxel"`` (2.x units; scalar spacing).
    **engine_options
        Further keywords for :class:`diffct_mlx.engine.Projector` (Torch
        Siddon), e.g. ``devices``, ``schedule``, ``partition``,
        ``distributed``. On MLX only their defaults are accepted.
    """

    def __init__(
        self,
        trajectory,
        volume_shape,
        detector_shape,
        *,
        beam: str = "cone",
        detector_spacing: Any = 1.0,
        voxel_spacing: Any = 1.0,
        projector_mode: str = "siddon",
        precision: str = "fast",
        detector_surface: Any = None,
        view_chunk_size: int | None = None,
        volume_chunk_shape: Any = None,
        detector_chunk_shape: Any = None,
        backend: str = "auto",
        detector_convention: str = "centered",
        line_integrals: str = "physical",
        **engine_options: Any,
    ):
        self.beam = _check(beam, BEAMS, "beam")
        self.projector_mode = _check(projector_mode, PROJECTOR_MODES, "projector_mode")
        self.precision = _check(precision, PRECISIONS, "precision")
        if _BACKEND == "mlx" and self.precision == "float64":
            warnings.warn(
                'precision="float64" runs in float32 on the MLX backend (Metal has no '
                "float64).", RuntimeWarning, stacklevel=2)
            self.precision = "float32"
        self.detector_convention = _check(
            detector_convention, DETECTOR_CONVENTIONS, "detector_convention")
        self.line_integrals = _check(line_integrals, LINE_INTEGRAL_UNITS, "line_integrals")

        ndim = _NDIM[self.beam]
        self.volume_shape = tuple(int(s) for s in volume_shape)
        if len(self.volume_shape) != ndim:
            raise ValueError(f"beam={self.beam!r} needs a {ndim}D volume_shape, got {volume_shape!r}")
        det = (detector_shape,) if isinstance(detector_shape, numbers.Integral) else detector_shape
        self.detector_shape = tuple(int(s) for s in det)
        if len(self.detector_shape) != ndim - 1:
            raise ValueError(
                f"beam={self.beam!r} needs a {ndim - 1}D detector_shape, got {detector_shape!r}")

        self.trajectory = tuple(trajectory)
        if len(self.trajectory) != _N_FRAMES[self.beam]:
            raise ValueError(
                f"beam={self.beam!r} needs {_N_FRAMES[self.beam]} trajectory arrays, "
                f"got {len(self.trajectory)}")
        self.n_views = int(self.trajectory[0].shape[0])

        spacing = _floats(detector_spacing, ndim - 1, "detector_spacing")
        self.detector_spacing = spacing[0] if ndim == 2 else spacing
        self._pitches = _floats(voxel_spacing, ndim, "voxel_spacing")
        self.per_axis_spacing = len(set(self._pitches)) > 1
        self.voxel_spacing = self._pitches if self.per_axis_spacing else self._pitches[0]
        if self.line_integrals == "voxel" and self.per_axis_spacing:
            raise ValueError('line_integrals="voxel" needs a scalar voxel_spacing')
        self._scale = 1.0 / self._pitches[0] if self.line_integrals == "voxel" else 1.0

        if detector_surface is not None and self.detector_convention != "centered":
            raise ValueError("detector_surface defines its own pixel grid; use "
                             'detector_convention="centered"')
        if view_chunk_size is not None and int(view_chunk_size) < 1:
            raise ValueError("view_chunk_size must be a positive integer")
        self.detector_surface = detector_surface
        self.view_chunk_size = None if view_chunk_size is None else int(view_chunk_size)
        # ``None`` entries keep an axis whole; the engine wants full sizes there.
        self.volume_chunk_shape = None if volume_chunk_shape is None else tuple(
            n if c is None else min(c, n) for n, c in zip(
                self.volume_shape, _chunk_shape(volume_chunk_shape, ndim, "volume_chunk_shape")))
        self.detector_chunk_shape = None if detector_chunk_shape is None else tuple(
            n if c is None else min(c, n) for n, c in zip(
                self.detector_shape,
                _chunk_shape(detector_chunk_shape, ndim - 1, "detector_chunk_shape")))
        self._backend_option = backend
        self._engine_options = dict(engine_options)
        self._subsets: dict = {}
        self._engine = None
        self._path = None

        if _BACKEND == "torch" and self.projector_mode == "siddon":
            self._engine = self._build_engine()
            self.path = "engine"
        else:
            self._path = self._select_path()
            self.path = self._path
            self._volume_axis_blocks = [
                _blocks(n, c) for n, c in zip(
                    self.volume_shape, _chunk_shape(self.volume_chunk_shape, ndim, ""))]
            self._volume_blocks = list(itertools.product(*self._volume_axis_blocks))
            self._detector_axis_blocks = [
                _blocks(n, c) for n, c in zip(
                    self.detector_shape, _chunk_shape(self.detector_chunk_shape, ndim - 1, ""))]
            self._detector_blocks = list(itertools.product(*self._detector_axis_blocks))

    # -- shapes ---------------------------------------------------------------

    @property
    def projection_shape(self) -> tuple[int, ...]:
        """Shape of a sinogram: ``(n_views, *detector_shape)``."""
        return (self.n_views,) + self.detector_shape

    def __repr__(self) -> str:
        return (f"Projector(beam={self.beam!r}, projector_mode={self.projector_mode!r}, "
                f"volume_shape={self.volume_shape}, projection_shape={self.projection_shape}, "
                f"precision={self.precision!r}, path={self.path!r})")

    # -- conventions ----------------------------------------------------------

    def _kernel_trajectory(self, trajectory):
        """Shift the detector centre from ``detector_convention`` to the kernel grid."""
        if self.detector_convention == "centered":
            return tuple(trajectory)
        frames = list(trajectory)
        if _NDIM[self.beam] == 2:
            frames[1] = kernel_detector_center(
                frames[1], frames[2], self.detector_spacing, convention=self.detector_convention)
        else:
            du, dv = self.detector_spacing
            frames[1] = kernel_detector_center(
                frames[1], frames[2], du, frames[3], dv, convention=self.detector_convention)
        return tuple(frames)

    def _scaled(self, y):
        return y if self._scale == 1.0 else y * self._scale

    # -- path selection -------------------------------------------------------

    def _unsupported(self, what: str, why: str = ""):
        where = ("the MLX backend" if _BACKEND == "mlx"
                 else f'projector_mode="{self.projector_mode}" on the Torch backend')
        raise NotImplementedError(f"{what} is not available on {where}.{why}")

    def _select_path(self) -> str:
        for key, default in _DEVICE_OPTIONS.items():
            value = self._engine_options.pop(key, default)
            if value is not None and value is not default and value != default:
                self._unsupported(f"{key}={value!r}", " Multi-device and distributed execution "
                                  "need the Torch engine (Siddon model).")
        if self._engine_options:
            raise TypeError(f"unexpected keyword(s): {', '.join(sorted(self._engine_options))}")
        backend = str(self._backend_option).strip().lower()
        # Flat parallel/fan/cone rays with scalar spacing use the frame-based
        # Metal kernels; everything else in float32 the Metal ray kernels.
        needs_rays = (self.beam == "parallel3d" or self.detector_surface is not None
                      or self.precision == "float32" or self.per_axis_spacing)

        if self.projector_mode == "footprint":
            for flag, what in ((self.beam == "parallel3d", 'beam="parallel3d"'),
                               (self.detector_surface is not None, "detector_surface"),
                               (self.precision == "float64", 'precision="float64"'),
                               (self.per_axis_spacing, "unequal per-axis voxel_spacing")):
                if flag:
                    raise NotImplementedError(
                        f'{what} is not available for projector_mode="footprint" (on either '
                        'backend); use the Siddon model.')
            native = "metal" if _BACKEND == "mlx" else "numba"
            if backend not in ("auto", native):
                raise ValueError(f'projector_mode="footprint" needs backend="auto" or "{native}"')
            return "kernel"

        if backend in ("auto", "metal"):
            return "rays" if needs_rays else "kernel"
        raise ValueError(f'backend must be "auto" or "metal" on MLX, got {backend!r}')

    # -- Torch Siddon: the engine Projector -----------------------------------

    def _build_engine(self):
        from . import engine

        trajectory = self._kernel_trajectory(self.trajectory)
        if self.detector_convention != "centered":
            # Rebuilt per call when the trajectory is learnable.
            trajectory = tuple(t.detach() for t in trajectory)
        return engine.Projector(
            trajectory, self.volume_shape, self.detector_shape, beam=self.beam,
            detector_spacing=self.detector_spacing, voxel_spacing=self.voxel_spacing,
            precision=self.precision, detector_surface=self.detector_surface,
            view_chunk_size=self.view_chunk_size, volume_chunk_shape=self.volume_chunk_shape,
            detector_chunk_shape=self.detector_chunk_shape, backend=self._backend_option,
            **self._engine_options,
        )

    def _engine_call(self, method, *args, trajectory=None):
        override = trajectory
        if override is None and self.detector_convention != "centered" \
                and _requires_grad(self.trajectory):
            override = self.trajectory
        if override is not None:
            override = self._kernel_trajectory(override)
        fn = getattr(self._engine, method)
        return fn(*args) if override is None else fn(*args, trajectory=override)

    # -- Metal kernel and ray paths --------------------------------------------

    def _work(self, a):
        """Backend array in the working dtype."""
        if not _is_backend_array(a):
            a = _b.as_array(np.asarray(a))
        if _BACKEND == "mlx" and a.dtype != mx.float32:
            a = a.astype(mx.float32)
        return a

    def _frames(self, trajectory):
        frames = self._kernel_trajectory(self.trajectory if trajectory is None else trajectory)
        if _BACKEND == "mlx":
            frames = tuple(self._work(f) for f in frames)
        return frames

    def _const(self, values, like):
        if _BACKEND == "mlx":
            return mx.array(values, dtype=like.dtype)
        import torch

        return torch.tensor(values, dtype=like.dtype, device=like.device)

    def _shift_volume(self, frames, block):
        """Move the trajectory so that the volume block is centred on the origin."""
        if all(_length(s) == n for s, n in zip(block, self.volume_shape)):
            return frames
        centre = [(s.start + _length(s) / 2.0 - n / 2.0) * p
                  for s, n, p in zip(block, self.volume_shape, self._pitches)][::-1]
        out = list(frames)
        for i in _POSITIONS[self.beam]:
            out[i] = out[i] - self._const(centre, out[i])
        return tuple(out)

    def _shift_detector(self, frames, pixels):
        """Flat detector: centre and shape of the pixel block ``pixels``."""
        shape = tuple(_length(s) for s in pixels)
        if shape == self.detector_shape:
            return frames, shape
        spacing = self.detector_spacing if _NDIM[self.beam] == 3 else (self.detector_spacing,)
        out = list(frames)
        for k, (s, n, pitch) in enumerate(zip(pixels, self.detector_shape, spacing)):
            offset = (s.start + (_length(s) - 1) / 2.0 - (n - 1) / 2.0) * pitch
            if offset:
                out[1] = out[1] + offset * out[2 + k]
        return tuple(out), shape

    def _kernel(self, adjoint, x, frames, block_shape, detector_shape, vs):
        if self.projector_mode == "siddon":
            pair = {"parallel": (_p.parallel_forward, _p.parallel_backward),
                    "fan": (_p.fan_forward, _p.fan_backward),
                    "cone": (_p.cone_forward, _p.cone_backward)}[self.beam]
        else:
            pair = {"parallel": (_p.parallel_forward_footprint, _p.parallel_backward_footprint),
                    "fan": (_p.fan_forward_footprint, _p.fan_backward_footprint),
                    "cone": (_p.cone_forward_footprint, _p.cone_backward_footprint)}[self.beam]
        if self.beam == "cone":
            (nu, nv), (du, dv) = detector_shape, self.detector_spacing
            if adjoint:
                return pair[1](x, *frames, *block_shape, du, dv, vs)
            return pair[0](x, *frames, nu, nv, du, dv, vs)
        if adjoint:
            return pair[1](x, *frames, self.detector_spacing, *block_shape, vs)
        return pair[0](x, *frames, detector_shape[0], self.detector_spacing, vs)

    def _run(self, adjoint, x, frames, view_slice, block_shape, pixels, surface_parameters):
        if self._path == "rays":
            impl = _rays
            kw = dict(pixels=pixels, surface=self.detector_surface,
                      surface_parameters=surface_parameters,
                      view_ids=mx.arange(view_slice.start, view_slice.stop),
                      n_total_views=self.n_views)
            if adjoint:
                return impl.backproject(x, self.beam, frames, block_shape, self.detector_shape,
                                        self.detector_spacing, self._pitches, **kw)
            return impl.project(x, self.beam, frames, self.detector_shape,
                                self.detector_spacing, self._pitches, **kw)
        frames, detector_shape = self._shift_detector(frames, pixels)
        return self._kernel(adjoint, x, frames, block_shape, detector_shape, self._pitches[0])

    def _check_footprint_device(self, x):
        if _BACKEND == "torch" and self.projector_mode == "footprint" \
                and not getattr(x, "is_cuda", False):
            raise NotImplementedError(
                'projector_mode="footprint" on the Torch backend needs CUDA tensors '
                "(the footprint kernels are numba-CUDA only); use the Siddon model "
                "on CPU or MPS.")

    def _view_blocks(self):
        return _blocks(self.n_views, self.view_chunk_size)

    def _project_views(self, x, frames, surface_parameters):
        """Yield ``(view_slice, projection)`` per view block."""
        for vs in self._view_blocks():
            fv = tuple(f[vs] for f in frames)
            parts = {}
            for pixels in self._detector_blocks:
                acc = None
                for block in self._volume_blocks:
                    xb = self._work(x[block])
                    part = self._run(False, xb, self._shift_volume(fv, block), vs,
                                     tuple(_length(s) for s in block), pixels, surface_parameters)
                    acc = part if acc is None else acc + part
                parts[pixels] = acc
            yield vs, _assemble(parts, self._detector_axis_blocks, 1)

    def _backproject_blocks(self, y, frames, surface_parameters):
        """Yield ``(volume_block, backprojection)`` per volume block."""
        for block in self._volume_blocks:
            acc = None
            for vs in self._view_blocks():
                fv = self._shift_volume(tuple(f[vs] for f in frames), block)
                for pixels in self._detector_blocks:
                    yb = self._work(y[(vs,) + pixels])
                    part = self._run(True, yb, fv, vs, tuple(_length(s) for s in block),
                                     pixels, surface_parameters)
                    acc = part if acc is None else acc + part
            yield block, acc

    def _kernel_call(self, adjoint, x, trajectory, surface_parameters):
        self._check_footprint_device(x)
        core = self.projection_shape if adjoint else self.volume_shape
        if len(x.shape) > len(core):
            lead = tuple(x.shape[:len(x.shape) - len(core)])
            flat = x.reshape((-1,) + core)
            out = xp.stack([self._kernel_call(adjoint, flat[i], trajectory, surface_parameters)
                            for i in range(flat.shape[0])], axis=0)
            return out.reshape(lead + tuple(out.shape[1:]))
        frames = self._frames(trajectory)
        if adjoint:
            parts = dict(self._backproject_blocks(x, frames, surface_parameters))
            return _assemble(parts, self._volume_axis_blocks, 0)
        return _concat([p for _, p in self._project_views(x, frames, surface_parameters)], 0)

    # -- public API -----------------------------------------------------------

    def _check_surface_parameters(self, surface_parameters):
        if surface_parameters is not None and _BACKEND == "torch":
            raise TypeError("On Torch the engine reads surface parameters live from the "
                            "ParameterizedSurface tensors; give them requires_grad instead.")

    def project(self, volume, *, trajectory=None, surface_parameters=None):
        """Forward projection ``A x``: ``volume_shape`` to ``projection_shape``.

        ``trajectory`` replaces the constructor trajectory for this call (same
        view count) and ``surface_parameters`` the parameters of a
        ``ParameterizedSurface`` (MLX), e.g. to take gradients with ``mx.grad``.
        Leading batch dimensions are kept.
        """
        self._check_surface_parameters(surface_parameters)
        if self._engine is not None:
            y = self._engine_call("project", volume, trajectory=trajectory)
        else:
            y = self._kernel_call(False, volume, trajectory, surface_parameters)
        return self._scaled(y)

    __call__ = project

    def backproject(self, sinogram, *, trajectory=None, surface_parameters=None):
        """Matched adjoint ``A^T y``: ``projection_shape`` to ``volume_shape``."""
        self._check_surface_parameters(surface_parameters)
        if self._engine is not None:
            x = self._engine_call("backproject", sinogram, trajectory=trajectory)
        else:
            x = self._kernel_call(True, sinogram, trajectory, surface_parameters)
        return self._scaled(x)

    def project_into(self, volume, output, *, trajectory=None):
        """Project block by block into ``output`` and return it (no autograd).

        ``volume`` and ``output`` may be NumPy arrays or memmaps (or any array
        with NumPy slicing), so neither has to fit in device memory; one view
        block is computed at a time. On the Torch engine path this is the
        engine's ``project_into`` (tensors and engine stores).
        """
        if self._engine is not None:
            if self._scale != 1.0:
                raise NotImplementedError('project_into with line_integrals="voxel"')
            return self._engine_call("project_into", volume, output, trajectory=trajectory)
        frames = self._frames(trajectory)
        for vs, part in self._project_views(volume, frames, None):
            self._write(output, vs, self._scaled(part))
        return output

    def backproject_into(self, sinogram, output, *, trajectory=None):
        """Backproject volume block by volume block into ``output`` (no autograd)."""
        if self._engine is not None:
            if self._scale != 1.0:
                raise NotImplementedError('backproject_into with line_integrals="voxel"')
            return self._engine_call("backproject_into", sinogram, output, trajectory=trajectory)
        frames = self._frames(trajectory)
        for block, part in self._backproject_blocks(sinogram, frames, None):
            self._write(output, block, self._scaled(part))
        return output

    @staticmethod
    def _write(output, index, value):
        if _is_backend_array(output):
            output[index] = value.astype(output.dtype) if _BACKEND == "mlx" else value.to(output.dtype)
        else:
            output[index] = _b.to_numpy(value)

    def subset(self, views) -> "Projector":
        """Projector restricted to ``views`` (a slice, list or index array)."""
        if self.detector_surface is not None:
            raise NotImplementedError(
                "subset() renumbers the views, which a detector_surface sampler "
                "may depend on; build a Projector for the subset trajectory instead.")
        if isinstance(views, slice):
            key = ("slice", views.start, views.stop, views.step)
        elif hasattr(views, "shape"):
            key = ("index",) + tuple(int(v) for v in _b.to_numpy(views).reshape(-1))
        else:
            key = ("index",) + tuple(int(v) for v in views)
        if key not in self._subsets:
            index = views if isinstance(views, slice) or hasattr(views, "shape") \
                else _b.as_array(list(views), dtype=xp.int64,
                                 device=_b.device_of(self.trajectory[0]))
            self._subsets[key] = Projector(
                tuple(t[index] for t in self.trajectory), self.volume_shape,
                self.detector_shape, beam=self.beam,
                detector_spacing=self.detector_spacing, voxel_spacing=self.voxel_spacing,
                projector_mode=self.projector_mode, precision=self.precision,
                view_chunk_size=self.view_chunk_size,
                volume_chunk_shape=self.volume_chunk_shape,
                detector_chunk_shape=self.detector_chunk_shape,
                backend=self._backend_option,
                detector_convention=self.detector_convention,
                line_integrals=self.line_integrals,
                **self._engine_options,
            )
        return self._subsets[key]

    def as_operator(self) -> ProjectionOperator:
        """This projector as a :class:`~diffct_mlx.operators.ProjectionOperator`.

        The operator works with ``A @ x``, ``A.T @ y``, the operator algebra,
        ``A.subset(views)`` and every solver in the registry.
        """
        def fwd(x, views):
            return (self if views is None else self.subset(views)).project(x)

        def adj(y, views):
            return (self if views is None else self.subset(views)).backproject(y)

        return ProjectionOperator(
            fwd, adj, domain_shape=self.volume_shape, range_shape=self.projection_shape,
            n_views=self.n_views, beam=self.beam, voxel_spacing=self.voxel_spacing,
            line_integrals=self.line_integrals,
        )
