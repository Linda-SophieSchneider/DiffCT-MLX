"""Detector pixel positions on MLX: flat panels and parameterized surfaces.

World positions of the detector pixels for every view, in plain MLX code so
that gradients reach the trajectory and the surface parameters. Conventions
match the engine: pixel ``i`` at ``(i + 1/2 - N/2)·pitch`` and, for curved
detectors, ``center + o_u·det_u + o_v·det_v + o_n·cross(det_u, det_v)`` (3D)
or ``center + o_u·det_u + o_n·(det_u_y, -det_u_x)`` (2D).
"""

from __future__ import annotations

import mlx.core as mx

_THREE_D = ("cone", "parallel3d")


class ParameterizedSurface:
    """Detector surface sampled from global pixel and view coordinates (MLX).

    ``sampler(u, v, view_indices, *parameters)`` returns local ``(u, v, n)``
    offsets of shape ``(*pixels, 3)`` (shared by all views) or
    ``(len(view_indices), *pixels, 3)``. ``u`` and ``v`` are the physical
    pixel-centre coordinates of the selected pixels, ``ij``-indexed; for 2D
    beams ``v`` is zero and the middle offset must be zero. The world point is
    ``center + o_u·det_u + o_v·det_v + o_n·cross(det_u, det_v)`` (3D) or
    ``center + o_u·det_u + o_n·(det_u_y, -det_u_x)`` (2D), as in the engine.

    ``parameters`` are MLX arrays. Their gradients are available through
    ``Projector.project(x, surface_parameters=params)`` under ``mx.grad``.
    """

    def __init__(self, sampler, parameters=()):
        if not callable(sampler):
            raise TypeError("surface sampler must be callable")
        self.sampler = sampler
        self.parameters = tuple(parameters)
        for p in self.parameters:
            if not isinstance(p, mx.array) or not mx.issubdtype(p.dtype, mx.floating):
                raise TypeError("surface parameters must be floating-point MLX arrays")

    def __call__(self, u, v, view_indices, *parameters):
        params = parameters if parameters else self.parameters
        return self.sampler(u, v, view_indices, *params)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _c(value, dtype):
    """A scalar constant in ``dtype``; MLX rounds Python floats to float32."""
    return mx.array(value, dtype=dtype)


def _grid(n, pitch, sel, dtype):
    start, stop, _ = sel.indices(n)
    return (mx.arange(start, stop, dtype=dtype) + 0.5 - n / 2.0) * _c(pitch, dtype)


def _cross(a, b):
    return mx.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


def _surface_offsets(surface, u, v, view_ids, n_total_views, pixel_sel, full_grids,
                     surface_parameters):
    """Sample local offsets for the selected views and pixels."""
    if isinstance(surface, ParameterizedSurface):
        offsets = surface(u, v, view_ids, *(surface_parameters or ()))
        return offsets
    # Two-argument callback: complete grids, per-view offsets for all views.
    offsets = surface(*full_grids)
    pix = tuple(pixel_sel)
    n_pix = len(pix)
    if offsets.ndim == n_pix + 2:
        offsets = offsets[view_ids]
    return offsets[(slice(None),) * (offsets.ndim - n_pix - 1) + pix]


def pixel_points(beam, frames, detector_shape, detector_spacing, *, pixels=None,
                 surface=None, surface_parameters=None, view_ids=None,
                 n_total_views=None, dtype=mx.float32):
    """World positions of the selected detector pixels, ``(views, *pixels, k)``."""
    three = beam in _THREE_D
    spacing = detector_spacing if three else (detector_spacing,)
    shape = tuple(detector_shape)
    pixels = pixels or tuple(slice(None) for _ in shape)
    center, axis_u = frames[1], frames[2]
    grids = [_grid(n, p, s, dtype) for n, p, s in zip(shape, spacing, pixels)]
    if surface is None:
        if three:
            axis_v = frames[3]
            return (center[:, None, None, :]
                    + grids[0][None, :, None, None] * axis_u[:, None, None, :]
                    + grids[1][None, None, :, None] * axis_v[:, None, None, :])
        return center[:, None, :] + grids[0][None, :, None] * axis_u[:, None, :]

    full = [_grid(n, p, slice(None), dtype) for n, p in zip(shape, spacing)]
    if three:
        ug, vg = mx.meshgrid(grids[0], grids[1], indexing="ij")
        fu, fv = mx.meshgrid(full[0], full[1], indexing="ij")
    else:
        ug, vg = grids[0], mx.zeros_like(grids[0])
        fu, fv = full[0], mx.zeros_like(full[0])
    off = _surface_offsets(surface, ug, vg, view_ids, n_total_views, pixels,
                           (fu, fv), surface_parameters).astype(dtype)
    pixel_ndim = len(shape)
    if off.ndim == pixel_ndim + 1:
        off = off[None]
    expand = (slice(None),) + (None,) * pixel_ndim
    if three:
        axis_v = frames[3]
        normal = _cross(axis_u, axis_v)
        return (center[expand] + off[..., 0:1] * axis_u[expand]
                + off[..., 1:2] * axis_v[expand] + off[..., 2:3] * normal[expand])
    normal = mx.stack([axis_u[:, 1], -axis_u[:, 0]], axis=-1)
    return center[expand] + off[..., 0:1] * axis_u[expand] + off[..., 2:3] * normal[expand]
