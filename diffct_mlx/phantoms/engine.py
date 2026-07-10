"""Analytic phantom engine: primitives you can voxelize *and* project exactly.

Build a phantom from geometric primitives (ellipsoids — which cover spheres and,
with rotation, the Shepp-Logan / FORBILD family), then either

* :meth:`Phantom.voxelize` it to a ground-truth volume, or
* :meth:`Phantom.project` it **analytically** (closed-form line integrals via
  ray/quadric intersection) — an exact sinogram with no discretization error, so
  you can validate a numerical projector against the truth.

Physical convention matches the projectors: the volume ``(D, H, W)`` is centered
at the origin with axes ``(z, y, x)`` and voxel positions
``(index - (n-1)/2) * voxel_spacing``; primitive ``center`` / ``half_axes`` are in
the same world units.

Implemented entirely through the backend ``xp`` namespace (elementwise +
slicing), so it runs on any backend the projectors run on — no backend-specific
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..backend import active as _b

xp = _b.xp

__all__ = ["Ellipsoid", "Phantom", "shepp_logan_phantom"]


def _rotation_matrix(angles_deg) -> np.ndarray:
    """ZYX intrinsic rotation matrix (degrees). Scalar = rotation about z."""
    if np.isscalar(angles_deg):
        az, ay, ax = float(angles_deg), 0.0, 0.0
    else:
        vals = list(angles_deg) + [0.0, 0.0, 0.0]
        az, ay, ax = float(vals[0]), float(vals[1]), float(vals[2])
    az, ay, ax = math.radians(az), math.radians(ay), math.radians(ax)
    cz, sz = math.cos(az), math.sin(az)
    cy, sy = math.cos(ay), math.sin(ay)
    cx, sx = math.cos(ax), math.sin(ax)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    return rz @ ry @ rx


@dataclass
class Ellipsoid:
    """A (rotated) ellipsoid of additive attenuation ``value``.

    ``center`` and ``half_axes`` are ``(x, y, z)`` in world units; ``angles_deg``
    is a scalar (rotation about z) or a ``(z, y, x)`` Euler triple.
    """

    center: tuple[float, float, float]
    half_axes: tuple[float, float, float]
    value: float = 1.0
    angles_deg: object = 0.0

    def _center(self) -> np.ndarray:
        return np.asarray(self.center, np.float64).reshape(3)

    def _matrix(self) -> np.ndarray:
        # World -> unit-sphere frame: normalize by half-axes after de-rotating.
        a = np.asarray(self.half_axes, np.float64).reshape(3)
        rot = _rotation_matrix(self.angles_deg)          # local -> world
        return np.diag(1.0 / np.maximum(a, 1e-12)) @ rot.T   # world -> normalized


@dataclass
class Phantom:
    """A collection of primitives (currently ellipsoids)."""

    primitives: list = field(default_factory=list)

    def add(self, primitive: Ellipsoid) -> "Phantom":
        self.primitives.append(primitive)
        return self

    def _prims(self):
        return [(e._matrix(), e._center(), float(e.value)) for e in self.primitives]

    # -- voxelization ------------------------------------------------------
    def voxelize(self, shape, voxel_spacing: float = 1.0):
        """Rasterize to a ground-truth volume ``(D, H, W)`` (axes z, y, x)."""
        D, H, W = (int(s) for s in shape)
        vs = float(voxel_spacing)
        # Separable world-coordinate axes, broadcast to (D, H, W).
        z = xp.reshape((xp.arange(D) - (D - 1) / 2.0) * vs, (D, 1, 1))
        y = xp.reshape((xp.arange(H) - (H - 1) / 2.0) * vs, (1, H, 1))
        x = xp.reshape((xp.arange(W) - (W - 1) / 2.0) * vs, (1, 1, W))
        volume = xp.zeros((D, H, W), dtype=_b.float32)
        for M, c, value in self._prims():
            m = M.tolist()
            dx, dy, dz = x - float(c[0]), y - float(c[1]), z - float(c[2])
            qx = m[0][0] * dx + m[0][1] * dy + m[0][2] * dz
            qy = m[1][0] * dx + m[1][1] * dy + m[1][2] * dz
            qz = m[2][0] * dx + m[2][1] * dy + m[2][2] * dz
            sq = qx * qx + qy * qy + qz * qz
            volume = volume + xp.where(sq <= 1.0, float(value), 0.0)
        return volume

    # -- analytic projection ----------------------------------------------
    def project(self, src_pos, det_center, det_u_vec, det_v_vec, *,
                detector_shape, du: float = 1.0, dv: float = 1.0):
        """Exact cone-beam sinogram ``(n_views, det_u, det_v)`` by ray/quadric intersection."""
        src = _b.to_numpy(src_pos).astype(np.float64)
        dc = _b.to_numpy(det_center).astype(np.float64)
        uu = _b.to_numpy(det_u_vec).astype(np.float64)
        vv = _b.to_numpy(det_v_vec).astype(np.float64)
        n_views = src.shape[0]
        nu, nv = int(detector_shape[0]), int(detector_shape[1])
        iu = xp.reshape((xp.arange(nu) - (nu - 1) / 2.0) * du, (nu, 1))
        iv = xp.reshape((xp.arange(nv) - (nv - 1) / 2.0) * dv, (1, nv))
        prims = self._prims()

        rows = []
        for v in range(n_views):
            sx, sy, sz = float(src[v, 0]), float(src[v, 1]), float(src[v, 2])
            # detector pixel world coordinates (nu, nv)
            px = float(dc[v, 0]) + iu * float(uu[v, 0]) + iv * float(vv[v, 0])
            py = float(dc[v, 1]) + iu * float(uu[v, 1]) + iv * float(vv[v, 1])
            pz = float(dc[v, 2]) + iu * float(uu[v, 2]) + iv * float(vv[v, 2])
            dx, dy, dz = px - sx, py - sy, pz - sz
            length = xp.sqrt(dx * dx + dy * dy + dz * dz)
            length = xp.maximum(length, 1e-12)
            ux, uy, uz = dx / length, dy / length, dz / length      # unit ray directions
            acc = xp.zeros((nu, nv), dtype=_b.float32)
            for M, c, value in prims:
                m = M.tolist()
                # o' = M (src - c)  (host scalars)
                scx, scy, scz = sx - float(c[0]), sy - float(c[1]), sz - float(c[2])
                ox = m[0][0] * scx + m[0][1] * scy + m[0][2] * scz
                oy = m[1][0] * scx + m[1][1] * scy + m[1][2] * scz
                oz = m[2][0] * scx + m[2][1] * scy + m[2][2] * scz
                # d' = M dir  (nu, nv)
                dpx = m[0][0] * ux + m[0][1] * uy + m[0][2] * uz
                dpy = m[1][0] * ux + m[1][1] * uy + m[1][2] * uz
                dpz = m[2][0] * ux + m[2][1] * uy + m[2][2] * uz
                A = dpx * dpx + dpy * dpy + dpz * dpz
                B = 2.0 * (dpx * ox + dpy * oy + dpz * oz)
                C = float(ox * ox + oy * oy + oz * oz - 1.0)
                disc = B * B - 4.0 * A * C
                sq = xp.sqrt(xp.maximum(disc, 0.0))
                A_safe = xp.maximum(A, 1e-12)
                t1 = (-B - sq) / (2.0 * A_safe)
                t2 = (-B + sq) / (2.0 * A_safe)
                lo = xp.maximum(xp.minimum(t1, t2), 0.0)
                hi = xp.minimum(xp.maximum(t1, t2), length)
                chord = xp.maximum(hi - lo, 0.0)
                acc = acc + xp.where(disc > 0.0, float(value) * chord, 0.0)
            rows.append(acc)
        return xp.stack(rows, axis=0)


# Standard 3D Shepp-Logan ellipsoids (Kak & Slaney), in normalized [-1, 1] coords:
# (a, b, c, x0, y0, z0, phi_deg, value)  with axes (x, y, z).
_SHEPP_LOGAN_3D = [
    (0.6900, 0.920, 0.900, 0.0, 0.0, 0.00, 0.0, 1.0),
    (0.6624, 0.874, 0.880, 0.0, 0.0, 0.00, 0.0, -0.8),
    (0.4100, 0.160, 0.210, -0.22, 0.0, -0.25, 108.0, -0.2),
    (0.3100, 0.110, 0.220, 0.22, 0.0, -0.25, 72.0, -0.2),
    (0.2100, 0.250, 0.500, 0.0, 0.35, -0.25, 0.0, 0.1),
    (0.0460, 0.046, 0.046, 0.0, 0.10, -0.25, 0.0, 0.1),
    (0.0460, 0.023, 0.020, -0.08, -0.65, -0.25, 0.0, 0.1),
    (0.0460, 0.023, 0.020, 0.06, -0.65, -0.25, 90.0, 0.1),
    (0.0560, 0.040, 0.100, 0.06, -0.105, 0.625, 90.0, 0.2),
    (0.0560, 0.056, 0.100, 0.0, 0.10, 0.625, 0.0, -0.2),
]


def shepp_logan_phantom(fov_radius: float = 1.0) -> Phantom:
    """The standard 3D Shepp-Logan head as an analytic :class:`Phantom`.

    Ellipsoids are scaled by ``fov_radius`` (the half-extent of the field of
    view in world units) — set it to ``(min(shape) / 2) * voxel_spacing`` so the
    voxelized phantom fills the volume and its analytic projections match the
    same physical geometry you reconstruct with.

    .. note::
       This table is the Kak–Slaney 3D variant, while
       :func:`~diffct_mlx.phantoms.shepp_logan_3d` voxelizes the slightly
       different Toft ``phantom3d`` table — the two volumes are not identical
       by design. Compare analytic projections against ``Phantom.voxelize`` of
       the *same* phantom object, not against ``shepp_logan_3d``.
    """
    r = float(fov_radius)
    prims = []
    for a, b, c, x0, y0, z0, phi, val in _SHEPP_LOGAN_3D:
        prims.append(Ellipsoid(
            center=(x0 * r, y0 * r, z0 * r),
            half_axes=(a * r, b * r, c * r),
            value=val,
            angles_deg=phi,
        ))
    return Phantom(prims)
