"""Detector and unit conventions of the projectors, and converters between them.

Since 3.0 the kernels on both backends (Torch/CUDA engine and MLX/Metal) use
the conventions of upstream diffct >= 2.0:

* **Detector pixels** — pixel ``i`` of an ``N``-pixel row sits at
  ``det_center + (i - (N-1)/2) * du * u``: ``det_center`` is the geometric
  centre of the panel (``"centered"``).
* **Line integrals** — physical, in ``mu * length`` with the length unit of
  the geometry (``"physical"``).

DiffCT-MLX 2.x kernels placed pixel ``i`` at ``(i - N/2) * du``, half a pixel
off the panel centre (``"legacy"``; 2.3's name ``"kernel"`` is accepted as an
alias), and returned line integrals in voxel units (``"voxel"``). The operator
factories take ``detector_convention=`` and ``line_integrals=`` to reproduce
2.x results; :func:`kernel_detector_center` does the detector conversion for
direct calls of the low-level projectors and the analytical FBP/FDK.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DETECTOR_CONVENTIONS",
    "LINE_INTEGRAL_UNITS",
    "kernel_detector_center",
    "line_integral_scale",
]

#: ``"centered"``: pixel i at (i - (N-1)/2)·du (kernel native since 3.0).
#: ``"legacy"``: pixel i at (i - N/2)·du (DiffCT-MLX 2.x kernels).
DETECTOR_CONVENTIONS = ("centered", "legacy")

#: ``"physical"``: line integrals in mu·length (kernel native since 3.0).
#: ``"voxel"``: line integrals in voxel units (DiffCT-MLX 2.x kernels).
LINE_INTEGRAL_UNITS = ("physical", "voxel")

_ALIASES = {"kernel": "legacy"}  # DiffCT-MLX 2.3 name for the 2.x kernel grid


def _check(value: str, allowed: tuple[str, ...], name: str) -> str:
    key = str(value).strip().lower()
    key = _ALIASES.get(key, key) if name == "detector_convention" else key
    if key not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return key


def kernel_detector_center(
    det_center: Any,
    det_u_vec: Any,
    du: float,
    det_v_vec: Any = None,
    dv: float | None = None,
    *,
    convention: str = "legacy",
) -> Any:
    """Return the ``det_center`` the kernels need for a given detector convention.

    ``det_center`` is interpreted in ``convention``. ``"centered"`` is the
    kernels' own grid and is returned unchanged. For ``"legacy"`` (the 2.x
    grid, pixel ``i`` at ``(i - N/2)·du``) the result is shifted by ``-du/2``
    along ``u`` and, for a 2D detector, ``-dv/2`` along ``v``, so the kernels
    sample the same pixel centres as DiffCT-MLX 2.x did.

    Works for numpy, torch and MLX arrays (plain arithmetic, autograd-safe).
    For a parallel beam pass ``det_origin`` as ``det_center``.
    """
    key = _check(convention, DETECTOR_CONVENTIONS, "detector_convention")
    if key == "centered":
        return det_center
    shifted = det_center - (0.5 * float(du)) * det_u_vec
    if det_v_vec is not None:
        if dv is None:
            raise ValueError("dv is required when det_v_vec is given")
        shifted = shifted - (0.5 * float(dv)) * det_v_vec
    return shifted


def line_integral_scale(line_integrals: str, voxel_spacing: float) -> float:
    """Factor that turns kernel output (physical) into ``line_integrals`` units."""
    key = _check(line_integrals, LINE_INTEGRAL_UNITS, "line_integrals")
    return 1.0 / float(voxel_spacing) if key == "voxel" else 1.0
