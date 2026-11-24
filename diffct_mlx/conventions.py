"""Detector and unit conventions of the projectors, and converters between them.

The kernels on both backends (Torch/CUDA and MLX/Metal) share the conventions
of the upstream diffct ``dev`` line they were imported from (v1.3.3.dev0):

* **Detector pixels** — pixel ``i`` of an ``N``-pixel row sits at
  ``det_center + (i - N/2) * du * u``. ``det_center`` is therefore the centre of
  pixel ``N/2``, half a pixel off the geometric centre of the panel
  (``"kernel"`` convention).
* **Line integrals** — returned in voxel units. Multiply by ``voxel_spacing``
  to get physical line integrals (``mu * mm``).

Upstream diffct >= 2.0.0, DiffCT-MLX 1.x and scanner geometry files instead
treat ``det_center`` as the geometric panel centre, with pixel ``i`` at
``(i - (N - 1)/2) * du`` (``"centered"`` convention), and return physical
line integrals. The operator factories take
``detector_convention=`` and ``line_integrals=`` to select either convention;
:func:`kernel_detector_center` does the detector conversion for direct calls
of the low-level projectors and the analytical FBP/FDK.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DETECTOR_CONVENTIONS",
    "LINE_INTEGRAL_UNITS",
    "kernel_detector_center",
    "line_integral_scale",
]

#: ``"kernel"``: pixel i at (i - N/2)·du (kernel native, diffct dev line).
#: ``"centered"``: pixel i at (i - (N-1)/2)·du (det_center = panel centre).
DETECTOR_CONVENTIONS = ("kernel", "centered")

#: ``"voxel"``: line integrals in voxel units (kernel native).
#: ``"physical"``: line integrals in the physical unit of ``voxel_spacing``.
LINE_INTEGRAL_UNITS = ("voxel", "physical")


def _check(value: str, allowed: tuple[str, ...], name: str) -> str:
    key = str(value).strip().lower()
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
    convention: str = "centered",
) -> Any:
    """Return the ``det_center`` the kernels need for a given detector convention.

    ``det_center`` is interpreted in ``convention``. For ``"centered"`` (the
    geometric panel centre, pixel ``i`` at ``(i - (N-1)/2)·du``) the result is
    shifted by ``+du/2`` along ``u`` and, for a 2D detector, ``+dv/2`` along
    ``v``, so the kernels' ``(i - N/2)·du`` grid lands on the same pixel
    centres. For ``"kernel"`` it is returned unchanged.

    Works for numpy, torch and MLX arrays (plain arithmetic, autograd-safe).
    For a parallel beam pass ``det_origin`` as ``det_center``.
    """
    key = _check(convention, DETECTOR_CONVENTIONS, "detector_convention")
    if key == "kernel":
        return det_center
    shifted = det_center + (0.5 * float(du)) * det_u_vec
    if det_v_vec is not None:
        if dv is None:
            raise ValueError("dv is required when det_v_vec is given")
        shifted = shifted + (0.5 * float(dv)) * det_v_vec
    return shifted


def line_integral_scale(line_integrals: str, voxel_spacing: float) -> float:
    """Factor that turns kernel output (voxel units) into ``line_integrals`` units."""
    key = _check(line_integrals, LINE_INTEGRAL_UNITS, "line_integrals")
    return float(voxel_spacing) if key == "physical" else 1.0
