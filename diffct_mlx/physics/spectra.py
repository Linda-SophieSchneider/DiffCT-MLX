"""Physically-based example X-ray spectra + material attenuation (preset library).

A curated, ready-to-use set of tube spectra (tungsten anode, industrial /
micro-CT range 40-225 kVp) and material attenuation curves ``mu(E)``. The data
were generated **offline** with the TASMICS-validated LLNL XrayPhysics model and
embedded as a static ``.npz``; XrayPhysics is **not** imported at runtime.

Typical use::

    from diffct_mlx.physics import spectra
    spec = spectra.tube_spectrum(160, filtration=[("Cu", 1.0)])   # 160 kVp + 1 mm Cu
    mu   = spectra.material_attenuation("bone", spec.energies)     # for beam hardening
    sino = dct.simulate_scan(phantom, A, spectrum=spec, material_attenuation=mu)

    spec = spectra.preset("industrial_225kVp_Sn1mm")               # named preset

All spectra share one energy grid (keV); ``material_attenuation`` returns
``1/cm`` on that grid (or resampled to any energies you pass).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from .simulate import Spectrum

__all__ = [
    "available_kvps",
    "available_materials",
    "tube_spectrum",
    "material_attenuation",
    "energy_grid",
    "PRESETS",
    "list_presets",
    "preset",
]


@lru_cache(maxsize=1)
def _data() -> dict:
    with files("diffct_mlx.physics").joinpath("data/spectra.npz").open("rb") as handle:
        npz = np.load(handle)
        return {key: npz[key] for key in npz.files}


def energy_grid() -> np.ndarray:
    """The common energy grid (keV) shared by all spectra and attenuation tables."""
    return _data()["energies"].astype(np.float64)


def available_kvps() -> list[float]:
    """Tube voltages (kVp) with an embedded base spectrum (W anode)."""
    return [float(v) for v in _data()["kvps"]]


def available_materials() -> list[str]:
    """Materials with an embedded attenuation curve ``mu(E)``."""
    return [str(v) for v in _data()["material_names"]]


def _base_weights(kvp: float) -> np.ndarray:
    data = _data()
    kvps = data["kvps"].astype(np.float64)
    spectra = data["spectra"].astype(np.float64)
    energies = data["energies"].astype(np.float64)
    kvp = float(kvp)
    if kvp < kvps.min() - 1e-6 or kvp > kvps.max() + 1e-6:
        raise ValueError(
            f"kVp {kvp} out of embedded range [{kvps.min():.0f}, {kvps.max():.0f}]; "
            f"available: {[int(v) for v in kvps]}."
        )
    hit = np.where(np.abs(kvps - kvp) < 1e-6)[0]
    if hit.size:
        weights = spectra[int(hit[0])].copy()
    else:  # linear interpolation between the two bracketing kVp spectra
        hi = int(np.searchsorted(kvps, kvp))
        lo = hi - 1
        t = (kvp - kvps[lo]) / (kvps[hi] - kvps[lo])
        weights = (1.0 - t) * spectra[lo] + t * spectra[hi]
    weights[energies > kvp] = 0.0
    total = weights.sum()
    return weights / (total if total > 0 else 1.0)


def material_attenuation(material: str, energies=None) -> np.ndarray:
    """Linear attenuation ``mu(E)`` [1/cm] for ``material`` (see :func:`available_materials`).

    Returned on the native energy grid, or resampled to ``energies`` if given —
    pass ``spectrum.energies`` to align it for :func:`~diffct_mlx.simulate_scan`
    / :func:`~diffct_mlx.physics.apply_beam_hardening`.
    """
    data = _data()
    names = [str(v) for v in data["material_names"]]
    if material not in names:
        raise ValueError(f"Unknown material {material!r}. Available: {names}.")
    mu = data["material_mu"][names.index(material)].astype(np.float64)
    if energies is None:
        return mu
    return np.interp(np.asarray(energies, dtype=np.float64), data["energies"].astype(np.float64), mu)


def tube_spectrum(kvp: float, filtration=None, anode: str = "W") -> Spectrum:
    """Build a tungsten-anode tube :class:`Spectrum` at ``kvp``, optionally filtered.

    ``filtration`` is a list of ``(material, thickness_mm)`` (or a dict) applied as
    ``exp(-mu(E) * t)`` using the embedded attenuation tables — e.g.
    ``[("Cu", 1.0), ("Sn", 0.5)]``. Only the tungsten anode is embedded.
    """
    if str(anode).strip().upper() not in ("W", "TUNGSTEN"):
        raise ValueError("Only the tungsten (W) anode is embedded; regenerate the library for others.")
    energies = _data()["energies"].astype(np.float64)
    weights = _base_weights(kvp)
    if filtration:
        items = filtration.items() if isinstance(filtration, dict) else filtration
        for material, thickness_mm in items:
            mu = material_attenuation(material)          # 1/cm on the native grid
            weights = weights * np.exp(-mu * (float(thickness_mm) / 10.0))  # mm -> cm
        total = weights.sum()
        weights = weights / (total if total > 0 else 1.0)
    return Spectrum(energies, weights)


#: Named ready-to-use presets: name -> (kvp, filtration).
PRESETS: dict[str, tuple[float, list | None]] = {
    "microCT_40kVp": (40, None),
    "microCT_60kVp": (60, None),
    "microCT_90kVp_Al1mm": (90, [("Al", 1.0)]),
    "microCT_100kVp_Al2mm": (100, [("Al", 2.0)]),
    "industrial_120kVp_Cu0.5mm": (120, [("Cu", 0.5)]),
    "industrial_160kVp_Cu1mm": (160, [("Cu", 1.0)]),
    "industrial_200kVp_Sn0.5mm": (200, [("Sn", 0.5)]),
    "industrial_225kVp_Sn1mm": (225, [("Sn", 1.0)]),
}


def list_presets() -> list[str]:
    """Names of the ready-to-use spectrum presets."""
    return sorted(PRESETS)


def preset(name: str) -> Spectrum:
    """Return a named preset :class:`Spectrum` (see :func:`list_presets`)."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Available: {list_presets()}.")
    kvp, filtration = PRESETS[name]
    return tube_spectrum(kvp, filtration)
