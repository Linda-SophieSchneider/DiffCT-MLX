"""Offline generator for the embedded spectrum + attenuation library.

Regenerates ``diffct_mlx/physics/data/spectra.npz`` from the TASMICS-validated
LLNL **XrayPhysics** model. This is an OFFLINE, one-time step — the shipped
package loads the ``.npz`` and never imports ``xrayphysics`` at runtime, so
XrayPhysics stays a pure reference (not a dependency).

Prerequisites (build XrayPhysics once):
    pip install cmake                       # if cmake is missing
    cd /path/to/XrayPhysics && rm -rf build && mkdir build && cd build
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && cmake --build . -j8
    # -> build/lib/libxrayphysics.so

Run:
    XRAYPHYSICS_SRC=/path/to/XrayPhysics/src python tools/generate_spectra.py

The library covers the industrial / micro-CT range (40-225 kVp, W anode) plus
material attenuation curves for common filters and objects. Adjust ``KVPS`` /
``MATERIALS`` below and re-run to extend it.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Locate the XrayPhysics Python wrapper (set XRAYPHYSICS_SRC, else a sensible default).
_src = os.environ.get("XRAYPHYSICS_SRC")
if _src:
    sys.path.insert(0, _src)
try:
    from xrayphysics import xrayPhysics
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import xrayphysics. Build XrayPhysics (see this file's docstring) "
        "and set XRAYPHYSICS_SRC to its 'src' directory."
    ) from exc

OUT = Path(__file__).resolve().parent.parent / "diffct_mlx" / "physics" / "data" / "spectra.npz"

ENERGIES = np.arange(5.0, 225.0 + 1e-6, 1.0)          # keV, common grid
KVPS = [40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225]
TAKE_OFF_DEG = 11.0
ANODE_Z = 74                                          # tungsten
# (name, formula-or-library-name, density g/cm^3 or None for the built-in value)
MATERIALS = [
    ("water", "water", None), ("PMMA", "C5H8O2", 1.18), ("adipose", "adipose_tissue", None),
    ("bone", "bone_cortical", None), ("graphite", "C", None), ("Al", "Al", None),
    ("Ti", "Ti", None), ("Fe", "Fe", None), ("Cu", "Cu", None), ("Sn", "Sn", None),
    ("W", "W", None), ("Pb", "Pb", None),
]


def main():
    xp = xrayPhysics()
    if xp.libxrayphysics is None:
        raise SystemExit("libxrayphysics failed to load (build it first).")

    spectra = np.zeros((len(KVPS), ENERGIES.size), dtype=np.float64)
    for i, kv in enumerate(KVPS):
        e_k, s_k = xp.simulateSpectra(float(kv), TAKE_OFF_DEG, ANODE_Z)
        s = np.interp(ENERGIES, np.asarray(e_k).ravel(), np.asarray(s_k).ravel(), left=0.0, right=0.0)
        s[ENERGIES > kv] = 0.0
        spectra[i] = s / (s.sum() or 1.0)

    names, mus = [], []
    for name, formula, density in MATERIALS:
        mu = np.array([xp.mu(formula, float(e)) if density is None else xp.mu(formula, float(e), density)
                       for e in ENERGIES], dtype=np.float64)
        if np.all(np.isfinite(mu)) and np.any(mu > 0):
            names.append(name)
            mus.append(mu)
        else:
            print(f"  skip {name} ({formula}): non-physical mu")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        energies=ENERGIES.astype(np.float32),
        kvps=np.array(KVPS, dtype=np.float32),
        spectra=spectra.astype(np.float32),
        material_names=np.array(names),
        material_mu=np.array(mus, dtype=np.float32),
        anode_Z=np.int64(ANODE_Z),
        take_off_deg=np.float32(TAKE_OFF_DEG),
    )
    print(f"wrote {OUT}  (spectra {spectra.shape}, materials {len(names)})")


if __name__ == "__main__":
    main()
