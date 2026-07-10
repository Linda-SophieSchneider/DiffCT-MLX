"""Forward acquisition simulation: phantom -> realistic projection data.

The symmetric counterpart to :mod:`diffct_mlx.physics.corrections`. Where those
*undo* acquisition effects, these *apply* them, so you can generate realistic CT
data from a phantom:

* polychromatic **beam hardening** driven by a **user-supplied spectrum** (a real
  CT tube spectrum you measured, or a synthetic one) and a material attenuation
  curve you provide,
* **flat-field forward** (Beer-Lambert to intensity/counts),
* **Poisson** photon noise,
* forward **detector blur**, **scatter**, and **ring** (detector-gain) effects,
* a one-call :func:`simulate_scan` that chains them on top of a projection
  operator.

Everything runs on the active backend (GPU-resident). No external physics
library is used: measured/tabulated spectra and material attenuation are inputs,
so this stays self-contained (LLNL XrayPhysics is only a reference for computing
such curves offline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import NAME as _BACKEND
from ..backend import active as _b
from ..filters import box_filter

xp = _b.xp

__all__ = [
    "Spectrum",
    "apply_beam_hardening",
    "flat_field_forward",
    "add_poisson_noise",
    "apply_detector_blur",
    "add_scatter",
    "add_rings",
    "simulate_scan",
]

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Spectrum (accepts measured data)
# ---------------------------------------------------------------------------
@dataclass
class Spectrum:
    """An X-ray spectrum: photon weight per energy bin.

    ``energies`` are in keV, ``fluence`` the (relative) photon count per bin.
    Absolute scaling is irrelevant — only the shape matters — so it is always
    used normalized. Build one straight from a **measured** tube spectrum::

        spec = Spectrum(energies_keV, measured_counts)     # your CT machine
        spec = Spectrum.from_file("spectrum.csv")          # 2-column E,I
        spec = Spectrum.kramers(kv_peak=120)               # synthetic fallback
        spec = Spectrum.monochromatic(70.0)                # ideal / no hardening
    """

    energies: np.ndarray
    fluence: np.ndarray

    def __post_init__(self):
        self.energies = np.asarray(self.energies, dtype=np.float64).ravel()
        self.fluence = np.asarray(self.fluence, dtype=np.float64).ravel()
        if self.energies.shape != self.fluence.shape:
            raise ValueError(
                f"energies and fluence must have equal length, got "
                f"{self.energies.shape} vs {self.fluence.shape}."
            )
        if self.energies.size == 0:
            raise ValueError("Spectrum must have at least one energy bin.")

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_arrays(cls, energies, fluence) -> "Spectrum":
        return cls(np.asarray(energies), np.asarray(fluence))

    @classmethod
    def from_file(cls, path, delimiter=None) -> "Spectrum":
        """Load a 2-column (energy_keV, intensity) text/CSV file."""
        data = np.loadtxt(path, delimiter=delimiter)
        data = np.atleast_2d(data)
        return cls(data[:, 0], data[:, 1])

    @classmethod
    def monochromatic(cls, energy_keV: float) -> "Spectrum":
        return cls(np.array([float(energy_keV)]), np.array([1.0]))

    @classmethod
    def kramers(cls, kv_peak: float, num_bins: int = 120, e_min: float = 5.0,
                filter_attenuation=None) -> "Spectrum":
        """Synthetic bremsstrahlung tube spectrum (Kramers' law).

        ``N(E) ~ (kVp/E - 1)`` for ``E <= kVp``. Optional ``filter_attenuation``
        is a per-bin ``mu*thickness`` applied as ``exp(-mu*t)`` (added filtration).
        A convenience for when no measured spectrum is available.
        """
        energies = np.linspace(float(e_min), float(kv_peak), int(num_bins))
        fluence = np.clip(kv_peak / np.maximum(energies, _EPS) - 1.0, 0.0, None)
        if filter_attenuation is not None:
            fluence = fluence * np.exp(-np.asarray(filter_attenuation, dtype=np.float64).ravel())
        return cls(energies, fluence)

    # -- queries -----------------------------------------------------------
    def normalized(self) -> np.ndarray:
        total = float(np.sum(self.fluence))
        return self.fluence / (total if total > 0 else 1.0)

    def mean_energy(self) -> float:
        w = self.normalized()
        return float(np.sum(self.energies * w))

    def is_monochromatic(self) -> bool:
        return self.energies.size == 1


# ---------------------------------------------------------------------------
# Beam hardening (polychromatic forward model)
# ---------------------------------------------------------------------------
def apply_beam_hardening(p_reference, spectrum: Spectrum, material_attenuation,
                         reference_energy: float | None = None):
    r"""Harden a reference-energy line integral through a polychromatic spectrum.

    The phantom is taken to hold linear attenuation at ``reference_energy`` (so
    ``p_reference = A @ x`` is the monochromatic line integral). The measured,
    beam-hardened line integral is

    .. math:: p = -\log \sum_k S(E_k)\, \exp(-r_k\, p_\text{ref}),\quad
              r_k = \mu(E_k)/\mu(E_\text{ref})

    with ``S`` the normalized spectrum. ``material_attenuation`` is the object
    material's attenuation ``mu(E)`` sampled at ``spectrum.energies`` (any units —
    only the ratio matters); you supply it (measured or tabulated). By default
    ``reference_energy`` is the beam's *effective* energy (``mu_ref`` = the
    spectrum-weighted mean attenuation), so the low-attenuation limit is unbiased;
    pass an explicit energy to treat the phantom as ``mu`` at that energy instead.
    """
    if spectrum.is_monochromatic():
        return xp.array(p_reference, dtype=_b.float32)

    weights = spectrum.normalized()
    mu = np.asarray(material_attenuation, dtype=np.float64).ravel()
    if mu.shape != spectrum.energies.shape:
        raise ValueError(
            "material_attenuation must be sampled at spectrum.energies: "
            f"got {mu.shape}, expected {spectrum.energies.shape}."
        )
    if reference_energy is None:
        # Effective energy: mu_ref = spectrum-weighted mean attenuation, so the
        # low-attenuation limit is unbiased (initial slope 1) and beam hardening
        # is a pure downward nonlinearity. Pass reference_energy to interpret the
        # phantom as mu at a specific energy instead.
        mu_ref = float(np.sum(weights * mu))
    else:
        mu_ref = float(np.interp(float(reference_energy), spectrum.energies, mu))
    ratios = mu / (mu_ref if mu_ref != 0 else 1.0)

    p_ref = xp.array(p_reference, dtype=_b.float32)
    intensity = xp.zeros_like(p_ref)
    for weight, ratio in zip(weights, ratios):
        if weight <= 0.0:
            continue
        intensity = intensity + float(weight) * xp.exp(-float(ratio) * p_ref)
    return -xp.log(xp.maximum(intensity, _EPS))


# ---------------------------------------------------------------------------
# Intensity / counts, noise, detector effects (forward)
# ---------------------------------------------------------------------------
def flat_field_forward(p, I0: float = 1.0e5, dark: float = 0.0):
    """Beer-Lambert forward: line integral ``p`` -> detected intensity ``I0 e^{-p} + dark``."""
    p = xp.array(p, dtype=_b.float32)
    return float(I0) * xp.exp(-p) + float(dark)


def add_poisson_noise(counts, seed: int | None = None):
    """Sample Poisson photon noise for expected ``counts`` (elementwise, >= 0)."""
    if _BACKEND == "torch":
        import torch

        t = counts if isinstance(counts, torch.Tensor) else torch.as_tensor(_b.to_numpy(counts))
        t = torch.clamp(t.to(dtype=torch.float32), min=0.0)
        if seed is not None:
            gen = torch.Generator(device=t.device)
            gen.manual_seed(int(seed))
            return torch.poisson(t, generator=gen)
        return torch.poisson(t)
    arr = np.maximum(_b.to_numpy(counts), 0.0)
    rng = np.random.default_rng(seed)
    return _b.as_array(rng.poisson(arr).astype(np.float32))


def _gaussian_transfer(n, sigma):
    freqs = xp.fft.fftfreq(n)
    return xp.exp(-2.0 * (np.pi ** 2) * (float(sigma) ** 2) * (freqs * freqs))


def apply_detector_blur(sinogram, sigma: float = 1.0):
    """Forward separable Gaussian detector blur (inverse of :func:`detector_deblur`)."""
    out = xp.array(sinogram, dtype=_b.float32)
    for axis in range(1, out.ndim):
        n = int(out.shape[axis])
        h = _gaussian_transfer(n, sigma)
        shape = [1] * out.ndim
        shape[axis] = n
        h = xp.reshape(h, shape)
        out = xp.real(xp.fft.ifft(xp.fft.fft(out, axis=axis) * h, axis=axis))
    return out


def add_scatter(intensity, fraction: float = 0.05, radius: int = 12):
    """Add a broad low-pass scatter floor to the intensity (inverse of scatter_correction).

    The blur acts per view in the detector plane only (views must not mix).
    """
    intensity = xp.array(intensity, dtype=_b.float32)
    detector_axes = tuple(range(1, intensity.ndim))
    return intensity + float(fraction) * box_filter(intensity, radius, axes=detector_axes)


def add_rings(intensity, strength: float = 0.01, seed: int = 0, gain=None):
    """Apply a per-detector multiplicative gain (ring source) to the intensity.

    ``gain`` (detector-shaped) is used if given; otherwise a random gain
    ``1 + strength * N(0,1)`` per detector element is generated with ``seed``.
    """
    intensity = xp.array(intensity, dtype=_b.float32)
    det_shape = tuple(int(s) for s in intensity.shape[1:])
    if gain is None:
        rng = np.random.default_rng(seed)
        gain = (1.0 + float(strength) * rng.standard_normal(det_shape)).astype(np.float32)
    gain = xp.array(gain, dtype=_b.float32)
    return intensity * gain[None]


# ---------------------------------------------------------------------------
# One-call scan simulation
# ---------------------------------------------------------------------------
def simulate_scan(
    phantom,
    operator,
    *,
    spectrum: Spectrum | None = None,
    material_attenuation=None,
    reference_energy: float | None = None,
    I0: float = 1.0e5,
    dark: float = 0.0,
    detector_blur_sigma: float = 0.0,
    scatter_fraction: float = 0.0,
    scatter_radius: int = 12,
    ring_strength: float = 0.0,
    poisson: bool = True,
    seed: int = 0,
    return_counts: bool = False,
):
    """Simulate a realistic scan: phantom -> (beam hardening) -> counts -> sinogram.

    ``operator`` is a projection :class:`~diffct_mlx.operators.LinearOperator`
    (from ``make_*_operator``). With ``spectrum`` **and** ``material_attenuation``
    given, polychromatic beam hardening is applied; otherwise the projection is
    monochromatic. Effects are applied in the intensity domain (blur -> scatter ->
    ring gain), then Poisson noise, then the negative-log brings it back to an
    attenuation sinogram ready for reconstruction.

    Returns the attenuation sinogram (default) or the raw photon counts
    (``return_counts=True``).
    """
    if float(I0) <= 0.0:
        raise ValueError(f"I0 (incident photon count) must be positive, got {I0!r}.")
    x = xp.array(phantom, dtype=_b.float32)
    p = operator @ x                                    # reference-energy line integrals
    if spectrum is not None and material_attenuation is not None:
        p = apply_beam_hardening(p, spectrum, material_attenuation, reference_energy)

    intensity = flat_field_forward(p, I0=I0, dark=0.0)  # add dark after gain, below
    if detector_blur_sigma > 0.0:
        intensity = apply_detector_blur(intensity, detector_blur_sigma)
    if scatter_fraction > 0.0:
        intensity = add_scatter(intensity, scatter_fraction, scatter_radius)
    if ring_strength > 0.0:
        intensity = add_rings(intensity, ring_strength, seed)
    intensity = intensity + float(dark)

    counts = add_poisson_noise(intensity, seed=seed) if poisson else intensity
    if return_counts:
        return counts

    transmission = xp.maximum(counts - float(dark), _EPS) / float(I0)
    return -xp.log(transmission)
