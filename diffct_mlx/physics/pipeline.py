"""Composable preprocessing pipeline for projection data.

A pipeline is an ordered list of steps, each a callable ``data -> data``. Steps
may be the :class:`PreprocessingStep` wrappers below (which hold parameters and
call the GPU-native corrections), plain functions, or lambdas — so users mix
built-ins with their own transforms freely::

    from diffct_mlx.physics import (
        PreprocessingPipeline, FlatField, RingRemoval, BeamHardening)

    prep = PreprocessingPipeline([
        FlatField(flat=flat, dark=dark),        # -> attenuation line integrals
        RingRemoval(radius=8),
        BeamHardening(coefficients=coeffs),
    ])
    sinogram = prep(raw_projections)
"""

from __future__ import annotations

from typing import Callable, Sequence

from . import corrections as _c

__all__ = [
    "PreprocessingStep",
    "PreprocessingPipeline",
    "FlatField",
    "RingRemoval",
    "BadPixel",
    "BeamHardening",
    "Deblur",
    "Scatter",
    "MetalArtifactReduction",
]


class PreprocessingStep:
    """Base class: a parameterized, callable projection-domain transform."""

    def __call__(self, data):  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PreprocessingPipeline:
    """Apply a sequence of steps (``PreprocessingStep`` or plain callables)."""

    def __init__(self, steps: Sequence[Callable] | None = None):
        self.steps: list[Callable] = list(steps) if steps else []

    def append(self, step: Callable) -> "PreprocessingPipeline":
        self.steps.append(step)
        return self

    def __add__(self, step: Callable) -> "PreprocessingPipeline":
        return PreprocessingPipeline(self.steps + [step])

    def __call__(self, data):
        for step in self.steps:
            data = step(data)
        return data

    def __repr__(self) -> str:
        return "PreprocessingPipeline([" + ", ".join(repr(s) for s in self.steps) + "])"


class FlatField(PreprocessingStep):
    """Flat/dark-field gain correction (+ optional negative-log)."""

    def __init__(self, flat, dark=None, *, clip_min: float = 0.0, log: bool = True):
        self.flat, self.dark, self.clip_min, self.log = flat, dark, clip_min, log

    def __call__(self, raw):
        return _c.flat_field(raw, self.flat, self.dark, clip_min=self.clip_min, log=self.log)


class RingRemoval(PreprocessingStep):
    """Detector mean-curve ring/stripe removal."""

    def __init__(self, radius: int = 8, strength: float = 1.0):
        self.radius, self.strength = int(radius), float(strength)

    def __call__(self, sino):
        return _c.ring_removal(sino, radius=self.radius, strength=self.strength)


class BadPixel(PreprocessingStep):
    """Median-based bad-pixel / outlier replacement."""

    def __init__(self, threshold: float = 4.0, radius: int = 1):
        self.threshold, self.radius = float(threshold), int(radius)

    def __call__(self, sino):
        return _c.bad_pixel_correction(sino, threshold=self.threshold, radius=self.radius)


class BeamHardening(PreprocessingStep):
    """Polynomial single-material beam-hardening correction."""

    def __init__(self, coefficients):
        self.coefficients = list(coefficients)

    def __call__(self, sino):
        return _c.beam_hardening_polynomial(sino, self.coefficients)


class Deblur(PreprocessingStep):
    """Separable Gaussian detector deblur (Wiener)."""

    def __init__(self, sigma: float = 1.0, reg: float = 1e-2):
        self.sigma, self.reg = float(sigma), float(reg)

    def __call__(self, sino):
        return _c.detector_deblur(sino, sigma=self.sigma, reg=self.reg)


class Scatter(PreprocessingStep):
    """First-order convolutional scatter subtraction (intensity domain)."""

    def __init__(self, fraction: float = 0.05, radius: int = 12):
        self.fraction, self.radius = float(fraction), int(radius)

    def __call__(self, intensity):
        return _c.scatter_correction(intensity, fraction=self.fraction, radius=self.radius)


class MetalArtifactReduction(PreprocessingStep):
    """Metal-trace sinogram inpainting (MAR). ``mask`` is the metal trace."""

    def __init__(self, mask, iterations: int = 40, radius: int = 1):
        self.mask, self.iterations, self.radius = mask, int(iterations), int(radius)

    def __call__(self, sino):
        return _c.mar_inpaint(sino, self.mask, iterations=self.iterations, radius=self.radius)
