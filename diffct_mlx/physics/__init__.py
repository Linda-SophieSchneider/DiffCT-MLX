"""Physics & preprocessing: GPU-native projection-domain corrections.

The corrections (flat-field, ring removal, bad-pixel, beam-hardening, deblur,
scatter, MAR) and the :class:`PreprocessingPipeline` run entirely on the active
backend and have **no external dependency**.

Note: the LLNL *XrayPhysics* package is kept in the tree only as an offline
**reference** for the underlying spectral physics (tube spectra, cross-sections,
beam-hardening / dual-energy LUTs). It is intentionally **not** imported or
integrated here — beam-hardening coefficients are supplied by the caller (e.g.
from an empirical water calibration).
"""

from .corrections import (
    bad_pixel_correction,
    beam_hardening_polynomial,
    detector_deblur,
    flat_field,
    mar_inpaint,
    ring_removal,
    scatter_correction,
)
from .pipeline import (
    BadPixel,
    BeamHardening,
    Deblur,
    FlatField,
    MetalArtifactReduction,
    PreprocessingPipeline,
    PreprocessingStep,
    RingRemoval,
    Scatter,
)
from .simulate import (
    Spectrum,
    add_poisson_noise,
    add_rings,
    add_scatter,
    apply_beam_hardening,
    apply_detector_blur,
    flat_field_forward,
    simulate_scan,
)
from . import spectra
from .spectra import material_attenuation, preset, tube_spectrum

__all__ = [
    # corrections
    "flat_field",
    "ring_removal",
    "bad_pixel_correction",
    "beam_hardening_polynomial",
    "detector_deblur",
    "scatter_correction",
    "mar_inpaint",
    # pipeline
    "PreprocessingStep",
    "PreprocessingPipeline",
    "FlatField",
    "RingRemoval",
    "BadPixel",
    "BeamHardening",
    "Deblur",
    "Scatter",
    "MetalArtifactReduction",
    # forward simulation
    "Spectrum",
    "apply_beam_hardening",
    "flat_field_forward",
    "add_poisson_noise",
    "apply_detector_blur",
    "add_scatter",
    "add_rings",
    "simulate_scan",
    # example spectra + attenuation library (offline-generated, embedded)
    "spectra",
    "tube_spectrum",
    "material_attenuation",
    "preset",
]
