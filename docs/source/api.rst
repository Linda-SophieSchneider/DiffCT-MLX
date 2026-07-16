API Reference
=============

DiffCT-MLX ships two importable packages. ``diffct_mlx`` is the unified,
auto-backend API most users should import; ``diffct`` is the low-level
Torch/numba-CUDA engine it wraps on NVIDIA systems.

The unified package: ``diffct_mlx``
-----------------------------------

.. rubric:: Backend

``diffct_mlx.backend`` selects the compute engine at import time (``torch`` or
``mlx``; override with ``DIFFCT_BACKEND``). ``dct.backend`` is the chosen
name. The backend-neutral array namespace ``diffct_mlx.backend.xp`` mirrors
``mlx.core`` on both engines and is what all shared reconstruction code is
written against.

.. rubric:: Projectors (``diffct_mlx.projectors``, re-exported at top level)

Functional operators with identical signatures on both backends:
``parallel_forward`` / ``parallel_backward``, ``fan_forward`` /
``fan_backward``, ``cone_forward`` / ``cone_backward``, and the
separable-footprint variants ``*_forward_footprint`` / ``*_backward_footprint``
(``cone_backward_footprint`` additionally accepts ``indices=`` for sparse
region-of-interest evaluation). All flow gradients.

.. rubric:: Geometry (``diffct_mlx.geometry``, re-exported at top level)

Trajectory generators: ``circular_trajectory_3d``, ``spiral_trajectory_3d``
(helical), ``sinusoidal_trajectory_3d``, ``saddle_trajectory_3d``,
``random_trajectory_3d``, ``custom_trajectory_3d``,
``laminography_trajectory_3d``, plus the 2D fan/parallel counterparts and
``load_arbitrary_cone_geometry_from_json`` for measured-scan geometry files.

.. rubric:: Operator algebra (``diffct_mlx.operators``)

``LinearOperator`` with ``A @ x``, ``A.T``, composition, sums and scaling;
``ProjectionOperator`` with ``.subset(views)`` for ordered-subset methods;
builders ``make_parallel_2d_operator``, ``make_fan_2d_operator``,
``make_cone_3d_operator`` (choose ``projector_mode="siddon"`` or
``"footprint"``). ``IdentityOperator``, ``DiagonalOperator``,
``FunctionOperator`` compose into pipelines; everything stays differentiable.

.. rubric:: Functionals & denoisers (``diffct_mlx.functionals``, ``diffct_mlx.filters``)

Objectives/regularizers/constraints: ``SquaredL2``, ``Tikhonov``, ``L1Norm``,
``LpNorm``, ``Huber``, ``TotalVariation``, ``AdaptiveWeightedTV``,
``NonNegativity``, ``Box``, ``soft_threshold``; edge-preserving denoisers
``Bilateral``, ``Guided``, ``Median``, ``HistogramSparsity``, ``Azimuthal``,
``DictionarySparsity`` and ``RegularizerSequence`` for Plug-and-Play priors.

.. rubric:: Reconstruction (``diffct_mlx.reconstruction_algorithms``, re-exported at top level)

Analytic: ``reconstruct_fbp`` / ``reconstruct_fdk`` (+ ``FBPParameters`` /
``FDKParameters``) and ``reconstruct_case_fdk`` for a case's quantitative
physical FDK path. Iterative: ``reconstruct_sart`` / ``reconstruct_sirt``
(+ normalized SART), ``reconstruct_tv_pocs`` / ``reconstruct_asd_pocs`` /
``reconstruct_awtv_pocs``, ``reconstruct_dart``. Case builders bundling data +
matched operators: ``build_parallel_2d_case``, ``build_fan_2d_case``,
``build_cone_3d_case``, ``build_measured_cone_3d_case``,
``build_npy_cone_3d_case``. Solver registry: ``reconstruct(name, A, b, ...)``,
``register_algorithm``, ``list_algorithms`` with ``landweber``, ``cgls``,
``pcg``, ``ls`` / ``wls`` / ``rls`` / ``rwls``, ``dls`` / ``rdls``, ``mlem`` /
``osem``, ``mltr``.

.. rubric:: Quantitative FDK and units

Cone-beam cases expose ``fdk_weight``, ``fdk_filter`` and
``fdk_back_project`` when the active backend provides the physical FDK gather
path. ``reconstruct_case_fdk(case)`` combines cosine weighting, trapezoidal
per-view angular weights, a zero-padded physical ramp and the voxel-driven
``(SID/U)^2`` backprojector; its result is attenuation in ``1/mm``.

Projectors themselves integrate in voxel units, so physical line integrals are
``(A @ x) * A.voxel_spacing``. ``simulate_scan`` performs this conversion
before transmission and beam-hardening physics. Iterative reconstruction of a
physical measured sinogram returns attenuation per voxel; divide by
``case.voxel_spacing`` to obtain ``1/mm``.

.. rubric:: Physics & simulation (``diffct_mlx.physics``)

Corrections: ``flat_field``, ``ring_removal``, ``bad_pixel_correction``,
``beam_hardening_polynomial``, ``detector_deblur``, ``scatter_correction``,
``mar_inpaint`` and the composable ``PreprocessingPipeline``. Simulation:
``Spectrum``, ``simulate_scan``, ``apply_beam_hardening``,
``add_poisson_noise``; embedded spectrum library ``physics.spectra``
(``tube_spectrum``, ``material_attenuation``, ``preset``, 40–225 kVp).

.. rubric:: Sinogram-domain tools (``diffct_mlx.rebinning``)

``apply_parker_weighting`` (short scans), ``apply_offset_weighting``
(extended FOV), ``extend_truncation``, ``fan_to_parallel``,
``curved_to_flat`` / ``flat_to_curved``.

.. rubric:: Phantoms & calibration (``diffct_mlx.phantoms``, ``diffct_mlx.calibration``)

``shepp_logan_2d`` / ``shepp_logan_3d`` voxel phantoms; the analytic engine
``Ellipsoid`` / ``Phantom`` / ``shepp_logan_phantom`` with exact closed-form
cone projections for ground truth. Self-calibration:
``estimate_center_of_rotation``, ``refine_center_by_sharpness``,
``apply_center_offset``.

.. rubric:: Out-of-core & multi-GPU (``diffct_mlx.orchestration``; CUDA only)

``chunked_cone_forward`` / ``chunked_cone_backward`` / ``chunked_cone_fdk``,
``chunked_sirt``, ``chunked_os_sart``, ``mgpu_cone_forward`` /
``mgpu_cone_backproject`` / ``mgpu_sirt``, storage helpers ``open_memmap`` /
``open_zarr`` / ``set_out_of_core_dir`` / ``set_out_of_core_backend``, and the
shadow-geometry primitives ``row_range_for_zslab`` / ``slice_range_for_vband``.

The low-level engine: ``diffct``
--------------------------------

The vendored Torch/numba-CUDA engine is organised into focused modules that can be combined to build differentiable CT pipelines:

- ``diffct.projectors`` – PyTorch ``autograd.Function`` implementations for forward and backward projectors
- ``diffct.geometry`` – helpers to generate detector/source trajectories for 2D and 3D scans
- ``diffct.analytical`` – analytical FBP / FDK building blocks (ramp filter, cosine weights, Parker weights, voxel-driven backprojection wrappers)
- ``diffct.kernels`` – CUDA kernel primitives (Siddon + separable-footprint forward / adjoint, FBP/FDK gather) for advanced users
- ``diffct.footprint`` – separable-footprint autograd Functions, incl. the sparse cone backprojection
- ``diffct.utils`` – CUDA device management and tensor bridge utilities
- ``diffct.constants`` – low-level configuration values for advanced tuning
- ``diffct.differentiable`` – deprecated compatibility shim that re-exports the public API

Core Projector Functions
------------------------

.. currentmodule:: diffct

.. autoclass:: ParallelProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ParallelBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FanProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FanBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConeProjectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConeBackprojectorFunction
   :members:
   :undoc-members:
   :show-inheritance:

Geometry Helpers
----------------

.. currentmodule:: diffct.geometry

**3D trajectories**

.. autofunction:: circular_trajectory_3d
.. autofunction:: random_trajectory_3d
.. autofunction:: spiral_trajectory_3d
.. autofunction:: sinusoidal_trajectory_3d
.. autofunction:: saddle_trajectory_3d
.. autofunction:: custom_trajectory_3d

**2D trajectories**

.. autofunction:: circular_trajectory_2d_parallel
.. autofunction:: sinusoidal_trajectory_2d_parallel
.. autofunction:: custom_trajectory_2d_parallel
.. autofunction:: circular_trajectory_2d_fan
.. autofunction:: sinusoidal_trajectory_2d_fan
.. autofunction:: custom_trajectory_2d_fan

Analytical Reconstruction Helpers
---------------------------------

.. currentmodule:: diffct.analytical

These helpers build the per-view pre-weights, angle-integration weights,
filter, and backprojection pieces of an analytical FBP / FDK pipeline.
They are plain functions (no autograd state) and can be freely mixed
with the autograd operators above. Every helper is trajectory-agnostic:
it takes the same ``(src_pos, det_center, det_u_vec[, det_v_vec])``
arrays that the projector / backprojector Functions consume, so the
same code path works for circular and non-circular trajectories.

.. autofunction:: detector_coordinates_1d

.. autofunction:: angular_integration_weights

.. autofunction:: fan_cosine_weights

.. autofunction:: cone_cosine_weights

.. autofunction:: parker_weights

.. autofunction:: ramp_filter_1d

.. autofunction:: parallel_weighted_backproject

.. autofunction:: fan_weighted_backproject

.. autofunction:: cone_weighted_backproject


Ramp Filter Options
-------------------

``ramp_filter_1d`` is a generic 1D ramp filter used by every analytical
reconstruction example. Its call signature is::

    ramp_filter_1d(sinogram_tensor, dim=-1, sample_spacing=1.0,
                   pad_factor=1, window=None, use_rfft=True)

``sample_spacing``
    Physical detector-cell spacing along ``dim`` (e.g. ``du`` for the
    cone-beam case). The filter is rescaled by ``1 / sample_spacing``
    internally so the output is in physical units and the
    reconstruction amplitude stays calibrated when detector pitch
    changes. Pass ``1.0`` to reproduce sample-unit behaviour.

``pad_factor``
    Zero-pad the signal to ``pad_factor * N`` samples along ``dim``
    before the FFT. Common values:

    - ``1`` (default): no padding, fastest but prone to circular
      convolution wrap-around at the detector edges.
    - ``2``: recommended for cone-beam FDK, good trade-off.
    - ``4``: stricter padding for objects close to the detector edge.

``window``
    Frequency-domain apodization multiplied onto the bare Ram-Lak ramp:

    - ``None`` or ``"ram-lak"``: unwindowed Ram-Lak, sharpest, highest noise.
    - ``"hann"`` / ``"hanning"``: Hann window, strong high-frequency suppression.
    - ``"hamming"``: slightly milder than Hann.
    - ``"cosine"``: half-cosine rolloff.
    - ``"shepp-logan"``: ``sinc`` rolloff, classical choice.

``use_rfft``
    Use ``torch.fft.rfft`` / ``irfft`` when the input is real. Defaults
    to ``True``; set to ``False`` only if you need the complex FFT path.


Analytical FBP / FDK architecture
---------------------------------

Each of the three analytical backprojection helpers
(``parallel_weighted_backproject``, ``fan_weighted_backproject``,
``cone_weighted_backproject``) dispatches to a dedicated voxel-driven
gather kernel, separate from the Siddon-based ray-driven scatter
kernels that drive autograd. The autograd kernels are the pure
adjoints ``P^T`` of the forward projectors (no distance weighting,
no magnification scale), so the autograd classes form self-consistent
forward / backward adjoint pairs. The analytical helpers on top apply
the appropriate FBP / FDK distance weighting and the analytical
scale so the returned image is already amplitude-calibrated.

Scale factors applied inside the analytical helpers:

- ``parallel_weighted_backproject`` multiplies by ``1 / (2 * pi)``
  (Fourier-convention constant; parallel beam has no source and no
  magnification so there is no ``(sid/U)^2`` weight).
- ``fan_weighted_backproject`` applies a per-voxel ``(sid_n / U_n)^2``
  weight inside the gather kernel and a ``sdd_mean / (2 * pi * sid_mean)``
  scale on top.
- ``cone_weighted_backproject`` applies a per-voxel ``(sid_n / U_n)^2``
  weight inside the gather kernel and a ``sdd_mean / (2 * pi * sid_mean)``
  scale on top.

``sid_n`` and ``U_n`` are measured in the per-view detector-normal
direction. For a canonical circular orbit ``sid_n`` reduces to the
classical scalar ``sid``, and ``U_n`` to the classical
``sid + x * sin(beta) - y * cos(beta)``. For non-circular trajectories
(spiral, saddle, random) the helpers keep working by computing the
per-view quantities directly from the trajectory arrays; the result
is the standard heuristic generalisation of FDK beyond the circle.

Utilities
---------

.. currentmodule:: diffct.utils

.. autoclass:: DeviceManager
   :members:

.. autoclass:: TorchCUDABridge
   :members:

Additional helper functions (prefixed with an underscore) remain available for advanced integrations that require direct control over CUDA streams.

Constants
---------

.. currentmodule:: diffct.constants

.. autodata:: _DTYPE
.. autodata:: _TPB_2D
.. autodata:: _TPB_3D
.. autodata:: _FASTMATH_DECORATOR
.. autodata:: _INF
.. autodata:: _EPSILON

These values mirror the defaults used by the CUDA kernels. They are exposed for power users who need to fine-tune launch parameters or numeric tolerances; most applications should rely on the defaults.

Backward Compatibility
----------------------

.. currentmodule:: diffct.differentiable

``diffct.differentiable`` continues to expose the legacy API surface for existing code bases. New projects should import from ``diffct`` (top level) or the specific submodules shown above.

Usage Notes
-----------

- All projector operators accept tensors on CUDA devices and return results on the same device. Use ``diffct.utils.DeviceManager`` helpers when integrating into larger code bases.
- Geometry helper functions build the ``ray_dir``, ``det_origin``, and detector orientation vectors expected by the projector operators.
- Gradients flow through both forward and backward passes; set ``requires_grad=True`` on inputs that participate in optimisation loops.
- Ensure tensors are contiguous and use consistent dtype (``torch.float32``) for maximum kernel performance.
