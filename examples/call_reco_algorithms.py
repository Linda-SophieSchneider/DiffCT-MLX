"""Compare reconstruction algorithms across 2D and 3D CT geometries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import diffct_mlx


def _default_measured_cone_config() -> diffct_mlx.MeasuredConeDataConfig:
    """Return measured-data loading and preprocessing settings for this example.

    This helper only configures how the TIFF projections and geometry are read:
    sample-data paths, log-domain conversion, per-view air normalization, and
    air-baseline subtraction. It does not change any reconstruction algorithm
    parameters.
    """
    sample_root = Path(__file__).resolve().parent / "non_circular_trajectory" / "sample_data"
    real_data_dir = sample_root / "sim_obj_1_tif"
    return diffct_mlx.MeasuredConeDataConfig(
        data_dir=real_data_dir,
        volume_shape=(256, 256, 256),
        voxel_spacing_mm=0.45,
        target_view_count=25,
        projector_mode="footprint",
        trajectory_json_path=real_data_dir / "sim_obj_1_geometry_diffct.json",
        reference_volume_path=sample_root / "reko" / "sim_obj_1_firefly.npy",
        reference_meta_path=sample_root / "reko" / "sim_obj_1_diffct.json",
        log_transform=True,
        revert=False,
        viewwise_i0=True,
        air_border_px=16,
        subtract_air_baseline=True,
        air_baseline_percentile=15.0,
    )


def _apply_measured_iterative_example_settings(case: diffct_mlx.ReconstructionCase) -> diffct_mlx.ReconstructionCase:
    """Override reconstruction parameters for the measured-data comparison run.

    This helper only adjusts iterative reconstruction settings on an already
    built case, for example SART step size, positivity handling, detector-border
    masking, and TV-family regularization strengths. It does not modify how the
    measured projections were loaded or preprocessed.
    """
    return replace(
        case,
        iterative_iteration_count=20,
        iterative_sart_iteration_count=1,
        iterative_projection_subset_count=8,
        iterative_normalized_sart_relaxation=0.6,
        iterative_backprojection_scale=None,
        pocs_iterative_update_method="normalized_sart",
        iterative_positivity_mode="final",
        iterative_detector_border_u=8,
        iterative_detector_border_v=8,
        iterative_volume_border_width=0,
        iterative_voxel_sensitivity_normalization=True,
        iterative_preserve_unmasked_computed_projection=True,

        tv_reg_iteration_count=1,
        tv_alpha=0.015,

        asd_reg_iteration_count=1,
        asd_alpha=0.015,
        asd_epsilon=0.02,

        awtv_reg_iteration_count=1,
        awtv_alpha=0.01,
        awtv_epsilon=0.02,
        awtv_delta=0.003,
    )


def _compute_metrics(reconstruction, reference):
    """Compute image-quality metrics when a reference is available."""
    if reference is None:
        return None
    reconstruction_np = np.asarray(reconstruction)
    reference_np = np.asarray(reference)
    mse = float(np.mean((reconstruction_np - reference_np) ** 2))
    dynamic_range = float(reference_np.max() - reference_np.min())
    if dynamic_range <= 0.0:
        dynamic_range = 1.0
    psnr = float("inf") if mse <= 0.0 else 10.0 * np.log10((dynamic_range**2) / mse)
    return {"mse": mse, "psnr": psnr}


def _print_result(case_name: str, algorithm_name: str, reconstruction, reference) -> None:
    """Print a compact summary for one reconstruction result."""
    reconstruction_np = np.asarray(reconstruction)
    metrics = _compute_metrics(reconstruction, reference)
    if metrics is None:
        print(
            f"{case_name:<18} {algorithm_name:<10} "
            f"shape={tuple(reconstruction_np.shape)!s:<16} "
            f"range=[{reconstruction_np.min():.4f}, {reconstruction_np.max():.4f}]"
        )
        return
    print(
        f"{case_name:<18} {algorithm_name:<10} "
        f"shape={tuple(reconstruction_np.shape)!s:<16} "
        f"range=[{reconstruction_np.min():.4f}, {reconstruction_np.max():.4f}] "
        f"mse={metrics['mse']:.6f} "
        f"psnr={metrics['psnr']:.2f} dB"
    )


def _print_peak_voxel_summary(case_name: str, algorithm_name: str, reconstruction, *, top_k: int = 5) -> None:
    """Print the location of the brightest voxels for quick artifact checks."""
    reconstruction_np = np.asarray(reconstruction)
    if reconstruction_np.ndim < 2:
        return

    max_value = float(np.max(reconstruction_np))
    max_coords = np.argwhere(reconstruction_np == max_value)
    shape = reconstruction_np.shape

    def _is_border(coord) -> bool:
        return any(int(index) in {0, int(size) - 1} for index, size in zip(coord, shape))

    print(
        f"{case_name:<18} {algorithm_name:<10} "
        f"peak={max_value:.4f} "
        f"max_voxel_count={len(max_coords)} "
        f"first_max={tuple(int(v) for v in max_coords[0])} "
        f"border={_is_border(max_coords[0])}"
    )

    flat = reconstruction_np.reshape(-1)
    top_count = min(int(top_k), int(flat.size))
    if top_count <= 0:
        return
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    top_entries = sorted(
        (
            float(flat[int(index)]),
            tuple(int(v) for v in np.unravel_index(int(index), shape)),
        )
        for index in top_indices
    )[::-1]
    print(
        f"{case_name:<18} {algorithm_name:<10} "
        f"top{top_count}={[(round(value, 4), coord, _is_border(coord)) for value, coord in top_entries]}"
    )


def _print_measured_reconstruction_settings(
    case: diffct_mlx.ReconstructionCase,
    config: diffct_mlx.MeasuredConeDataConfig,
) -> None:
    """Print the measured-data geometry and reconstruction grid used for this run."""
    geometry_payload = json.loads(Path(config.trajectory_json_path).read_text(encoding="utf-8"))
    detector_payload = geometry_payload["detector"]
    source_payload = geometry_payload["source"]

    det_u_binning = max(1, int(np.ceil(detector_payload["num_pixels"]["u"] / config.target_detector_shape[0])))
    det_v_binning = max(1, int(np.ceil(detector_payload["num_pixels"]["v"] / config.target_detector_shape[1])))
    measured_du = float(detector_payload["pixel_size_mm"]["u"]) * det_u_binning
    measured_dv = float(detector_payload["pixel_size_mm"]["v"]) * det_v_binning

    raw_det_v = int(case.sinogram.shape[1])
    raw_det_u = int(case.sinogram.shape[2])
    _, _, detector_pitch_u_mm, detector_pitch_v_mm, detector_pixels_u, detector_pixels_v = (
        diffct_mlx.apply_detector_geometry_convention(
            du=measured_du,
            dv=measured_dv,
            det_u=raw_det_u,
            det_v=raw_det_v,
            flip_u=config.flip_u,
            flip_v=config.flip_v,
            transpose_uv=config.transpose_uv,
        )
    )

    voxel_spacing_mm = None
    if config.voxel_spacing_mm is not None:
        voxel_spacing_mm = float(config.voxel_spacing_mm)
    elif config.reference_meta_path is not None:
        reference_meta_path = Path(config.reference_meta_path)
        if reference_meta_path.exists():
            reference_meta = json.loads(reference_meta_path.read_text(encoding="utf-8"))
            source_shape = tuple(int(value) for value in reference_meta["shape_zyx"])
            resize_factors = tuple(src / dst for src, dst in zip(source_shape, config.volume_shape))
            voxel_spacing_mm = float(reference_meta["voxel_size_mm"]) * resize_factors[0]
    if voxel_spacing_mm is None:
        voxel_spacing_mm = diffct_mlx.auto_voxel_spacing_from_detector(
            config.volume_shape,
            (detector_pixels_u, detector_pixels_v),
            detector_pitch_u_mm,
            detector_pitch_v_mm,
            magnification=float(source_payload["magnification"]),
            fov_margin_mm=config.measured_fov_margin_mm,
        )

    print(
        "Reconstruction settings: "
        f"views={int(case.sinogram.shape[0])}, "
        f"volume_shape={tuple(case.volume_shape)}, "
        f"voxel_size={voxel_spacing_mm:.4f} mm"
    )
    print(
        "Projection geometry: "
        f"source-isocenter={float(source_payload['source_to_isocenter_distance_mm']):.2f} mm, "
        f"source-detector={float(source_payload['source_to_detector_distance_mm']):.2f} mm, "
        f"magnification={float(source_payload['magnification']):.4f}x"
    )
    print(
        "Detector: "
        f"pixels_u={int(detector_pixels_u)}, pixels_v={int(detector_pixels_v)}, "
        f"pixel_size_u={detector_pitch_u_mm:.4f} mm, "
        f"pixel_size_v={detector_pitch_v_mm:.4f} mm"
    )


def _finalize_reconstruction(reconstruction):
    """Materialize a finished reconstruction without changing its physical scale."""
    return np.asarray(reconstruction, dtype=np.float32)


def _iterative_results(case: diffct_mlx.ReconstructionCase) -> dict[str, object]:
    """Run the iterative algorithms for one geometry case."""
    shared_reco_params = diffct_mlx.ReconstructionParameters(
        volume_shape=case.volume_shape,
        iteration_count=case.iterative_iteration_count,
        sart_iteration_count=case.iterative_sart_iteration_count,
        iterative_update_method=case.pocs_iterative_update_method,
        enforce_positivity=True,
        positivity_mode=case.iterative_positivity_mode,
        preserve_unmasked_computed_projection=case.iterative_preserve_unmasked_computed_projection,
        detector_border_u=case.iterative_detector_border_u,
        detector_border_v=case.iterative_detector_border_v,
        volume_border_width=case.iterative_volume_border_width,
        volume_support_mask=case.iterative_volume_support_mask,
        volume_support_mask_mode=case.iterative_volume_support_mask_mode,
        voxel_sensitivity_normalization=case.iterative_voxel_sensitivity_normalization,
        projection_subset_count=case.iterative_projection_subset_count,
        projection_weights=case.iterative_projection_weights,
        normalized_sart_relaxation=case.iterative_normalized_sart_relaxation,
        backprojection_scale=case.iterative_backprojection_scale,
    )
    sart_params = diffct_mlx.SARTParameters(
        volume_shape=case.volume_shape,
        iteration_count=case.iterative_iteration_count,
        sart_iteration_count=case.iterative_sart_iteration_count,
        enforce_positivity=True,
        positivity_mode=case.iterative_positivity_mode,
        preserve_unmasked_computed_projection=case.iterative_preserve_unmasked_computed_projection,
        detector_border_u=case.iterative_detector_border_u,
        detector_border_v=case.iterative_detector_border_v,
        volume_border_width=case.iterative_volume_border_width,
        volume_support_mask=case.iterative_volume_support_mask,
        volume_support_mask_mode=case.iterative_volume_support_mask_mode,
        voxel_sensitivity_normalization=case.iterative_voxel_sensitivity_normalization,
        projection_subset_count=case.iterative_projection_subset_count,
        projection_weights=case.iterative_projection_weights,
        normalized_sart_relaxation=case.iterative_normalized_sart_relaxation,
        iterative_update_method=case.pocs_iterative_update_method,
        backprojection_scale=case.iterative_backprojection_scale,
    )
    sirt_params = diffct_mlx.SIRTParameters(
        volume_shape=case.volume_shape,
        iteration_count=case.sirt_iteration_count,
        sart_iteration_count=case.iterative_sart_iteration_count,
        enforce_positivity=True,
        positivity_mode=case.iterative_positivity_mode,
        preserve_unmasked_computed_projection=case.iterative_preserve_unmasked_computed_projection,
        detector_border_u=case.iterative_detector_border_u,
        detector_border_v=case.iterative_detector_border_v,
        volume_border_width=case.iterative_volume_border_width,
        volume_support_mask=case.iterative_volume_support_mask,
        volume_support_mask_mode=case.iterative_volume_support_mask_mode,
        voxel_sensitivity_normalization=case.iterative_voxel_sensitivity_normalization,
        projection_subset_count=case.iterative_projection_subset_count,
        projection_weights=case.iterative_projection_weights,
        normalized_sart_relaxation=case.iterative_normalized_sart_relaxation,
        backprojection_scale=(
            case.iterative_backprojection_scale if case.iterative_backprojection_scale is not None else 1.0
        ),
    )
    measured_projections = [case.sinogram[i] for i in range(case.sinogram.shape[0])]

    return {
        "SART": 
            diffct_mlx.run_sart(
                measured_projections,
                case.forward_single,
                case.back_single,
                sart_params,
                show_progress=True,
            ),
        "SIRT": 
            diffct_mlx.run_sirt(
                measured_projections,
                case.forward_single,
                case.back_single,
                sirt_params,
                show_progress=True,
            ),
        "TV-POCS": 
            diffct_mlx.run_tv_pocs(
                measured_projections,
                case.forward_single,
                case.back_single,
                shared_reco_params,
                diffct_mlx.TVPOCSParameters(
                    reg_iteration_count=case.tv_reg_iteration_count,
                    alpha=case.tv_alpha,
                ),
                show_progress=True,
        ),
        "ASD-POCS": 
            diffct_mlx.run_asd_pocs(
                measured_projections,
                case.forward_single,
                case.back_single,
                shared_reco_params,
                diffct_mlx.ASDPOCSParameters(
                    reg_iteration_count=case.asd_reg_iteration_count,
                    alpha=case.asd_alpha,
                    epsilon=case.asd_epsilon,
                    beta=1.0,
                ),
                show_progress=True,
            ),

        "AwTV-POCS":
            diffct_mlx.run_awtv_pocs(
                measured_projections,
                case.forward_single,
                case.back_single,
                shared_reco_params,
                diffct_mlx.AwTVPOCSParameters(
                    reg_iteration_count=case.awtv_reg_iteration_count,
                    alpha=case.awtv_alpha,
                    epsilon=case.awtv_epsilon,
                    delta=case.awtv_delta,
                    beta=1.0,
                ),
                show_progress=True,
            ),
    }


def _fbp_result(case: diffct_mlx.ReconstructionCase):
    """Run generic FBP for one case."""
    if not case.supports_fbp or case.fbp_normalization_scale is None:
        raise ValueError(f"FBP is not configured for case {case.name!r}.")
    return _finalize_reconstruction(
        diffct_mlx.run_fbp(
            case.sinogram,
            back_project=case.back_project_all,
            params=diffct_mlx.FBPParameters(
                normalization_scale=case.fbp_normalization_scale,
                filter_axis=1,
            ),
            weight_projections=case.fbp_weight,
        ),
    )


def _fdk_result(case: diffct_mlx.ReconstructionCase):
    """Run generic FDK for one case."""
    if not case.supports_fdk or case.fdk_normalization_scale is None:
        raise ValueError(f"FDK is not configured for case {case.name!r}.")
    return _finalize_reconstruction(
        diffct_mlx.run_fdk(
            case.sinogram,
            back_project=case.back_project_all,
            params=diffct_mlx.FDKParameters(
                normalization_scale=case.fdk_normalization_scale,
                filter_axis=1,
            ),
            weight_projections=case.fdk_weight,
        ),
    )


def _plot_comparison_results(
    case: diffct_mlx.ReconstructionCase,
    results: dict[str, object],
    output_name: str,
    names: list[str],
) -> None:
    """Plot comparisons in a layout similar to the iterative cone example."""
    n = len(names)
    reference_np = None if case.reference is None else np.asarray(case.reference)
    reference_slice = None
    if reference_np is not None:
        reference_slice = reference_np[reference_np.shape[0] // 2] if reference_np.ndim == 3 else reference_np

    measured_panel = None
    if reference_slice is None:
        sinogram_np = np.asarray(case.sinogram)
        measured_panel = sinogram_np[sinogram_np.shape[0] // 2] if sinogram_np.ndim == 3 else sinogram_np

    fig = plt.figure(figsize=(4.5 * n, 10))

    for index, name in enumerate(names):
        reconstruction_np = np.asarray(results[name])
        reconstruction_slice = (
            reconstruction_np[reconstruction_np.shape[0] // 2] if reconstruction_np.ndim == 3 else reconstruction_np
        )
        display_max = float(np.max(reconstruction_slice))
        if display_max <= 0.0:
            display_max = 1.0

        plt.subplot(3, n, index + 1)
        metrics = _compute_metrics(results[name], case.reference)
        if reference_slice is not None and metrics is not None:
            error_map = np.abs(reconstruction_slice - reference_slice)
            plt.imshow(error_map, cmap="magma")
            plt.title(f"{name} Error\nMSE: {metrics['mse']:.4e}\nPSNR: {metrics['psnr']:.2f} dB")
        else:
            plt.axis("off")
            plt.title(f"{name}\nNo reference")
        plt.axis("off")

        plt.subplot(3, n, n + index + 1)
        if reference_slice is not None:
            plt.imshow(reference_slice, cmap="gray", vmin=0, vmax=1)
            plt.title(case.reference_title or "Reference")
        elif measured_panel is not None:
            plt.imshow(measured_panel, cmap="gray")
            plt.title("Measured view")
        else:
            plt.axis("off")
            plt.title("No reference")
        plt.axis("off")

        plt.subplot(3, n, 2 * n + index + 1)
        plt.imshow(reconstruction_slice, cmap="gray", vmin=0, vmax=display_max)
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.show()


def _filename_part(value: str) -> str:
    """Return a filesystem-friendly lowercase token."""
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _save_reconstructions(case_name: str, results: dict[str, object]) -> None:
    """Persist reconstruction volumes for offline inspection."""
    output_dir = Path("saved_reconstructions")
    output_dir.mkdir(exist_ok=True)
    case_part = _filename_part(case_name)
    for method_name, reconstruction in results.items():
        output_path = output_dir / f"{case_part}_{_filename_part(method_name)}.npy"
        np.save(output_path, np.asarray(reconstruction))
        print(f"Saved {method_name} reconstruction to {output_path.resolve()}")


def compare_2d_parallel() -> dict[str, object]:
    """Compare FBP and iterative methods on 2D parallel-beam Shepp-Logan data."""
    case = diffct_mlx.build_parallel_2d_case()
    ordered_names = ["FBP", "SART", "SIRT", "TV-POCS", "ASD-POCS", "AwTV-POCS"]
    results = {"FBP": _fbp_result(case)}
    results.update(_iterative_results(case))

    print("\n2D Parallel")
    for name in ordered_names:
        _print_result(case.name, name, results[name], case.reference)

    _plot_comparison_results(case, results, "compare_2d_parallel.png", ordered_names)
    return results


def compare_2d_fan() -> dict[str, object]:
    """Compare FBP and iterative methods on 2D fan-beam Shepp-Logan data."""
    case = diffct_mlx.build_fan_2d_case()
    ordered_names = ["FBP", "SART", "SIRT", "TV-POCS", "ASD-POCS", "AwTV-POCS"]
    results = {"FBP": _fbp_result(case)}
    results.update(_iterative_results(case))

    print("\n2D Fan")
    for name in ordered_names:
        _print_result(case.name, name, results[name], case.reference)

    _plot_comparison_results(case, results, "compare_2d_fan.png", ordered_names)
    return results


def compare_3d_cone(
    *,
    data_source: str = "shepp_logan",
    measured_config: diffct_mlx.MeasuredConeDataConfig | None = None,
    case: diffct_mlx.ReconstructionCase | None = None,
    output_name: str | None = None,
) -> dict[str, object]:
    """Compare cone-beam reconstruction algorithms for synthetic or measured data.

    Pass a pre-built *case* (e.g. from _build_case_from_npy in reconstruct_sim1.py)
    to bypass TIFF loading entirely.  *data_source* and *measured_config* are
    ignored when *case* is provided.
    """
    resolved_measured_config = measured_config
    if case is not None:
        resolved_output_name = output_name or "compare_3d_cone_npy.png"
        print(f"\n3D Cone (pre-built case: {case.name!r})")
    elif data_source == "shepp_logan":
        case = diffct_mlx.build_cone_3d_case()
        resolved_output_name = output_name or "compare_3d_cone.png"
    elif data_source == "measured":
        resolved_measured_config = measured_config or _default_measured_cone_config()
        case = diffct_mlx.build_measured_cone_3d_case(resolved_measured_config)
        case = _apply_measured_iterative_example_settings(case)
        resolved_output_name = output_name or "compare_3d_cone_measured.png"
    else:
        raise ValueError("data_source must be 'shepp_logan', 'measured', or pass case= directly.")

    results: dict[str, object] = {}
    ordered_names: list[str] = []
    if case.supports_fdk:
        results["FDK"] = _fdk_result(case)
        ordered_names.append("FDK")
    if case.supports_fbp:
        results["FBP"] = _fbp_result(case)
        ordered_names.append("FBP")

    if data_source == "measured" and not case.supports_fbp and not case.supports_fdk:
        print("\n3D Cone (Measured)")
        print("Skipping FBP/FDK: the loaded measured trajectory is arbitrary rather than circular.")
        _print_measured_reconstruction_settings(case, resolved_measured_config)
    elif case is None:
        print("\n3D Cone")

    iterative = _iterative_results(case)
    results.update(iterative)
    ordered_names.extend(["SART", "SIRT", "TV-POCS", "ASD-POCS", "AwTV-POCS"])

    for name in ordered_names:
        _print_result(case.name, name, results[name], case.reference)
        if np.asarray(results[name]).ndim == 3:
            _print_peak_voxel_summary(case.name, name, results[name])

    _save_reconstructions(case.name, {name: results[name] for name in ordered_names})
    _plot_comparison_results(case, results, resolved_output_name, ordered_names)
    return results


def main() -> None:
    # compare_2d_parallel()
    # compare_2d_fan()
    # compare_3d_cone(data_source="shepp_logan")

    # --- npy projections (occlusion stress stages) ---
    STRESS_DIR = Path(__file__).resolve().parents[2] / "results" / "occlusion_stress" / "sim_obj_1_firefly_occlusion_stress_roi25_q95"
    GEOMETRY_JSON = Path(__file__).resolve().parent / "non_circular_trajectory" / "sample_data" / "sim_obj_1_tif" / "sim_obj_1_geometry_diffct.json"
    for stage in ("none", "mild", "moderate", "severe"):
        npy_path = STRESS_DIR / stage / "projections_line_integral.npy"
        case = diffct_mlx.build_npy_cone_3d_case(diffct_mlx.NpyProjectionsConfig(
            projections_npy_path=npy_path,
            geometry_json_path=GEOMETRY_JSON,
            volume_shape=(384, 384, 384),
            transpose_uv=True,
        ))
        compare_3d_cone(case=case, output_name=f"compare_3d_cone_npy_{stage}.png")

    # print("\nNote: the measured-data comparison may take several minutes to run.")
    # compare_3d_cone(data_source="measured", measured_config=_default_measured_cone_config())


if __name__ == "__main__":
    main()
