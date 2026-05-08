from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@unittest.skipUnless(
    os.environ.get("DIFFCT_TEST_USE_REAL_MLX") == "1",
    "Set DIFFCT_TEST_USE_REAL_MLX=1 on Apple Silicon to run footprint projector checks.",
)
class ConeFootprintProjectorTests(unittest.TestCase):
    def test_parallel_forward_and_backward_are_adjoint(self) -> None:
        import numpy as np
        import mlx.core as mx

        import diffct_mlx

        ray_dir = mx.array([[1.0, 0.0], [0.70710677, 0.70710677]], dtype=mx.float32)
        det_origin = mx.array([[0.0, 0.0], [-7.0710678, 7.0710678]], dtype=mx.float32)
        det_u_vec = mx.array([[0.0, 1.0], [-0.70710677, 0.70710677]], dtype=mx.float32)

        image = mx.array(np.arange(25, dtype=np.float32).reshape(5, 5) / 24.0, dtype=mx.float32)
        sinogram = mx.array(np.linspace(-0.4, 0.3, num=2 * 7, dtype=np.float32).reshape(2, 7), dtype=mx.float32)

        forward = diffct_mlx.parallel_forward_footprint(
            image,
            ray_dir,
            det_origin,
            det_u_vec,
            num_detectors=7,
            detector_spacing=1.5,
            voxel_spacing=1.0,
        )
        backward = diffct_mlx.parallel_backward_footprint(
            sinogram,
            ray_dir,
            det_origin,
            det_u_vec,
            detector_spacing=1.5,
            H=5,
            W=5,
            voxel_spacing=1.0,
        )

        lhs = float(np.sum(np.asarray(forward) * np.asarray(sinogram)))
        rhs = float(np.sum(np.asarray(image) * np.asarray(backward)))
        scale = max(1.0, abs(lhs), abs(rhs))
        self.assertLess(abs(lhs - rhs) / scale, 5e-4)

    def test_fan_forward_and_backward_are_adjoint(self) -> None:
        import numpy as np
        import mlx.core as mx

        import diffct_mlx

        src_pos = mx.array([[0.0, -80.0], [56.568542, -56.568542]], dtype=mx.float32)
        det_center = mx.array([[0.0, 80.0], [-56.568542, 56.568542]], dtype=mx.float32)
        det_u_vec = mx.array([[1.0, 0.0], [0.70710677, 0.70710677]], dtype=mx.float32)

        image = mx.array(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0, dtype=mx.float32)
        sinogram = mx.array(np.linspace(-0.25, 0.2, num=2 * 6, dtype=np.float32).reshape(2, 6), dtype=mx.float32)

        forward = diffct_mlx.fan_forward_footprint(
            image,
            src_pos,
            det_center,
            det_u_vec,
            num_detectors=6,
            detector_spacing=2.0,
            voxel_spacing=1.0,
        )
        backward = diffct_mlx.fan_backward_footprint(
            sinogram,
            src_pos,
            det_center,
            det_u_vec,
            detector_spacing=2.0,
            H=4,
            W=4,
            voxel_spacing=1.0,
        )

        lhs = float(np.sum(np.asarray(forward) * np.asarray(sinogram)))
        rhs = float(np.sum(np.asarray(image) * np.asarray(backward)))
        scale = max(1.0, abs(lhs), abs(rhs))
        self.assertLess(abs(lhs - rhs) / scale, 5e-4)

    def test_forward_and_backward_are_adjoint(self) -> None:
        import numpy as np
        import mlx.core as mx

        import diffct_mlx

        src_pos = mx.array([[0.0, -120.0, 0.0], [84.85, -84.85, 0.0]], dtype=mx.float32)
        det_center = mx.array([[0.0, 120.0, 0.0], [-84.85, 84.85, 0.0]], dtype=mx.float32)
        det_u_vec = mx.array([[1.0, 0.0, 0.0], [0.70710677, 0.70710677, 0.0]], dtype=mx.float32)
        det_v_vec = mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32)

        volume = mx.array(np.arange(64, dtype=np.float32).reshape(4, 4, 4) / 63.0, dtype=mx.float32)
        sinogram = mx.array(np.linspace(-0.25, 0.35, num=2 * 5 * 5, dtype=np.float32).reshape(2, 5, 5), dtype=mx.float32)

        forward = diffct_mlx.cone_forward_footprint(
            volume,
            src_pos,
            det_center,
            det_u_vec,
            det_v_vec,
            det_u=5,
            det_v=5,
            du=2.0,
            dv=2.0,
            voxel_spacing=1.5,
        )
        backward = diffct_mlx.cone_backward_footprint(
            sinogram,
            src_pos,
            det_center,
            det_u_vec,
            det_v_vec,
            D=4,
            H=4,
            W=4,
            du=2.0,
            dv=2.0,
            voxel_spacing=1.5,
        )

        lhs = float(np.sum(np.asarray(forward) * np.asarray(sinogram)))
        rhs = float(np.sum(np.asarray(volume) * np.asarray(backward)))

        scale = max(1.0, abs(lhs), abs(rhs))
        self.assertLess(abs(lhs - rhs) / scale, 5e-4)
