"""Regression tests for the geometry-VJP zero-gradient fix.

Before this fix, every ``*_forward.vjp`` / ``*_backward.vjp`` returned ``None``
for the geometry-array primals (``src_pos`` / ``ray_dir``, ``det_center`` /
``det_origin``, ``det_u_vec``, ``det_v_vec``). MLX 0.31 did not collapse those
``None`` returns into zeros and instead propagated the sinogram-shaped
cotangent into the wrong slot, causing ``mx.grad`` to raise a broadcast error
the moment a downstream graph leaf depended on geometry.

These tests verify the minimal contract: ``mx.grad`` over each geometry array
returns an array with the *same shape as the primal* and all entries equal to
zero. Analytical non-zero gradients are the subject of a follow-up PR.
"""

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
    "Set DIFFCT_TEST_USE_REAL_MLX=1 on Apple Silicon to run VJP shape checks.",
)
class GeometryVJPShapeTests(unittest.TestCase):
    """Each forward projector must yield zero-gradients of matching shape."""

    def _assert_zero_grad(self, primal, grad) -> None:
        import mlx.core as mx

        self.assertEqual(tuple(grad.shape), tuple(primal.shape))
        self.assertEqual(float(mx.sum(mx.abs(grad))), 0.0)

    # --- 2D parallel beam -------------------------------------------------
    def _parallel_inputs(self):
        import mlx.core as mx
        import numpy as np

        image = mx.array(np.ones((8, 8), dtype=np.float32))
        ray_dir = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        det_origin = mx.array([[0.0, -4.0], [-4.0, 0.0]], dtype=mx.float32)
        det_u_vec = mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.float32)
        return image, ray_dir, det_origin, det_u_vec

    def test_parallel_forward_geometry_grads_are_zero(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        image, ray_dir, det_origin, det_u_vec = self._parallel_inputs()

        def loss(rd, do, du):
            sino = diffct_mlx.parallel_forward(image, rd, do, du, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2))(ray_dir, det_origin, det_u_vec)
        for primal, grad in zip((ray_dir, det_origin, det_u_vec), grads):
            self._assert_zero_grad(primal, grad)

    def test_parallel_forward_footprint_geometry_grads_are_zero(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        image, ray_dir, det_origin, det_u_vec = self._parallel_inputs()

        def loss(rd, do, du):
            sino = diffct_mlx.parallel_forward_footprint(image, rd, do, du, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2))(ray_dir, det_origin, det_u_vec)
        for primal, grad in zip((ray_dir, det_origin, det_u_vec), grads):
            self._assert_zero_grad(primal, grad)

    # --- 2D fan beam ------------------------------------------------------
    def _fan_inputs(self):
        import mlx.core as mx
        import numpy as np

        image = mx.array(np.ones((8, 8), dtype=np.float32))
        src_pos = mx.array([[0.0, 12.0], [12.0, 0.0]], dtype=mx.float32)
        det_center = mx.array([[0.0, -12.0], [-12.0, 0.0]], dtype=mx.float32)
        det_u_vec = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        return image, src_pos, det_center, det_u_vec

    def test_fan_forward_geometry_grads_are_zero(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        image, src_pos, det_center, det_u_vec = self._fan_inputs()

        def loss(sp, dc, du):
            sino = diffct_mlx.fan_forward(image, sp, dc, du, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2))(src_pos, det_center, det_u_vec)
        for primal, grad in zip((src_pos, det_center, det_u_vec), grads):
            self._assert_zero_grad(primal, grad)

    def test_fan_forward_footprint_geometry_grads_are_zero(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        image, src_pos, det_center, det_u_vec = self._fan_inputs()

        def loss(sp, dc, du):
            sino = diffct_mlx.fan_forward_footprint(image, sp, dc, du, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2))(src_pos, det_center, det_u_vec)
        for primal, grad in zip((src_pos, det_center, det_u_vec), grads):
            self._assert_zero_grad(primal, grad)

    # --- 3D cone beam -----------------------------------------------------
    def _cone_inputs(self):
        import mlx.core as mx
        import numpy as np

        volume = mx.array(np.ones((8, 8, 8), dtype=np.float32))
        src_pos, det_center, det_u_vec, det_v_vec = (
            mx.array(arr, dtype=mx.float32) for arr in (
                [[0.0, 12.0, 0.0], [12.0, 0.0, 0.0]],
                [[0.0, -12.0, 0.0], [-12.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            )
        )
        return volume, src_pos, det_center, det_u_vec, det_v_vec

    def test_cone_forward_geometry_grads_are_zero(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        volume, src_pos, det_center, det_u_vec, det_v_vec = self._cone_inputs()

        def loss(sp, dc, du, dv):
            sino = diffct_mlx.cone_forward(volume, sp, dc, du, dv, 8, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2, 3))(
            src_pos, det_center, det_u_vec, det_v_vec
        )
        for primal, grad in zip((src_pos, det_center, det_u_vec, det_v_vec), grads):
            self._assert_zero_grad(primal, grad)

    def test_cone_forward_footprint_geometry_grads_are_zero(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        volume, src_pos, det_center, det_u_vec, det_v_vec = self._cone_inputs()

        def loss(sp, dc, du, dv):
            sino = diffct_mlx.cone_forward_footprint(volume, sp, dc, du, dv, 8, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2, 3))(
            src_pos, det_center, det_u_vec, det_v_vec
        )
        for primal, grad in zip((src_pos, det_center, det_u_vec, det_v_vec), grads):
            self._assert_zero_grad(primal, grad)

    # --- volume gradients still work --------------------------------------
    def test_cone_forward_volume_grad_unchanged(self) -> None:
        """Sanity: the volume-input gradient path must keep working."""
        import mlx.core as mx
        import diffct_mlx

        volume, src_pos, det_center, det_u_vec, det_v_vec = self._cone_inputs()

        def loss(v):
            sino = diffct_mlx.cone_forward(
                v, src_pos, det_center, det_u_vec, det_v_vec, 8, 8
            )
            return mx.sum(sino)

        grad = mx.grad(loss)(volume)
        self.assertEqual(tuple(grad.shape), tuple(volume.shape))
        self.assertGreater(float(mx.sum(mx.abs(grad))), 0.0)


if __name__ == "__main__":
    unittest.main()
