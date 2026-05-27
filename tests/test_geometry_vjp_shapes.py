"""Regression tests for geometry-VJP correctness.

History
-------
Before the first fix, every ``*_forward.vjp`` / ``*_backward.vjp`` returned
``None`` for geometry-array primals.  MLX 0.31 did not collapse those ``None``
returns into zeros and instead propagated the sinogram-shaped cotangent into
the wrong slot, causing ``mx.grad`` to raise a broadcast error.

After that fix, geometry VJPs returned explicit ``zeros_like``, making
``mx.grad`` structurally correct but numerically zero.

Current state
-------------
``cone_forward`` has two VJP paths controlled by ``DIFFCT_GEOMETRY_VJP``:

* **Flag off (default)**: ``src_pos`` uses finite-difference (6 forward passes,
  ±ε per axis).  ``det_center``, ``det_u_vec``, ``det_v_vec`` are zero.
* **Flag on**: all four geometry primals use the analytical Siddon kernel
  (``_cone_geometry_grad_impl``).  All gradients should be non-zero on a
  non-trivial input.  FD gradient-check tests verify correctness.

``cone_forward_footprint`` still uses the FD path for ``src_pos``; det_*
remain zero (footprint analytical kernel deferred to a later PR).
2-D projectors (parallel, fan) still return zero for all geometry arrays.

Contract tested here
--------------------
* Every geometry gradient has the same shape as its primal.
* Shape-and-sign tests for the FD path (flag off).
* Finite-difference correctness checks for the analytical path (flag on,
  parameterised over all 4 geometry primals × 3 coordinate axes).
* The volume-gradient path is unaffected.
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
    """Geometry gradients must have the correct shape and expected zero/nonzero values."""

    def _assert_zero_grad(self, primal, grad) -> None:
        import mlx.core as mx

        self.assertEqual(tuple(grad.shape), tuple(primal.shape))
        self.assertEqual(float(mx.sum(mx.abs(grad))), 0.0)

    def _assert_nonzero_grad(self, primal, grad) -> None:
        import mlx.core as mx

        self.assertEqual(tuple(grad.shape), tuple(primal.shape))
        self.assertGreater(float(mx.sum(mx.abs(grad))), 0.0)

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

    def test_cone_forward_geometry_grads_fd_path(self) -> None:
        """Flag off: src_pos non-zero (FD); det_* zero."""
        import mlx.core as mx
        import diffct_mlx

        volume, src_pos, det_center, det_u_vec, det_v_vec = self._cone_inputs()

        def loss(sp, dc, du, dv):
            sino = diffct_mlx.cone_forward(volume, sp, dc, du, dv, 8, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2, 3))(
            src_pos, det_center, det_u_vec, det_v_vec
        )
        grad_src, grad_dc, grad_du, grad_dv = grads
        self._assert_nonzero_grad(src_pos, grad_src)
        for primal, grad in zip((det_center, det_u_vec, det_v_vec), (grad_dc, grad_du, grad_dv)):
            self._assert_zero_grad(primal, grad)

    @unittest.skipUnless(
        os.environ.get("DIFFCT_GEOMETRY_VJP") == "1",
        "Set DIFFCT_GEOMETRY_VJP=1 to run analytical geometry gradient tests.",
    )
    def test_cone_forward_analytical_all_grads_nonzero(self) -> None:
        """Flag on: all 4 geometry primals must carry non-zero gradients."""
        import mlx.core as mx
        import diffct_mlx

        volume, src_pos, det_center, det_u_vec, det_v_vec = self._cone_inputs()

        def loss(sp, dc, du, dv):
            sino = diffct_mlx.cone_forward(volume, sp, dc, du, dv, 8, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2, 3))(
            src_pos, det_center, det_u_vec, det_v_vec
        )
        for primal, grad, name in zip(
            (src_pos, det_center, det_u_vec, det_v_vec),
            grads,
            ("src_pos", "det_center", "det_u_vec", "det_v_vec"),
        ):
            with self.subTest(primal=name):
                self._assert_nonzero_grad(primal, grad)

    @unittest.skipUnless(
        os.environ.get("DIFFCT_GEOMETRY_VJP") == "1",
        "Set DIFFCT_GEOMETRY_VJP=1 to run analytical geometry gradient tests.",
    )
    def test_cone_forward_analytical_fd_check(self) -> None:
        """Flag on: analytical gradient matches central-difference FD for all
        geometry primals × 3 coordinate axes.  Tolerance 2e-3 (float32, FD
        step 1e-3, non-trivial phantom)."""
        import mlx.core as mx
        import numpy as np
        import diffct_mlx

        # Small non-trivial volume (ramp in x)
        vol_np = np.zeros((8, 8, 8), dtype=np.float32)
        for i in range(8):
            vol_np[i, :, :] = float(i) * 0.1
        volume = mx.array(vol_np)

        src_pos    = mx.array([[0.0, 20.0, 0.0], [20.0, 0.0, 0.0]], dtype=mx.float32)
        det_center = mx.array([[0.0, -20.0, 0.0], [-20.0, 0.0, 0.0]], dtype=mx.float32)
        det_u_vec  = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=mx.float32)
        det_v_vec  = mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32)

        primals_map = {
            "src_pos":    src_pos,
            "det_center": det_center,
            "det_u_vec":  det_u_vec,
            "det_v_vec":  det_v_vec,
        }
        argnum_map = {"src_pos": 0, "det_center": 1, "det_u_vec": 2, "det_v_vec": 3}

        def loss_fn(sp, dc, du, dv):
            sino = diffct_mlx.cone_forward(volume, sp, dc, du, dv, 8, 8)
            return mx.sum(sino)

        # Analytical gradients for all 4 primals
        all_grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3))(
            src_pos, det_center, det_u_vec, det_v_vec
        )
        anal_grads = dict(zip(primals_map.keys(), all_grads))

        eps = 1e-3
        for name, primal in primals_map.items():
            primal_np = np.array(primal.tolist())
            for iview in range(primal.shape[0]):
                for axis in range(3):
                    p_plus  = primal_np.copy(); p_plus[iview, axis]  += eps
                    p_minus = primal_np.copy(); p_minus[iview, axis] -= eps

                    def _loss_perturbed(p_arr, _name=name, _argnum=argnum_map[name]):
                        args = [src_pos, det_center, det_u_vec, det_v_vec]
                        args[_argnum] = mx.array(p_arr, dtype=mx.float32)
                        return float(loss_fn(*args))

                    fd_val = (_loss_perturbed(p_plus) - _loss_perturbed(p_minus)) / (2 * eps)
                    anal_val = float(mx.array(anal_grads[name].tolist()[iview][axis]))

                    with self.subTest(primal=name, view=iview, axis=axis):
                        self.assertAlmostEqual(
                            anal_val, fd_val,
                            delta=2e-3,
                            msg=f"{name}[{iview},{axis}]: analytical={anal_val:.6f}, FD={fd_val:.6f}",
                        )

    def test_cone_forward_footprint_geometry_grads_fd_path(self) -> None:
        """src_pos gradient is non-zero (FD); det_* gradients are zero."""
        import mlx.core as mx
        import diffct_mlx

        volume, src_pos, det_center, det_u_vec, det_v_vec = self._cone_inputs()

        def loss(sp, dc, du, dv):
            sino = diffct_mlx.cone_forward_footprint(volume, sp, dc, du, dv, 8, 8)
            return mx.sum(sino)

        grads = mx.grad(loss, argnums=(0, 1, 2, 3))(
            src_pos, det_center, det_u_vec, det_v_vec
        )
        grad_src, grad_dc, grad_du, grad_dv = grads
        self._assert_nonzero_grad(src_pos, grad_src)
        for primal, grad in zip((det_center, det_u_vec, det_v_vec), (grad_dc, grad_du, grad_dv)):
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
