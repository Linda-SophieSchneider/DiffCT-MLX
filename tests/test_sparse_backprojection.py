from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SparseIndexNormalizationTests(unittest.TestCase):
    def test_normalize_sparse_indices_preserves_order(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        indices = np.array([11, 3, 7, 0], dtype=np.int64)
        normalized = _normalize_sparse_indices(indices, D=2, H=3, W=2)

        np.testing.assert_array_equal(normalized, np.array([11, 3, 7, 0], dtype=np.int32))

    def test_normalize_sparse_indices_accepts_empty_input(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        normalized = _normalize_sparse_indices(np.array([], dtype=np.int64), D=4, H=4, W=4)

        self.assertEqual(normalized.dtype, np.int32)
        self.assertEqual(normalized.shape, (0,))

    def test_normalize_sparse_indices_rejects_non_integer_arrays(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        with self.assertRaisesRegex(TypeError, "integer"):
            _normalize_sparse_indices(np.array([0.0, 1.0], dtype=np.float32), D=2, H=2, W=2)

    def test_normalize_sparse_indices_rejects_non_1d_inputs(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        with self.assertRaisesRegex(ValueError, "1D"):
            _normalize_sparse_indices(np.array([[0, 1], [2, 3]], dtype=np.int64), D=2, H=2, W=2)

    def test_normalize_sparse_indices_rejects_negative_values(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        with self.assertRaisesRegex(ValueError, "negative"):
            _normalize_sparse_indices(np.array([0, -1], dtype=np.int64), D=2, H=2, W=2)

    def test_normalize_sparse_indices_rejects_out_of_bounds_values(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        with self.assertRaisesRegex(ValueError, "out of bounds"):
            _normalize_sparse_indices(np.array([0, 8], dtype=np.int64), D=2, H=2, W=2)

    def test_normalize_sparse_indices_rejects_duplicates(self) -> None:
        from diffct_mlx.projectors import _normalize_sparse_indices

        with self.assertRaisesRegex(ValueError, "duplicate"):
            _normalize_sparse_indices(np.array([1, 1], dtype=np.int64), D=2, H=2, W=2)


@unittest.skipUnless(
    os.environ.get("DIFFCT_TEST_USE_REAL_MLX") == "1",
    "Set DIFFCT_TEST_USE_REAL_MLX=1 on Apple Silicon to run sparse backprojection checks.",
)
class SparseConeBackprojectionTests(unittest.TestCase):
    def _geometry(self):
        import mlx.core as mx

        src_pos = mx.array([[0.0, -120.0, 0.0], [84.85, -84.85, 0.0]], dtype=mx.float32)
        det_center = mx.array([[0.0, 120.0, 0.0], [-84.85, 84.85, 0.0]], dtype=mx.float32)
        det_u_vec = mx.array([[1.0, 0.0, 0.0], [0.70710677, 0.70710677, 0.0]], dtype=mx.float32)
        det_v_vec = mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32)
        return src_pos, det_center, det_u_vec, det_v_vec

    def test_sparse_matches_dense_samples(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        src_pos, det_center, det_u_vec, det_v_vec = self._geometry()
        sinogram = mx.array(
            np.linspace(-0.25, 0.35, num=2 * 5 * 5, dtype=np.float32).reshape(2, 5, 5),
            dtype=mx.float32,
        )
        indices = np.array([0, 7, 16, 21, 42, 63], dtype=np.int64)

        dense = diffct_mlx.cone_backward_footprint(
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
        sparse = diffct_mlx.cone_backward_footprint(
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
            indices=indices,
        )

        dense_samples = np.asarray(dense).reshape(-1)[indices]
        np.testing.assert_allclose(np.asarray(sparse), dense_samples, rtol=5e-5, atol=5e-5)

    def test_sparse_empty_indices_returns_empty_vector(self) -> None:
        import mlx.core as mx
        import diffct_mlx

        src_pos, det_center, det_u_vec, det_v_vec = self._geometry()
        sinogram = mx.zeros((2, 5, 5), dtype=mx.float32)

        sparse = diffct_mlx.cone_backward_footprint(
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
            indices=np.array([], dtype=np.int64),
        )

        self.assertEqual(tuple(sparse.shape), (0,))


if __name__ == "__main__":
    unittest.main()
