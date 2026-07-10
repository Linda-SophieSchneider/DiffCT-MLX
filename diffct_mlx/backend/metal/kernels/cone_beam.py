"""Metal kernels for 3D cone beam projections.

This module contains Metal kernel source strings implementing the Siddon
ray-tracing method for 3D cone beam forward projection and backprojection,
optimized for Apple Silicon GPUs.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants for Metal kernels
# ---------------------------------------------------------------------------
_EPSILON_STR = "1e-6f"
_INF_STR = "1e30f"

# ============================================================================
# 3D Cone Beam Forward Projection Metal Kernel
# ============================================================================

_CONE_3D_FORWARD_SOURCE = """
    // Thread position: (iview, iu, iv)
    uint iview = thread_position_in_grid.x;
    uint iu    = thread_position_in_grid.y;
    uint iv    = thread_position_in_grid.z;

    // Read scalar parameters
    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)iview >= n_views || (int)iu >= n_u || (int)iv >= n_v) return;

    float eps = """ + _EPSILON_STR + """;

    // === 3D CONE BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iview * 3 + 0] / voxel_spacing;
    float src_y = src_pos[iview * 3 + 1] / voxel_spacing;
    float src_z = src_pos[iview * 3 + 2] / voxel_spacing;

    float det_cx = det_center[iview * 3 + 0] / voxel_spacing;
    float det_cy = det_center[iview * 3 + 1] / voxel_spacing;
    float det_cz = det_center[iview * 3 + 2] / voxel_spacing;

    float u_vec_x = det_u_vec_arr[iview * 3 + 0];
    float u_vec_y = det_u_vec_arr[iview * 3 + 1];
    float u_vec_z = det_u_vec_arr[iview * 3 + 2];

    float v_vec_x = det_v_vec_arr[iview * 3 + 0];
    float v_vec_y = det_v_vec_arr[iview * 3 + 1];
    float v_vec_z = det_v_vec_arr[iview * 3 + 2];

    float u_offset = ((float)iu - (float)n_u * 0.5f) * du / voxel_spacing;
    float v_offset = ((float)iv - (float)n_v * 0.5f) * dv / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x;
    float det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y;
    float det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z;

    // === 3D RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float dir_z = det_z - src_z;
    float length = metal::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (length < eps) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    // === 3D RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (metal::abs(dir_z) > eps) {
        float tz1 = (-cz - src_z) / dir_z;
        float tz2 = ( cz - src_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (src_z < -cz || src_z > cz) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (t_min >= t_max) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    // === 3D SIDDON TRAVERSAL ===
    float accum = 0.0f;
    float t = t_min;

    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);
    int iz = (int)metal::floor(src_z + t * dir_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > eps) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > eps) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > eps) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - src_z) * inv_dir_z : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                // Trilinear interpolation at midpoint
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;
                float mid_z = src_z + t_mid * dir_z + cz;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                int iz0 = (int)metal::floor(mid_z);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;
                float dz = mid_z - (float)iz0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));
                iz0 = metal::max(0, metal::min(iz0, Nz - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;
                float omdz = 1.0f - dz;

                // Volume layout: (Nx, Ny, Nz) = WHD permuted
                int base = ix0 * Ny * Nz + iy0 * Nz + iz0;
                int yz_stride = Ny * Nz;

                float val = vol[base]                         * omdx * omdy * omdz +
                            vol[base + yz_stride]             * dx   * omdy * omdz +
                            vol[base + Nz]                    * omdx * dy   * omdz +
                            vol[base + 1]                     * omdx * omdy * dz   +
                            vol[base + yz_stride + Nz]        * dx   * dy   * omdz +
                            vol[base + yz_stride + 1]         * dx   * omdy * dz   +
                            vol[base + Nz + 1]                * omdx * dy   * dz   +
                            vol[base + yz_stride + Nz + 1]    * dx   * dy   * dz;

                accum += val * seg_len * voxel_spacing;
            }
        }

        // 3D boundary crossing
        if (txn <= tyn && txn <= tzn) {
            t = txn;
            ix += step_x;
            txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = tyn;
            iy += step_y;
            tyn += dt_y;
        } else {
            t = tzn;
            iz += step_z;
            tzn += dt_z;
        }
    }

    sino[iview * n_u * n_v + iu * n_v + iv] = accum;
"""

cone_3d_forward_kernel = mx.fast.metal_kernel(
    name="cone_3d_forward",
    input_names=["vol", "src_pos", "det_center", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["sino"],
    source=_CONE_3D_FORWARD_SOURCE,
)

# ============================================================================
# 3D Cone Beam Backprojection Metal Kernel
# ============================================================================

_CONE_3D_BACKWARD_SOURCE = """
    // Thread position: (iview, iu, iv)
    uint iview = thread_position_in_grid.x;
    uint iu    = thread_position_in_grid.y;
    uint iv    = thread_position_in_grid.z;

    // Read scalar parameters
    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)iview >= n_views || (int)iu >= n_u || (int)iv >= n_v) return;

    float eps = """ + _EPSILON_STR + """;

    float g = sino[iview * n_u * n_v + iu * n_v + iv];

    // === 3D CONE BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iview * 3 + 0] / voxel_spacing;
    float src_y = src_pos[iview * 3 + 1] / voxel_spacing;
    float src_z = src_pos[iview * 3 + 2] / voxel_spacing;

    float det_cx = det_center_arr[iview * 3 + 0] / voxel_spacing;
    float det_cy = det_center_arr[iview * 3 + 1] / voxel_spacing;
    float det_cz = det_center_arr[iview * 3 + 2] / voxel_spacing;

    float u_vec_x = det_u_vec_arr[iview * 3 + 0];
    float u_vec_y = det_u_vec_arr[iview * 3 + 1];
    float u_vec_z = det_u_vec_arr[iview * 3 + 2];

    float v_vec_x = det_v_vec_arr[iview * 3 + 0];
    float v_vec_y = det_v_vec_arr[iview * 3 + 1];
    float v_vec_z = det_v_vec_arr[iview * 3 + 2];

    float u_offset = ((float)iu - (float)n_u * 0.5f) * du / voxel_spacing;
    float v_offset = ((float)iv - (float)n_v * 0.5f) * dv / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x;
    float det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y;
    float det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z;

    // === 3D RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float dir_z = det_z - src_z;
    float length = metal::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    // === 3D RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) { return; }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) { return; }

    if (metal::abs(dir_z) > eps) {
        float tz1 = (-cz - src_z) / dir_z;
        float tz2 = ( cz - src_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (src_z < -cz || src_z > cz) { return; }

    if (t_min >= t_max) return;

    // === 3D SIDDON TRAVERSAL ===
    float t = t_min;
    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);
    int iz = (int)metal::floor(src_z + t * dir_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > eps) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > eps) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > eps) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - src_z) * inv_dir_z : """ + _INF_STR + """;

    int yz_stride = Ny * Nz;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;
                float mid_z = src_z + t_mid * dir_z + cz;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                int iz0 = (int)metal::floor(mid_z);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;
                float dz = mid_z - (float)iz0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));
                iz0 = metal::max(0, metal::min(iz0, Nz - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;
                float omdz = 1.0f - dz;
                float cval = g * seg_len;

                int base = ix0 * yz_stride + iy0 * Nz + iz0;

                // Atomic backprojection with trilinear weights
                atomic_fetch_add_explicit(&grad_vol[base],                         cval * omdx * omdy * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride],             cval * dx   * omdy * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + Nz],                    cval * omdx * dy   * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + 1],                     cval * omdx * omdy * dz,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride + Nz],        cval * dx   * dy   * omdz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride + 1],         cval * dx   * omdy * dz,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + Nz + 1],                cval * omdx * dy   * dz,   memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_vol[base + yz_stride + Nz + 1],    cval * dx   * dy   * dz,   memory_order_relaxed);
            }
        }

        if (txn <= tyn && txn <= tzn) {
            t = txn;
            ix += step_x;
            txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = tyn;
            iy += step_y;
            tyn += dt_y;
        } else {
            t = tzn;
            iz += step_z;
            tzn += dt_z;
        }
    }
"""

cone_3d_backward_kernel = mx.fast.metal_kernel(
    name="cone_3d_backward",
    input_names=["sino", "src_pos", "det_center_arr", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["grad_vol"],
    source=_CONE_3D_BACKWARD_SOURCE,
    atomic_outputs=True,
)

# ============================================================================
# 3D Cone Beam Footprint-Matched Projector Pair
# ============================================================================

_CONE_3D_FOOTPRINT_FORWARD_SOURCE = """
    // Thread position: (iview, iy, iz)
    uint iview = thread_position_in_grid.x;
    uint iy    = thread_position_in_grid.y;
    uint iz    = thread_position_in_grid.z;

    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)iview >= n_views || (int)iy >= Ny || (int)iz >= Nz) return;

    float eps = """ + _EPSILON_STR + """;
    float center_u = (float)n_u * 0.5f;
    float center_v = (float)n_v * 0.5f;
    int yz_stride = Ny * Nz;

    float src_x = src_pos[iview * 3 + 0];
    float src_y = src_pos[iview * 3 + 1];
    float src_z = src_pos[iview * 3 + 2];

    float det_cx = det_center[iview * 3 + 0];
    float det_cy = det_center[iview * 3 + 1];
    float det_cz = det_center[iview * 3 + 2];

    float u_vec_x = det_u_vec_arr[iview * 3 + 0];
    float u_vec_y = det_u_vec_arr[iview * 3 + 1];
    float u_vec_z = det_u_vec_arr[iview * 3 + 2];

    float v_vec_x = det_v_vec_arr[iview * 3 + 0];
    float v_vec_y = det_v_vec_arr[iview * 3 + 1];
    float v_vec_z = det_v_vec_arr[iview * 3 + 2];

    float n_x = u_vec_y * v_vec_z - u_vec_z * v_vec_y;
    float n_y = u_vec_z * v_vec_x - u_vec_x * v_vec_z;
    float n_z = u_vec_x * v_vec_y - u_vec_y * v_vec_x;

    float plane_dx = det_cx - src_x;
    float plane_dy = det_cy - src_y;
    float plane_dz = det_cz - src_z;
    float plane_dist = plane_dx * n_x + plane_dy * n_y + plane_dz * n_z;
    if (metal::abs(plane_dist) <= eps) return;

    float py = (((float)iy + 0.5f) - cy) * voxel_spacing;
    float pz = (((float)iz + 0.5f) - cz) * voxel_spacing;

    for (int ix = 0; ix < Nx; ++ix) {
        int base = ix * yz_stride + (int)iy * Nz + (int)iz;
        float val = vol[base];
        if (metal::abs(val) <= eps) continue;

        float px = (((float)ix + 0.5f) - cx) * voxel_spacing;
        float rx = px - src_x;
        float ry = py - src_y;
        float rz = pz - src_z;

        float denom = rx * n_x + ry * n_y + rz * n_z;
        if (metal::abs(denom) <= eps) continue;

        float t = plane_dist / denom;
        if (t <= 0.0f) continue;

        float qx = src_x + t * rx;
        float qy = src_y + t * ry;
        float qz = src_z + t * rz;

        float rel_x = qx - det_cx;
        float rel_y = qy - det_cy;
        float rel_z = qz - det_cz;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y + rel_z * u_vec_z;
        float v0 = rel_x * v_vec_x + rel_y * v_vec_y + rel_z * v_vec_z;

        float ru = rx * u_vec_x + ry * u_vec_y + rz * u_vec_z;
        float rv = rx * v_vec_x + ry * v_vec_y + rz * v_vec_z;
        float inv_denom = 1.0f / denom;

        float grad_ux = t * u_vec_x - t * ru * inv_denom * n_x;
        float grad_uy = t * u_vec_y - t * ru * inv_denom * n_y;
        float grad_uz = t * u_vec_z - t * ru * inv_denom * n_z;

        float grad_vx = t * v_vec_x - t * rv * inv_denom * n_x;
        float grad_vy = t * v_vec_y - t * rv * inv_denom * n_y;
        float grad_vz = t * v_vec_z - t * rv * inv_denom * n_z;

        float half_u = 0.5f * voxel_spacing * (
            metal::abs(grad_ux) + metal::abs(grad_uy) + metal::abs(grad_uz)
        );
        float half_v = 0.5f * voxel_spacing * (
            metal::abs(grad_vx) + metal::abs(grad_vy) + metal::abs(grad_vz)
        );
        half_u = metal::max(half_u, 0.5f * du);
        half_v = metal::max(half_v, 0.5f * dv);

        float width_u = metal::max(2.0f * half_u, du);
        float width_v = metal::max(2.0f * half_v, dv);

        float support_u = half_u + 0.5f * du;
        float support_v = half_v + 0.5f * dv;

        int iu_min = (int)metal::floor((u0 - support_u) / du + center_u);
        int iu_max = (int)metal::ceil((u0 + support_u) / du + center_u);
        int iv_min = (int)metal::floor((v0 - support_v) / dv + center_v);
        int iv_max = (int)metal::ceil((v0 + support_v) / dv + center_v);

        iu_min = metal::max(0, iu_min);
        iv_min = metal::max(0, iv_min);
        iu_max = metal::min(n_u - 1, iu_max);
        iv_max = metal::min(n_v - 1, iv_max);

        float fp_u_lo = u0 - half_u;
        float fp_u_hi = u0 + half_u;
        float fp_v_lo = v0 - half_v;
        float fp_v_hi = v0 + half_v;
        float scaled_val = val * voxel_spacing;

        for (int iu = iu_min; iu <= iu_max; ++iu) {
            float pixel_u = ((float)iu - center_u) * du;
            float pixel_u_lo = pixel_u - 0.5f * du;
            float pixel_u_hi = pixel_u + 0.5f * du;
            float overlap_u = metal::min(pixel_u_hi, fp_u_hi) - metal::max(pixel_u_lo, fp_u_lo);
            if (overlap_u <= eps) continue;
            float wu = overlap_u / width_u;

            for (int iv = iv_min; iv <= iv_max; ++iv) {
                float pixel_v = ((float)iv - center_v) * dv;
                float pixel_v_lo = pixel_v - 0.5f * dv;
                float pixel_v_hi = pixel_v + 0.5f * dv;
                float overlap_v = metal::min(pixel_v_hi, fp_v_hi) - metal::max(pixel_v_lo, fp_v_lo);
                if (overlap_v <= eps) continue;
                float wv = overlap_v / width_v;

                atomic_fetch_add_explicit(
                    &sino[iview * n_u * n_v + iu * n_v + iv],
                    scaled_val * wu * wv,
                    memory_order_relaxed
                );
            }
        }
    }
"""

cone_3d_footprint_forward_kernel = mx.fast.metal_kernel(
    name="cone_3d_footprint_forward",
    input_names=["vol", "src_pos", "det_center", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["sino"],
    source=_CONE_3D_FOOTPRINT_FORWARD_SOURCE,
    atomic_outputs=True,
)

_CONE_3D_FOOTPRINT_BACKWARD_SOURCE = """
    // Thread position: (ix, iy, iz)
    uint ix = thread_position_in_grid.x;
    uint iy = thread_position_in_grid.y;
    uint iz = thread_position_in_grid.z;

    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)ix >= Nx || (int)iy >= Ny || (int)iz >= Nz) return;

    float eps = """ + _EPSILON_STR + """;
    float center_u = (float)n_u * 0.5f;
    float center_v = (float)n_v * 0.5f;
    int yz_stride = Ny * Nz;

    float px = (((float)ix + 0.5f) - cx) * voxel_spacing;
    float py = (((float)iy + 0.5f) - cy) * voxel_spacing;
    float pz = (((float)iz + 0.5f) - cz) * voxel_spacing;

    float accum = 0.0f;

    for (int iview = 0; iview < n_views; ++iview) {
        float src_x = src_pos[iview * 3 + 0];
        float src_y = src_pos[iview * 3 + 1];
        float src_z = src_pos[iview * 3 + 2];

        float det_cx = det_center_arr[iview * 3 + 0];
        float det_cy = det_center_arr[iview * 3 + 1];
        float det_cz = det_center_arr[iview * 3 + 2];

        float u_vec_x = det_u_vec_arr[iview * 3 + 0];
        float u_vec_y = det_u_vec_arr[iview * 3 + 1];
        float u_vec_z = det_u_vec_arr[iview * 3 + 2];

        float v_vec_x = det_v_vec_arr[iview * 3 + 0];
        float v_vec_y = det_v_vec_arr[iview * 3 + 1];
        float v_vec_z = det_v_vec_arr[iview * 3 + 2];

        float n_x = u_vec_y * v_vec_z - u_vec_z * v_vec_y;
        float n_y = u_vec_z * v_vec_x - u_vec_x * v_vec_z;
        float n_z = u_vec_x * v_vec_y - u_vec_y * v_vec_x;

        float plane_dx = det_cx - src_x;
        float plane_dy = det_cy - src_y;
        float plane_dz = det_cz - src_z;
        float plane_dist = plane_dx * n_x + plane_dy * n_y + plane_dz * n_z;
        if (metal::abs(plane_dist) <= eps) continue;

        float rx = px - src_x;
        float ry = py - src_y;
        float rz = pz - src_z;
        float denom = rx * n_x + ry * n_y + rz * n_z;
        if (metal::abs(denom) <= eps) continue;

        float t = plane_dist / denom;
        if (t <= 0.0f) continue;

        float qx = src_x + t * rx;
        float qy = src_y + t * ry;
        float qz = src_z + t * rz;

        float rel_x = qx - det_cx;
        float rel_y = qy - det_cy;
        float rel_z = qz - det_cz;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y + rel_z * u_vec_z;
        float v0 = rel_x * v_vec_x + rel_y * v_vec_y + rel_z * v_vec_z;

        float ru = rx * u_vec_x + ry * u_vec_y + rz * u_vec_z;
        float rv = rx * v_vec_x + ry * v_vec_y + rz * v_vec_z;
        float inv_denom = 1.0f / denom;

        float grad_ux = t * u_vec_x - t * ru * inv_denom * n_x;
        float grad_uy = t * u_vec_y - t * ru * inv_denom * n_y;
        float grad_uz = t * u_vec_z - t * ru * inv_denom * n_z;

        float grad_vx = t * v_vec_x - t * rv * inv_denom * n_x;
        float grad_vy = t * v_vec_y - t * rv * inv_denom * n_y;
        float grad_vz = t * v_vec_z - t * rv * inv_denom * n_z;

        float half_u = 0.5f * voxel_spacing * (
            metal::abs(grad_ux) + metal::abs(grad_uy) + metal::abs(grad_uz)
        );
        float half_v = 0.5f * voxel_spacing * (
            metal::abs(grad_vx) + metal::abs(grad_vy) + metal::abs(grad_vz)
        );
        half_u = metal::max(half_u, 0.5f * du);
        half_v = metal::max(half_v, 0.5f * dv);

        float width_u = metal::max(2.0f * half_u, du);
        float width_v = metal::max(2.0f * half_v, dv);

        float support_u = half_u + 0.5f * du;
        float support_v = half_v + 0.5f * dv;

        int iu_min = (int)metal::floor((u0 - support_u) / du + center_u);
        int iu_max = (int)metal::ceil((u0 + support_u) / du + center_u);
        int iv_min = (int)metal::floor((v0 - support_v) / dv + center_v);
        int iv_max = (int)metal::ceil((v0 + support_v) / dv + center_v);

        iu_min = metal::max(0, iu_min);
        iv_min = metal::max(0, iv_min);
        iu_max = metal::min(n_u - 1, iu_max);
        iv_max = metal::min(n_v - 1, iv_max);

        float fp_u_lo = u0 - half_u;
        float fp_u_hi = u0 + half_u;
        float fp_v_lo = v0 - half_v;
        float fp_v_hi = v0 + half_v;

        for (int iu = iu_min; iu <= iu_max; ++iu) {
            float pixel_u = ((float)iu - center_u) * du;
            float pixel_u_lo = pixel_u - 0.5f * du;
            float pixel_u_hi = pixel_u + 0.5f * du;
            float overlap_u = metal::min(pixel_u_hi, fp_u_hi) - metal::max(pixel_u_lo, fp_u_lo);
            if (overlap_u <= eps) continue;
            float wu = overlap_u / width_u;

            for (int iv = iv_min; iv <= iv_max; ++iv) {
                float pixel_v = ((float)iv - center_v) * dv;
                float pixel_v_lo = pixel_v - 0.5f * dv;
                float pixel_v_hi = pixel_v + 0.5f * dv;
                float overlap_v = metal::min(pixel_v_hi, fp_v_hi) - metal::max(pixel_v_lo, fp_v_lo);
                if (overlap_v <= eps) continue;
                float wv = overlap_v / width_v;
                accum += sino[iview * n_u * n_v + iu * n_v + iv] * wu * wv * voxel_spacing;
            }
        }
    }

    grad_vol[(int)ix * yz_stride + (int)iy * Nz + (int)iz] = accum;
"""

cone_3d_footprint_backward_kernel = mx.fast.metal_kernel(
    name="cone_3d_footprint_backward",
    input_names=["sino", "src_pos", "det_center_arr", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["grad_vol"],
    source=_CONE_3D_FOOTPRINT_BACKWARD_SOURCE,
)

_CONE_3D_FOOTPRINT_BACKWARD_SPARSE_SOURCE = """
    uint isample = thread_position_in_grid.x;

    int n_samples = sparse_params[0];
    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)isample >= n_samples) return;

    int flat_idx = indices[(int)isample];
    int ix = flat_idx % Nx;
    int iy = (flat_idx / Nx) % Ny;
    int iz = flat_idx / (Nx * Ny);

    float eps = """ + _EPSILON_STR + """;
    float center_u = (float)n_u * 0.5f;
    float center_v = (float)n_v * 0.5f;

    float px = (((float)ix + 0.5f) - cx) * voxel_spacing;
    float py = (((float)iy + 0.5f) - cy) * voxel_spacing;
    float pz = (((float)iz + 0.5f) - cz) * voxel_spacing;

    float accum = 0.0f;

    for (int iview = 0; iview < n_views; ++iview) {
        float src_x = src_pos[iview * 3 + 0];
        float src_y = src_pos[iview * 3 + 1];
        float src_z = src_pos[iview * 3 + 2];

        float det_cx = det_center_arr[iview * 3 + 0];
        float det_cy = det_center_arr[iview * 3 + 1];
        float det_cz = det_center_arr[iview * 3 + 2];

        float u_vec_x = det_u_vec_arr[iview * 3 + 0];
        float u_vec_y = det_u_vec_arr[iview * 3 + 1];
        float u_vec_z = det_u_vec_arr[iview * 3 + 2];

        float v_vec_x = det_v_vec_arr[iview * 3 + 0];
        float v_vec_y = det_v_vec_arr[iview * 3 + 1];
        float v_vec_z = det_v_vec_arr[iview * 3 + 2];

        float n_x = u_vec_y * v_vec_z - u_vec_z * v_vec_y;
        float n_y = u_vec_z * v_vec_x - u_vec_x * v_vec_z;
        float n_z = u_vec_x * v_vec_y - u_vec_y * v_vec_x;

        float plane_dx = det_cx - src_x;
        float plane_dy = det_cy - src_y;
        float plane_dz = det_cz - src_z;
        float plane_dist = plane_dx * n_x + plane_dy * n_y + plane_dz * n_z;
        if (metal::abs(plane_dist) <= eps) continue;

        float rx = px - src_x;
        float ry = py - src_y;
        float rz = pz - src_z;
        float denom = rx * n_x + ry * n_y + rz * n_z;
        if (metal::abs(denom) <= eps) continue;

        float t = plane_dist / denom;
        if (t <= 0.0f) continue;

        float qx = src_x + t * rx;
        float qy = src_y + t * ry;
        float qz = src_z + t * rz;

        float rel_x = qx - det_cx;
        float rel_y = qy - det_cy;
        float rel_z = qz - det_cz;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y + rel_z * u_vec_z;
        float v0 = rel_x * v_vec_x + rel_y * v_vec_y + rel_z * v_vec_z;

        float ru = rx * u_vec_x + ry * u_vec_y + rz * u_vec_z;
        float rv = rx * v_vec_x + ry * v_vec_y + rz * v_vec_z;
        float inv_denom = 1.0f / denom;

        float grad_ux = t * u_vec_x - t * ru * inv_denom * n_x;
        float grad_uy = t * u_vec_y - t * ru * inv_denom * n_y;
        float grad_uz = t * u_vec_z - t * ru * inv_denom * n_z;

        float grad_vx = t * v_vec_x - t * rv * inv_denom * n_x;
        float grad_vy = t * v_vec_y - t * rv * inv_denom * n_y;
        float grad_vz = t * v_vec_z - t * rv * inv_denom * n_z;

        float half_u = 0.5f * voxel_spacing * (
            metal::abs(grad_ux) + metal::abs(grad_uy) + metal::abs(grad_uz)
        );
        float half_v = 0.5f * voxel_spacing * (
            metal::abs(grad_vx) + metal::abs(grad_vy) + metal::abs(grad_vz)
        );
        half_u = metal::max(half_u, 0.5f * du);
        half_v = metal::max(half_v, 0.5f * dv);

        float width_u = metal::max(2.0f * half_u, du);
        float width_v = metal::max(2.0f * half_v, dv);

        float support_u = half_u + 0.5f * du;
        float support_v = half_v + 0.5f * dv;

        int iu_min = (int)metal::floor((u0 - support_u) / du + center_u);
        int iu_max = (int)metal::ceil((u0 + support_u) / du + center_u);
        int iv_min = (int)metal::floor((v0 - support_v) / dv + center_v);
        int iv_max = (int)metal::ceil((v0 + support_v) / dv + center_v);

        iu_min = metal::max(0, iu_min);
        iv_min = metal::max(0, iv_min);
        iu_max = metal::min(n_u - 1, iu_max);
        iv_max = metal::min(n_v - 1, iv_max);

        float fp_u_lo = u0 - half_u;
        float fp_u_hi = u0 + half_u;
        float fp_v_lo = v0 - half_v;
        float fp_v_hi = v0 + half_v;

        for (int iu = iu_min; iu <= iu_max; ++iu) {
            float pixel_u = ((float)iu - center_u) * du;
            float pixel_u_lo = pixel_u - 0.5f * du;
            float pixel_u_hi = pixel_u + 0.5f * du;
            float overlap_u = metal::min(pixel_u_hi, fp_u_hi) - metal::max(pixel_u_lo, fp_u_lo);
            if (overlap_u <= eps) continue;
            float wu = overlap_u / width_u;

            for (int iv = iv_min; iv <= iv_max; ++iv) {
                float pixel_v = ((float)iv - center_v) * dv;
                float pixel_v_lo = pixel_v - 0.5f * dv;
                float pixel_v_hi = pixel_v + 0.5f * dv;
                float overlap_v = metal::min(pixel_v_hi, fp_v_hi) - metal::max(pixel_v_lo, fp_v_lo);
                if (overlap_v <= eps) continue;
                float wv = overlap_v / width_v;
                accum += sino[iview * n_u * n_v + iu * n_v + iv] * wu * wv * voxel_spacing;
            }
        }
    }

    grad_sparse[(int)isample] = accum;
"""

cone_3d_footprint_backward_sparse_kernel = mx.fast.metal_kernel(
    name="cone_3d_footprint_backward_sparse",
    input_names=[
        "sino",
        "src_pos",
        "det_center_arr",
        "det_u_vec_arr",
        "det_v_vec_arr",
        "indices",
        "sparse_params",
        "params",
        "fparams",
    ],
    output_names=["grad_sparse"],
    source=_CONE_3D_FOOTPRINT_BACKWARD_SPARSE_SOURCE,
)

# ============================================================================
# 3D Cone Beam Analytical Geometry Gradient Kernel
#
# Computes VJP of cone_forward w.r.t. src_pos, det_center, det_u_vec,
# det_v_vec analytically via the Reynolds-transport gradient of the
# Siddon ray integral.
#
# Derivation (voxel-space parametrisation, λ = t/L where L = ray length):
#   a_p = voxel_spacing * Σ_seg  μ(x(t_mid)) * seg_len
#   ∂a_p/∂s_mm  = −d̂ · A/L + G1    (d̂ = unit ray dir, A = Σ μ*seg_len)
#   ∂a_p/∂p_mm  = +d̂ · A/L + G2    (p = detector pixel world position)
#   G1 = Σ seg_len * (1−λ_mid) * ∇μ_vox(x_mid)
#   G2 = Σ seg_len *    λ_mid  * ∇μ_vox(x_mid)
# Chain rule then yields det_center / det_u_vec / det_v_vec gradients from
# ∂a_p/∂p_mm via p = det_center + u_off_mm*u_vec + v_off_mm*v_vec.
# ============================================================================

_CONE_3D_GEOMETRY_GRAD_SOURCE = """
    uint iview = thread_position_in_grid.x;
    uint iu    = thread_position_in_grid.y;
    uint iv    = thread_position_in_grid.z;

    int n_views = params[0];
    int n_u     = params[1];
    int n_v     = params[2];
    int Nx      = params[3];
    int Ny      = params[4];
    int Nz      = params[5];

    float du            = fparams[0];
    float dv            = fparams[1];
    float cx            = fparams[2];
    float cy            = fparams[3];
    float cz            = fparams[4];
    float voxel_spacing = fparams[5];

    if ((int)iview >= n_views || (int)iu >= n_u || (int)iv >= n_v) return;

    float g = cotangent[iview * n_u * n_v + iu * n_v + iv];
    if (g == 0.0f) return;

    float eps = """ + _EPSILON_STR + """;

    // === GEOMETRY SETUP (identical to forward kernel) ===
    float src_x = src_pos[iview * 3 + 0] / voxel_spacing;
    float src_y = src_pos[iview * 3 + 1] / voxel_spacing;
    float src_z = src_pos[iview * 3 + 2] / voxel_spacing;

    float det_cx = det_center[iview * 3 + 0] / voxel_spacing;
    float det_cy = det_center[iview * 3 + 1] / voxel_spacing;
    float det_cz = det_center[iview * 3 + 2] / voxel_spacing;

    float u_vec_x = det_u_vec_arr[iview * 3 + 0];
    float u_vec_y = det_u_vec_arr[iview * 3 + 1];
    float u_vec_z = det_u_vec_arr[iview * 3 + 2];

    float v_vec_x = det_v_vec_arr[iview * 3 + 0];
    float v_vec_y = det_v_vec_arr[iview * 3 + 1];
    float v_vec_z = det_v_vec_arr[iview * 3 + 2];

    // Detector pixel offsets: voxel-space for ray, mm for chain-rule
    float center_u = (float)n_u * 0.5f;
    float center_v = (float)n_v * 0.5f;
    float u_off_vox = ((float)iu - center_u) * du / voxel_spacing;
    float v_off_vox = ((float)iv - center_v) * dv / voxel_spacing;
    float u_off_mm  = ((float)iu - center_u) * du;
    float v_off_mm  = ((float)iv - center_v) * dv;

    float det_x = det_cx + u_off_vox * u_vec_x + v_off_vox * v_vec_x;
    float det_y = det_cy + u_off_vox * u_vec_y + v_off_vox * v_vec_y;
    float det_z = det_cz + u_off_vox * u_vec_z + v_off_vox * v_vec_z;

    // === RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float dir_z = det_z - src_z;
    float length = metal::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    // === RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) { return; }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) { return; }

    if (metal::abs(dir_z) > eps) {
        float tz1 = (-cz - src_z) / dir_z;
        float tz2 = ( cz - src_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (src_z < -cz || src_z > cz) { return; }

    if (t_min >= t_max) return;

    // === SIDDON TRAVERSAL WITH G1/G2 ACCUMULATORS ===
    float A   = 0.0f;
    float G1x = 0.0f, G1y = 0.0f, G1z = 0.0f;
    float G2x = 0.0f, G2y = 0.0f, G2z = 0.0f;

    float t = t_min;
    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);
    int iz = (int)metal::floor(src_z + t * dir_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > eps) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > eps) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > eps) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - src_z) * inv_dir_z : """ + _INF_STR + """;

    int yz_stride = Ny * Nz;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                float t_mid = t + seg_len * 0.5f;
                float mid_x = src_x + t_mid * dir_x + cx;
                float mid_y = src_y + t_mid * dir_y + cy;
                float mid_z = src_z + t_mid * dir_z + cz;

                int ix0 = (int)metal::floor(mid_x);
                int iy0 = (int)metal::floor(mid_y);
                int iz0 = (int)metal::floor(mid_z);
                float dx = mid_x - (float)ix0;
                float dy = mid_y - (float)iy0;
                float dz = mid_z - (float)iz0;

                ix0 = metal::max(0, metal::min(ix0, Nx - 2));
                iy0 = metal::max(0, metal::min(iy0, Ny - 2));
                iz0 = metal::max(0, metal::min(iz0, Nz - 2));

                float omdx = 1.0f - dx;
                float omdy = 1.0f - dy;
                float omdz = 1.0f - dz;

                int base = ix0 * yz_stride + iy0 * Nz + iz0;

                // Trilinear voxel values (same as forward kernel)
                float v000 = vol[base];
                float v100 = vol[base + yz_stride];
                float v010 = vol[base + Nz];
                float v001 = vol[base + 1];
                float v110 = vol[base + yz_stride + Nz];
                float v101 = vol[base + yz_stride + 1];
                float v011 = vol[base + Nz + 1];
                float v111 = vol[base + yz_stride + Nz + 1];

                float val = v000 * omdx*omdy*omdz + v100 * dx*omdy*omdz
                          + v010 * omdx*dy*omdz   + v001 * omdx*omdy*dz
                          + v110 * dx*dy*omdz      + v101 * dx*omdy*dz
                          + v011 * omdx*dy*dz      + v111 * dx*dy*dz;

                A += val * seg_len;

                // Trilinear gradient of μ in kernel (voxel-index) space
                float dmu_x = omdy*omdz*(v100-v000) + dy*omdz*(v110-v010)
                            + omdy*dz*(v101-v001)   + dy*dz*(v111-v011);
                float dmu_y = omdx*omdz*(v010-v000) + dx*omdz*(v110-v100)
                            + omdx*dz*(v011-v001)   + dx*dz*(v111-v101);
                float dmu_z = omdx*omdy*(v001-v000) + dx*omdy*(v101-v100)
                            + omdx*dy*(v011-v010)   + dx*dy*(v111-v110);

                // λ = t_mid / L; weights for source-end and detector-end
                float lam   = t_mid * inv_len;
                float w1    = seg_len * (1.0f - lam);
                float w2    = seg_len * lam;

                G1x += w1 * dmu_x;  G1y += w1 * dmu_y;  G1z += w1 * dmu_z;
                G2x += w2 * dmu_x;  G2y += w2 * dmu_y;  G2z += w2 * dmu_z;
            }
        }

        if (txn <= tyn && txn <= tzn) {
            t = txn; ix += step_x; txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = tyn; iy += step_y; tyn += dt_y;
        } else {
            t = tzn; iz += step_z; tzn += dt_z;
        }
    }

    // === ASSEMBLE GEOMETRY GRADIENTS ===
    //
    // ∂a_p/∂s_mm  = −d̂ · A/L + G1   (source position, world mm)
    // ∂a_p/∂p_mm  = +d̂ · A/L + G2   (detector pixel, world mm)
    //
    // d̂ = (dir_x, dir_y, dir_z) — same unit vector in voxel and world space
    // (voxel_spacing cancels: d̂_vox = d̂_mm for uniform scaling)
    float A_over_L = A * inv_len;

    float grad_sx = g * (-dir_x * A_over_L + G1x);
    float grad_sy = g * (-dir_y * A_over_L + G1y);
    float grad_sz = g * (-dir_z * A_over_L + G1z);

    float grad_px = g * ( dir_x * A_over_L + G2x);
    float grad_py = g * ( dir_y * A_over_L + G2y);
    float grad_pz = g * ( dir_z * A_over_L + G2z);

    // grad_src_pos (per view)
    atomic_fetch_add_explicit(&grad_src_pos[iview*3+0], grad_sx, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_src_pos[iview*3+1], grad_sy, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_src_pos[iview*3+2], grad_sz, memory_order_relaxed);

    // grad_det_center = grad_p  (∂p_mm/∂det_center_mm = I)
    atomic_fetch_add_explicit(&grad_det_center[iview*3+0], grad_px, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iview*3+1], grad_py, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iview*3+2], grad_pz, memory_order_relaxed);

    // grad_det_u_vec = u_off_mm * grad_p  (∂p_mm/∂u_vec = u_off_mm·I)
    atomic_fetch_add_explicit(&grad_det_u_vec[iview*3+0], u_off_mm * grad_px, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iview*3+1], u_off_mm * grad_py, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iview*3+2], u_off_mm * grad_pz, memory_order_relaxed);

    // grad_det_v_vec = v_off_mm * grad_p  (∂p_mm/∂v_vec = v_off_mm·I)
    atomic_fetch_add_explicit(&grad_det_v_vec[iview*3+0], v_off_mm * grad_px, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_v_vec[iview*3+1], v_off_mm * grad_py, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_v_vec[iview*3+2], v_off_mm * grad_pz, memory_order_relaxed);
"""

cone_3d_geometry_grad_kernel = mx.fast.metal_kernel(
    name="cone_3d_geometry_grad",
    input_names=["vol", "cotangent", "src_pos", "det_center", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["grad_src_pos", "grad_det_center", "grad_det_u_vec", "grad_det_v_vec"],
    source=_CONE_3D_GEOMETRY_GRAD_SOURCE,
    atomic_outputs=True,
)
