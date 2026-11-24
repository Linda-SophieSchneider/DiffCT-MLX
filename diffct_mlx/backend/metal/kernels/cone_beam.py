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
_INF_STR = "1e30f"  # finite sentinel: fast math may assume no infinities
_TINY_STR = "1e-20f"

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
    // Direction components at or below tiny count as axis-parallel.
    float tiny = """ + _TINY_STR + """;

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

    float u_offset = ((float)iu - ((float)n_u - 1.0f) * 0.5f) * du / voxel_spacing;
    float v_offset = ((float)iv - ((float)n_v - 1.0f) * 0.5f) * dv / voxel_spacing;

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
    // Integrate only from the source to the detector pixel, t in [0, length].
    // The ray parameter is measured from the endpoint nearer the volume
    // centre, so a distant source does not make the parameters coarse.
    bool from_src = (src_x * src_x + src_y * src_y + src_z * src_z) <= (det_x * det_x + det_y * det_y + det_z * det_z);
    float org_x = from_src ? src_x : det_x;
    float org_y = from_src ? src_y : det_y;
    float org_z = from_src ? src_z : det_z;
    float t_min = from_src ? 0.0f : -length;
    float t_max = from_src ? length : 0.0f;

    if (metal::abs(dir_x) > tiny) {
        float tx1 = (-cx - org_x) / dir_x;
        float tx2 = ( cx - org_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (org_x < -cx || org_x > cx) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > tiny) {
        float ty1 = (-cy - org_y) / dir_y;
        float ty2 = ( cy - org_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (org_y < -cy || org_y > cy) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (metal::abs(dir_z) > tiny) {
        float tz1 = (-cz - org_z) / dir_z;
        float tz2 = ( cz - org_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (org_z < -cz || org_z > cz) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    if (t_min >= t_max) {
        sino[iview * n_u * n_v + iu * n_v + iv] = 0.0f;
        return;
    }

    // === 3D SIDDON TRAVERSAL ===
    float accum = 0.0f;
    // Restart the traversal at the volume entry point.
    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float ent_z = org_z + t_min * dir_z;
    float t_end = t_max - t_min;
    float t = 0.0f;

    int ix = (int)metal::floor(ent_x + cx);
    int iy = (int)metal::floor(ent_y + cy);
    int iz = (int)metal::floor(ent_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > tiny) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > tiny) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - ent_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - ent_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > tiny) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - ent_z) * inv_dir_z : """ + _INF_STR + """;

    while (t < t_end) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_end);
            float seg_len = t_next - t;

            if (seg_len > 0.0f) {
                // Cell-constant accumulation (matches the CUDA Siddon kernel:
                // cell k spans [k - c, k + 1 - c), center at k + 0.5 - c).
                // Volume layout: (Nx, Ny, Nz) = WHD permuted.
                accum += vol[ix * Ny * Nz + iy * Nz + iz] * seg_len;
            }
        }

        // 3D boundary crossing
        if (txn <= tyn && txn <= tzn) {
            t = metal::max(t, txn);
            ix += step_x;
            txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = metal::max(t, tyn);
            iy += step_y;
            tyn += dt_y;
        } else {
            t = metal::max(t, tzn);
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
    // Direction components at or below tiny count as axis-parallel.
    float tiny = """ + _TINY_STR + """;

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

    float u_offset = ((float)iu - ((float)n_u - 1.0f) * 0.5f) * du / voxel_spacing;
    float v_offset = ((float)iv - ((float)n_v - 1.0f) * 0.5f) * dv / voxel_spacing;

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
    // Integrate only from the source to the detector pixel, t in [0, length].
    // The ray parameter is measured from the endpoint nearer the volume
    // centre, so a distant source does not make the parameters coarse.
    bool from_src = (src_x * src_x + src_y * src_y + src_z * src_z) <= (det_x * det_x + det_y * det_y + det_z * det_z);
    float org_x = from_src ? src_x : det_x;
    float org_y = from_src ? src_y : det_y;
    float org_z = from_src ? src_z : det_z;
    float t_min = from_src ? 0.0f : -length;
    float t_max = from_src ? length : 0.0f;

    if (metal::abs(dir_x) > tiny) {
        float tx1 = (-cx - org_x) / dir_x;
        float tx2 = ( cx - org_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (org_x < -cx || org_x > cx) { return; }

    if (metal::abs(dir_y) > tiny) {
        float ty1 = (-cy - org_y) / dir_y;
        float ty2 = ( cy - org_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (org_y < -cy || org_y > cy) { return; }

    if (metal::abs(dir_z) > tiny) {
        float tz1 = (-cz - org_z) / dir_z;
        float tz2 = ( cz - org_z) / dir_z;
        t_min = metal::max(t_min, metal::min(tz1, tz2));
        t_max = metal::min(t_max, metal::max(tz1, tz2));
    } else if (org_z < -cz || org_z > cz) { return; }

    if (t_min >= t_max) return;

    // === 3D SIDDON TRAVERSAL ===
    // Restart the traversal at the volume entry point.
    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float ent_z = org_z + t_min * dir_z;
    float t_end = t_max - t_min;
    float t = 0.0f;
    int ix = (int)metal::floor(ent_x + cx);
    int iy = (int)metal::floor(ent_y + cy);
    int iz = (int)metal::floor(ent_z + cz);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > tiny) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;
    float dt_z = (metal::abs(dir_z) > tiny) ? metal::abs(inv_dir_z) : """ + _INF_STR + """;

    float txn = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - ent_x) * inv_dir_x : """ + _INF_STR + """;
    float tyn = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - ent_y) * inv_dir_y : """ + _INF_STR + """;
    float tzn = (metal::abs(dir_z) > tiny) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - ent_z) * inv_dir_z : """ + _INF_STR + """;

    int yz_stride = Ny * Nz;

    while (t < t_end) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_end);
            float seg_len = t_next - t;

            if (seg_len > 0.0f) {
                // Cell-constant adjoint of the forward kernel (atomic: several
                // rays may hit the same cell)
                atomic_fetch_add_explicit(&grad_vol[ix * yz_stride + iy * Nz + iz], g * seg_len, memory_order_relaxed);
            }
        }

        if (txn <= tyn && txn <= tzn) {
            t = metal::max(t, txn);
            ix += step_x;
            txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = metal::max(t, tyn);
            iy += step_y;
            tyn += dt_y;
        } else {
            t = metal::max(t, tzn);
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
    float center_u = ((float)n_u - 1.0f) * 0.5f;
    float center_v = ((float)n_v - 1.0f) * 0.5f;
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
        // Mass-conserving footprint amplitude (wrapper scales by voxel_spacing): line integral averaged over the
        // detector cell, on the same scale as the Siddon projector.
        float ray_len = metal::sqrt(rx * rx + ry * ry + rz * rz);
        float amp = (voxel_spacing * voxel_spacing / (du * dv)) * t * t * ray_len * metal::abs(inv_denom);
        float scaled_val = val * amp;

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
    float center_u = ((float)n_u - 1.0f) * 0.5f;
    float center_v = ((float)n_v - 1.0f) * 0.5f;
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

        // Mass-conserving footprint amplitude (wrapper scales by voxel_spacing): line integral averaged over the
        // detector cell, on the same scale as the Siddon projector.
        float ray_len = metal::sqrt(rx * rx + ry * ry + rz * rz);
        float amp = (voxel_spacing * voxel_spacing / (du * dv)) * t * t * ray_len * metal::abs(inv_denom);

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
                accum += sino[iview * n_u * n_v + iu * n_v + iv] * wu * wv * amp;
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
    float center_u = ((float)n_u - 1.0f) * 0.5f;
    float center_v = ((float)n_v - 1.0f) * 0.5f;

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

        // Mass-conserving footprint amplitude (wrapper scales by voxel_spacing): line integral averaged over the
        // detector cell, on the same scale as the Siddon projector.
        float ray_len = metal::sqrt(rx * rx + ry * ry + rz * rz);
        float amp = (voxel_spacing * voxel_spacing / (du * dv)) * t * t * ray_len * metal::abs(inv_denom);

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
                accum += sino[iview * n_u * n_v + iu * n_v + iv] * wu * wv * amp;
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
# 3D Cone Beam Exact Geometry Gradient Kernel
#
# Port of the engine's _cone_3d_geometry_vjp_kernel (upstream diffct 2.x,
# kernels/geometry_vjp.py). For a cell-constant volume the Siddon projection
# of one ray is
#
#     p = s * |D| * sum_k f_k (tau_{k+1} - tau_k),   D = B - A,
#
# with A the source, B the detector pixel (voxel coordinates), s the voxel
# spacing and tau the fractions along D at which the ray crosses cell faces.
# Collecting terms gives sum_j c_j tau_j with c_j = f_before - f_after at
# every crossed face (-f_first at an entry face, +f_last at an exit face).
# A face x_a = b has tau = (b - A_a) / D_a, so
#
#     d tau / d A_a = (tau - 1) / D_a,   d tau / d B_a = -tau / D_a.
#
# Segment ends at the source or the detector move no face and contribute
# nothing. With S = sum c_j tau_j the length term adds -S*d to A and +S*d
# to B. The result is the derivative of the physical projection with respect
# to the world-space source and detector pixel; det_u_vec / det_v_vec follow
# from B = det_center + u_phys*u + v_phys*v. The traversal is the forward
# kernel's, so the gradient belongs to exactly the projected function.
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
    float tiny = """ + _TINY_STR + """;
    float big = """ + _INF_STR + """;

    // === GEOMETRY SETUP (identical to the forward kernel) ===
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

    float u_phys = ((float)iu - ((float)n_u - 1.0f) * 0.5f) * du;
    float v_phys = ((float)iv - ((float)n_v - 1.0f) * 0.5f) * dv;
    float u_offset = u_phys / voxel_spacing;
    float v_offset = v_phys / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x;
    float det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y;
    float det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z;

    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float dir_z = det_z - src_z;
    float length = metal::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    // === SEGMENT CLIPPING, remembering which faces bound it ===
    bool from_src = (src_x * src_x + src_y * src_y + src_z * src_z) <= (det_x * det_x + det_y * det_y + det_z * det_z);
    float org_x = from_src ? src_x : det_x;
    float org_y = from_src ? src_y : det_y;
    float org_z = from_src ? src_z : det_z;
    float t_min = from_src ? 0.0f : -length;
    float t_max = from_src ? length : 0.0f;
    float t_a0 = t_min;   // parameter of the source
    float t_b0 = t_max;   // parameter of the detector pixel
    int ent_axis = -1;
    int ex_axis = -1;

    if (metal::abs(dir_x) > tiny) {
        float t1 = (-cx - org_x) / dir_x;
        float t2 = ( cx - org_x) / dir_x;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 0; }
        if (hi < t_max) { t_max = hi; ex_axis = 0; }
    } else if (org_x < -cx || org_x > cx) {
        return;
    }
    if (metal::abs(dir_y) > tiny) {
        float t1 = (-cy - org_y) / dir_y;
        float t2 = ( cy - org_y) / dir_y;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 1; }
        if (hi < t_max) { t_max = hi; ex_axis = 1; }
    } else if (org_y < -cy || org_y > cy) {
        return;
    }
    if (metal::abs(dir_z) > tiny) {
        float t1 = (-cz - org_z) / dir_z;
        float t2 = ( cz - org_z) / dir_z;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 2; }
        if (hi < t_max) { t_max = hi; ex_axis = 2; }
    } else if (org_z < -cz || org_z > cz) {
        return;
    }
    if (t_min >= t_max) return;

    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float ent_z = org_z + t_min * dir_z;
    float t_end = t_max - t_min;
    float entry_tau = (t_min - t_a0) * inv_len;
    float entry_tau_m1 = (t_min - t_b0) * inv_len;

    int ix = (int)metal::floor(ent_x + cx);
    int iy = (int)metal::floor(ent_y + cy);
    int iz = (int)metal::floor(ent_z + cz);
    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    int step_z = (dir_z >= 0.0f) ? 1 : -1;
    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float inv_dir_z = (metal::abs(dir_z) > tiny) ? (1.0f / dir_z) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : big;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : big;
    float dt_z = (metal::abs(dir_z) > tiny) ? metal::abs(inv_dir_z) : big;
    float txn = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - ent_x) * inv_dir_x : big;
    float tyn = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - ent_y) * inv_dir_y : big;
    float tzn = (metal::abs(dir_z) > tiny) ? ((float)(iz + (step_z > 0 ? 1 : 0)) - cz - ent_z) * inv_dir_z : big;

    // === TRAVERSAL WITH FACE TERMS ===
    float accum = 0.0f;
    float ga_x = 0.0f, ga_y = 0.0f, ga_z = 0.0f;
    float gb_x = 0.0f, gb_y = 0.0f, gb_z = 0.0f;
    float f_prev = 0.0f;
    int pend_axis = ent_axis;
    float pend_t = 0.0f;
    bool first = true;
    float t = 0.0f;
    while (t < t_end) {
        float f = 0.0f;
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            f = vol[ix * Ny * Nz + iy * Nz + iz];
        }
        if (pend_axis >= 0) {
            float c = first ? -f : (f_prev - f);
            float dtau = pend_t * inv_len;
            float da = c * (entry_tau_m1 + dtau);
            float db = -c * (entry_tau + dtau);
            if (pend_axis == 0)      { ga_x += da / dir_x; gb_x += db / dir_x; }
            else if (pend_axis == 1) { ga_y += da / dir_y; gb_y += db / dir_y; }
            else                     { ga_z += da / dir_z; gb_z += db / dir_z; }
        }
        first = false;
        float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_end);
        float seg_len = t_next - t;
        if (seg_len > 0.0f) accum += f * seg_len;
        f_prev = f;
        if (txn <= tyn && txn <= tzn) {
            t = metal::max(t, txn); pend_t = t; pend_axis = 0; ix += step_x; txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = metal::max(t, tyn); pend_t = t; pend_axis = 1; iy += step_y; tyn += dt_y;
        } else {
            t = metal::max(t, tzn); pend_t = t; pend_axis = 2; iz += step_z; tzn += dt_z;
        }
    }
    if (ex_axis >= 0) {
        float dtau = t_end * inv_len;
        float da = f_prev * (entry_tau_m1 + dtau);
        float db = -f_prev * (entry_tau + dtau);
        if (ex_axis == 0)      { ga_x += da / dir_x; gb_x += db / dir_x; }
        else if (ex_axis == 1) { ga_y += da / dir_y; gb_y += db / dir_y; }
        else                   { ga_z += da / dir_z; gb_z += db / dir_z; }
    }

    // Length term: d|D|/dA = -d, d|D|/dB = +d, times S = sum c_j tau_j.
    float S = accum * inv_len;
    ga_x -= S * dir_x; ga_y -= S * dir_y; ga_z -= S * dir_z;
    gb_x += S * dir_x; gb_y += S * dir_y; gb_z += S * dir_z;

    atomic_fetch_add_explicit(&grad_src_pos[iview*3+0], g * ga_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_src_pos[iview*3+1], g * ga_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_src_pos[iview*3+2], g * ga_z, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iview*3+0], g * gb_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iview*3+1], g * gb_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iview*3+2], g * gb_z, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iview*3+0], g * u_phys * gb_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iview*3+1], g * u_phys * gb_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iview*3+2], g * u_phys * gb_z, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_v_vec[iview*3+0], g * v_phys * gb_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_v_vec[iview*3+1], g * v_phys * gb_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_v_vec[iview*3+2], g * v_phys * gb_z, memory_order_relaxed);
"""

cone_3d_geometry_grad_kernel = mx.fast.metal_kernel(
    name="cone_3d_geometry_grad",
    input_names=["vol", "cotangent", "src_pos", "det_center", "det_u_vec_arr", "det_v_vec_arr", "params", "fparams"],
    output_names=["grad_src_pos", "grad_det_center", "grad_det_u_vec", "grad_det_v_vec"],
    source=_CONE_3D_GEOMETRY_GRAD_SOURCE,
    atomic_outputs=True,
)
