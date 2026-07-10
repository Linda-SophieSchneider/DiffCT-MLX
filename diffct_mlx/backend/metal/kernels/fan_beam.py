"""Metal kernels for 2D fan beam projections.

This module contains Metal kernel source strings implementing the Siddon
ray-tracing method for 2D fan beam forward projection and backprojection,
optimized for Apple Silicon GPUs.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants for Metal kernels
# ---------------------------------------------------------------------------
_EPSILON_STR = "1e-6f"
_INF_STR = "1e30f"

# ============================================================================
# 2D Fan Beam Forward Projection Metal Kernel
# ============================================================================

_FAN_2D_FORWARD_SOURCE = """
    // Thread position: (iang, idet)
    uint iang = thread_position_in_grid.x;
    uint idet = thread_position_in_grid.y;

    // Read scalar parameters
    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)idet >= n_det) return;

    float eps = """ + _EPSILON_STR + """;

    // === 2D FAN BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;

    float det_cx = det_center[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - (float)n_det * 0.5f) * det_spacing / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x;
    float det_y = det_cy + u_offset * u_vec_y;

    // === RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float length = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (length < eps) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;

    // === RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    if (metal::abs(dir_x) > eps) {
        float tx1 = (-cx - src_x) / dir_x;
        float tx2 = ( cx - src_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (src_x < -cx || src_x > cx) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > eps) {
        float ty1 = (-cy - src_y) / dir_y;
        float ty2 = ( cy - src_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (src_y < -cy || src_y > cy) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (t_min >= t_max) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    // === SIDDON TRAVERSAL ===
    float accum = 0.0f;
    float t = t_min;

    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                // Cell-constant accumulation (matches the CUDA Siddon kernel:
                // cell k spans [k - c, k + 1 - c), center at k + 0.5 - c)
                accum += image[iy * Nx + ix] * seg_len;
            }
        }

        if (tx <= ty) {
            t = tx;
            ix += step_x;
            tx += dt_x;
        } else {
            t = ty;
            iy += step_y;
            ty += dt_y;
        }
    }

    sino[iang * n_det + idet] = accum;
"""

fan_2d_forward_kernel = mx.fast.metal_kernel(
    name="fan_2d_forward",
    input_names=["image", "src_pos", "det_center", "det_u_vec", "params", "fparams"],
    output_names=["sino"],
    source=_FAN_2D_FORWARD_SOURCE,
)

# ============================================================================
# 2D Fan Beam Backprojection Metal Kernel
# ============================================================================

_FAN_2D_BACKWARD_SOURCE = """
    // Thread position: (iang, idet)
    uint iang = thread_position_in_grid.x;
    uint idet = thread_position_in_grid.y;

    // Read scalar parameters
    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)idet >= n_det) return;

    float eps = """ + _EPSILON_STR + """;

    float val = sino[iang * n_det + idet];

    // === 2D FAN BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;

    float det_cx = det_center_arr[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center_arr[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - (float)n_det * 0.5f) * det_spacing / voxel_spacing;

    float det_x = det_cx + u_offset * u_vec_x;
    float det_y = det_cy + u_offset * u_vec_y;

    // === RAY DIRECTION ===
    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float length = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;

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

    if (t_min >= t_max) return;

    // === SIDDON TRAVERSAL ===
    float t = t_min;
    int ix = (int)metal::floor(src_x + t * dir_x + cx);
    int iy = (int)metal::floor(src_y + t * dir_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > eps) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > eps) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > eps) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > eps) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > eps) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - src_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > eps) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - src_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_max);
            float seg_len = t_next - t;

            if (seg_len > eps) {
                // Cell-constant adjoint of the forward kernel (atomic: several
                // rays may hit the same cell)
                atomic_fetch_add_explicit(&grad_image[iy * Nx + ix], val * seg_len, memory_order_relaxed);
            }
        }

        if (tx <= ty) {
            t = tx;
            ix += step_x;
            tx += dt_x;
        } else {
            t = ty;
            iy += step_y;
            ty += dt_y;
        }
    }
"""

fan_2d_backward_kernel = mx.fast.metal_kernel(
    name="fan_2d_backward",
    input_names=["sino", "src_pos", "det_center_arr", "det_u_vec", "params", "fparams"],
    output_names=["grad_image"],
    source=_FAN_2D_BACKWARD_SOURCE,
    atomic_outputs=True,
)

# ============================================================================
# 2D Fan Beam Footprint-Matched Projector Pair
# ============================================================================

_FAN_2D_FOOTPRINT_FORWARD_SOURCE = """
    // Thread position: (iang, iy)
    uint iang = thread_position_in_grid.x;
    uint iy   = thread_position_in_grid.y;

    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)iy >= Ny) return;

    float eps = """ + _EPSILON_STR + """;
    float det_spacing_vox = det_spacing / voxel_spacing;
    float center_u = (float)n_det * 0.5f;

    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;
    float det_cx = det_center[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center[iang * 2 + 1] / voxel_spacing;
    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];
    float n_x = -u_vec_y;
    float n_y = u_vec_x;

    float plane_dx = det_cx - src_x;
    float plane_dy = det_cy - src_y;
    float plane_dist = plane_dx * n_x + plane_dy * n_y;
    if (metal::abs(plane_dist) <= eps) return;

    float py = ((float)iy + 0.5f) - cy;

    for (int ix = 0; ix < Nx; ++ix) {
        float val = image[(int)iy * Nx + ix];
        if (metal::abs(val) <= eps) continue;

        float px = ((float)ix + 0.5f) - cx;
        float rx = px - src_x;
        float ry = py - src_y;
        float denom = rx * n_x + ry * n_y;
        if (metal::abs(denom) <= eps) continue;

        float t = plane_dist / denom;
        if (t <= 0.0f) continue;

        float qx = src_x + t * rx;
        float qy = src_y + t * ry;
        float rel_x = qx - det_cx;
        float rel_y = qy - det_cy;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y;

        float ru = rx * u_vec_x + ry * u_vec_y;
        float inv_denom = 1.0f / denom;
        float grad_ux = t * u_vec_x - t * ru * inv_denom * n_x;
        float grad_uy = t * u_vec_y - t * ru * inv_denom * n_y;
        float half_u = 0.5f * (metal::abs(grad_ux) + metal::abs(grad_uy));
        half_u = metal::max(half_u, 0.5f * det_spacing_vox);
        float width_u = metal::max(2.0f * half_u, det_spacing_vox);
        float support_u = half_u + 0.5f * det_spacing_vox;

        float ray_len = metal::sqrt(rx * rx + ry * ry);
        if (ray_len <= eps) continue;
        float dir_x = rx / ray_len;
        float dir_y = ry / ray_len;
        float l_phi = 1.0f / metal::max(metal::max(metal::abs(dir_x), metal::abs(dir_y)), eps);

        int idet_min = (int)metal::floor(u0 / det_spacing_vox + center_u - support_u / det_spacing_vox);
        int idet_max = (int)metal::ceil(u0 / det_spacing_vox + center_u + support_u / det_spacing_vox);
        idet_min = metal::max(0, idet_min);
        idet_max = metal::min(n_det - 1, idet_max);

        float fp_lo = u0 - half_u;
        float fp_hi = u0 + half_u;
        float scaled_val = val * l_phi;

        for (int idet = idet_min; idet <= idet_max; ++idet) {
            float pixel_u = ((float)idet - center_u) * det_spacing_vox;
            float pixel_lo = pixel_u - 0.5f * det_spacing_vox;
            float pixel_hi = pixel_u + 0.5f * det_spacing_vox;
            float overlap = metal::min(pixel_hi, fp_hi) - metal::max(pixel_lo, fp_lo);
            if (overlap <= eps) continue;
            float weight = overlap / width_u;
            atomic_fetch_add_explicit(&sino[iang * n_det + idet], scaled_val * weight, memory_order_relaxed);
        }
    }
"""

fan_2d_footprint_forward_kernel = mx.fast.metal_kernel(
    name="fan_2d_footprint_forward",
    input_names=["image", "src_pos", "det_center", "det_u_vec", "params", "fparams"],
    output_names=["sino"],
    source=_FAN_2D_FOOTPRINT_FORWARD_SOURCE,
    atomic_outputs=True,
)

_FAN_2D_FOOTPRINT_BACKWARD_SOURCE = """
    // Thread position: (ix, iy)
    uint ix = thread_position_in_grid.x;
    uint iy = thread_position_in_grid.y;

    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)ix >= Nx || (int)iy >= Ny) return;

    float eps = """ + _EPSILON_STR + """;
    float det_spacing_vox = det_spacing / voxel_spacing;
    float center_u = (float)n_det * 0.5f;
    float px = ((float)ix + 0.5f) - cx;
    float py = ((float)iy + 0.5f) - cy;

    float accum = 0.0f;

    for (int iang = 0; iang < n_ang; ++iang) {
        float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
        float src_y = src_pos[iang * 2 + 1] / voxel_spacing;
        float det_cx = det_center_arr[iang * 2 + 0] / voxel_spacing;
        float det_cy = det_center_arr[iang * 2 + 1] / voxel_spacing;
        float u_vec_x = det_u_vec[iang * 2 + 0];
        float u_vec_y = det_u_vec[iang * 2 + 1];
        float n_x = -u_vec_y;
        float n_y = u_vec_x;

        float plane_dx = det_cx - src_x;
        float plane_dy = det_cy - src_y;
        float plane_dist = plane_dx * n_x + plane_dy * n_y;
        if (metal::abs(plane_dist) <= eps) continue;

        float rx = px - src_x;
        float ry = py - src_y;
        float denom = rx * n_x + ry * n_y;
        if (metal::abs(denom) <= eps) continue;

        float t = plane_dist / denom;
        if (t <= 0.0f) continue;

        float qx = src_x + t * rx;
        float qy = src_y + t * ry;
        float rel_x = qx - det_cx;
        float rel_y = qy - det_cy;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y;

        float ru = rx * u_vec_x + ry * u_vec_y;
        float inv_denom = 1.0f / denom;
        float grad_ux = t * u_vec_x - t * ru * inv_denom * n_x;
        float grad_uy = t * u_vec_y - t * ru * inv_denom * n_y;
        float half_u = 0.5f * (metal::abs(grad_ux) + metal::abs(grad_uy));
        half_u = metal::max(half_u, 0.5f * det_spacing_vox);
        float width_u = metal::max(2.0f * half_u, det_spacing_vox);
        float support_u = half_u + 0.5f * det_spacing_vox;

        float ray_len = metal::sqrt(rx * rx + ry * ry);
        if (ray_len <= eps) continue;
        float dir_x = rx / ray_len;
        float dir_y = ry / ray_len;
        float l_phi = 1.0f / metal::max(metal::max(metal::abs(dir_x), metal::abs(dir_y)), eps);

        int idet_min = (int)metal::floor(u0 / det_spacing_vox + center_u - support_u / det_spacing_vox);
        int idet_max = (int)metal::ceil(u0 / det_spacing_vox + center_u + support_u / det_spacing_vox);
        idet_min = metal::max(0, idet_min);
        idet_max = metal::min(n_det - 1, idet_max);

        float fp_lo = u0 - half_u;
        float fp_hi = u0 + half_u;

        for (int idet = idet_min; idet <= idet_max; ++idet) {
            float pixel_u = ((float)idet - center_u) * det_spacing_vox;
            float pixel_lo = pixel_u - 0.5f * det_spacing_vox;
            float pixel_hi = pixel_u + 0.5f * det_spacing_vox;
            float overlap = metal::min(pixel_hi, fp_hi) - metal::max(pixel_lo, fp_lo);
            if (overlap <= eps) continue;
            float weight = overlap / width_u;
            accum += sino[iang * n_det + idet] * weight * l_phi;
        }
    }

    grad_image[(int)iy * Nx + (int)ix] = accum;
"""

fan_2d_footprint_backward_kernel = mx.fast.metal_kernel(
    name="fan_2d_footprint_backward",
    input_names=["sino", "src_pos", "det_center_arr", "det_u_vec", "params", "fparams"],
    output_names=["grad_image"],
    source=_FAN_2D_FOOTPRINT_BACKWARD_SOURCE,
)
