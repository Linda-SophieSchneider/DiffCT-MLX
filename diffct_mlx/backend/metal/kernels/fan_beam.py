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
_INF_STR = "1e30f"  # finite sentinel: fast math may assume no infinities
_TINY_STR = "1e-20f"

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
    // Direction components at or below tiny count as axis-parallel.
    float tiny = """ + _TINY_STR + """;

    // === 2D FAN BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;

    float det_cx = det_center[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - ((float)n_det - 1.0f) * 0.5f) * det_spacing / voxel_spacing;

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
    // Integrate only from the source to the detector pixel, t in [0, length].
    // The ray parameter is measured from the endpoint nearer the volume
    // centre, so a distant source does not make the parameters coarse.
    bool from_src = (src_x * src_x + src_y * src_y) <= (det_x * det_x + det_y * det_y);
    float org_x = from_src ? src_x : det_x;
    float org_y = from_src ? src_y : det_y;
    float t_min = from_src ? 0.0f : -length;
    float t_max = from_src ? length : 0.0f;

    if (metal::abs(dir_x) > tiny) {
        float tx1 = (-cx - org_x) / dir_x;
        float tx2 = ( cx - org_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (org_x < -cx || org_x > cx) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > tiny) {
        float ty1 = (-cy - org_y) / dir_y;
        float ty2 = ( cy - org_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (org_y < -cy || org_y > cy) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (t_min >= t_max) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    // === SIDDON TRAVERSAL ===
    float accum = 0.0f;
    // Restart the traversal at the volume entry point.
    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float t_end = t_max - t_min;
    float t = 0.0f;

    int ix = (int)metal::floor(ent_x + cx);
    int iy = (int)metal::floor(ent_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - ent_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - ent_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_end) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_end);
            float seg_len = t_next - t;

            if (seg_len > 0.0f) {
                // Cell-constant accumulation (matches the CUDA Siddon kernel:
                // cell k spans [k - c, k + 1 - c), center at k + 0.5 - c)
                accum += image[iy * Nx + ix] * seg_len;
            }
        }

        if (tx <= ty) {
            t = metal::max(t, tx);
            ix += step_x;
            tx += dt_x;
        } else {
            t = metal::max(t, ty);
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
    // Direction components at or below tiny count as axis-parallel.
    float tiny = """ + _TINY_STR + """;

    float val = sino[iang * n_det + idet];

    // === 2D FAN BEAM GEOMETRY SETUP ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;

    float det_cx = det_center_arr[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center_arr[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - ((float)n_det - 1.0f) * 0.5f) * det_spacing / voxel_spacing;

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
    // Integrate only from the source to the detector pixel, t in [0, length].
    // The ray parameter is measured from the endpoint nearer the volume
    // centre, so a distant source does not make the parameters coarse.
    bool from_src = (src_x * src_x + src_y * src_y) <= (det_x * det_x + det_y * det_y);
    float org_x = from_src ? src_x : det_x;
    float org_y = from_src ? src_y : det_y;
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

    if (t_min >= t_max) return;

    // === SIDDON TRAVERSAL ===
    // Restart the traversal at the volume entry point.
    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float t_end = t_max - t_min;
    float t = 0.0f;
    int ix = (int)metal::floor(ent_x + cx);
    int iy = (int)metal::floor(ent_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - ent_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - ent_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_end) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_end);
            float seg_len = t_next - t;

            if (seg_len > 0.0f) {
                // Cell-constant adjoint of the forward kernel (atomic: several
                // rays may hit the same cell)
                atomic_fetch_add_explicit(&grad_image[iy * Nx + ix], val * seg_len, memory_order_relaxed);
            }
        }

        if (tx <= ty) {
            t = metal::max(t, tx);
            ix += step_x;
            tx += dt_x;
        } else {
            t = metal::max(t, ty);
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
    float center_u = ((float)n_det - 1.0f) * 0.5f;

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
        // Mass-conserving footprint amplitude (wrapper scales by voxel_spacing): line integral averaged over the
        // detector cell, on the same scale as the Siddon projector.
        float amp = t * ray_len / (metal::abs(denom) * det_spacing_vox);

        int idet_min = (int)metal::floor(u0 / det_spacing_vox + center_u - support_u / det_spacing_vox);
        int idet_max = (int)metal::ceil(u0 / det_spacing_vox + center_u + support_u / det_spacing_vox);
        idet_min = metal::max(0, idet_min);
        idet_max = metal::min(n_det - 1, idet_max);

        float fp_lo = u0 - half_u;
        float fp_hi = u0 + half_u;
        float scaled_val = val * amp;

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
    float center_u = ((float)n_det - 1.0f) * 0.5f;
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
        // Mass-conserving footprint amplitude (wrapper scales by voxel_spacing): line integral averaged over the
        // detector cell, on the same scale as the Siddon projector.
        float amp = t * ray_len / (metal::abs(denom) * det_spacing_vox);

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
            accum += sino[iang * n_det + idet] * weight * amp;
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

# ============================================================================
# 2D Fan Beam Geometry Gradient Metal Kernel
# ============================================================================
# Exact VJP of the cell-constant Siddon fan projection with respect to
# src_pos, det_center and det_u_vec. Port of the engine's
# ``_fan_2d_geometry_vjp_kernel`` (scalar voxel spacing). With D = B - A in
# voxel units, the projection is s * |D| * sum_j c_j tau_j, where c_j is the
# value step at face j (-f_first at the entry face, +f_last at the exit face).
# A face x_a = b gives tau = (b - A_a) / D_a, so d tau/dA_a = (tau - 1)/D_a and
# d tau/dB_a = -tau/D_a; d|D|/dA = -d, d|D|/dB = +d. Segment ends at the
# source or the detector pixel contribute nothing.

_FAN_2D_GEOMETRY_GRAD_SOURCE = """
    uint iang = thread_position_in_grid.x;
    uint idet = thread_position_in_grid.y;

    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)idet >= n_det) return;

    float g = cotangent[iang * n_det + idet];
    if (g == 0.0f) return;

    float eps = """ + _EPSILON_STR + """;
    float tiny = """ + _TINY_STR + """;
    float big = """ + _INF_STR + """;

    // === GEOMETRY SETUP (identical to the forward kernel) ===
    float src_x = src_pos[iang * 2 + 0] / voxel_spacing;
    float src_y = src_pos[iang * 2 + 1] / voxel_spacing;
    float det_cx = det_center[iang * 2 + 0] / voxel_spacing;
    float det_cy = det_center[iang * 2 + 1] / voxel_spacing;
    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_phys = ((float)idet - ((float)n_det - 1.0f) * 0.5f) * det_spacing;
    float u_offset = u_phys / voxel_spacing;
    float det_x = det_cx + u_offset * u_vec_x;
    float det_y = det_cy + u_offset * u_vec_y;

    float dir_x = det_x - src_x;
    float dir_y = det_y - src_y;
    float length = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (length < eps) return;
    float inv_len = 1.0f / length;
    dir_x *= inv_len;
    dir_y *= inv_len;

    // === SEGMENT CLIPPING, remembering which faces bound it ===
    bool from_src = (src_x * src_x + src_y * src_y) <= (det_x * det_x + det_y * det_y);
    float org_x = from_src ? src_x : det_x;
    float org_y = from_src ? src_y : det_y;
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
    if (t_min >= t_max) return;

    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float t_end = t_max - t_min;
    float entry_tau = (t_min - t_a0) * inv_len;
    float entry_tau_m1 = (t_min - t_b0) * inv_len;

    int ix = (int)metal::floor(ent_x + cx);
    int iy = (int)metal::floor(ent_y + cy);
    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;
    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : big;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : big;
    float txn = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - ent_x) * inv_dir_x : big;
    float tyn = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - ent_y) * inv_dir_y : big;

    // === TRAVERSAL WITH FACE TERMS ===
    float accum = 0.0f;
    float ga_x = 0.0f, ga_y = 0.0f;
    float gb_x = 0.0f, gb_y = 0.0f;
    float f_prev = 0.0f;
    int pend_axis = ent_axis;
    float pend_t = 0.0f;
    bool first = true;
    float t = 0.0f;
    while (t < t_end) {
        float f = 0.0f;
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            f = image[iy * Nx + ix];
        }
        if (pend_axis >= 0) {
            float c = first ? -f : (f_prev - f);
            float dtau = pend_t * inv_len;
            float da = c * (entry_tau_m1 + dtau);
            float db = -c * (entry_tau + dtau);
            if (pend_axis == 0) { ga_x += da / dir_x; gb_x += db / dir_x; }
            else                { ga_y += da / dir_y; gb_y += db / dir_y; }
        }
        first = false;
        float t_next = metal::min(metal::min(txn, tyn), t_end);
        float seg_len = t_next - t;
        if (seg_len > 0.0f) accum += f * seg_len;
        f_prev = f;
        if (txn <= tyn) {
            t = metal::max(t, txn); pend_t = t; pend_axis = 0; ix += step_x; txn += dt_x;
        } else {
            t = metal::max(t, tyn); pend_t = t; pend_axis = 1; iy += step_y; tyn += dt_y;
        }
    }
    if (ex_axis >= 0) {
        float dtau = t_end * inv_len;
        float da = f_prev * (entry_tau_m1 + dtau);
        float db = -f_prev * (entry_tau + dtau);
        if (ex_axis == 0) { ga_x += da / dir_x; gb_x += db / dir_x; }
        else              { ga_y += da / dir_y; gb_y += db / dir_y; }
    }

    // Length term: d|D|/dA = -d, d|D|/dB = +d, times S = sum c_j tau_j.
    float S = accum * inv_len;
    ga_x -= S * dir_x; ga_y -= S * dir_y;
    gb_x += S * dir_x; gb_y += S * dir_y;

    // p = s * accum and A = A_phys / s, so the physical derivative is ga.
    atomic_fetch_add_explicit(&grad_src_pos[iang*2+0], g * ga_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_src_pos[iang*2+1], g * ga_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iang*2+0], g * gb_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_center[iang*2+1], g * gb_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iang*2+0], g * u_phys * gb_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iang*2+1], g * u_phys * gb_y, memory_order_relaxed);
"""

fan_2d_geometry_grad_kernel = mx.fast.metal_kernel(
    name="fan_2d_geometry_grad",
    input_names=["image", "cotangent", "src_pos", "det_center", "det_u_vec", "params", "fparams"],
    output_names=["grad_src_pos", "grad_det_center", "grad_det_u_vec"],
    source=_FAN_2D_GEOMETRY_GRAD_SOURCE,
    atomic_outputs=True,
)
