"""Metal kernels for 2D parallel beam projections.

This module contains Metal kernel source strings implementing the Siddon
ray-tracing method for 2D parallel beam forward projection and backprojection,
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
# 2D Parallel Beam Forward Projection Metal Kernel
# ============================================================================

_PARALLEL_2D_FORWARD_SOURCE = """
    // Thread position: (iang, idet)
    uint iang = thread_position_in_grid.x;
    uint idet = thread_position_in_grid.y;

    // Read scalar parameters from constant buffers
    int n_ang = params[0];
    int n_det = params[1];
    int Nx    = params[2];
    int Ny    = params[3];

    float det_spacing   = fparams[0];
    float cx            = fparams[1];
    float cy            = fparams[2];
    float voxel_spacing = fparams[3];

    if ((int)iang >= n_ang || (int)idet >= n_det) return;

    // === 2D PARALLEL BEAM GEOMETRY SETUP ===
    float dir_x = ray_dir[iang * 2 + 0];
    float dir_y = ray_dir[iang * 2 + 1];
    // Normalize the direction: segment lengths are differences of the ray
    // parameter, so a non-unit ray_dir would scale every line integral.
    float dir_norm = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (dir_norm <= 0.0f) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }
    dir_x /= dir_norm;
    dir_y /= dir_norm;

    float det_ox = det_origin[iang * 2 + 0] / voxel_spacing;
    float det_oy = det_origin[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - ((float)n_det - 1.0f) * 0.5f) * det_spacing / voxel_spacing;

    float pnt_x = det_ox + u_offset * u_vec_x;
    float pnt_y = det_oy + u_offset * u_vec_y;

    // === RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;

    float eps = """ + _EPSILON_STR + """;
    // Direction components at or below tiny count as axis-parallel.
    float tiny = """ + _TINY_STR + """;

    if (metal::abs(dir_x) > tiny) {
        float tx1 = (-cx - pnt_x) / dir_x;
        float tx2 = ( cx - pnt_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (pnt_x < -cx || pnt_x > cx) {
        sino[iang * n_det + idet] = 0.0f;
        return;
    }

    if (metal::abs(dir_y) > tiny) {
        float ty1 = (-cy - pnt_y) / dir_y;
        float ty2 = ( cy - pnt_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (pnt_y < -cy || pnt_y > cy) {
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

    int ix = (int)metal::floor(pnt_x + t * dir_x + cx);
    int iy = (int)metal::floor(pnt_y + t * dir_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - pnt_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - pnt_y) * inv_dir_y : """ + _INF_STR + """;

    // Main traversal loop
    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_max);
            float seg_len = t_next - t;

            if (seg_len > 0.0f) {
                // Cell-constant accumulation (matches the CUDA Siddon kernel:
                // cell k spans [k - c, k + 1 - c), center at k + 0.5 - c)
                accum += image[iy * Nx + ix] * seg_len;
            }
        }

        // Advance to next voxel
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

parallel_2d_forward_kernel = mx.fast.metal_kernel(
    name="parallel_2d_forward",
    input_names=["image", "ray_dir", "det_origin", "det_u_vec", "params", "fparams"],
    output_names=["sino"],
    source=_PARALLEL_2D_FORWARD_SOURCE,
)

# ============================================================================
# 2D Parallel Beam Backprojection Metal Kernel
# ============================================================================

_PARALLEL_2D_BACKWARD_SOURCE = """
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

    float val = sino[iang * n_det + idet];

    // === 2D PARALLEL BEAM GEOMETRY SETUP ===
    float dir_x = ray_dir[iang * 2 + 0];
    float dir_y = ray_dir[iang * 2 + 1];
    // Normalize the direction: segment lengths are differences of the ray
    // parameter, so a non-unit ray_dir would scale every line integral.
    float dir_norm = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (dir_norm <= 0.0f) { return; }
    dir_x /= dir_norm;
    dir_y /= dir_norm;

    float det_ox = det_origin_arr[iang * 2 + 0] / voxel_spacing;
    float det_oy = det_origin_arr[iang * 2 + 1] / voxel_spacing;

    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float u_offset = ((float)idet - ((float)n_det - 1.0f) * 0.5f) * det_spacing / voxel_spacing;

    float pnt_x = det_ox + u_offset * u_vec_x;
    float pnt_y = det_oy + u_offset * u_vec_y;

    // === RAY-VOLUME INTERSECTION ===
    float t_min = -""" + _INF_STR + """;
    float t_max =  """ + _INF_STR + """;
    float eps = """ + _EPSILON_STR + """;
    // Direction components at or below tiny count as axis-parallel.
    float tiny = """ + _TINY_STR + """;

    if (metal::abs(dir_x) > tiny) {
        float tx1 = (-cx - pnt_x) / dir_x;
        float tx2 = ( cx - pnt_x) / dir_x;
        t_min = metal::max(t_min, metal::min(tx1, tx2));
        t_max = metal::min(t_max, metal::max(tx1, tx2));
    } else if (pnt_x < -cx || pnt_x > cx) { return; }

    if (metal::abs(dir_y) > tiny) {
        float ty1 = (-cy - pnt_y) / dir_y;
        float ty2 = ( cy - pnt_y) / dir_y;
        t_min = metal::max(t_min, metal::min(ty1, ty2));
        t_max = metal::min(t_max, metal::max(ty1, ty2));
    } else if (pnt_y < -cy || pnt_y > cy) { return; }

    if (t_min >= t_max) return;

    // === SIDDON TRAVERSAL ===
    float t = t_min;
    int ix = (int)metal::floor(pnt_x + t * dir_x + cx);
    int iy = (int)metal::floor(pnt_y + t * dir_y + cy);

    int step_x = (dir_x >= 0.0f) ? 1 : -1;
    int step_y = (dir_y >= 0.0f) ? 1 : -1;

    float inv_dir_x = (metal::abs(dir_x) > tiny) ? (1.0f / dir_x) : 0.0f;
    float inv_dir_y = (metal::abs(dir_y) > tiny) ? (1.0f / dir_y) : 0.0f;
    float dt_x = (metal::abs(dir_x) > tiny) ? metal::abs(inv_dir_x) : """ + _INF_STR + """;
    float dt_y = (metal::abs(dir_y) > tiny) ? metal::abs(inv_dir_y) : """ + _INF_STR + """;

    float tx = (metal::abs(dir_x) > tiny) ? ((float)(ix + (step_x > 0 ? 1 : 0)) - cx - pnt_x) * inv_dir_x : """ + _INF_STR + """;
    float ty = (metal::abs(dir_y) > tiny) ? ((float)(iy + (step_y > 0 ? 1 : 0)) - cy - pnt_y) * inv_dir_y : """ + _INF_STR + """;

    while (t < t_max) {
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny) {
            float t_next = metal::min(metal::min(tx, ty), t_max);
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

parallel_2d_backward_kernel = mx.fast.metal_kernel(
    name="parallel_2d_backward",
    input_names=["sino", "ray_dir", "det_origin_arr", "det_u_vec", "params", "fparams"],
    output_names=["grad_image"],
    source=_PARALLEL_2D_BACKWARD_SOURCE,
    atomic_outputs=True,
)

# ============================================================================
# 2D Parallel Beam Footprint-Matched Projector Pair
# ============================================================================

_PARALLEL_2D_FOOTPRINT_FORWARD_SOURCE = """
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

    float dir_x = ray_dir[iang * 2 + 0];
    float dir_y = ray_dir[iang * 2 + 1];
    float det_ox = det_origin[iang * 2 + 0] / voxel_spacing;
    float det_oy = det_origin[iang * 2 + 1] / voxel_spacing;
    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];

    float half_u = 0.5f * (metal::abs(u_vec_x) + metal::abs(u_vec_y));
    half_u = metal::max(half_u, 0.5f * det_spacing_vox);
    float width_u = metal::max(2.0f * half_u, det_spacing_vox);
    float support_u = half_u + 0.5f * det_spacing_vox;
    float py = ((float)iy + 0.5f) - cy;

    for (int ix = 0; ix < Nx; ++ix) {
        float val = image[(int)iy * Nx + ix];
        if (metal::abs(val) <= eps) continue;

        float px = ((float)ix + 0.5f) - cx;
        float rel_x = px - det_ox;
        float rel_y = py - det_oy;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y;

        int idet_min = (int)metal::floor(u0 / det_spacing_vox + center_u - support_u / det_spacing_vox);
        int idet_max = (int)metal::ceil(u0 / det_spacing_vox + center_u + support_u / det_spacing_vox);
        idet_min = metal::max(0, idet_min);
        idet_max = metal::min(n_det - 1, idet_max);

        float fp_lo = u0 - half_u;
        float fp_hi = u0 + half_u;
        // Mass-conserving footprint amplitude (wrapper scales by voxel_spacing): line integral averaged over the
        // detector cell, on the same scale as the Siddon projector.
        float scaled_val = val / det_spacing_vox;

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

parallel_2d_footprint_forward_kernel = mx.fast.metal_kernel(
    name="parallel_2d_footprint_forward",
    input_names=["image", "ray_dir", "det_origin", "det_u_vec", "params", "fparams"],
    output_names=["sino"],
    source=_PARALLEL_2D_FOOTPRINT_FORWARD_SOURCE,
    atomic_outputs=True,
)

_PARALLEL_2D_FOOTPRINT_BACKWARD_SOURCE = """
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
        float dir_x = ray_dir[iang * 2 + 0];
        float dir_y = ray_dir[iang * 2 + 1];
        float det_ox = det_origin_arr[iang * 2 + 0] / voxel_spacing;
        float det_oy = det_origin_arr[iang * 2 + 1] / voxel_spacing;
        float u_vec_x = det_u_vec[iang * 2 + 0];
        float u_vec_y = det_u_vec[iang * 2 + 1];

        float half_u = 0.5f * (metal::abs(u_vec_x) + metal::abs(u_vec_y));
        half_u = metal::max(half_u, 0.5f * det_spacing_vox);
        float width_u = metal::max(2.0f * half_u, det_spacing_vox);
        float support_u = half_u + 0.5f * det_spacing_vox;

        float rel_x = px - det_ox;
        float rel_y = py - det_oy;
        float u0 = rel_x * u_vec_x + rel_y * u_vec_y;

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
            accum += sino[iang * n_det + idet] * weight / det_spacing_vox;
        }
    }

    grad_image[(int)iy * Nx + (int)ix] = accum;
"""

parallel_2d_footprint_backward_kernel = mx.fast.metal_kernel(
    name="parallel_2d_footprint_backward",
    input_names=["sino", "ray_dir", "det_origin_arr", "det_u_vec", "params", "fparams"],
    output_names=["grad_image"],
    source=_PARALLEL_2D_FOOTPRINT_BACKWARD_SOURCE,
)

# ============================================================================
# 2D Parallel Beam Geometry Gradient Metal Kernel
# ============================================================================
# Exact VJP of the cell-constant Siddon parallel projection with respect to
# ray_dir, det_origin and det_u_vec. Port of the engine's
# ``_parallel_2d_geometry_vjp_kernel`` (scalar voxel spacing). The ray is
# P + t r over the whole line, P the detector pixel in voxels and r the unit
# direction. The projection is s * sum_j c_j t_j with c_j the value step at
# face j. A face x_a = b gives t = (b - P_a) / r_a, so dt/dP_a = -1/r_a and
# dt/dr_a = -t/r_a. The direction gradient is pulled back through the
# normalisation r = R / |R|.

_PARALLEL_2D_GEOMETRY_GRAD_SOURCE = """
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

    float tiny = """ + _TINY_STR + """;
    float big = """ + _INF_STR + """;

    // === GEOMETRY SETUP (identical to the forward kernel) ===
    float dir_x = ray_dir[iang * 2 + 0];
    float dir_y = ray_dir[iang * 2 + 1];
    float dir_norm = metal::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (dir_norm <= 0.0f) return;
    dir_x /= dir_norm;
    dir_y /= dir_norm;

    float det_ox = det_origin[iang * 2 + 0] / voxel_spacing;
    float det_oy = det_origin[iang * 2 + 1] / voxel_spacing;
    float u_vec_x = det_u_vec[iang * 2 + 0];
    float u_vec_y = det_u_vec[iang * 2 + 1];
    float u_phys = ((float)idet - ((float)n_det - 1.0f) * 0.5f) * det_spacing;
    float u_offset = u_phys / voxel_spacing;
    float pnt_x = det_ox + u_offset * u_vec_x;
    float pnt_y = det_oy + u_offset * u_vec_y;

    // === LINE CLIPPING, remembering which faces bound it ===
    float t_min = -big;
    float t_max = big;
    int ent_axis = -1;
    int ex_axis = -1;
    if (metal::abs(dir_x) > tiny) {
        float t1 = (-cx - pnt_x) / dir_x;
        float t2 = ( cx - pnt_x) / dir_x;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 0; }
        if (hi < t_max) { t_max = hi; ex_axis = 0; }
    } else if (pnt_x < -cx || pnt_x > cx) {
        return;
    }
    if (metal::abs(dir_y) > tiny) {
        float t1 = (-cy - pnt_y) / dir_y;
        float t2 = ( cy - pnt_y) / dir_y;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 1; }
        if (hi < t_max) { t_max = hi; ex_axis = 1; }
    } else if (pnt_y < -cy || pnt_y > cy) {
        return;
    }
    if (t_min >= t_max) return;

    // Traverse from the entry point; t is local, t_min is added afterwards.
    float ent_x = pnt_x + t_min * dir_x;
    float ent_y = pnt_y + t_min * dir_y;
    float t_end = t_max - t_min;

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

    // Per axis: C = sum of face coefficients, CT = sum of coefficient * t.
    float c_x = 0.0f, c_y = 0.0f;
    float ct_x = 0.0f, ct_y = 0.0f;
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
            if (pend_axis == 0) { c_x += c; ct_x += c * pend_t; }
            else                { c_y += c; ct_y += c * pend_t; }
        }
        first = false;
        f_prev = f;
        if (txn <= tyn) {
            t = metal::max(t, txn); pend_t = t; pend_axis = 0; ix += step_x; txn += dt_x;
        } else {
            t = metal::max(t, tyn); pend_t = t; pend_axis = 1; iy += step_y; tyn += dt_y;
        }
    }
    if (ex_axis == 0)      { c_x += f_prev; ct_x += f_prev * t_end; }
    else if (ex_axis == 1) { c_y += f_prev; ct_y += f_prev * t_end; }

    // Physical derivatives: dp/dP_phys = -C_a/r_a; dp/dr_a = -s (t_min C_a + CT_a)/r_a.
    float gp_x = 0.0f, gp_y = 0.0f, gr_x = 0.0f, gr_y = 0.0f;
    if (metal::abs(dir_x) > tiny) {
        gp_x = -c_x / dir_x;
        gr_x = -voxel_spacing * (t_min * c_x + ct_x) / dir_x;
    }
    if (metal::abs(dir_y) > tiny) {
        gp_y = -c_y / dir_y;
        gr_y = -voxel_spacing * (t_min * c_y + ct_y) / dir_y;
    }
    float radial = gr_x * dir_x + gr_y * dir_y;
    gr_x = (gr_x - radial * dir_x) / dir_norm;
    gr_y = (gr_y - radial * dir_y) / dir_norm;

    atomic_fetch_add_explicit(&grad_ray_dir[iang*2+0], g * gr_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_ray_dir[iang*2+1], g * gr_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_origin[iang*2+0], g * gp_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_origin[iang*2+1], g * gp_y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iang*2+0], g * u_phys * gp_x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_det_u_vec[iang*2+1], g * u_phys * gp_y, memory_order_relaxed);
"""

parallel_2d_geometry_grad_kernel = mx.fast.metal_kernel(
    name="parallel_2d_geometry_grad",
    input_names=["image", "cotangent", "ray_dir", "det_origin", "det_u_vec", "params", "fparams"],
    output_names=["grad_ray_dir", "grad_det_origin", "grad_det_u_vec"],
    source=_PARALLEL_2D_GEOMETRY_GRAD_SOURCE,
    atomic_outputs=True,
)
