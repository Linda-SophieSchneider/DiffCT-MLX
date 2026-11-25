"""Metal kernels for arbitrary rays: one thread per ray.

The flat-detector kernels in this package build every ray from per-view frames.
These kernels take each ray explicitly, so the same code serves curved
detectors (per-pixel positions), ``parallel3d`` beams and per-axis voxel
spacing:

* finite rays (fan, cone): ``A`` is the source, ``B`` the detector pixel,
* infinite rays (parallel, parallel3d): ``A`` is a point on the line and
  ``B`` the unit direction, in world coordinates.

The volume is ``(Nz, Ny, Nx)`` in C order (``vol[(iz*Ny + iy)*Nx + ix]``),
centred on the origin with per-axis pitches ``(sx, sy, sz)``. 2D problems use
``Nz = 1`` with zero z coordinates. Projections are physical line integrals.

The traversal is the forward kernels' Siddon walk in voxel coordinates, from
the endpoint nearer the volume centre for finite rays. The physical length of
a voxel-space step is ``|B - A| / |(B - A) / s|`` (finite) or ``1 / |r / s|``
(infinite).

The geometry VJP is the exact face-crossing derivative of the cell-constant
model (the engine's ``geometry_vjp`` kernels), written in world coordinates so
it holds for per-axis pitches: with ``tau`` the fraction along ``A -> B`` and
``c_j`` the value step at face ``j``,

    p = |B - A| * S,  S = sum_j c_j tau_j,
    dp/dA_a = -S d_a + sum_{faces on a} c (tau - 1) / d_a,
    dp/dB_a = +S d_a - sum_{faces on a} c tau / d_a,

with ``d = (B - A) / |B - A|``. For infinite rays ``p = sum_j c_j t_j`` with
``t`` the world parameter along the unit direction ``r``, so
``dp/dA_a = -C_a / r_a`` and ``dp/dr_a = -(sum_{faces on a} c t) / r_a``.
The gradient is returned per ray; MLX autodiff carries it back to the
trajectory and to detector-surface parameters.
"""

import mlx.core as mx

_EPSILON_STR = "1e-6f"
_INF_STR = "1e30f"
_TINY_STR = "1e-20f"

# Shared setup: voxel-space origin, unit direction, clipping, entry state.
# Defines ax..az (A in voxels), dir_x..dir_z (unit, voxel space), length
# (voxel length of the finite segment), factor (world length per voxel step),
# org, t_min/t_max, ent_axis/ex_axis, and for finite rays the parameters of A
# (t_a0) and B (t_b0).
_SETUP = """
    uint r = thread_position_in_grid.x;
    int n_rays = params[0];
    int Nx = params[1];
    int Ny = params[2];
    int Nz = params[3];
    int finite = params[4];
    float sx = fparams[0];
    float sy = fparams[1];
    float sz = fparams[2];
    if ((int)r >= n_rays) return;

    float eps = """ + _EPSILON_STR + """;
    float tiny = """ + _TINY_STR + """;
    float big = """ + _INF_STR + """;
    float cx = (float)Nx * 0.5f;
    float cy = (float)Ny * 0.5f;
    float cz = (float)Nz * 0.5f;

    float Ax = A[r * 3 + 0], Ay = A[r * 3 + 1], Az = A[r * 3 + 2];
    float Bx = B[r * 3 + 0], By = B[r * 3 + 1], Bz = B[r * 3 + 2];
    float ax = Ax / sx, ay = Ay / sy, az = Az / sz;

    float dir_x, dir_y, dir_z, length = 0.0f, factor, org_x, org_y, org_z;
    float t_min, t_max, t_a0 = 0.0f, t_b0 = 0.0f;
    float wd_x = 0.0f, wd_y = 0.0f, wd_z = 0.0f;   // world unit direction
    bool ok = true;
    if (finite) {
        float bx = Bx / sx, by = By / sy, bz = Bz / sz;
        dir_x = bx - ax; dir_y = by - ay; dir_z = bz - az;
        length = metal::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
        float wx = Bx - Ax, wy = By - Ay, wz = Bz - Az;
        float wlen = metal::sqrt(wx * wx + wy * wy + wz * wz);
        if (length < eps || wlen <= 0.0f) {
            ok = false;
            length = 1.0f; wlen = 1.0f;
        }
        factor = wlen / length;
        wd_x = wx / wlen; wd_y = wy / wlen; wd_z = wz / wlen;
        dir_x /= length; dir_y /= length; dir_z /= length;
        bool from_a = (ax * ax + ay * ay + az * az) <= (bx * bx + by * by + bz * bz);
        org_x = from_a ? ax : bx;
        org_y = from_a ? ay : by;
        org_z = from_a ? az : bz;
        t_min = from_a ? 0.0f : -length;
        t_max = from_a ? length : 0.0f;
        t_a0 = t_min;
        t_b0 = t_max;
    } else {
        wd_x = Bx; wd_y = By; wd_z = Bz;
        dir_x = Bx / sx; dir_y = By / sy; dir_z = Bz / sz;
        float n = metal::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
        if (n <= 0.0f) { ok = false; n = 1.0f; }
        factor = 1.0f / n;
        dir_x /= n; dir_y /= n; dir_z /= n;
        org_x = ax; org_y = ay; org_z = az;
        t_min = -big;
        t_max = big;
    }

    int ent_axis = -1;
    int ex_axis = -1;
    if (metal::abs(dir_x) > tiny) {
        float t1 = (-cx - org_x) / dir_x, t2 = (cx - org_x) / dir_x;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 0; }
        if (hi < t_max) { t_max = hi; ex_axis = 0; }
    } else if (org_x < -cx || org_x > cx) { ok = false; }
    if (metal::abs(dir_y) > tiny) {
        float t1 = (-cy - org_y) / dir_y, t2 = (cy - org_y) / dir_y;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 1; }
        if (hi < t_max) { t_max = hi; ex_axis = 1; }
    } else if (org_y < -cy || org_y > cy) { ok = false; }
    if (metal::abs(dir_z) > tiny) {
        float t1 = (-cz - org_z) / dir_z, t2 = (cz - org_z) / dir_z;
        float lo = metal::min(t1, t2), hi = metal::max(t1, t2);
        if (lo > t_min) { t_min = lo; ent_axis = 2; }
        if (hi < t_max) { t_max = hi; ex_axis = 2; }
    } else if (org_z < -cz || org_z > cz) { ok = false; }
    if (t_min >= t_max) ok = false;

    float ent_x = org_x + t_min * dir_x;
    float ent_y = org_y + t_min * dir_y;
    float ent_z = org_z + t_min * dir_z;
    float t_end = t_max - t_min;

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
"""

_STEP = """
        if (txn <= tyn && txn <= tzn) {
            t = metal::max(t, txn); pend_t = t; pend_axis = 0; ix += step_x; txn += dt_x;
        } else if (tyn <= txn && tyn <= tzn) {
            t = metal::max(t, tyn); pend_t = t; pend_axis = 1; iy += step_y; tyn += dt_y;
        } else {
            t = metal::max(t, tzn); pend_t = t; pend_axis = 2; iz += step_z; tzn += dt_z;
        }
"""

_RAY_FORWARD_SOURCE = _SETUP + """
    float accum = 0.0f;
    if (ok) {
        float t = 0.0f;
        float pend_t = 0.0f;
        int pend_axis = -1;
        while (t < t_end) {
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_end);
            float seg = t_next - t;
            if (seg > 0.0f && ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
                accum += vol[(iz * Ny + iy) * Nx + ix] * seg;
            }
""" + _STEP + """
        }
    }
    out[r] = accum * factor;
"""

_RAY_BACKWARD_SOURCE = _SETUP + """
    if (!ok) return;
    float w = y[r] * factor;
    if (w == 0.0f) return;
    float t = 0.0f;
    float pend_t = 0.0f;
    int pend_axis = -1;
    while (t < t_end) {
        float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_end);
        float seg = t_next - t;
        if (seg > 0.0f && ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
            atomic_fetch_add_explicit(&out[(iz * Ny + iy) * Nx + ix], w * seg, memory_order_relaxed);
        }
""" + _STEP + """
    }
"""

_RAY_GEOMETRY_GRAD_SOURCE = _SETUP + """
    float gA_x = 0.0f, gA_y = 0.0f, gA_z = 0.0f;
    float gB_x = 0.0f, gB_y = 0.0f, gB_z = 0.0f;
    float g = cotangent[r];
    if (ok && g != 0.0f) {
        float accum = 0.0f;
        float f_prev = 0.0f;
        int pend_axis = ent_axis;
        float pend_t = 0.0f;
        bool first = true;
        float t = 0.0f;
        // Per axis: C = sum of face coefficients c, CT = sum of c * value,
        // where value is tau (finite) or the local voxel parameter (infinite).
        float c_x = 0.0f, c_y = 0.0f, c_z = 0.0f;
        float ct_x = 0.0f, ct_y = 0.0f, ct_z = 0.0f;
        float inv_len = finite ? 1.0f / length : 0.0f;
        float entry_tau = finite ? (t_min - t_a0) * inv_len : 0.0f;
        while (t < t_end) {
            float f = 0.0f;
            if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
                f = vol[(iz * Ny + iy) * Nx + ix];
            }
            if (pend_axis >= 0) {
                float c = first ? -f : (f_prev - f);
                float v = finite ? entry_tau + pend_t * inv_len : pend_t;
                if (pend_axis == 0)      { c_x += c; ct_x += c * v; }
                else if (pend_axis == 1) { c_y += c; ct_y += c * v; }
                else                     { c_z += c; ct_z += c * v; }
            }
            first = false;
            float t_next = metal::min(metal::min(metal::min(txn, tyn), tzn), t_end);
            float seg = t_next - t;
            if (seg > 0.0f) accum += f * seg;
            f_prev = f;
""" + _STEP + """
        }
        if (ex_axis >= 0) {
            float v = finite ? entry_tau + t_end * inv_len : t_end;
            if (ex_axis == 0)      { c_x += f_prev; ct_x += f_prev * v; }
            else if (ex_axis == 1) { c_y += f_prev; ct_y += f_prev * v; }
            else                   { c_z += f_prev; ct_z += f_prev * v; }
        }
        if (finite) {
            // sum c (tau - 1) = CT - C and sum c tau = CT, divided by d_a.
            float S = accum * inv_len;
            if (metal::abs(wd_x) > tiny) { gA_x = (ct_x - c_x) / wd_x; gB_x = -ct_x / wd_x; }
            if (metal::abs(wd_y) > tiny) { gA_y = (ct_y - c_y) / wd_y; gB_y = -ct_y / wd_y; }
            if (metal::abs(wd_z) > tiny) { gA_z = (ct_z - c_z) / wd_z; gB_z = -ct_z / wd_z; }
            gA_x -= S * wd_x; gA_y -= S * wd_y; gA_z -= S * wd_z;
            gB_x += S * wd_x; gB_y += S * wd_y; gB_z += S * wd_z;
        } else {
            // World parameter t = factor * (t_min + local); r is the unit world direction.
            if (metal::abs(wd_x) > tiny) {
                gA_x = -c_x / wd_x;
                gB_x = -factor * (t_min * c_x + ct_x) / wd_x;
            }
            if (metal::abs(wd_y) > tiny) {
                gA_y = -c_y / wd_y;
                gB_y = -factor * (t_min * c_y + ct_y) / wd_y;
            }
            if (metal::abs(wd_z) > tiny) {
                gA_z = -c_z / wd_z;
                gB_z = -factor * (t_min * c_z + ct_z) / wd_z;
            }
        }
    }
    grad_A[r * 3 + 0] = g * gA_x; grad_A[r * 3 + 1] = g * gA_y; grad_A[r * 3 + 2] = g * gA_z;
    grad_B[r * 3 + 0] = g * gB_x; grad_B[r * 3 + 1] = g * gB_y; grad_B[r * 3 + 2] = g * gB_z;
"""

ray_forward_kernel = mx.fast.metal_kernel(
    name="ray_forward",
    input_names=["vol", "A", "B", "params", "fparams"],
    output_names=["out"],
    source=_RAY_FORWARD_SOURCE,
)

ray_backward_kernel = mx.fast.metal_kernel(
    name="ray_backward",
    input_names=["y", "A", "B", "params", "fparams"],
    output_names=["out"],
    source=_RAY_BACKWARD_SOURCE,
    atomic_outputs=True,
)

ray_geometry_grad_kernel = mx.fast.metal_kernel(
    name="ray_geometry_grad",
    input_names=["vol", "cotangent", "A", "B", "params", "fparams"],
    output_names=["grad_A", "grad_B"],
    source=_RAY_GEOMETRY_GRAD_SOURCE,
)
