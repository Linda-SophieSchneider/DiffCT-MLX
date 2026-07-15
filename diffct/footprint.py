"""PyTorch autograd functions for separable-footprint CT projections.

Footprint counterparts of the Siddon projectors in :mod:`diffct.projectors`,
sharing the same per-view arbitrary-trajectory calling convention so they are
drop-in replacements. Currently: 2D parallel beam (fan / cone follow).
"""

import torch

from .constants import _DTYPE
from .projectors import (
    _FD_REL_EPS_POS,
    _FD_REL_EPS_VEC,
    _fd_geometry_grads,
)
from .utils import (
    DeviceManager,
    TorchCUDABridge,
    _get_numba_external_stream_for,
    _grid_2d,
    _grid_3d,
)
from .kernels.parallel_footprint import (
    _parallel_2d_footprint_forward_kernel,
    _parallel_2d_footprint_backward_kernel,
)
from .kernels.fan_footprint import (
    _fan_2d_footprint_forward_kernel,
    _fan_2d_footprint_backward_kernel,
)
from .kernels.cone_footprint import (
    _cone_3d_footprint_forward_kernel,
    _cone_3d_footprint_backward_kernel,
    _cone_3d_footprint_backward_sparse_kernel,
)


def _to_f32_contig(t):
    return t.to(dtype=torch.float32).contiguous()


class ParallelFootprintProjectorFunction(torch.autograd.Function):
    """Differentiable 2D parallel-beam separable-footprint forward projection."""

    @staticmethod
    def forward(ctx, image, ray_dir, det_origin, det_u_vec, num_detectors,
                detector_spacing=1.0, voxel_spacing=1.0):
        device = DeviceManager.get_device(image)
        image = _to_f32_contig(DeviceManager.ensure_device(image, device))
        ray_dir = _to_f32_contig(DeviceManager.ensure_device(ray_dir, device))
        det_origin = _to_f32_contig(DeviceManager.ensure_device(det_origin, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        Ny, Nx = image.shape
        n_views = ray_dir.shape[0]
        sinogram = torch.zeros((n_views, num_detectors), dtype=image.dtype, device=device)

        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_ray = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_do = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, Ny)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _parallel_2d_footprint_forward_kernel[grid, tpb, numba_stream](
            d_image, Nx, Ny, d_sino, n_views, num_detectors,
            _DTYPE(detector_spacing), d_ray, d_do, d_du, cx, cy, _DTYPE(voxel_spacing)
        )

        # image saved for the (optional) finite-difference geometry gradients.
        ctx.save_for_backward(image, ray_dir, det_origin, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        image, ray_dir, det_origin, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, Ny, Nx, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = _to_f32_contig(DeviceManager.ensure_device(grad_sinogram, device))
        ray_dir = _to_f32_contig(DeviceManager.ensure_device(ray_dir, device))
        det_origin = _to_f32_contig(DeviceManager.ensure_device(det_origin, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        n_views = ray_dir.shape[0]
        grad_image = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_ray = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_do = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(Nx, Ny)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _parallel_2d_footprint_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_views, num_detectors, d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_ray, d_do, d_du, cx, cy, _DTYPE(voxel_spacing)
        )

        grad_ray = grad_do = grad_duv = None
        if any(ctx.needs_input_grad[1:4]):
            def _launch(ray_d, det_o, det_uv):
                with torch.no_grad():
                    return ParallelFootprintProjectorFunction.apply(
                        image, ray_d, det_o, det_uv,
                        num_detectors, detector_spacing, voxel_spacing)

            grad_ray, grad_do, grad_duv = _fd_geometry_grads(
                _launch, [ray_dir, det_origin, det_u_vec],
                ctx.needs_input_grad[1:4], grad_sinogram,
                [_FD_REL_EPS_VEC, _FD_REL_EPS_POS, _FD_REL_EPS_VEC])

        return grad_image, grad_ray, grad_do, grad_duv, None, None, None


class ParallelFootprintBackprojectorFunction(torch.autograd.Function):
    """Differentiable 2D parallel-beam separable-footprint backprojection (adjoint)."""

    @staticmethod
    def forward(ctx, sinogram, ray_dir, det_origin, det_u_vec,
                detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
        device = DeviceManager.get_device(sinogram)
        sinogram = _to_f32_contig(DeviceManager.ensure_device(sinogram, device))
        ray_dir = _to_f32_contig(DeviceManager.ensure_device(ray_dir, device))
        det_origin = _to_f32_contig(DeviceManager.ensure_device(det_origin, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        n_views, num_detectors = sinogram.shape
        image = torch.zeros((H, W), dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_ray = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_do = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(W, H)
        cx, cy = _DTYPE(W * 0.5), _DTYPE(H * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _parallel_2d_footprint_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_views, num_detectors, d_image, W, H,
            _DTYPE(detector_spacing), d_ray, d_do, d_du, cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(ray_dir, det_origin, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, H, W, voxel_spacing)
        return image

    @staticmethod
    def backward(ctx, grad_image):
        ray_dir, det_origin, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, H, W, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_image)
        grad_image = _to_f32_contig(DeviceManager.ensure_device(grad_image, device))
        ray_dir = _to_f32_contig(DeviceManager.ensure_device(ray_dir, device))
        det_origin = _to_f32_contig(DeviceManager.ensure_device(det_origin, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        n_views = ray_dir.shape[0]
        grad_sino = torch.zeros((n_views, num_detectors), dtype=grad_image.dtype, device=device)

        d_grad_image = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_ray = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_do = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, H)
        cx, cy = _DTYPE(W * 0.5), _DTYPE(H * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _parallel_2d_footprint_forward_kernel[grid, tpb, numba_stream](
            d_grad_image, W, H, d_grad_sino, n_views, num_detectors,
            _DTYPE(detector_spacing), d_ray, d_do, d_du, cx, cy, _DTYPE(voxel_spacing)
        )
        return grad_sino, None, None, None, None, None, None, None


class FanFootprintProjectorFunction(torch.autograd.Function):
    """Differentiable 2D fan-beam separable-footprint forward projection."""

    @staticmethod
    def forward(ctx, image, src_pos, det_center, det_u_vec, num_detectors,
                detector_spacing=1.0, voxel_spacing=1.0):
        device = DeviceManager.get_device(image)
        image = _to_f32_contig(DeviceManager.ensure_device(image, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        Ny, Nx = image.shape
        n_views = src_pos.shape[0]
        sinogram = torch.zeros((n_views, num_detectors), dtype=image.dtype, device=device)

        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, Ny)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _fan_2d_footprint_forward_kernel[grid, tpb, numba_stream](
            d_image, Nx, Ny, d_sino, n_views, num_detectors,
            _DTYPE(detector_spacing), d_src, d_dc, d_du, cx, cy, _DTYPE(voxel_spacing)
        )

        # image saved for the (optional) finite-difference geometry gradients.
        ctx.save_for_backward(image, src_pos, det_center, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        image, src_pos, det_center, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, Ny, Nx, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = _to_f32_contig(DeviceManager.ensure_device(grad_sinogram, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        n_views = src_pos.shape[0]
        grad_image = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(Nx, Ny)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _fan_2d_footprint_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_views, num_detectors, d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_src, d_dc, d_du, cx, cy, _DTYPE(voxel_spacing)
        )
        grad_src = grad_dc = grad_duv = None
        if any(ctx.needs_input_grad[1:4]):
            def _launch(src_p, det_c, det_uv):
                with torch.no_grad():
                    return FanFootprintProjectorFunction.apply(
                        image, src_p, det_c, det_uv,
                        num_detectors, detector_spacing, voxel_spacing)

            grad_src, grad_dc, grad_duv = _fd_geometry_grads(
                _launch, [src_pos, det_center, det_u_vec],
                ctx.needs_input_grad[1:4], grad_sinogram,
                [_FD_REL_EPS_POS, _FD_REL_EPS_POS, _FD_REL_EPS_VEC])

        return grad_image, grad_src, grad_dc, grad_duv, None, None, None


class FanFootprintBackprojectorFunction(torch.autograd.Function):
    """Differentiable 2D fan-beam separable-footprint backprojection (adjoint)."""

    @staticmethod
    def forward(ctx, sinogram, src_pos, det_center, det_u_vec,
                detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
        device = DeviceManager.get_device(sinogram)
        sinogram = _to_f32_contig(DeviceManager.ensure_device(sinogram, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        n_views, num_detectors = sinogram.shape
        image = torch.zeros((H, W), dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(W, H)
        cx, cy = _DTYPE(W * 0.5), _DTYPE(H * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _fan_2d_footprint_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_views, num_detectors, d_image, W, H,
            _DTYPE(detector_spacing), d_src, d_dc, d_du, cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(src_pos, det_center, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, H, W, voxel_spacing)
        return image

    @staticmethod
    def backward(ctx, grad_image):
        src_pos, det_center, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, H, W, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_image)
        grad_image = _to_f32_contig(DeviceManager.ensure_device(grad_image, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))

        n_views = src_pos.shape[0]
        grad_sino = torch.zeros((n_views, num_detectors), dtype=grad_image.dtype, device=device)

        d_grad_image = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, H)
        cx, cy = _DTYPE(W * 0.5), _DTYPE(H * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _fan_2d_footprint_forward_kernel[grid, tpb, numba_stream](
            d_grad_image, W, H, d_grad_sino, n_views, num_detectors,
            _DTYPE(detector_spacing), d_src, d_dc, d_du, cx, cy, _DTYPE(voxel_spacing)
        )
        return grad_sino, None, None, None, None, None, None, None


class ConeFootprintProjectorFunction(torch.autograd.Function):
    """Differentiable 3D cone-beam separable-footprint forward projection."""

    @staticmethod
    def forward(ctx, volume, src_pos, det_center, det_u_vec, det_v_vec,
                det_u, det_v, du, dv, voxel_spacing=1.0):
        device = DeviceManager.get_device(volume)
        volume = _to_f32_contig(DeviceManager.ensure_device(volume, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))
        det_v_vec = _to_f32_contig(DeviceManager.ensure_device(det_v_vec, device))

        D, H, W = volume.shape
        n_views = src_pos.shape[0]
        sino = torch.zeros((n_views, det_u, det_v), dtype=volume.dtype, device=device)

        volume_perm = volume.permute(2, 1, 0).contiguous()  # (W, H, D) = (Nx, Ny, Nz)
        d_vol = TorchCUDABridge.tensor_to_cuda_array(volume_perm)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sino)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_dv = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(n_views, H, D)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _cone_3d_footprint_forward_kernel[grid, tpb, numba_stream](
            d_vol, W, H, D, d_sino, n_views, det_u, det_v,
            _DTYPE(du), _DTYPE(dv), d_src, d_dc, d_du, d_dv,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        # volume saved for the (optional) finite-difference geometry gradients.
        ctx.save_for_backward(volume, src_pos, det_center, det_u_vec, det_v_vec)
        ctx.intermediate = (D, H, W, det_u, det_v, du, dv, voxel_spacing)
        return sino

    @staticmethod
    def backward(ctx, grad_sino):
        volume, src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        D, H, W, det_u, det_v, du, dv, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_sino)
        grad_sino = _to_f32_contig(DeviceManager.ensure_device(grad_sino, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))
        det_v_vec = _to_f32_contig(DeviceManager.ensure_device(det_v_vec, device))

        n_views = src_pos.shape[0]
        grad_vol_perm = torch.zeros((W, H, D), dtype=grad_sino.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_grad_vol = TorchCUDABridge.tensor_to_cuda_array(grad_vol_perm)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_dv = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(W, H, D)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _cone_3d_footprint_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_views, det_u, det_v, d_grad_vol, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_src, d_dc, d_du, d_dv,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )
        grad_volume = grad_vol_perm.permute(2, 1, 0).contiguous()  # (D, H, W)
        grad_src = grad_dc = grad_duv = grad_dvv = None
        if any(ctx.needs_input_grad[1:5]):
            # Finite-difference geometry VJPs of the footprint forward model
            # (the separable footprint is piecewise-smooth in the geometry, so
            # central differences give well-behaved smoothed gradients).
            def _launch(src_p, det_c, det_uv, det_vv):
                with torch.no_grad():
                    return ConeFootprintProjectorFunction.apply(
                        volume, src_p, det_c, det_uv, det_vv,
                        det_u, det_v, du, dv, voxel_spacing)

            grad_src, grad_dc, grad_duv, grad_dvv = _fd_geometry_grads(
                _launch, [src_pos, det_center, det_u_vec, det_v_vec],
                ctx.needs_input_grad[1:5], grad_sino,
                [_FD_REL_EPS_POS, _FD_REL_EPS_POS, _FD_REL_EPS_VEC, _FD_REL_EPS_VEC])

        return grad_volume, grad_src, grad_dc, grad_duv, grad_dvv, None, None, None, None, None


class ConeFootprintBackprojectorFunction(torch.autograd.Function):
    """Differentiable 3D cone-beam separable-footprint backprojection (adjoint)."""

    @staticmethod
    def forward(ctx, sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                D, H, W, du, dv, voxel_spacing=1.0):
        device = DeviceManager.get_device(sinogram)
        sinogram = _to_f32_contig(DeviceManager.ensure_device(sinogram, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))
        det_v_vec = _to_f32_contig(DeviceManager.ensure_device(det_v_vec, device))

        n_views, n_u, n_v = sinogram.shape
        vol_perm = torch.zeros((W, H, D), dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_vol = TorchCUDABridge.tensor_to_cuda_array(vol_perm)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_dv = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(W, H, D)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _cone_3d_footprint_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_u, n_v, d_vol, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_src, d_dc, d_du, d_dv,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )
        reco = vol_perm.permute(2, 1, 0).contiguous()  # (D, H, W)

        ctx.save_for_backward(src_pos, det_center, det_u_vec, det_v_vec)
        ctx.intermediate = (D, H, W, n_u, n_v, du, dv, voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_volume):
        src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        D, H, W, n_u, n_v, du, dv, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_volume)
        grad_volume = _to_f32_contig(DeviceManager.ensure_device(grad_volume, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))
        det_v_vec = _to_f32_contig(DeviceManager.ensure_device(det_v_vec, device))

        n_views = src_pos.shape[0]
        grad_vol_perm = grad_volume.permute(2, 1, 0).contiguous()  # (W, H, D)
        grad_sino = torch.zeros((n_views, n_u, n_v), dtype=grad_volume.dtype, device=device)

        d_grad_vol = TorchCUDABridge.tensor_to_cuda_array(grad_vol_perm)
        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_dv = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(n_views, H, D)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _cone_3d_footprint_forward_kernel[grid, tpb, numba_stream](
            d_grad_vol, W, H, D, d_grad_sino, n_views, n_u, n_v,
            _DTYPE(du), _DTYPE(dv), d_src, d_dc, d_du, d_dv,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )
        return grad_sino, None, None, None, None, None, None, None, None, None, None


class ConeFootprintBackprojectorSparseFunction(torch.autograd.Function):
    """Sparse cone-beam footprint backprojection.

    Evaluates the footprint adjoint only at the flattened (D, H, W) C-order voxel
    ``indices`` and returns a 1D vector (one value per requested voxel, in the
    given order). The gradient scatters the cotangent back to a dense volume and
    forward-projects it, matching the dense operator's adjoint.
    """

    @staticmethod
    def forward(ctx, sinogram, indices, src_pos, det_center, det_u_vec, det_v_vec,
                D, H, W, du, dv, voxel_spacing=1.0):
        device = DeviceManager.get_device(sinogram)
        sinogram = _to_f32_contig(DeviceManager.ensure_device(sinogram, device))
        src_pos = _to_f32_contig(DeviceManager.ensure_device(src_pos, device))
        det_center = _to_f32_contig(DeviceManager.ensure_device(det_center, device))
        det_u_vec = _to_f32_contig(DeviceManager.ensure_device(det_u_vec, device))
        det_v_vec = _to_f32_contig(DeviceManager.ensure_device(det_v_vec, device))
        n_voxels = int(D) * int(H) * int(W)
        if n_voxels > 2**31 - 1:
            raise ValueError(
                f"sparse cone backprojection: volume has {n_voxels} voxels, "
                "which exceeds int32 flat indexing (max ~1290^3). Chunk the "
                "volume (e.g. diffct_mlx.orchestration) instead."
            )
        indices = DeviceManager.ensure_device(indices, device).to(torch.int32).contiguous()

        n_views, n_u, n_v = sinogram.shape
        n_samples = int(indices.shape[0])
        out = torch.zeros((n_samples,), dtype=sinogram.dtype, device=device)
        if n_samples == 0:
            ctx.save_for_backward(indices, src_pos, det_center, det_u_vec, det_v_vec)
            ctx.intermediate = (D, H, W, n_u, n_v, du, dv, voxel_spacing)
            return out

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_out = TorchCUDABridge.tensor_to_cuda_array(out)
        d_idx = TorchCUDABridge.tensor_to_cuda_array(indices)
        d_src = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_dc = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_du = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_dv = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        tpb = 128
        blocks = (n_samples + tpb - 1) // tpb
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)
        numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream())
        _cone_3d_footprint_backward_sparse_kernel[blocks, tpb, numba_stream](
            d_sino, n_views, n_u, n_v, d_out, n_samples, d_idx, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_src, d_dc, d_du, d_dv,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(indices, src_pos, det_center, det_u_vec, det_v_vec)
        ctx.intermediate = (D, H, W, n_u, n_v, du, dv, voxel_spacing)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        indices, src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        D, H, W, n_u, n_v, du, dv, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_out)
        grad_out = _to_f32_contig(DeviceManager.ensure_device(grad_out, device))

        # Scatter the sparse cotangent back to a dense (D, H, W) volume, then
        # forward-project it to obtain the sinogram gradient.
        dense_flat = torch.zeros(D * H * W, dtype=grad_out.dtype, device=device)
        dense_flat.index_add_(0, indices.to(torch.long), grad_out)
        dense = dense_flat.view(D, H, W)

        grad_sino = ConeFootprintProjectorFunction.apply(
            dense, src_pos, det_center, det_u_vec, det_v_vec,
            n_u, n_v, du, dv, voxel_spacing
        )
        return (grad_sino, None, None, None, None, None,
                None, None, None, None, None, None)
