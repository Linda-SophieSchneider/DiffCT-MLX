"""PyTorch autograd functions for separable-footprint CT projections.

Footprint counterparts of the Siddon projectors in :mod:`diffct.projectors`,
sharing the same per-view arbitrary-trajectory calling convention so they are
drop-in replacements. Currently: 2D parallel beam (fan / cone follow).
"""

import torch

from .constants import _DTYPE
from .utils import (
    DeviceManager,
    TorchCUDABridge,
    _get_numba_external_stream_for,
    _grid_2d,
)
from .kernels.parallel_footprint import (
    _parallel_2d_footprint_forward_kernel,
    _parallel_2d_footprint_backward_kernel,
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

        ctx.save_for_backward(ray_dir, det_origin, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        ray_dir, det_origin, det_u_vec = ctx.saved_tensors
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
        return grad_image, None, None, None, None, None, None


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
