from __future__ import annotations

import numpy as np
import torch

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu import (
    GpuGaussianModel,
    project_gaussians_orthographic_gpu,
)


def make_model() -> GaussianModel:
    return GaussianModel(
        means=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        covariances=np.array(
            [
                np.diag([1.0, 4.0, 9.0]),
                np.diag([4.0, 1.0, 2.0]),
            ],
            dtype=np.float64,
        ),
        weights=np.array(
            [0.4, 0.6],
            dtype=np.float64,
        ),
        n_particles=100,
        attribute_means=np.array(
            [10.0, 20.0],
            dtype=np.float64,
        ),
        position_attribute_cross_covariances=np.zeros(
            (2, 3),
            dtype=np.float64,
        ),
        attribute_variances=np.ones(
            2,
            dtype=np.float64,
        ),
        attribute_name="test",
        box_size=10.0,
    )


def make_camera() -> OrthographicCamera:
    return OrthographicCamera(
        position=np.array(
            [0.0, 0.0, 10.0],
            dtype=np.float64,
        ),
        target=np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=10.0,
        image_width=100,
        image_height=100,
    )


def test_center_projects_to_image_center() -> None:
    model = make_model()
    camera = make_camera()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
    )

    torch.testing.assert_close(
        projected.means_pixel[0],
        torch.tensor(
            [50.0, 50.0],
            dtype=torch.float32,
        ),
    )


def test_orthographic_pixel_scaling() -> None:
    model = make_model()
    camera = make_camera()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
    )

    # 100 pixels / 10 world units = 10 pixels per world unit.
    #
    # World point (2, 1) becomes:
    #
    # x = 50 + 2*10 = 70
    # y = 50 - 1*10 = 40
    torch.testing.assert_close(
        projected.means_pixel[1],
        torch.tensor(
            [70.0, 40.0],
            dtype=torch.float32,
        ),
    )


def test_covariance_pixel_scaling() -> None:
    model = make_model()
    camera = make_camera()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        covariance_regularization=0.0,
    )

    # First covariance has world/camera XY variances [1, 4].
    # Pixel scale is 10, so variances scale by 10² = 100.
    expected = torch.tensor(
        [
            [100.0, 0.0],
            [0.0, 400.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        projected.covariances_pixel[0],
        expected,
    )


def test_inverse_covariance() -> None:
    model = make_model()
    camera = make_camera()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        covariance_regularization=0.0,
    )

    expected = torch.tensor(
        [
            [0.01, 0.0],
            [0.0, 0.0025],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        projected.inverse_covariances_pixel[0],
        expected,
    )


def test_radius_uses_largest_eigenvalue() -> None:
    model = make_model()
    camera = make_camera()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        support_sigma=3.0,
        covariance_regularization=0.0,
    )

    # Largest pixel-space variance is 400.
    # Standard deviation is 20 pixels.
    # Three-sigma radius is 60 pixels.
    torch.testing.assert_close(
        projected.radii_pixel[0],
        torch.tensor(
            60.0,
            dtype=torch.float32,
        ),
    )


def test_output_stays_on_model_device() -> None:
    model = make_model()
    camera = make_camera()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
    )

    assert projected.device == gpu_model.device
    assert projected.means_pixel.is_contiguous()
    assert projected.covariances_pixel.is_contiguous()
    assert projected.radii_xy.dtype == torch.int32
    assert projected.valid.dtype == torch.bool