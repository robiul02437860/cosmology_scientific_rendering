from __future__ import annotations

import numpy as np
import pytest
import torch

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu import (
    GpuGaussianModel,
    project_conditional_attributes_orthographic_gpu,
    project_gaussians_orthographic_gpu,
)
from scientific_gsplat_renderer.projection.conditional_attribute import (
    project_conditional_attributes_orthographic,
)
from scientific_gsplat_renderer.projection.orthographic import (
    project_gaussians_orthographic,
)


def make_model() -> GaussianModel:
    return GaussianModel(
        means=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, -1.0],
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
        position_attribute_cross_covariances=np.array(
            [
                [2.0, -3.0, 4.0],
                [-1.0, 5.0, 2.0],
            ],
            dtype=np.float64,
        ),
        attribute_variances=np.array(
            [16.0, 25.0],
            dtype=np.float64,
        ),
        attribute_name="test_attribute",
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
        near=0.0,
        far=20.0,
    )


def test_conditional_projection_shapes() -> None:
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

    conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
        )
    )

    assert tuple(conditional.means.shape) == (2,)
    assert tuple(
        conditional.cross_covariances_camera.shape
    ) == (2, 3)
    assert tuple(
        conditional.cross_covariances_pixel.shape
    ) == (2, 2)
    assert tuple(
        conditional.slopes_pixel.shape
    ) == (2, 2)
    assert tuple(conditional.valid.shape) == (2,)

    assert tuple(
        conditional.conditional_parameters.shape
    ) == (2, 3)


def test_attribute_means_are_preserved() -> None:
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

    conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
        )
    )

    torch.testing.assert_close(
        conditional.means,
        torch.tensor(
            [10.0, 20.0],
            dtype=torch.float32,
        ),
    )


def test_conditional_parameters_channel_layout() -> None:
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

    conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
        )
    )

    parameters = (
        conditional.conditional_parameters
    )

    torch.testing.assert_close(
        parameters[:, 0],
        conditional.means,
    )

    torch.testing.assert_close(
        parameters[:, 1:],
        conditional.slopes_pixel,
    )


def test_gpu_matches_cpu_conditional_projection() -> None:
    model = make_model()
    camera = make_camera()

    cpu_projected = project_gaussians_orthographic(
        model,
        camera,
        minimum_eigenvalue=1.0e-6,
        minimum_pixel_variance=1.0e-4,
        sigma_extent=3.0,
    )

    cpu_conditional = (
        project_conditional_attributes_orthographic(
            model,
            cpu_projected,
            camera,
        )
    )

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
        minimum_eigenvalue=1.0e-6,
    )

    gpu_projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        minimum_eigenvalue=1.0e-6,
        minimum_pixel_variance=1.0e-4,
        sigma_extent=3.0,
    )

    gpu_conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            gpu_projected,
            camera,
        )
    )

    torch.testing.assert_close(
        gpu_conditional.means,
        torch.as_tensor(
            cpu_conditional.means,
            dtype=torch.float32,
        ),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    torch.testing.assert_close(
        gpu_conditional.cross_covariances_pixel,
        torch.as_tensor(
            cpu_conditional.cross_covariances_pixel,
            dtype=torch.float32,
        ),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    torch.testing.assert_close(
        gpu_conditional.slopes_pixel,
        torch.as_tensor(
            cpu_conditional.slopes_pixel,
            dtype=torch.float32,
        ),
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    assert torch.equal(
        gpu_conditional.valid,
        torch.as_tensor(
            cpu_conditional.valid,
            dtype=torch.bool,
        ),
    )


def test_slope_limit() -> None:
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

    conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
            slope_limit=0.01,
        )
    )

    assert torch.all(
        conditional.slopes_pixel <= 0.01
    )

    assert torch.all(
        conditional.slopes_pixel >= -0.01
    )


def test_invalid_slope_limit() -> None:
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

    with pytest.raises(
        ValueError,
        match="slope_limit must be positive",
    ):
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
            slope_limit=0.0,
        )