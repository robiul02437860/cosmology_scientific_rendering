from __future__ import annotations

import numpy as np
import pytest
import torch

from scientific_gsplat_renderer.data.gaussian_model import GaussianModel
from scientific_gsplat_renderer.gpu import GpuGaussianModel


def make_model() -> GaussianModel:
    return GaussianModel(
        means=np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        ),
        covariances=np.array(
            [
                np.eye(3),
                2.0 * np.eye(3),
            ],
            dtype=np.float64,
        ),
        weights=np.array(
            [0.25, 0.75],
            dtype=np.float64,
        ),
        n_particles=100,
        attribute_means=np.array(
            [10.0, 20.0],
            dtype=np.float64,
        ),
        position_attribute_cross_covariances=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float64,
        ),
        attribute_variances=np.array(
            [4.0, 9.0],
            dtype=np.float64,
        ),
        attribute_name="test_attribute",
        box_size=100.0,
    )


def test_gpu_model_cpu_device() -> None:
    model = make_model()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    assert gpu_model.device.type == "cpu"
    assert gpu_model.dtype == torch.float32
    assert gpu_model.n_gaussians == 2

    torch.testing.assert_close(
        gpu_model.means,
        torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        gpu_model.masses,
        torch.tensor(
            [25.0, 75.0],
            dtype=torch.float32,
        ),
    )


def test_gpu_model_metadata() -> None:
    model = make_model()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    assert gpu_model.n_particles == 100
    assert gpu_model.attribute_name == "test_attribute"
    assert gpu_model.box_size == 100.0


def test_gpu_model_tensors_are_contiguous() -> None:
    model = make_model()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
    )

    assert gpu_model.means.is_contiguous()
    assert gpu_model.covariances.is_contiguous()
    assert gpu_model.weights.is_contiguous()
    assert gpu_model.masses.is_contiguous()
    assert gpu_model.attribute_means.is_contiguous()
    assert (
        gpu_model.position_attribute_cross_covariances.is_contiguous()
    )
    assert gpu_model.attribute_variances.is_contiguous()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_gpu_model_cuda_device() -> None:
    model = make_model()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cuda",
    )

    assert gpu_model.device.type == "cuda"
    assert gpu_model.dtype == torch.float32
    assert gpu_model.n_gaussians == 2

    torch.testing.assert_close(
        gpu_model.masses.cpu(),
        torch.tensor(
            [25.0, 75.0],
            dtype=torch.float32,
        ),
    )


def test_stabilized_covariances_are_cached() -> None:
    model = make_model()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
        minimum_eigenvalue=1.0e-6,
    )

    assert tuple(
        gpu_model.stabilized_covariances.shape
    ) == (2, 3, 3)

    assert (
        gpu_model.stabilized_covariances
        .is_contiguous()
    )

    eigenvalues = torch.linalg.eigvalsh(
        gpu_model.stabilized_covariances
    )

    assert torch.all(
        eigenvalues >= 1.0e-6
    )


def test_stabilization_threshold_must_match() -> None:
    model = make_model()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cpu",
        minimum_eigenvalue=1.0e-6,
    )

    gpu_model.require_stabilization(
        1.0e-6
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        gpu_model.require_stabilization(
            1.0e-4
        )