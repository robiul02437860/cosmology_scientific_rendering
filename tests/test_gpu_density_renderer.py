from __future__ import annotations

from math import exp, pi

import pytest
import torch

from scientific_gsplat_renderer.gpu.density_renderer import (
    GpuDensityRenderResult,
    render_density_gpu,
)
from scientific_gsplat_renderer.gpu.orthographic_projection import (
    GpuProjectedGaussians,
)
from scientific_gsplat_renderer.gpu.tile_intersections import (
    build_gpu_tile_intersections,
)


def make_single_gaussian(
    *,
    device: str | torch.device,
    mass: float = 1.0,
) -> GpuProjectedGaussians:
    """Create one unit-covariance Gaussian centered on pixel (2, 2)."""

    resolved_device = torch.device(device)

    # Pixel (2, 2) has pixel-center coordinates (2.5, 2.5).
    means_pixel = torch.tensor(
        [[2.5, 2.5]],
        dtype=torch.float32,
        device=resolved_device,
    )

    means_camera = torch.tensor(
        [[0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=resolved_device,
    )

    covariances_camera = (
        torch.eye(
            3,
            dtype=torch.float32,
            device=resolved_device,
        )
        .unsqueeze(0)
        .contiguous()
    )

    covariances_pixel = (
        torch.eye(
            2,
            dtype=torch.float32,
            device=resolved_device,
        )
        .unsqueeze(0)
        .contiguous()
    )

    inverse_covariances_pixel = (
        covariances_pixel.clone()
    )

    # Four pixels comfortably cover the three-sigma support.
    radii_pixel = torch.tensor(
        [4.0],
        dtype=torch.float32,
        device=resolved_device,
    )

    radii_xy = torch.tensor(
        [[4, 4]],
        dtype=torch.int32,
        device=resolved_device,
    )

    depths = torch.tensor(
        [1.0],
        dtype=torch.float32,
        device=resolved_device,
    )

    masses = torch.tensor(
        [mass],
        dtype=torch.float32,
        device=resolved_device,
    )

    valid = torch.tensor(
        [True],
        dtype=torch.bool,
        device=resolved_device,
    )

    return GpuProjectedGaussians(
        means_camera=means_camera,
        means_pixel=means_pixel,
        covariances_camera=covariances_camera,
        covariances_pixel=covariances_pixel,
        inverse_covariances_pixel=(
            inverse_covariances_pixel
        ),
        radii_pixel=radii_pixel,
        radii_xy=radii_xy,
        depths=depths,
        masses=masses,
        valid=valid,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_single_gaussian_normalized_density() -> None:
    projected = make_single_gaussian(
        device="cuda",
        mass=1.0,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_density_gpu(
        projected,
        intersections,
        normalize_gaussian_mass=True,
    )

    assert isinstance(
        result,
        GpuDensityRenderResult,
    )

    assert result.device.type == "cuda"
    assert tuple(result.density.shape) == (5, 5)

    assert result.input_gaussians == 1
    assert result.valid_gaussians == 1
    assert result.intersections == 1

    assert torch.isfinite(
        result.density
    ).all()

    # For covariance I and mass 1:
    #
    # amplitude = 1 / (2*pi*sqrt(det(I)))
    #           = 1 / (2*pi)
    expected_center = 1.0 / (
        2.0 * pi
    )

    # One pixel to the right has delta=(1,0), so:
    #
    # density = amplitude * exp(-0.5)
    expected_right = (
        expected_center
        * exp(-0.5)
    )

    assert result.density[2, 2].item() == pytest.approx(
        expected_center,
        rel=1.0e-5,
        abs=1.0e-7,
    )

    assert result.density[2, 3].item() == pytest.approx(
        expected_right,
        rel=1.0e-5,
        abs=1.0e-7,
    )

    assert result.density[2, 1].item() == pytest.approx(
        expected_right,
        rel=1.0e-5,
        abs=1.0e-7,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_single_gaussian_unnormalized_density() -> None:
    projected = make_single_gaussian(
        device="cuda",
        mass=2.0,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_density_gpu(
        projected,
        intersections,
        normalize_gaussian_mass=False,
    )

    # Without mass normalization, the Gaussian mass is used directly as
    # its peak amplitude. Therefore the center value is exactly 2.
    assert result.density[2, 2].item() == pytest.approx(
        2.0,
        rel=1.0e-6,
        abs=1.0e-7,
    )

    assert result.density[2, 3].item() == pytest.approx(
        2.0 * exp(-0.5),
        rel=1.0e-5,
        abs=1.0e-7,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_density_is_spatially_symmetric() -> None:
    projected = make_single_gaussian(
        device="cuda",
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_density_gpu(
        projected,
        intersections,
    )

    density = result.density

    torch.testing.assert_close(
        density[2, 1],
        density[2, 3],
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    torch.testing.assert_close(
        density[1, 2],
        density[3, 2],
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    torch.testing.assert_close(
        density[1, 1],
        density[3, 3],
        rtol=1.0e-6,
        atol=1.0e-7,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_density_scales_linearly_with_mass() -> None:
    projected_one = make_single_gaussian(
        device="cuda",
        mass=1.0,
    )

    projected_three = make_single_gaussian(
        device="cuda",
        mass=3.0,
    )

    intersections_one = build_gpu_tile_intersections(
        projected_one,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    intersections_three = build_gpu_tile_intersections(
        projected_three,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result_one = render_density_gpu(
        projected_one,
        intersections_one,
    )

    result_three = render_density_gpu(
        projected_three,
        intersections_three,
    )

    torch.testing.assert_close(
        result_three.density,
        3.0 * result_one.density,
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_cpu_tensors_are_rejected() -> None:
    projected = make_single_gaussian(
        device="cpu",
    )

    # The empty-intersection construction works on CPU, allowing us to
    # verify that the renderer itself rejects non-CUDA input cleanly.
    projected.valid[:] = False

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    with pytest.raises(
        ValueError,
        match="requires CUDA tensors",
    ):
        render_density_gpu(
            projected,
            intersections,
        )