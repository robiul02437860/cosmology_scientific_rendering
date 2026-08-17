from __future__ import annotations

from math import exp, pi

import pytest
import torch

from scientific_gsplat_renderer.gpu import (
    GpuProjectedConditionalAttributes,
    GpuProjectedGaussians,
    build_gpu_tile_intersections,
    render_conditional_attribute_gpu,
)


def make_single_gaussian(
    *,
    device: str | torch.device,
    mass: float = 1.0,
    attribute_mean: float = 10.0,
    slope_x: float = 2.0,
    slope_y: float = 3.0,
) -> tuple[
    GpuProjectedGaussians,
    GpuProjectedConditionalAttributes,
]:
    resolved_device = torch.device(device)

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

    projected_valid = torch.tensor(
        [True],
        dtype=torch.bool,
        device=resolved_device,
    )

    projected = GpuProjectedGaussians(
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
        valid=projected_valid,
    )

    means = torch.tensor(
        [attribute_mean],
        dtype=torch.float32,
        device=resolved_device,
    )

    cross_covariances_camera = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=resolved_device,
    )

    cross_covariances_pixel = torch.zeros(
        (1, 2),
        dtype=torch.float32,
        device=resolved_device,
    )

    slopes_pixel = torch.tensor(
        [[slope_x, slope_y]],
        dtype=torch.float32,
        device=resolved_device,
    )

    conditional_valid = torch.tensor(
        [True],
        dtype=torch.bool,
        device=resolved_device,
    )

    conditional = GpuProjectedConditionalAttributes(
        means=means,
        cross_covariances_camera=(
            cross_covariances_camera
        ),
        cross_covariances_pixel=(
            cross_covariances_pixel
        ),
        slopes_pixel=slopes_pixel,
        valid=conditional_valid,
    )

    return projected, conditional


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_single_gaussian_conditional_gradient() -> None:
    projected, conditional = make_single_gaussian(
        device="cuda",
        attribute_mean=10.0,
        slope_x=2.0,
        slope_y=3.0,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        normalize_gaussian_mass=True,
        relative_density_threshold=0.0,
    )

    assert tuple(result.density.shape) == (5, 5)
    assert tuple(result.attribute.shape) == (5, 5)
    assert tuple(result.valid_mask.shape) == (5, 5)

    # Pixel centers are:
    #
    # center: (2.5, 2.5)
    # right : (3.5, 2.5)
    # left  : (1.5, 2.5)
    # down  : (2.5, 3.5)
    # up    : (2.5, 1.5)
    #
    # a(p) = 10 + 2*dx + 3*dy
    assert result.attribute[2, 2].item() == pytest.approx(
        10.0,
        rel=1.0e-6,
        abs=1.0e-6,
    )

    assert result.attribute[2, 3].item() == pytest.approx(
        12.0,
        rel=1.0e-6,
        abs=1.0e-6,
    )

    assert result.attribute[2, 1].item() == pytest.approx(
        8.0,
        rel=1.0e-6,
        abs=1.0e-6,
    )

    assert result.attribute[3, 2].item() == pytest.approx(
        13.0,
        rel=1.0e-6,
        abs=1.0e-6,
    )

    assert result.attribute[1, 2].item() == pytest.approx(
        7.0,
        rel=1.0e-6,
        abs=1.0e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_zero_slope_reproduces_attribute_mean() -> None:
    projected, conditional = make_single_gaussian(
        device="cuda",
        attribute_mean=12.0,
        slope_x=0.0,
        slope_y=0.0,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        relative_density_threshold=0.0,
    )

    valid_values = result.attribute[
        result.valid_mask
    ]

    torch.testing.assert_close(
        valid_values,
        torch.full_like(
            valid_values,
            12.0,
        ),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_density_matches_analytic_gaussian() -> None:
    projected, conditional = make_single_gaussian(
        device="cuda",
        mass=1.0,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        normalize_gaussian_mass=True,
        relative_density_threshold=0.0,
    )

    expected_center = 1.0 / (
        2.0 * pi
    )

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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_attribute_numerator_equals_density_times_attribute() -> None:
    projected, conditional = make_single_gaussian(
        device="cuda",
        attribute_mean=10.0,
        slope_x=2.0,
        slope_y=3.0,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        relative_density_threshold=0.0,
    )

    reconstructed = (
        result.density[result.valid_mask]
        * result.attribute[result.valid_mask]
    )

    torch.testing.assert_close(
        result.attribute_numerator[
            result.valid_mask
        ],
        reconstructed,
        rtol=1.0e-5,
        atol=1.0e-7,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_relative_density_threshold_masks_low_density_pixels() -> None:
    projected, conditional = make_single_gaussian(
        device="cuda",
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=5,
        image_height=5,
        tile_size=16,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        relative_density_threshold=0.9,
    )

    assert result.valid_mask[2, 2].item()

    assert int(
        result.valid_mask.sum().item()
    ) < 25

    assert torch.isnan(
        result.attribute[
            ~result.valid_mask
        ]
    ).all()


def test_cpu_tensors_are_rejected() -> None:
    projected, conditional = make_single_gaussian(
        device="cpu",
    )

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
        render_conditional_attribute_gpu(
            projected,
            conditional,
            intersections,
        )