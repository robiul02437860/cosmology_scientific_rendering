from __future__ import annotations

import pytest
import torch

from scientific_gsplat_renderer.gpu.orthographic_projection import (
    GpuProjectedGaussians,
)
from scientific_gsplat_renderer.gpu.tile_intersections import (
    GpuTileIntersections,
    build_gpu_tile_intersections,
)


def make_projected(
    *,
    device: str | torch.device,
) -> GpuProjectedGaussians:
    resolved_device = torch.device(device)

    means_camera = torch.tensor(
        [
            [8.0, 8.0, 1.0],
            [24.0, 8.0, 2.0],
            [8.0, 24.0, 3.0],
        ],
        dtype=torch.float32,
        device=resolved_device,
    )

    means_pixel = torch.tensor(
        [
            [8.0, 8.0],
            [24.0, 8.0],
            [8.0, 24.0],
        ],
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
        .repeat(3, 1, 1)
        .contiguous()
    )

    covariances_pixel = (
        torch.eye(
            2,
            dtype=torch.float32,
            device=resolved_device,
        )
        .unsqueeze(0)
        .repeat(3, 1, 1)
        .contiguous()
    )

    inverse_covariances_pixel = (
        covariances_pixel.clone()
    )

    # Each Gaussian fits inside one 16x16 tile:
    #
    # Gaussian 0 -> tile (0, 0)
    # Gaussian 1 -> tile (1, 0)
    # Gaussian 2 -> tile (0, 1)
    radii_pixel = torch.tensor(
        [2.0, 2.0, 2.0],
        dtype=torch.float32,
        device=resolved_device,
    )

    radii_integer = torch.tensor(
        [2, 2, 2],
        dtype=torch.int32,
        device=resolved_device,
    )

    radii_xy = torch.stack(
        (
            radii_integer,
            radii_integer,
        ),
        dim=-1,
    )

    depths = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
        device=resolved_device,
    )

    masses = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
        device=resolved_device,
    )

    valid = torch.tensor(
        [True, True, True],
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
def test_build_gpu_tile_intersections() -> None:
    projected = make_projected(
        device="cuda",
    )

    result = build_gpu_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    assert isinstance(
        result,
        GpuTileIntersections,
    )

    assert result.device.type == "cuda"

    assert result.tile_size == 16
    assert result.tile_width == 2
    assert result.tile_height == 2
    assert result.n_tiles == 4

    assert result.image_width == 32
    assert result.image_height == 32

    assert result.n_valid_gaussians == 3
    assert result.n_intersections == 3

    assert tuple(
        result.isect_offsets.shape
    ) == (1, 2, 2)

    assert result.flatten_ids.dtype == torch.int32
    assert result.valid_gaussian_ids.dtype == torch.int64

    returned_ids = set(
        result.flatten_ids
        .detach()
        .cpu()
        .tolist()
    )

    assert returned_ids == {
        0,
        1,
        2,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_invalid_gaussian_is_filtered() -> None:
    projected = make_projected(
        device="cuda",
    )

    projected.valid[1] = False

    result = build_gpu_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    assert result.n_valid_gaussians == 2
    assert result.n_intersections == 2

    returned_ids = set(
        result.flatten_ids
        .detach()
        .cpu()
        .tolist()
    )

    assert returned_ids == {
        0,
        2,
    }

    valid_ids = (
        result.valid_gaussian_ids
        .detach()
        .cpu()
        .tolist()
    )

    assert valid_ids == [
        0,
        2,
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_empty_valid_mask() -> None:
    projected = make_projected(
        device="cuda",
    )

    projected.valid[:] = False

    result = build_gpu_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    assert result.n_valid_gaussians == 0
    assert result.n_intersections == 0

    assert result.isect_ids.numel() == 0
    assert result.flatten_ids.numel() == 0

    assert tuple(
        result.isect_offsets.shape
    ) == (1, 2, 2)

    assert torch.count_nonzero(
        result.isect_offsets
    ).item() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_large_gaussian_intersects_multiple_tiles() -> None:
    projected = make_projected(
        device="cuda",
    )

    # Keep only Gaussian 0 and give it support extending across all four
    # tiles of the 32x32 image.
    projected.valid[:] = False
    projected.valid[0] = True

    projected.means_pixel[0] = torch.tensor(
        [16.0, 16.0],
        dtype=torch.float32,
        device="cuda",
    )

    projected.radii_pixel[0] = 15.0

    projected.radii_xy[0] = torch.tensor(
        [15, 15],
        dtype=torch.int32,
        device="cuda",
    )

    result = build_gpu_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    assert result.n_valid_gaussians == 1
    assert result.n_intersections == 4

    returned_ids = (
        result.flatten_ids
        .detach()
        .cpu()
        .tolist()
    )

    assert returned_ids == [
        0,
        0,
        0,
        0,
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_non_divisible_image_size() -> None:
    projected = make_projected(
        device="cuda",
    )

    result = build_gpu_tile_intersections(
        projected,
        image_width=35,
        image_height=33,
        tile_size=16,
    )

    assert result.tile_width == 3
    assert result.tile_height == 3
    assert result.n_tiles == 9

    assert tuple(
        result.isect_offsets.shape
    ) == (1, 3, 3)


def test_invalid_tile_size() -> None:
    projected = make_projected(
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="tile_size must be positive",
    ):
        build_gpu_tile_intersections(
            projected,
            image_width=32,
            image_height=32,
            tile_size=0,
        )