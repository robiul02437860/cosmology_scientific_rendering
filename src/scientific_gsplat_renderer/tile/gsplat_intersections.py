from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from numpy.typing import NDArray

from ..projection.orthographic import ProjectedGaussians


IntArray = NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class GsplatTileIntersections:
    tiles_per_gaussian: torch.Tensor
    isect_ids: torch.Tensor
    flatten_ids: torch.Tensor
    isect_offsets: torch.Tensor

    tile_width: int
    tile_height: int
    tile_size: int

    @property
    def n_intersections(self) -> int:
        return int(self.flatten_ids.numel())


def build_gsplat_intersections(
    projected: ProjectedGaussians,
    *,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    device: torch.device | str = "cuda",
) -> GsplatTileIntersections:
    """
    Build Gaussian-to-tile intersections using gsplat's CUDA backend.

    The input projection is produced by our scientific orthographic
    projection code. gsplat is used only for tile binning and sorting.
    """
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    valid = np.asarray(projected.valid, dtype=bool)

    if valid.ndim != 1:
        raise ValueError(
            f"projected.valid must have shape (N,), got {valid.shape}"
        )

    valid_indices = np.flatnonzero(valid)

    tile_width = math.ceil(image_width / tile_size)
    tile_height = math.ceil(image_height / tile_size)

    if valid_indices.size == 0:
        empty_int32 = torch.empty(
            (1, 0),
            dtype=torch.int32,
            device=device,
        )

        empty_int64 = torch.empty(
            (0,),
            dtype=torch.int64,
            device=device,
        )

        offsets = torch.zeros(
            (1, tile_height, tile_width),
            dtype=torch.int32,
            device=device,
        )

        return GsplatTileIntersections(
            tiles_per_gaussian=empty_int32,
            isect_ids=empty_int64,
            flatten_ids=torch.empty(
                (0,),
                dtype=torch.int32,
                device=device,
            ),
            isect_offsets=offsets,
            tile_width=tile_width,
            tile_height=tile_height,
            tile_size=tile_size,
        )

    means_np = np.asarray(
        projected.means_pixel[valid_indices],
        dtype=np.float32,
    )

    covariances_np = np.asarray(
        projected.covariances_pixel[valid_indices],
        dtype=np.float32,
    )

    depths_np = np.asarray(
        projected.depths[valid_indices],
        dtype=np.float32,
    )

    if means_np.shape != (valid_indices.size, 2):
        raise ValueError(
            "means_pixel must have shape (N, 2); "
            f"got {means_np.shape}"
        )

    if covariances_np.shape != (valid_indices.size, 2, 2):
        raise ValueError(
            "covariances_pixel must have shape (N, 2, 2); "
            f"got {covariances_np.shape}"
        )

    if depths_np.shape != (valid_indices.size,):
        raise ValueError(
            "depths must have shape (N,); "
            f"got {depths_np.shape}"
        )

    means2d = torch.as_tensor(
        means_np,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    depths = torch.as_tensor(
        depths_np,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    covariances = torch.as_tensor(
        covariances_np,
        dtype=torch.float32,
        device=device,
    )

    variance_x = covariances[:, 0, 0].clamp_min(0.0)
    variance_y = covariances[:, 1, 1].clamp_min(0.0)

    # Use the same truncation already represented by projected.radii_pixel.
    #
    # If radii_pixel is a circular conservative radius, repeat it for x/y.
    # This guarantees no Gaussian support is accidentally excluded.
    radius_np = np.asarray(
        projected.radii_pixel[valid_indices],
        dtype=np.float32,
    )

    if radius_np.shape != (valid_indices.size,):
        raise ValueError(
            "radii_pixel must have shape (N,); "
            f"got {radius_np.shape}"
        )

    radius = torch.as_tensor(
        radius_np,
        dtype=torch.float32,
        device=device,
    )

    radii = torch.ceil(
        torch.stack((radius, radius), dim=-1)
    ).to(torch.int32)

    radii = radii.clamp_min(1).unsqueeze(0)

    tiles_per_gaussian, isect_ids, flatten_ids = isect_tiles(
        means2d=means2d,
        radii=radii,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
    )

    isect_offsets = isect_offset_encode(
        isect_ids=isect_ids,
        n_images=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )

    return GsplatTileIntersections(
        tiles_per_gaussian=tiles_per_gaussian,
        isect_ids=isect_ids,
        flatten_ids=flatten_ids,
        isect_offsets=isect_offsets,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_size=tile_size,
    )