from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from gsplat.cuda import _wrapper

from .orthographic_projection import GpuProjectedGaussians


@dataclass(slots=True)
class GpuTileIntersections:
    """Tile-intersection data produced by gsplat.

    Parameters
    ----------
    isect_ids
        Encoded and sorted tile-intersection identifiers returned by
        ``gsplat.cuda._wrapper.isect_tiles``.

    flatten_ids
        Original Gaussian indices for all tile intersections.

        A Gaussian appears multiple times when its projected support overlaps
        multiple tiles.

    isect_offsets
        Start offsets for each image tile. Shape:

        ``(1, tile_height, tile_width)``

        The leading dimension is the camera batch dimension. This renderer
        currently projects one camera at a time.

    valid_gaussian_ids
        Original indices of all Gaussians that were passed to gsplat after
        applying the projection validity mask.

    tile_size
        Width and height of each square tile in pixels.

    tile_width
        Number of horizontal tiles.

    tile_height
        Number of vertical tiles.

    image_width
        Output image width in pixels.

    image_height
        Output image height in pixels.
    """

    isect_ids: Tensor
    flatten_ids: Tensor
    isect_offsets: Tensor

    valid_gaussian_ids: Tensor

    tile_size: int
    tile_width: int
    tile_height: int

    image_width: int
    image_height: int

    @property
    def device(self) -> torch.device:
        """Return the device containing the intersection tensors."""

        return self.flatten_ids.device

    @property
    def n_tiles(self) -> int:
        """Return the total number of image tiles."""

        return self.tile_width * self.tile_height

    @property
    def n_intersections(self) -> int:
        """Return the total number of Gaussian-tile intersections."""

        return int(self.flatten_ids.numel())

    @property
    def n_valid_gaussians(self) -> int:
        """Return the number of Gaussians submitted to gsplat."""

        return int(self.valid_gaussian_ids.numel())

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the tile grid as ``(tile_width, tile_height)``."""

        return self.tile_width, self.tile_height


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Validate and return a positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive, got {value}."
        )

    return value


def _empty_intersections(
    *,
    device: torch.device,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    valid_gaussian_ids: Tensor,
) -> GpuTileIntersections:
    """Construct an empty tile-intersection result."""

    return GpuTileIntersections(
        isect_ids=torch.empty(
            (0,),
            dtype=torch.int64,
            device=device,
        ),
        flatten_ids=torch.empty(
            (0,),
            dtype=torch.int32,
            device=device,
        ),
        isect_offsets=torch.zeros(
            (1, tile_height, tile_width),
            dtype=torch.int32,
            device=device,
        ),
        valid_gaussian_ids=valid_gaussian_ids,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        image_width=image_width,
        image_height=image_height,
    )


def build_gpu_tile_intersections(
    projected: GpuProjectedGaussians,
    *,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    sort: bool = True,
) -> GpuTileIntersections:
    """Build Gaussian-tile intersections using gsplat.

    Only projected Gaussians marked as valid are submitted to gsplat.
    The Gaussian IDs returned by this function are then mapped back to the
    original indices in ``projected``.

    Parameters
    ----------
    projected
        GPU-resident orthographically projected Gaussians.

    image_width
        Output image width in pixels.

    image_height
        Output image height in pixels.

    tile_size
        Width and height of each square tile in pixels.

    sort
        Ask gsplat to sort intersections by tile and depth.

    Returns
    -------
    GpuTileIntersections
        Tile lists and offsets suitable for the custom additive CUDA
        rasterizer.
    """

    image_width = _validate_positive_integer(
        image_width,
        name="image_width",
    )

    image_height = _validate_positive_integer(
        image_height,
        name="image_height",
    )

    tile_size = _validate_positive_integer(
        tile_size,
        name="tile_size",
    )

    if not isinstance(sort, bool):
        raise TypeError(
            f"sort must be a bool, got {type(sort).__name__}."
        )

    if projected.means_pixel.ndim != 2:
        raise ValueError(
            "projected.means_pixel must have shape (N, 2), "
            f"got {tuple(projected.means_pixel.shape)}."
        )

    if projected.means_pixel.shape[1] != 2:
        raise ValueError(
            "projected.means_pixel must have shape (N, 2), "
            f"got {tuple(projected.means_pixel.shape)}."
        )

    n_gaussians = projected.n_gaussians

    expected_radii_shape = (
        n_gaussians,
        2,
    )

    if tuple(projected.radii_xy.shape) != expected_radii_shape:
        raise ValueError(
            "projected.radii_xy must have shape "
            f"{expected_radii_shape}, "
            f"got {tuple(projected.radii_xy.shape)}."
        )

    if tuple(projected.depths.shape) != (n_gaussians,):
        raise ValueError(
            "projected.depths must have shape "
            f"({n_gaussians},), "
            f"got {tuple(projected.depths.shape)}."
        )

    if tuple(projected.valid.shape) != (n_gaussians,):
        raise ValueError(
            "projected.valid must have shape "
            f"({n_gaussians},), "
            f"got {tuple(projected.valid.shape)}."
        )

    device = projected.device

    if projected.radii_xy.device != device:
        raise ValueError(
            "projected.radii_xy must be on the same device as "
            "projected.means_pixel."
        )

    if projected.depths.device != device:
        raise ValueError(
            "projected.depths must be on the same device as "
            "projected.means_pixel."
        )

    if projected.valid.device != device:
        raise ValueError(
            "projected.valid must be on the same device as "
            "projected.means_pixel."
        )

    tile_width = math.ceil(
        image_width / tile_size
    )

    tile_height = math.ceil(
        image_height / tile_size
    )

    # Original Gaussian indices that survived projection and image culling.
    valid_gaussian_ids = torch.nonzero(
        projected.valid,
        as_tuple=False,
    ).squeeze(-1)

    valid_gaussian_ids = valid_gaussian_ids.to(
        dtype=torch.int64
    ).contiguous()

    if valid_gaussian_ids.numel() == 0:
        return _empty_intersections(
            device=device,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            image_width=image_width,
            image_height=image_height,
            valid_gaussian_ids=valid_gaussian_ids,
        )

    # Select only valid Gaussians before invoking gsplat.
    means2d = projected.means_pixel.index_select(
        0,
        valid_gaussian_ids,
    )

    radii = projected.radii_xy.index_select(
        0,
        valid_gaussian_ids,
    )

    depths = projected.depths.index_select(
        0,
        valid_gaussian_ids,
    )

    # gsplat expects a camera/image batch dimension when packed=False:
    #
    # means2d : (C, N, 2)
    # radii   : (C, N, 2)
    # depths  : (C, N)
    #
    # This renderer currently uses C = 1.
    means2d = means2d.unsqueeze(
        0
    ).contiguous()

    radii = radii.to(
        dtype=torch.int32
    ).unsqueeze(
        0
    ).contiguous()

    depths = depths.unsqueeze(
        0
    ).contiguous()

    (
    tiles_per_gaussian,
    isect_ids,
    compact_flatten_ids,
    ) = _wrapper.isect_tiles(
        means2d=means2d,
        radii=radii,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=sort,
        segmented=False,
        packed=False,
    )

    isect_offsets = _wrapper.isect_offset_encode(
        isect_ids=isect_ids,
        n_images=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )

    _ = tiles_per_gaussian

    isect_ids = isect_ids.contiguous()
    compact_flatten_ids = (
        compact_flatten_ids.contiguous()
    )
    isect_offsets = isect_offsets.contiguous()

    # Because C=1 and packed=False, flatten IDs index the compact valid
    # Gaussian array. Convert them back to original projected Gaussian IDs.
    compact_indices = compact_flatten_ids.to(
        dtype=torch.int64
    )

    if compact_indices.numel() > 0:
        minimum_index = int(
            compact_indices.min().item()
        )

        maximum_index = int(
            compact_indices.max().item()
        )

        if minimum_index < 0:
            raise RuntimeError(
                "gsplat returned a negative Gaussian index: "
                f"{minimum_index}."
            )

        if maximum_index >= valid_gaussian_ids.numel():
            raise RuntimeError(
                "gsplat returned a Gaussian index outside the compact "
                "valid-Gaussian array: "
                f"maximum={maximum_index}, "
                f"valid_count={valid_gaussian_ids.numel()}."
            )

    flatten_ids = valid_gaussian_ids.index_select(
        0,
        compact_indices,
    ).to(
        dtype=torch.int32
    ).contiguous()

    expected_offset_shape = (
        1,
        tile_height,
        tile_width,
    )

    if tuple(isect_offsets.shape) != expected_offset_shape:
        raise RuntimeError(
            "Unexpected gsplat intersection-offset shape. "
            f"Expected {expected_offset_shape}, "
            f"got {tuple(isect_offsets.shape)}."
        )

    if isect_ids.numel() != flatten_ids.numel():
        raise RuntimeError(
            "gsplat returned inconsistent intersection arrays: "
            f"{isect_ids.numel()} intersection IDs and "
            f"{flatten_ids.numel()} Gaussian IDs."
        )

    return GpuTileIntersections(
        isect_ids=isect_ids,
        flatten_ids=flatten_ids,
        isect_offsets=isect_offsets,
        valid_gaussian_ids=valid_gaussian_ids,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        image_width=image_width,
        image_height=image_height,
    )