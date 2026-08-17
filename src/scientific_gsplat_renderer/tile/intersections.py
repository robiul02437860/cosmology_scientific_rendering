from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scientific_gsplat_renderer.projection import (
    ProjectedGaussians,
)


IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class TileIntersections:
    """Mapping between image tiles and projected Gaussians.

    The tile-intersection structure uses a compressed representation.

    For tile ``t``, its Gaussian indices are stored in:

    ``gaussian_indices[offsets[t] : offsets[t + 1]]``

    Attributes
    ----------
    tile_size
        Width and height of each square tile in pixels.
    tiles_x
        Number of horizontal tiles.
    tiles_y
        Number of vertical tiles.
    offsets
        Start and end offsets for every tile, shape ``(n_tiles + 1,)``.
    gaussian_indices
        Flattened Gaussian indices for all tile intersections.
    """

    tile_size: int
    tiles_x: int
    tiles_y: int
    offsets: IntArray
    gaussian_indices: IntArray

    def __post_init__(self) -> None:
        self._validate()

    @property
    def n_tiles(self) -> int:
        """Return the total number of image tiles."""

        return self.tiles_x * self.tiles_y

    @property
    def n_intersections(self) -> int:
        """Return the number of Gaussian-tile intersections."""

        return int(self.gaussian_indices.size)

    @property
    def maximum_gaussians_per_tile(self) -> int:
        """Return the maximum number of Gaussians assigned to one tile."""

        counts = self.counts_per_tile

        if counts.size == 0:
            return 0

        return int(np.max(counts))

    @property
    def counts_per_tile(self) -> IntArray:
        """Return the number of Gaussian intersections per tile."""

        return np.diff(self.offsets)

    @property
    def mean_gaussians_per_tile(self) -> float:
        """Return the mean number of Gaussians assigned to a tile."""

        if self.n_tiles == 0:
            return 0.0

        return (
            float(self.n_intersections)
            / float(self.n_tiles)
        )

    def tile_id(
        self,
        tile_x: int,
        tile_y: int,
    ) -> int:
        """Return the flattened ID of a tile coordinate."""

        if isinstance(tile_x, bool) or not isinstance(
            tile_x,
            int,
        ):
            raise TypeError(
                "tile_x must be an integer"
            )

        if isinstance(tile_y, bool) or not isinstance(
            tile_y,
            int,
        ):
            raise TypeError(
                "tile_y must be an integer"
            )

        if not 0 <= tile_x < self.tiles_x:
            raise ValueError(
                f"tile_x must lie in [0, {self.tiles_x}), "
                f"got {tile_x}"
            )

        if not 0 <= tile_y < self.tiles_y:
            raise ValueError(
                f"tile_y must lie in [0, {self.tiles_y}), "
                f"got {tile_y}"
            )

        return (
            tile_y * self.tiles_x
            + tile_x
        )

    def gaussian_indices_for_tile(
        self,
        tile_x: int,
        tile_y: int,
    ) -> IntArray:
        """Return Gaussian indices intersecting one tile."""

        tile = self.tile_id(
            tile_x,
            tile_y,
        )

        start = int(self.offsets[tile])
        stop = int(self.offsets[tile + 1])

        return self.gaussian_indices[
            start:stop
        ]

    def _validate(self) -> None:
        for name, value in {
            "tile_size": self.tile_size,
            "tiles_x": self.tiles_x,
            "tiles_y": self.tiles_y,
        }.items():
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be positive, got {value}"
                )

        if self.offsets.ndim != 1:
            raise ValueError(
                "offsets must have shape (n_tiles + 1,), "
                f"got {self.offsets.shape}"
            )

        expected_offsets = self.n_tiles + 1

        if self.offsets.shape != (
            expected_offsets,
        ):
            raise ValueError(
                "offsets must have shape "
                f"({expected_offsets},), "
                f"got {self.offsets.shape}"
            )

        if self.gaussian_indices.ndim != 1:
            raise ValueError(
                "gaussian_indices must be one-dimensional, "
                f"got {self.gaussian_indices.shape}"
            )

        if not np.issubdtype(
            self.offsets.dtype,
            np.integer,
        ):
            raise TypeError(
                "offsets must have an integer dtype"
            )

        if not np.issubdtype(
            self.gaussian_indices.dtype,
            np.integer,
        ):
            raise TypeError(
                "gaussian_indices must have an integer dtype"
            )

        if int(self.offsets[0]) != 0:
            raise ValueError(
                "offsets must begin with zero"
            )

        if np.any(
            np.diff(self.offsets) < 0
        ):
            raise ValueError(
                "offsets must be nondecreasing"
            )

        if int(self.offsets[-1]) != int(
            self.gaussian_indices.size
        ):
            raise ValueError(
                "The final offset must equal the number "
                "of Gaussian indices"
            )

        if np.any(
            self.gaussian_indices < 0
        ):
            raise ValueError(
                "gaussian_indices must be nonnegative"
            )


def build_tile_intersections(
    projected: ProjectedGaussians,
    *,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    maximum_gaussians: int | None = None,
) -> TileIntersections:
    """Build Gaussian-to-tile intersections.

    A Gaussian is assigned to every tile touched by its conservative
    circular screen-space bounding box.

    Parameters
    ----------
    projected
        Projected Gaussian data.
    image_width
        Image width in pixels.
    image_height
        Image height in pixels.
    tile_size
        Width and height of every square tile.
    maximum_gaussians
        Optional limit on the number of valid Gaussians processed.

    Returns
    -------
    TileIntersections
        Compressed tile-to-Gaussian mapping.
    """

    _validate_build_parameters(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        maximum_gaussians=maximum_gaussians,
    )

    tiles_x = (
        image_width + tile_size - 1
    ) // tile_size

    tiles_y = (
        image_height + tile_size - 1
    ) // tile_size

    n_tiles = tiles_x * tiles_y

    tile_lists: list[list[int]] = [
        []
        for _ in range(n_tiles)
    ]

    valid_indices = np.flatnonzero(
        projected.valid
    )

    if (
        maximum_gaussians is not None
        and len(valid_indices) > maximum_gaussians
    ):
        selection_positions = np.linspace(
            0,
            len(valid_indices) - 1,
            num=maximum_gaussians,
            dtype=np.int64,
        )

        valid_indices = valid_indices[
            selection_positions
        ]

    for index_value in valid_indices:
        index = int(index_value)

        center_x = float(
            projected.means_pixel[index, 0]
        )

        center_y = float(
            projected.means_pixel[index, 1]
        )

        radius = float(
            projected.radii_pixel[index]
        )

        minimum_pixel_x = max(
            0,
            int(
                np.floor(
                    center_x - radius
                )
            ),
        )

        maximum_pixel_x = min(
            image_width - 1,
            int(
                np.ceil(
                    center_x + radius
                )
            ),
        )

        minimum_pixel_y = max(
            0,
            int(
                np.floor(
                    center_y - radius
                )
            ),
        )

        maximum_pixel_y = min(
            image_height - 1,
            int(
                np.ceil(
                    center_y + radius
                )
            ),
        )

        if minimum_pixel_x > maximum_pixel_x:
            continue

        if minimum_pixel_y > maximum_pixel_y:
            continue

        minimum_tile_x = (
            minimum_pixel_x // tile_size
        )

        maximum_tile_x = (
            maximum_pixel_x // tile_size
        )

        minimum_tile_y = (
            minimum_pixel_y // tile_size
        )

        maximum_tile_y = (
            maximum_pixel_y // tile_size
        )

        for tile_y in range(
            minimum_tile_y,
            maximum_tile_y + 1,
        ):
            row_offset = (
                tile_y * tiles_x
            )

            for tile_x in range(
                minimum_tile_x,
                maximum_tile_x + 1,
            ):
                tile = (
                    row_offset + tile_x
                )

                tile_lists[tile].append(
                    index
                )

    counts = np.fromiter(
        (
            len(indices)
            for indices in tile_lists
        ),
        dtype=np.int64,
        count=n_tiles,
    )

    offsets = np.empty(
        n_tiles + 1,
        dtype=np.int64,
    )

    offsets[0] = 0

    np.cumsum(
        counts,
        out=offsets[1:],
    )

    gaussian_indices = np.empty(
        int(offsets[-1]),
        dtype=np.int64,
    )

    cursor = 0

    for indices in tile_lists:
        next_cursor = (
            cursor + len(indices)
        )

        if indices:
            gaussian_indices[
                cursor:next_cursor
            ] = indices

        cursor = next_cursor

    return TileIntersections(
        tile_size=tile_size,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        offsets=offsets,
        gaussian_indices=gaussian_indices,
    )


def _validate_build_parameters(
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    maximum_gaussians: int | None,
) -> None:
    """Validate tile-intersection parameters."""

    for name, value in {
        "image_width": image_width,
        "image_height": image_height,
        "tile_size": tile_size,
    }.items():
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be positive, got {value}"
            )

    if maximum_gaussians is None:
        return

    if isinstance(
        maximum_gaussians,
        bool,
    ) or not isinstance(
        maximum_gaussians,
        int,
    ):
        raise TypeError(
            "maximum_gaussians must be an integer or None"
        )

    if maximum_gaussians <= 0:
        raise ValueError(
            "maximum_gaussians must be positive, "
            f"got {maximum_gaussians}"
        )