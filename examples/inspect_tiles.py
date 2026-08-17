from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from scientific_gsplat_renderer import (
    GaussianModel,
    OrthographicCamera,
    build_tile_intersections,
    project_gaussians_orthographic,
)


MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/"
    "simple_model.npz"
)

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
TILE_SIZE = 16


def main() -> None:
    print("Loading model...")

    model = GaussianModel.load(
        MODEL_PATH
    )

    minimum = np.min(
        model.means,
        axis=0,
    )

    maximum = np.max(
        model.means,
        axis=0,
    )

    center = 0.5 * (
        minimum + maximum
    )

    extent = maximum - minimum

    view_width = float(
        max(
            extent[0],
            extent[1],
        )
        * 1.02
    )

    camera_distance = float(
        max(extent)
    )

    camera = OrthographicCamera(
        position=np.array(
            [
                center[0],
                center[1],
                maximum[2] + camera_distance,
            ],
            dtype=np.float64,
        ),
        target=center,
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=view_width,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        near=0.0,
        far=3.0 * camera_distance,
    )

    projected = project_gaussians_orthographic(
        model,
        camera,
        sigma_extent=4.0,
    )

    print("Building tile intersections...")

    start = perf_counter()

    intersections = build_tile_intersections(
        projected,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        tile_size=TILE_SIZE,
    )

    elapsed = perf_counter() - start

    counts = intersections.counts_per_tile

    print("=" * 72)
    print("Tile Intersections")
    print("=" * 72)

    print(
        f"Image size             : "
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )

    print(
        f"Tile size              : "
        f"{TILE_SIZE} x {TILE_SIZE}"
    )

    print(
        f"Tile grid              : "
        f"{intersections.tiles_x} x "
        f"{intersections.tiles_y}"
    )

    print(
        f"Number of tiles        : "
        f"{intersections.n_tiles:,}"
    )

    print(
        f"Valid Gaussians        : "
        f"{projected.n_valid:,}"
    )

    print(
        f"Tile intersections     : "
        f"{intersections.n_intersections:,}"
    )

    print(
        f"Mean Gaussians/tile    : "
        f"{intersections.mean_gaussians_per_tile:.3f}"
    )

    print(
        f"Maximum Gaussians/tile : "
        f"{intersections.maximum_gaussians_per_tile:,}"
    )

    print(
        f"Empty tiles            : "
        f"{np.count_nonzero(counts == 0):,}"
    )

    print(
        f"Build time             : "
        f"{elapsed:.6f} seconds"
    )

    print(
        f"Index storage          : "
        f"{intersections.gaussian_indices.nbytes / 1e6:.3f} MB"
    )

    print(
        f"Offset storage         : "
        f"{intersections.offsets.nbytes / 1e6:.3f} MB"
    )


if __name__ == "__main__":
    main()