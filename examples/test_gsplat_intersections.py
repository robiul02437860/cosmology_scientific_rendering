from __future__ import annotations

import math

import torch

from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    dtype = torch.float32

    image_width = 600
    image_height = 600
    tile_size = 16

    tile_width = math.ceil(image_width / tile_size)
    tile_height = math.ceil(image_height / tile_size)

    # One image containing three projected Gaussians.
    means2d = torch.tensor(
        [
            [
                [100.0, 100.0],
                [300.0, 300.0],
                [500.0, 400.0],
            ]
        ],
        device=device,
        dtype=dtype,
    )

    # gsplat 1.5.3 expects a radius for each image axis: [..., N, 2].
    radii = torch.tensor(
        [
            [
                [20, 20],
                [40, 30],
                [15, 25],
            ]
        ],
        device=device,
        dtype=torch.int32,
    )

    depths = torch.tensor(
        [[1.0, 2.0, 3.0]],
        device=device,
        dtype=dtype,
    )

    (
        tiles_per_gaussian,
        isect_ids,
        flatten_ids,
    ) = isect_tiles(
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

    torch.cuda.synchronize()

    print("Device:", device)
    print("Image size:", (image_width, image_height))
    print("Tile size:", tile_size)
    print("Tile grid:", (tile_width, tile_height))
    print()
    print("means2d shape:", tuple(means2d.shape))
    print("radii shape:", tuple(radii.shape))
    print("depths shape:", tuple(depths.shape))
    print()
    print("tiles_per_gaussian:")
    print(tiles_per_gaussian.cpu())
    print()
    print("Number of intersections:", isect_ids.numel())
    print("isect_ids shape:", tuple(isect_ids.shape))
    print("flatten_ids shape:", tuple(flatten_ids.shape))
    print("isect_offsets shape:", tuple(isect_offsets.shape))
    print()
    print("First intersection IDs:")
    print(isect_ids[:20].cpu())
    print()
    print("First flattened Gaussian IDs:")
    print(flatten_ids[:20].cpu())

    assert tiles_per_gaussian.shape == (1, 3)
    assert flatten_ids.ndim == 1
    assert flatten_ids.numel() == isect_ids.numel()
    assert isect_offsets.shape == (1, tile_height, tile_width)

    print()
    print("gsplat tile-intersection test: OK")


if __name__ == "__main__":
    main()