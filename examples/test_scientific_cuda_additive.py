from __future__ import annotations

import torch

from gsplat.cuda import _wrapper


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    # One image, two identical Gaussians centered on one pixel.
    #
    # For the center pixel:
    #   sigma = 0
    #   exp(-sigma) = 1
    #
    # Therefore:
    #   accumulated weight = 2 + 3 = 5
    #   rendered value     = 10*2 + 20*3 = 80
    means2d = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.float32,
        device=device,
    )

    # conic = [A, B, C]
    # sigma = 0.5 * (A dx^2 + C dy^2) + B dx dy
    conics = torch.tensor(
        [[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )

    colors = torch.tensor(
        [[[10.0], [20.0]]],
        dtype=torch.float32,
        device=device,
    )

    opacities = torch.tensor(
        [[2.0, 3.0]],
        dtype=torch.float32,
        device=device,
    )

    tile_offsets = torch.tensor(
        [[[0]]],
        dtype=torch.int32,
        device=device,
    )

    flatten_ids = torch.tensor(
        [0, 1],
        dtype=torch.int32,
        device=device,
    )

    scientific_fwd = _wrapper._make_lazy_cuda_func(
        "rasterize_to_pixels_scientific_fwd"
    )

    renders, weights, last_ids = scientific_fwd(
        means2d,
        conics,
        colors,
        opacities,
        None,          # backgrounds
        None,          # masks
        1,             # image_width
        1,             # image_height
        1,             # tile_size
        tile_offsets,
        flatten_ids,
    )

    torch.cuda.synchronize()

    rendered_value = float(renders[0, 0, 0, 0].item())
    accumulated_weight = float(weights[0, 0, 0].item())
    last_id = int(last_ids[0, 0, 0].item())

    print("renders:", renders)
    print("weights:", weights)
    print("last_ids:", last_ids)

    print()
    print("Rendered value:", rendered_value)
    print("Expected value: 80.0")

    print("Accumulated weight:", accumulated_weight)
    print("Expected weight: 5.0")

    print("Last intersection ID:", last_id)
    print("Expected last ID: 1")

    torch.testing.assert_close(
        renders,
        torch.tensor([[[[80.0]]]], device=device),
    )

    torch.testing.assert_close(
    weights,
    torch.tensor([[[[5.0]]]], device=device),
    )

    assert last_id == 1

    print()
    print("Scientific additive CUDA test: PASSED")


if __name__ == "__main__":
    main()