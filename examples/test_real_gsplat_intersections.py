from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import torch

from scientific_gsplat_renderer import (
    GaussianModel,
    OrthographicCamera,
    project_gaussians_orthographic,
)
from scientific_gsplat_renderer.tile import build_gsplat_intersections


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "hacc_mpicosmo_full/full_speed_ratio_0_001/simple_model.npz"
)

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
TILE_SIZE = 16
VIEW_PADDING = 0.02


def create_camera(model: GaussianModel) -> OrthographicCamera:
    """
    Create a front-facing orthographic camera containing the full model.
    """

    means = np.asarray(model.means, dtype=np.float64)

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(
            f"Model means must have shape (N, 3), got {means.shape}."
        )

    model_minimum = means.min(axis=0)
    model_maximum = means.max(axis=0)

    center = 0.5 * (model_minimum + model_maximum)
    extent = model_maximum - model_minimum

    view_width = float(max(extent[0], extent[1]))
    view_width *= 1.0 + VIEW_PADDING

    camera_distance = max(
        float(extent[2]),
        view_width,
        1.0,
    )

    position = center + np.array(
        [0.0, 0.0, camera_distance],
        dtype=np.float64,
    )

    return OrthographicCamera(
        position=position,
        target=center,
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=view_width,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )


def project_model(
    model: GaussianModel,
    camera: OrthographicCamera,
):
    """
    Call project_gaussians_orthographic using the image-size arguments
    supported by the installed project version.
    """

    signature = inspect.signature(
        project_gaussians_orthographic
    )

    parameter_names = set(signature.parameters)

    kwargs: dict[str, int] = {}

    if "image_width" in parameter_names:
        kwargs["image_width"] = IMAGE_WIDTH

    if "image_height" in parameter_names:
        kwargs["image_height"] = IMAGE_HEIGHT

    if "width" in parameter_names:
        kwargs["width"] = IMAGE_WIDTH

    if "height" in parameter_names:
        kwargs["height"] = IMAGE_HEIGHT

    if "image_size" in parameter_names:
        kwargs["image_size"] = (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        )

    print(
        "Projection signature:",
        signature,
    )

    print(
        "Projection image arguments:",
        kwargs if kwargs else "none",
    )

    return project_gaussians_orthographic(
        model,
        camera,
        **kwargs,
    )


def main() -> None:
    print("=" * 80)
    print("Real-model gsplat tile-intersection test")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script on a CUDA-capable machine."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist:\n{MODEL_PATH}"
        )

    device = torch.device("cuda")

    print("Device:", torch.cuda.get_device_name(device))
    print("Model:", MODEL_PATH)
    print("Image size:", (IMAGE_WIDTH, IMAGE_HEIGHT))
    print("Tile size:", TILE_SIZE)

    # -----------------------------------------------------------------
    # 1. Load the model
    # -----------------------------------------------------------------

    print()
    print("Loading model...")

    start = time.perf_counter()
    model = GaussianModel.load(MODEL_PATH)
    load_seconds = time.perf_counter() - start

    print("Gaussians:", f"{model.n_gaussians:,}")
    print("Load time:", f"{load_seconds:.6f} seconds")

    # -----------------------------------------------------------------
    # 2. Create the camera
    # -----------------------------------------------------------------

    print()
    print("Creating camera...")

    camera = create_camera(model)

    print(
        "OrthographicCamera signature:",
        inspect.signature(OrthographicCamera),
    )
    print("Camera position:", camera.position)
    print("Camera target:", camera.target)
    print("Camera up:", camera.up)
    print("View width:", camera.view_width)
    print("Image size:", (IMAGE_WIDTH, IMAGE_HEIGHT))

    # -----------------------------------------------------------------
    # 3. Project the model
    # -----------------------------------------------------------------

    print()
    print("Projecting Gaussians...")

    start = time.perf_counter()
    projected = project_model(model, camera)
    projection_seconds = time.perf_counter() - start

    valid = np.asarray(projected.valid, dtype=bool)

    valid_count = int(np.count_nonzero(valid))
    invalid_count = int(projected.n_gaussians - valid_count)

    print("Projected Gaussians:", f"{projected.n_gaussians:,}")
    print("Valid Gaussians:", f"{valid_count:,}")
    print("Invalid Gaussians:", f"{invalid_count:,}")
    print(
        "Projection time:",
        f"{projection_seconds:.6f} seconds",
    )

    if valid_count == 0:
        raise RuntimeError(
            "Orthographic projection produced no valid Gaussians."
        )

    valid_radii = np.asarray(
        projected.radii_pixel[valid],
        dtype=np.float64,
    )

    print(
        "Pixel radius minimum:",
        f"{valid_radii.min():.6f}",
    )
    print(
        "Pixel radius mean:",
        f"{valid_radii.mean():.6f}",
    )
    print(
        "Pixel radius maximum:",
        f"{valid_radii.max():.6f}",
    )

    # -----------------------------------------------------------------
    # 4. Build gsplat intersections
    # -----------------------------------------------------------------

    print()
    print("Building gsplat tile intersections...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start = time.perf_counter()

    intersections = build_gsplat_intersections(
        projected,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        tile_size=TILE_SIZE,
        device=device,
    )

    torch.cuda.synchronize(device)
    intersection_seconds = time.perf_counter() - start

    # -----------------------------------------------------------------
    # 5. Calculate statistics
    # -----------------------------------------------------------------

    total_intersections = intersections.n_intersections

    tiles_per_gaussian = (
        intersections.tiles_per_gaussian
        .detach()
        .reshape(-1)
    )

    if tiles_per_gaussian.numel() > 0:
        minimum_tiles = int(
            tiles_per_gaussian.min().item()
        )
        maximum_tiles = int(
            tiles_per_gaussian.max().item()
        )
        mean_tiles = float(
            tiles_per_gaussian
            .to(torch.float32)
            .mean()
            .item()
        )
        counted_intersections = int(
            tiles_per_gaussian
            .to(torch.int64)
            .sum()
            .item()
        )
    else:
        minimum_tiles = 0
        maximum_tiles = 0
        mean_tiles = 0.0
        counted_intersections = 0

    current_memory_mib = (
        torch.cuda.memory_allocated(device) / 1024**2
    )

    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
    )

    # -----------------------------------------------------------------
    # 6. Print results
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("Intersection results")
    print("=" * 80)

    print(
        "Tile grid:",
        (
            intersections.tile_width,
            intersections.tile_height,
        ),
    )

    print(
        "Number of tiles:",
        intersections.tile_width
        * intersections.tile_height,
    )

    print(
        "Total intersections:",
        f"{total_intersections:,}",
    )

    print(
        "Average tiles per Gaussian:",
        f"{mean_tiles:.6f}",
    )

    print(
        "Minimum tiles per Gaussian:",
        minimum_tiles,
    )

    print(
        "Maximum tiles per Gaussian:",
        maximum_tiles,
    )

    print(
        "tiles_per_gaussian shape:",
        tuple(
            intersections.tiles_per_gaussian.shape
        ),
    )

    print(
        "isect_ids shape:",
        tuple(intersections.isect_ids.shape),
    )

    print(
        "flatten_ids shape:",
        tuple(intersections.flatten_ids.shape),
    )

    print(
        "isect_offsets shape:",
        tuple(intersections.isect_offsets.shape),
    )

    print(
        "Intersection construction time:",
        f"{intersection_seconds:.6f} seconds",
    )

    print(
        "Current CUDA memory:",
        f"{current_memory_mib:.2f} MiB",
    )

    print(
        "Peak CUDA memory:",
        f"{peak_memory_mib:.2f} MiB",
    )

    # -----------------------------------------------------------------
    # 7. Consistency checks
    # -----------------------------------------------------------------

    expected_offset_shape = (
        1,
        intersections.tile_height,
        intersections.tile_width,
    )

    if tuple(intersections.isect_offsets.shape) != expected_offset_shape:
        raise RuntimeError(
            "Unexpected isect_offsets shape: "
            f"{tuple(intersections.isect_offsets.shape)}; "
            f"expected {expected_offset_shape}."
        )

    if intersections.isect_ids.numel() != total_intersections:
        raise RuntimeError(
            "isect_ids count does not match the reported "
            "intersection count."
        )

    if intersections.flatten_ids.numel() != total_intersections:
        raise RuntimeError(
            "flatten_ids count does not match the reported "
            "intersection count."
        )

    if counted_intersections != total_intersections:
        raise RuntimeError(
            "The sum of tiles_per_gaussian does not match the "
            "number of intersections: "
            f"{counted_intersections:,} != "
            f"{total_intersections:,}."
        )

    if intersections.flatten_ids.numel() > 0:
        minimum_flatten_id = int(
            intersections.flatten_ids.min().item()
        )
        maximum_flatten_id = int(
            intersections.flatten_ids.max().item()
        )

        if minimum_flatten_id < 0:
            raise RuntimeError(
                "flatten_ids contains a negative Gaussian index."
            )

        if maximum_flatten_id >= valid_count:
            raise RuntimeError(
                "flatten_ids contains a Gaussian index outside "
                "the valid projected-Gaussian array: "
                f"{maximum_flatten_id} >= {valid_count}."
            )

    print()
    print("All consistency checks passed.")
    print("Real gsplat tile-intersection test: OK")


if __name__ == "__main__":
    main()