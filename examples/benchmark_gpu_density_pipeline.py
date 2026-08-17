from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu import (
    GpuGaussianModel,
    build_gpu_tile_intersections,
    project_gaussians_orthographic_gpu,
    render_density_gpu,
)


MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
TILE_SIZE = 16

MINIMUM_EIGENVALUE = 1.0e-6
MINIMUM_PIXEL_VARIANCE = 0.25
SIGMA_EXTENT = 3.0

WARMUP_FRAMES = 5
TIMED_FRAMES = 20


def build_camera(
    model: GaussianModel,
) -> OrthographicCamera:
    minimum = np.asarray(
        model.means.min(axis=0),
        dtype=np.float64,
    )

    maximum = np.asarray(
        model.means.max(axis=0),
        dtype=np.float64,
    )

    center = 0.5 * (
        minimum + maximum
    )

    span = maximum - minimum

    view_width = float(
        max(
            float(span[0]),
            float(span[1]),
        )
        * 1.02
    )

    camera_distance = float(
        span.max() * 2.0
    )

    return OrthographicCamera(
        position=np.array(
            [
                center[0],
                center[1],
                center[2] + camera_distance,
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
        far=2.0 * camera_distance,
    )


def make_event_pair() -> tuple[
    torch.cuda.Event,
    torch.cuda.Event,
]:
    return (
        torch.cuda.Event(
            enable_timing=True,
        ),
        torch.cuda.Event(
            enable_timing=True,
        ),
    )


def elapsed_seconds(
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> float:
    return (
        start.elapsed_time(end)
        / 1000.0
    )


def render_one_frame(
    gpu_model: GpuGaussianModel,
    camera: OrthographicCamera,
):
    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        minimum_eigenvalue=MINIMUM_EIGENVALUE,
        minimum_pixel_variance=MINIMUM_PIXEL_VARIANCE,
        sigma_extent=SIGMA_EXTENT,
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        tile_size=TILE_SIZE,
    )

    density_result = render_density_gpu(
        projected,
        intersections,
        normalize_gaussian_mass=True,
    )

    return (
        projected,
        intersections,
        density_result,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    print("=" * 80)
    print("Real-model GPU density pipeline benchmark")
    print("=" * 80)

    load_start = perf_counter()

    model = GaussianModel.load(
        MODEL_PATH
    )

    load_seconds = (
        perf_counter()
        - load_start
    )

    camera = build_camera(
        model
    )

    torch.cuda.synchronize(
        device
    )

    upload_start = perf_counter()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device=device,
    )

    gpu_model.synchronize()

    upload_seconds = (
        perf_counter()
        - upload_start
    )

    print(f"Model path              : {MODEL_PATH}")
    print(f"Gaussians               : {gpu_model.n_gaussians:,}")
    print(f"Particles represented   : {gpu_model.n_particles:,}")
    print(f"Image size              : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Tile size               : {TILE_SIZE}")
    print(f"Minimum pixel variance  : {MINIMUM_PIXEL_VARIANCE}")
    print(f"Sigma extent            : {SIGMA_EXTENT}")
    print(f"GPU                     : {torch.cuda.get_device_name(device)}")
    print()
    print(f"Model load              : {load_seconds:.6f} s")
    print(f"One-time GPU upload     : {upload_seconds:.6f} s")

    print()
    print(
        f"Warming up {WARMUP_FRAMES} frames..."
    )

    projected = None
    intersections = None
    density_result = None

    for _ in range(WARMUP_FRAMES):
        (
            projected,
            intersections,
            density_result,
        ) = render_one_frame(
            gpu_model,
            camera,
        )

    torch.cuda.synchronize(
        device
    )

    projection_start, projection_end = (
        make_event_pair()
    )

    intersection_start, intersection_end = (
        make_event_pair()
    )

    frame_start, frame_end = (
        make_event_pair()
    )

    projection_total = 0.0
    intersection_total = 0.0
    preparation_total = 0.0
    rasterization_total = 0.0
    frame_total = 0.0

    print(
        f"Timing {TIMED_FRAMES} frames..."
    )

    for _ in range(TIMED_FRAMES):
        frame_start.record()

        projection_start.record()

        projected = project_gaussians_orthographic_gpu(
            gpu_model,
            camera,
            minimum_eigenvalue=MINIMUM_EIGENVALUE,
            minimum_pixel_variance=MINIMUM_PIXEL_VARIANCE,
            sigma_extent=SIGMA_EXTENT,
        )

        projection_end.record()
        projection_end.synchronize()

        projection_total += elapsed_seconds(
            projection_start,
            projection_end,
        )

        intersection_start.record()

        intersections = build_gpu_tile_intersections(
            projected,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            tile_size=TILE_SIZE,
        )

        intersection_end.record()
        intersection_end.synchronize()

        intersection_total += elapsed_seconds(
            intersection_start,
            intersection_end,
        )

        density_result = render_density_gpu(
            projected,
            intersections,
            normalize_gaussian_mass=True,
        )

        preparation_total += (
            density_result.preparation_seconds
        )

        rasterization_total += (
            density_result.rasterization_seconds
        )

        frame_end.record()
        frame_end.synchronize()

        frame_total += elapsed_seconds(
            frame_start,
            frame_end,
        )

    if (
        projected is None
        or intersections is None
        or density_result is None
    ):
        raise RuntimeError(
            "Benchmark produced no frame."
        )

    projection_average = (
        projection_total
        / TIMED_FRAMES
    )

    intersection_average = (
        intersection_total
        / TIMED_FRAMES
    )

    preparation_average = (
        preparation_total
        / TIMED_FRAMES
    )

    rasterization_average = (
        rasterization_total
        / TIMED_FRAMES
    )

    frame_average = (
        frame_total
        / TIMED_FRAMES
    )

    fps = (
        1.0 / frame_average
        if frame_average > 0.0
        else float("inf")
    )

    density = density_result.density

    print()
    print("=" * 80)
    print("Average per-frame timings")
    print("=" * 80)

    print(
        f"GPU projection          : "
        f"{projection_average * 1000.0:.3f} ms"
    )

    print(
        f"Tile intersections      : "
        f"{intersection_average * 1000.0:.3f} ms"
    )

    print(
        f"Density preparation     : "
        f"{preparation_average * 1000.0:.3f} ms"
    )

    print(
        f"Density rasterization   : "
        f"{rasterization_average * 1000.0:.3f} ms"
    )

    print(
        f"Complete GPU frame      : "
        f"{frame_average * 1000.0:.3f} ms"
    )

    print(
        f"Frames per second       : "
        f"{fps:.2f}"
    )

    print()
    print("Final-frame statistics")
    print("-" * 80)

    print(
        f"Valid Gaussians         : "
        f"{projected.n_valid:,}"
    )

    print(
        f"Tile intersections      : "
        f"{intersections.n_intersections:,}"
    )

    print(
        f"Density shape           : "
        f"{tuple(density.shape)}"
    )

    print(
        f"Density minimum         : "
        f"{density.min().item():.12g}"
    )

    print(
        f"Density maximum         : "
        f"{density.max().item():.12g}"
    )

    print(
        f"Density mean            : "
        f"{density.mean().item():.12g}"
    )

    print(
        f"Density sum             : "
        f"{density.sum().item():.12g}"
    )


if __name__ == "__main__":
    main()