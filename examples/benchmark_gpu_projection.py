from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu.model import (
    GpuGaussianModel,
)
from scientific_gsplat_renderer.gpu.orthographic_projection import (
    GpuProjectedGaussians,
    project_gaussians_orthographic_gpu,
)
from scientific_gsplat_renderer.projection.orthographic import (
    ProjectedGaussians,
    project_gaussians_orthographic,
)


FloatArray = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)

IMAGE_SIZE = (600, 600)

VIEW_PADDING = 0.02

MINIMUM_EIGENVALUE = 1.0e-6
MINIMUM_PIXEL_VARIANCE = 1.0e-4
SIGMA_EXTENT = 3.0

GPU_WARMUP_ITERATIONS = 5
GPU_TIMING_ITERATIONS = 20

DEVICE = "cuda:0"


@dataclass(frozen=True, slots=True)
class ErrorStatistics:
    """Numerical comparison statistics."""

    maximum_absolute: float
    normalized_l2: float
    scale_relative_maximum: float


def separator() -> None:
    print("=" * 80)


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Copy a tensor to a NumPy array."""

    return (
        tensor
        .detach()
        .cpu()
        .numpy()
    )


def compute_error_statistics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    relative_floor: float = 1.0e-8,
) -> ErrorStatistics:
    """Compare candidate values against a reference array.

    Three metrics are returned:

    maximum_absolute
        Largest elementwise absolute error.

    normalized_l2
        Global L2 error divided by the reference L2 norm.

    scale_relative_maximum
        Largest absolute error divided by a global reference scale. Unlike
        naive elementwise relative error, this metric does not explode around
        reference values near zero.
    """

    reference_array = np.asarray(
        reference,
        dtype=np.float64,
    )

    candidate_array = np.asarray(
        candidate,
        dtype=np.float64,
    )

    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Cannot compare arrays with different shapes: "
            f"{reference_array.shape} and {candidate_array.shape}."
        )

    difference = (
        candidate_array
        - reference_array
    )

    absolute_difference = np.abs(
        difference
    )

    maximum_absolute = float(
        np.max(absolute_difference)
    )

    reference_norm = float(
        np.linalg.norm(
            reference_array.ravel()
        )
    )

    difference_norm = float(
        np.linalg.norm(
            difference.ravel()
        )
    )

    if reference_norm > 0.0:
        normalized_l2 = (
            difference_norm
            / reference_norm
        )
    else:
        normalized_l2 = difference_norm

    reference_scale = max(
        float(
            np.max(
                np.abs(reference_array)
            )
        ),
        relative_floor,
    )

    scale_relative_maximum = (
        maximum_absolute
        / reference_scale
    )

    return ErrorStatistics(
        maximum_absolute=maximum_absolute,
        normalized_l2=normalized_l2,
        scale_relative_maximum=(
            scale_relative_maximum
        ),
    )


def build_camera(
    model: GaussianModel,
) -> OrthographicCamera:
    """Construct a centered orthographic camera for the model."""

    means = np.asarray(
        model.means,
        dtype=np.float64,
    )

    minimum = np.min(
        means,
        axis=0,
    )

    maximum = np.max(
        means,
        axis=0,
    )

    center = 0.5 * (
        minimum + maximum
    )

    extent = maximum - minimum

    largest_xy_extent = float(
        max(
            extent[0],
            extent[1],
        )
    )

    view_width = (
        largest_xy_extent
        * (1.0 + VIEW_PADDING)
    )

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
        image_width=int(IMAGE_SIZE[0]),
        image_height=int(IMAGE_SIZE[1]),
        near=0.0,
        far=float("inf"),
    )


def time_cpu_projection(
    model: GaussianModel,
    camera: OrthographicCamera,
) -> tuple[ProjectedGaussians, float]:
    """Run and time the NumPy CPU projection."""

    start = perf_counter()

    projected = project_gaussians_orthographic(
        model,
        camera,
        minimum_eigenvalue=MINIMUM_EIGENVALUE,
        minimum_pixel_variance=(
            MINIMUM_PIXEL_VARIANCE
        ),
        sigma_extent=SIGMA_EXTENT,
    )

    elapsed = (
        perf_counter()
        - start
    )

    return projected, elapsed


def time_gpu_projection(
    model: GpuGaussianModel,
    camera: OrthographicCamera,
) -> tuple[GpuProjectedGaussians, float]:
    """Warm up and benchmark the GPU projection."""

    projected: GpuProjectedGaussians | None = None

    # Warm up PyTorch and CUDA kernels.
    for _ in range(
        GPU_WARMUP_ITERATIONS
    ):
        projected = (
            project_gaussians_orthographic_gpu(
                model,
                camera,
                minimum_eigenvalue=(
                    MINIMUM_EIGENVALUE
                ),
                minimum_pixel_variance=(
                    MINIMUM_PIXEL_VARIANCE
                ),
                sigma_extent=SIGMA_EXTENT,
            )
        )

    torch.cuda.synchronize(
        model.device
    )

    start_event = torch.cuda.Event(
        enable_timing=True
    )

    end_event = torch.cuda.Event(
        enable_timing=True
    )

    start_event.record()

    for _ in range(
        GPU_TIMING_ITERATIONS
    ):
        projected = (
            project_gaussians_orthographic_gpu(
                model,
                camera,
                minimum_eigenvalue=(
                    MINIMUM_EIGENVALUE
                ),
                minimum_pixel_variance=(
                    MINIMUM_PIXEL_VARIANCE
                ),
                sigma_extent=SIGMA_EXTENT,
            )
        )

    end_event.record()

    torch.cuda.synchronize(
        model.device
    )

    elapsed_milliseconds = (
        start_event.elapsed_time(
            end_event
        )
        / GPU_TIMING_ITERATIONS
    )

    if projected is None:
        raise RuntimeError(
            "GPU projection did not produce a result."
        )

    elapsed_seconds = (
        elapsed_milliseconds
        / 1000.0
    )

    return projected, elapsed_seconds


def print_error_row(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    """Print comparison metrics for one projected quantity."""

    statistics = compute_error_statistics(
        reference,
        candidate,
    )

    print(
        f"{name:<28}: "
        f"abs={statistics.maximum_absolute:12.8e}  "
        f"nL2={statistics.normalized_l2:12.8e}  "
        f"scale-rel={statistics.scale_relative_maximum:12.8e}"
    )


def main() -> None:
    separator()
    print(
        "CPU versus GPU orthographic projection"
    )
    separator()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in PyTorch."
        )

    device = torch.device(
        DEVICE
    )

    # ------------------------------------------------------------------
    # Load CPU model
    # ------------------------------------------------------------------

    model_load_start = perf_counter()

    cpu_model = GaussianModel.load(
        MODEL_PATH
    )

    model_load_seconds = (
        perf_counter()
        - model_load_start
    )

    camera = build_camera(
        cpu_model
    )

    # ------------------------------------------------------------------
    # Upload model once
    # ------------------------------------------------------------------

    torch.cuda.synchronize(
        device
    )

    upload_start = perf_counter()

    gpu_model = GpuGaussianModel.from_cpu(
        cpu_model,
        device=device,
    )

    torch.cuda.synchronize(
        device
    )

    upload_seconds = (
        perf_counter()
        - upload_start
    )

    # ------------------------------------------------------------------
    # CPU and GPU projection
    # ------------------------------------------------------------------

    cpu_projected, cpu_seconds = (
        time_cpu_projection(
            cpu_model,
            camera,
        )
    )

    gpu_projected, gpu_seconds = (
        time_gpu_projection(
            gpu_model,
            camera,
        )
    )

    # Copy the GPU results only after timing has completed.
    gpu_means_camera = tensor_to_numpy(
        gpu_projected.means_camera
    )

    gpu_means_pixel = tensor_to_numpy(
        gpu_projected.means_pixel
    )

    gpu_covariances_camera = tensor_to_numpy(
        gpu_projected.covariances_camera
    )

    gpu_covariances_pixel = tensor_to_numpy(
        gpu_projected.covariances_pixel
    )

    gpu_inverse_covariances_pixel = tensor_to_numpy(
        gpu_projected.inverse_covariances_pixel
    )

    gpu_radii_pixel = tensor_to_numpy(
        gpu_projected.radii_pixel
    )

    gpu_depths = tensor_to_numpy(
        gpu_projected.depths
    )

    gpu_valid = tensor_to_numpy(
        gpu_projected.valid
    ).astype(
        np.bool_,
        copy=False,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    speedup = (
        cpu_seconds / gpu_seconds
        if gpu_seconds > 0.0
        else float("inf")
    )

    print(
        f"Model path                 : {MODEL_PATH}"
    )

    print(
        f"Gaussians                  : "
        f"{cpu_model.n_gaussians:,}"
    )

    print(
        f"Particles                  : "
        f"{cpu_model.n_particles:,}"
    )

    print(
        f"Image size                 : "
        f"{camera.image_size}"
    )

    print(
        f"View width                 : "
        f"{camera.view_width:.6f}"
    )

    print(
        f"View height                : "
        f"{camera.view_height:.6f}"
    )

    print(
        f"GPU device                 : "
        f"{device}"
    )

    print()

    print(
        f"Model load                 : "
        f"{model_load_seconds:.6f} s"
    )

    print(
        f"One-time GPU upload        : "
        f"{upload_seconds:.6f} s"
    )

    print(
        f"CPU projection             : "
        f"{cpu_seconds:.6f} s"
    )

    print(
        f"GPU projection             : "
        f"{gpu_seconds:.6f} s"
    )

    print(
        f"GPU projection             : "
        f"{gpu_seconds * 1000.0:.3f} ms"
    )

    print(
        f"Projection speedup         : "
        f"{speedup:.2f}x"
    )

    print()
    print(
        "Numerical errors "
        "(absolute, normalized L2, scale-relative)"
    )

    print_error_row(
        "means_camera",
        cpu_projected.means_camera,
        gpu_means_camera,
    )

    print_error_row(
        "means_pixel",
        cpu_projected.means_pixel,
        gpu_means_pixel,
    )

    print_error_row(
        "covariances_camera",
        cpu_projected.covariances_camera,
        gpu_covariances_camera,
    )

    print_error_row(
        "covariances_pixel",
        cpu_projected.covariances_pixel,
        gpu_covariances_pixel,
    )

    print_error_row(
        "inverse covariances pixel",
        cpu_projected.inverse_covariances_pixel,
        gpu_inverse_covariances_pixel,
    )

    print_error_row(
        "radii_pixel",
        cpu_projected.radii_pixel,
        gpu_radii_pixel,
    )

    print_error_row(
        "depths",
        cpu_projected.depths,
        gpu_depths,
    )

    cpu_valid = np.asarray(
        cpu_projected.valid,
        dtype=np.bool_,
    )

    validity_disagreements = int(
        np.count_nonzero(
            cpu_valid != gpu_valid
        )
    )

    print()

    print(
        f"CPU valid Gaussians        : "
        f"{np.count_nonzero(cpu_valid):,}"
    )

    print(
        f"GPU valid Gaussians        : "
        f"{np.count_nonzero(gpu_valid):,}"
    )

    print(
        f"Validity disagreements     : "
        f"{validity_disagreements:,}"
    )

    separator()


if __name__ == "__main__":
    main()