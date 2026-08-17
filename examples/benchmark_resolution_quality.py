

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

try:
    from skimage.metrics import structural_similarity
except ImportError as error:
    raise ImportError(
        "scikit-image is required for SSIM. Install it with:\n"
        "    pip install scikit-image"
    ) from error

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu import (
    GpuConditionalRenderResult,
    GpuGaussianModel,
    GpuProjectedConditionalAttributes,
    GpuProjectedGaussians,
    GpuTileIntersections,
    build_gpu_tile_intersections,
    project_conditional_attributes_orthographic_gpu,
    project_gaussians_orthographic_gpu,
    render_conditional_attribute_gpu,
)


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/illustris3_parallel/full_94m_1pct_parallel/simple_model.npz"
    # "/home/robiul/Particle_flow/HACC_project/output/"
    # "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
    # "/home/robiul/Particle_flow/HACC_project/output/hacc_process_parallel_v2/full_hacc_1pct_components_4/simple_model.npz"
)

DEFAULT_GT_ROOT = Path(
    "ground_truth/resolution_benchmark/Illustris/full_1pct"
    # "ground_truth/resolution_benchmark/HACC/full_1pct"
)

DEFAULT_OUTPUT = Path(
    "outputs/resolution_quality_benchmark/Illustris/full_1pct"
    # "outputs/resolution_quality_benchmark/HACC/full_1pct"
)

DEFAULT_RESOLUTIONS = (
    1024,
    2048,
    4096,
)


@dataclass(frozen=True, slots=True)
class StageTiming:
    projection_ms: float
    conditional_projection_ms: float
    intersections_ms: float
    rendering_ms: float
    total_frame_ms: float
    fps: float


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    density_psnr: float
    density_ssim: float
    attribute_psnr: float
    attribute_ssim: float
    attribute_mae: float
    attribute_rmse: float
    attribute_common_pixels: int


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    resolution_label: str
    width: int
    height: int

    valid_gaussians: int
    tile_intersections: int

    density_psnr: float
    density_ssim: float

    attribute_psnr: float
    attribute_ssim: float
    attribute_mae: float
    attribute_rmse: float
    attribute_common_pixels: int

    projection_ms: float
    conditional_projection_ms: float
    intersections_ms: float
    rendering_ms: float
    total_frame_ms: float
    fps: float

    density_sum: float
    density_minimum: float
    density_maximum: float
    density_retained_fraction: float

    attribute_minimum: float
    attribute_maximum: float
    attribute_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GPU scientific Gaussian rendering quality and "
            "performance at multiple square output resolutions."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_ROOT,
        help=(
            "Ground-truth root. Expected directories such as "
            "GT_ROOT/1024, GT_ROOT/2048, and GT_ROOT/4096."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--view-padding",
        type=float,
        default=0.0,
        help=(
            "Fractional padding around the model XY bounds. "
            "This must match the value used when generating particle GT."
        ),
    )

    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--minimum-eigenvalue",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--minimum-pixel-variance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--sigma-extent",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help=(
            "Paper-style covariance smoothing. The model covariance is "
            "multiplied by (1 + beta) before projection."
        ),
    )

    parser.add_argument(
        "--blob",
        type=float,
        default=2.0,
        help=(
            "Isotropic screen-space Gaussian blob sigma in pixels. "
            "Use the same value used to generate particle GT."
        ),
    )

    parser.add_argument(
        "--relative-density-threshold",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--timed-frames",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--density-upper-percentile",
        type=float,
        default=99.9,
    )

    parser.add_argument(
        "--attribute-lower-percentile",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--attribute-upper-percentile",
        type=float,
        default=99.0,
    )

    parser.add_argument(
        "--comparison-error-percentile",
        type=float,
        default=99.5,
        help=(
            "Upper percentile used to scale absolute-error panels."
        ),
    )

    parser.add_argument(
        "--unnormalized-mass",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model does not exist: {args.model}"
        )

    if not args.gt_root.exists():
        raise FileNotFoundError(
            f"Ground-truth root does not exist: {args.gt_root}"
        )

    if (
        not math.isfinite(args.view_padding)
        or args.view_padding < 0.0
    ):
        raise ValueError(
            "view_padding must be finite and nonnegative."
        )

    if not args.resolutions:
        raise ValueError(
            "At least one resolution is required."
        )

    for resolution in args.resolutions:
        if resolution <= 0:
            raise ValueError(
                f"Resolution must be positive, got {resolution}."
            )

    numeric_positive = {
        "tile_size": float(args.tile_size),
        "minimum_eigenvalue": args.minimum_eigenvalue,
        "minimum_pixel_variance": args.minimum_pixel_variance,
        "sigma_extent": args.sigma_extent,
    }

    for name, value in numeric_positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and positive, got {value}."
            )

    for name, value in {
        "beta": args.beta,
        "blob": args.blob,
        "relative_density_threshold": (
            args.relative_density_threshold
        ),
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be finite and nonnegative, got {value}."
            )

    if args.warmup_frames < 0:
        raise ValueError(
            "warmup_frames must be nonnegative."
        )

    if args.timed_frames <= 0:
        raise ValueError(
            "timed_frames must be positive."
        )

    if not (
        0.0
        < args.density_upper_percentile
        <= 100.0
    ):
        raise ValueError(
            "density_upper_percentile must be in (0, 100]."
        )

    if not (
        0.0
        <= args.attribute_lower_percentile
        < args.attribute_upper_percentile
        <= 100.0
    ):
        raise ValueError(
            "Attribute percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )

    if not (
        0.0
        < args.comparison_error_percentile
        <= 100.0
    ):
        raise ValueError(
            "comparison_error_percentile must be in (0, 100]."
        )


def model_bounds(
    model: GaussianModel,
    *,
    view_padding: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
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
        * (1.0 + view_padding)
    )

    camera_distance = float(
        max(
            float(span.max()) * 2.0,
            1.0,
        )
    )

    return (
        minimum,
        maximum,
        center,
        view_width,
        camera_distance,
    )


def build_camera(
    *,
    center: np.ndarray,
    view_width: float,
    camera_distance: float,
    width: int,
    height: int,
) -> OrthographicCamera:
    return OrthographicCamera(
        position=np.array(
            [
                center[0],
                center[1],
                center[2] + camera_distance,
            ],
            dtype=np.float64,
        ),
        target=np.asarray(
            center,
            dtype=np.float64,
        ),
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=view_width,
        image_width=width,
        image_height=height,
        near=0.0,
        far=2.0 * camera_distance,
    )


def create_cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(
        enable_timing=True,
    )


def event_seconds(
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> float:
    return (
        start.elapsed_time(end)
        / 1000.0
    )


def render_complete_frame(
    gpu_model: GpuGaussianModel,
    camera: OrthographicCamera,
    *,
    width: int,
    height: int,
    tile_size: int,
    minimum_eigenvalue: float,
    minimum_pixel_variance: float,
    sigma_extent: float,
    beta: float,
    blob_sigma_pixels: float,
    normalize_gaussian_mass: bool,
    relative_density_threshold: float,
) -> tuple[
    GpuProjectedGaussians,
    GpuProjectedConditionalAttributes,
    GpuTileIntersections,
    GpuConditionalRenderResult,
]:
    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        minimum_eigenvalue=minimum_eigenvalue,
        minimum_pixel_variance=minimum_pixel_variance,
        sigma_extent=sigma_extent,
        beta=beta,
        blob_sigma_pixels=blob_sigma_pixels,
    )

    conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
        )
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        normalize_gaussian_mass=(
            normalize_gaussian_mass
        ),
        relative_density_threshold=(
            relative_density_threshold
        ),
    )

    return (
        projected,
        conditional,
        intersections,
        result,
    )


def benchmark_resolution(
    gpu_model: GpuGaussianModel,
    camera: OrthographicCamera,
    *,
    width: int,
    height: int,
    tile_size: int,
    minimum_eigenvalue: float,
    minimum_pixel_variance: float,
    sigma_extent: float,
    beta: float,
    blob_sigma_pixels: float,
    normalize_gaussian_mass: bool,
    relative_density_threshold: float,
    warmup_frames: int,
    timed_frames: int,
) -> tuple[
    StageTiming,
    GpuProjectedGaussians,
    GpuTileIntersections,
    GpuConditionalRenderResult,
]:
    (
        projected,
        _conditional,
        intersections,
        result,
    ) = render_complete_frame(
        gpu_model,
        camera,
        width=width,
        height=height,
        tile_size=tile_size,
        minimum_eigenvalue=minimum_eigenvalue,
        minimum_pixel_variance=minimum_pixel_variance,
        sigma_extent=sigma_extent,
        beta=beta,
        blob_sigma_pixels=blob_sigma_pixels,
        normalize_gaussian_mass=(
            normalize_gaussian_mass
        ),
        relative_density_threshold=(
            relative_density_threshold
        ),
    )

    gpu_model.synchronize()

    for _ in range(warmup_frames):
        (
            projected,
            _conditional,
            intersections,
            result,
        ) = render_complete_frame(
            gpu_model,
            camera,
            width=width,
            height=height,
            tile_size=tile_size,
            minimum_eigenvalue=minimum_eigenvalue,
            minimum_pixel_variance=(
                minimum_pixel_variance
            ),
            sigma_extent=sigma_extent,
            beta=beta,
            blob_sigma_pixels=(
                blob_sigma_pixels
            ),
            normalize_gaussian_mass=(
                normalize_gaussian_mass
            ),
            relative_density_threshold=(
                relative_density_threshold
            ),
        )

    gpu_model.synchronize()

    projection_total = 0.0
    conditional_total = 0.0
    intersections_total = 0.0
    rendering_total = 0.0
    frame_total = 0.0

    for _ in range(timed_frames):
        frame_start = create_cuda_event()
        frame_end = create_cuda_event()

        projection_start = create_cuda_event()
        projection_end = create_cuda_event()

        conditional_start = create_cuda_event()
        conditional_end = create_cuda_event()

        intersections_start = create_cuda_event()
        intersections_end = create_cuda_event()

        rendering_start = create_cuda_event()
        rendering_end = create_cuda_event()

        frame_start.record()

        projection_start.record()

        projected = project_gaussians_orthographic_gpu(
            gpu_model,
            camera,
            minimum_eigenvalue=minimum_eigenvalue,
            minimum_pixel_variance=(
                minimum_pixel_variance
            ),
            sigma_extent=sigma_extent,
            beta=beta,
            blob_sigma_pixels=(
                blob_sigma_pixels
            ),
        )

        projection_end.record()

        conditional_start.record()

        conditional = (
            project_conditional_attributes_orthographic_gpu(
                gpu_model,
                projected,
                camera,
            )
        )

        conditional_end.record()

        intersections_start.record()

        intersections = build_gpu_tile_intersections(
            projected,
            image_width=width,
            image_height=height,
            tile_size=tile_size,
        )

        intersections_end.record()

        rendering_start.record()

        result = render_conditional_attribute_gpu(
            projected,
            conditional,
            intersections,
            normalize_gaussian_mass=(
                normalize_gaussian_mass
            ),
            relative_density_threshold=(
                relative_density_threshold
            ),
        )

        rendering_end.record()

        frame_end.record()
        frame_end.synchronize()

        projection_total += event_seconds(
            projection_start,
            projection_end,
        )

        conditional_total += event_seconds(
            conditional_start,
            conditional_end,
        )

        intersections_total += event_seconds(
            intersections_start,
            intersections_end,
        )

        rendering_total += event_seconds(
            rendering_start,
            rendering_end,
        )

        frame_total += event_seconds(
            frame_start,
            frame_end,
        )

    n_frames = float(
        timed_frames
    )

    frame_seconds = (
        frame_total / n_frames
    )

    timing = StageTiming(
        projection_ms=(
            projection_total
            / n_frames
            * 1000.0
        ),
        conditional_projection_ms=(
            conditional_total
            / n_frames
            * 1000.0
        ),
        intersections_ms=(
            intersections_total
            / n_frames
            * 1000.0
        ),
        rendering_ms=(
            rendering_total
            / n_frames
            * 1000.0
        ),
        total_frame_ms=(
            frame_seconds * 1000.0
        ),
        fps=(
            1.0 / frame_seconds
            if frame_seconds > 0.0
            else float("inf")
        ),
    )

    return (
        timing,
        projected,
        intersections,
        result,
    )


def resolution_directory(
    gt_root: Path,
    resolution: int,
) -> Path:
    candidates = (
        gt_root / str(resolution),
        gt_root / f"{resolution}x{resolution}",
        gt_root / f"{resolution}_x_{resolution}",
        gt_root / f"{resolution}px",
    )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    expected = "\n".join(
        f"  {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not find a GT directory for resolution "
        f"{resolution}. Checked:\n{expected}"
    )


def find_existing_file(
    directory: Path,
    names: tuple[str, ...],
    *,
    required: bool,
) -> Path | None:
    for name in names:
        candidate = directory / name

        if candidate.exists():
            return candidate

    if required:
        expected = "\n".join(
            f"  {directory / name}"
            for name in names
        )

        raise FileNotFoundError(
            f"Required GT file not found. Checked:\n{expected}"
        )

    return None


def load_ground_truth(
    gt_root: Path,
    resolution: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    directory = resolution_directory(
        gt_root,
        resolution,
    )

    density_path = find_existing_file(
        directory,
        (
            "density.npy",
            "density_gt.npy",
            "gt_density.npy",
        ),
        required=True,
    )

    attribute_path = find_existing_file(
        directory,
        (
            "attribute.npy",
            "conditional_attribute.npy",
            "attribute_gt.npy",
            "gt_attribute.npy",
        ),
        required=True,
    )

    mask_path = find_existing_file(
        directory,
        (
            "valid_mask.npy",
            "attribute_valid_mask.npy",
            "mask.npy",
        ),
        required=False,
    )

    assert density_path is not None
    assert attribute_path is not None

    density = np.asarray(
        np.load(density_path),
        dtype=np.float64,
    )

    attribute = np.asarray(
        np.load(attribute_path),
        dtype=np.float64,
    )

    expected_shape = (
        resolution,
        resolution,
    )

    if density.shape != expected_shape:
        raise ValueError(
            f"GT density at {resolution} has shape "
            f"{density.shape}; expected {expected_shape}."
        )

    if attribute.shape != expected_shape:
        raise ValueError(
            f"GT attribute at {resolution} has shape "
            f"{attribute.shape}; expected {expected_shape}."
        )

    if mask_path is None:
        valid_mask = np.isfinite(
            attribute
        )
    else:
        valid_mask = np.asarray(
            np.load(mask_path),
            dtype=np.bool_,
        )

        if valid_mask.shape != expected_shape:
            raise ValueError(
                f"GT mask at {resolution} has shape "
                f"{valid_mask.shape}; expected {expected_shape}."
            )

    valid_mask &= np.isfinite(
        attribute
    )

    return (
        density,
        attribute,
        valid_mask,
    )


def normalize_density_pair(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    upper_percentile: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    valid_gt = ground_truth[
        np.isfinite(ground_truth)
        & (ground_truth > 0.0)
    ]

    if valid_gt.size == 0:
        raise ValueError(
            "GT density has no positive finite values."
        )

    upper = float(
        np.percentile(
            valid_gt,
            upper_percentile,
        )
    )

    upper = max(
        upper,
        1.0e-12,
    )

    denominator = max(
        math.log1p(upper),
        1.0e-12,
    )

    gt_normalized = np.clip(
        np.log1p(
            np.maximum(
                ground_truth,
                0.0,
            )
        )
        / denominator,
        0.0,
        1.0,
    )

    prediction_normalized = np.clip(
        np.log1p(
            np.maximum(
                prediction,
                0.0,
            )
        )
        / denominator,
        0.0,
        1.0,
    )

    return (
        np.nan_to_num(
            gt_normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        np.nan_to_num(
            prediction_normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
    )


def normalize_attribute_pair(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    common_mask: np.ndarray,
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    valid_gt = ground_truth[
        common_mask
        & np.isfinite(ground_truth)
    ]

    if valid_gt.size == 0:
        raise ValueError(
            "No common valid GT attribute pixels."
        )

    lower = float(
        np.percentile(
            valid_gt,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            valid_gt,
            upper_percentile,
        )
    )

    if upper <= lower:
        upper = lower + 1.0

    denominator = upper - lower

    gt_normalized = np.clip(
        (
            ground_truth - lower
        )
        / denominator,
        0.0,
        1.0,
    )

    prediction_normalized = np.clip(
        (
            prediction - lower
        )
        / denominator,
        0.0,
        1.0,
    )

    gt_normalized = np.where(
        common_mask,
        gt_normalized,
        0.0,
    )

    prediction_normalized = np.where(
        common_mask,
        prediction_normalized,
        0.0,
    )

    return (
        np.nan_to_num(
            gt_normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        np.nan_to_num(
            prediction_normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
    )


def psnr_from_mse(
    mse: float,
    *,
    data_range: float = 1.0,
) -> float:
    if mse < 0.0:
        raise ValueError(
            "MSE cannot be negative."
        )

    if mse == 0.0:
        return float("inf")

    return float(
        10.0
        * math.log10(
            (data_range * data_range)
            / mse
        )
    )


def full_image_psnr(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    difference = (
        prediction - reference
    )

    mse = float(
        np.mean(
            difference * difference
        )
    )

    return psnr_from_mse(
        mse,
        data_range=1.0,
    )


def masked_psnr(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        raise ValueError(
            "Cannot calculate masked PSNR with an empty mask."
        )

    difference = (
        prediction[mask]

        - reference[mask]
    )
    mse = float(
        np.mean(
            difference * difference
        )
    )

    return psnr_from_mse(
        mse,
        data_range=1.0,
    )


def mask_bounding_box(
    mask: np.ndarray,
    *,
    padding: int = 5,
) -> tuple[
    slice,
    slice,
]:
    rows, columns = np.nonzero(
        mask
    )

    if rows.size == 0:
        raise ValueError(
            "Cannot calculate a bounding box for an empty mask."
        )

    row_min = max(
        int(rows.min()) - padding,
        0,
    )

    row_max = min(
        int(rows.max()) + padding + 1,
        mask.shape[0],
    )

    column_min = max(
        int(columns.min()) - padding,
        0,
    )

    column_max = min(
        int(columns.max()) + padding + 1,
        mask.shape[1],
    )

    return (
        slice(row_min, row_max),
        slice(column_min, column_max),
    )


def calculate_quality_metrics(
    *,
    gt_density: np.ndarray,
    predicted_density: np.ndarray,
    gt_attribute: np.ndarray,
    predicted_attribute: np.ndarray,
    gt_valid_mask: np.ndarray,
    predicted_valid_mask: np.ndarray,
    density_upper_percentile: float,
    attribute_lower_percentile: float,
    attribute_upper_percentile: float,
) -> QualityMetrics:
    (
        gt_density_normalized,
        predicted_density_normalized,
    ) = normalize_density_pair(
        gt_density,
        predicted_density,
        upper_percentile=(
            density_upper_percentile
        ),
    )

    density_psnr = full_image_psnr(
        gt_density_normalized,
        predicted_density_normalized,
    )

    density_ssim = float(
        structural_similarity(
            gt_density_normalized,
            predicted_density_normalized,
            data_range=1.0,
        )
    )

    common_mask = (
        gt_valid_mask
        & predicted_valid_mask
        & np.isfinite(gt_attribute)
        & np.isfinite(predicted_attribute)
    )

    common_pixel_count = int(
        np.count_nonzero(
            common_mask
        )
    )

    if common_pixel_count == 0:
        raise ValueError(
            "There are no common valid attribute pixels."
        )

    (
        gt_attribute_normalized,
        predicted_attribute_normalized,
    ) = normalize_attribute_pair(
        gt_attribute,
        predicted_attribute,
        common_mask,
        lower_percentile=(
            attribute_lower_percentile
        ),
        upper_percentile=(
            attribute_upper_percentile
        ),
    )

    attribute_psnr = masked_psnr(
        gt_attribute_normalized,
        predicted_attribute_normalized,
        common_mask,
    )

    row_slice, column_slice = (
        mask_bounding_box(
            common_mask,
        )
    )

    attribute_ssim = float(
        structural_similarity(
            gt_attribute_normalized[
                row_slice,
                column_slice,
            ],
            predicted_attribute_normalized[
                row_slice,
                column_slice,
            ],
            data_range=1.0,
        )
    )

    raw_error = (
        predicted_attribute[common_mask]
        - gt_attribute[common_mask]
    )

    attribute_mae = float(
        np.mean(
            np.abs(raw_error)
        )
    )

    attribute_rmse = float(
        np.sqrt(
            np.mean(
                raw_error * raw_error
            )
        )
    )

    return QualityMetrics(
        density_psnr=density_psnr,
        density_ssim=density_ssim,
        attribute_psnr=attribute_psnr,
        attribute_ssim=attribute_ssim,
        attribute_mae=attribute_mae,
        attribute_rmse=attribute_rmse,
        attribute_common_pixels=(
            common_pixel_count
        ),
    )

def select_automatic_roi(
    image: np.ndarray,
    *,
    roi_fraction: float = 0.18,
    minimum_size: int = 128,
    maximum_size: int = 512,
) -> tuple[int, int, int]:
    """Select a detail-rich square ROI from a normalized GT image.

    The score combines local variance and local intensity so that the selected
    region tends to contain filament intersections, halos, and fine structure
    rather than empty or nearly uniform background.

    Returns
    -------
    x_start, y_start, roi_size
        Pixel coordinates suitable for array slicing and Rectangle drawing.
    """

    if image.ndim != 2:
        raise ValueError(
            f"image must be two-dimensional, got {image.shape}."
        )

    height, width = image.shape

    roi_size = int(
        round(
            min(height, width)
            * roi_fraction
        )
    )

    roi_size = max(
        minimum_size,
        roi_size,
    )

    roi_size = min(
        maximum_size,
        roi_size,
        height,
        width,
    )

    # Use an even crop size.
    if roi_size % 2 != 0:
        roi_size -= 1

    source = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(
        np.float64,
        copy=False,
    )

    # Local variance:
    #
    # Var(X) = E[X^2] - E[X]^2
    local_mean = uniform_filter(
        source,
        size=roi_size,
        mode="reflect",
    )

    local_mean_squared = uniform_filter(
        source * source,
        size=roi_size,
        mode="reflect",
    )

    local_variance = np.maximum(
        local_mean_squared
        - local_mean * local_mean,
        0.0,
    )

    def normalize_score(
        values: np.ndarray,
    ) -> np.ndarray:
        finite = values[
            np.isfinite(values)
        ]

        if finite.size == 0:
            return np.zeros_like(
                values,
                dtype=np.float64,
            )

        low = float(
            np.percentile(
                finite,
                1.0,
            )
        )

        high = float(
            np.percentile(
                finite,
                99.0,
            )
        )

        if high <= low:
            return np.zeros_like(
                values,
                dtype=np.float64,
            )

        return np.clip(
            (values - low)
            / (high - low),
            0.0,
            1.0,
        )

    variance_score = normalize_score(
        local_variance
    )

    intensity_score = normalize_score(
        local_mean
    )

    # Favor detailed regions while also avoiding completely dark background.
    score = (
        0.75 * variance_score
        + 0.25 * intensity_score
    )

    half = roi_size // 2

    # Exclude centers that would make the ROI cross an image boundary.
    score[:half, :] = -np.inf
    score[-half:, :] = -np.inf
    score[:, :half] = -np.inf
    score[:, -half:] = -np.inf

    if not np.isfinite(score).any():
        center_y = height // 2
        center_x = width // 2
    else:
        center_y, center_x = np.unravel_index(
            np.nanargmax(score),
            score.shape,
        )

    x_start = int(
        np.clip(
            center_x - half,
            0,
            width - roi_size,
        )
    )

    y_start = int(
        np.clip(
            center_y - half,
            0,
            height - roi_size,
        )
    )

    return (
        x_start,
        y_start,
        roi_size,
    )


# def save_gt_prediction_comparison(
#     path: Path,
#     *,
#     gt_density: np.ndarray,
#     predicted_density: np.ndarray,
#     gt_attribute: np.ndarray,
#     predicted_attribute: np.ndarray,
#     gt_valid_mask: np.ndarray,
#     predicted_valid_mask: np.ndarray,
#     density_upper_percentile: float,
#     attribute_lower_percentile: float,
#     attribute_upper_percentile: float,
#     error_percentile: float,
#     attribute_name: str,
#     density_psnr: float,
#     density_ssim: float,
#     attribute_psnr: float,
#     attribute_ssim: float,
# ) -> None:
#     path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     (
#         gt_density_normalized,
#         predicted_density_normalized,
#     ) = normalize_density_pair(
#         gt_density,
#         predicted_density,
#         upper_percentile=(
#             density_upper_percentile
#         ),
#     )

#     density_error = np.abs(
#         predicted_density_normalized
#         - gt_density_normalized
#     )

#     finite_density_error = density_error[
#         np.isfinite(density_error)
#     ]

#     density_error_maximum = max(
#         float(
#             np.percentile(
#                 finite_density_error,
#                 error_percentile,
#             )
#         ),
#         1.0e-8,
#     )

#     common_mask = (
#         gt_valid_mask
#         & predicted_valid_mask
#         & np.isfinite(gt_attribute)
#         & np.isfinite(predicted_attribute)
#     )

#     if not np.any(common_mask):
#         raise RuntimeError(
#             "No common valid attribute pixels for comparison."
#         )

#     finite_gt_attribute = gt_attribute[
#         gt_valid_mask
#         & np.isfinite(gt_attribute)
#     ]

#     attribute_lower = float(
#         np.percentile(
#             finite_gt_attribute,
#             attribute_lower_percentile,
#         )
#     )

#     attribute_upper = float(
#         np.percentile(
#             finite_gt_attribute,
#             attribute_upper_percentile,
#         )
#     )

#     if attribute_upper <= attribute_lower:
#         attribute_upper = attribute_lower + 1.0

#     gt_attribute_display = np.asarray(
#         gt_attribute,
#         dtype=np.float64,
#     ).copy()

#     predicted_attribute_display = np.asarray(
#         predicted_attribute,
#         dtype=np.float64,
#     ).copy()

#     gt_attribute_display[
#         ~gt_valid_mask
#     ] = np.nan

#     predicted_attribute_display[
#         ~predicted_valid_mask
#     ] = np.nan

#     attribute_error = np.full(
#         gt_attribute.shape,
#         np.nan,
#         dtype=np.float64,
#     )

#     attribute_error[common_mask] = np.abs(
#         predicted_attribute[common_mask]
#         - gt_attribute[common_mask]
#     )

#     finite_attribute_error = attribute_error[
#         np.isfinite(attribute_error)
#     ]

#     attribute_error_maximum = max(
#         float(
#             np.percentile(
#                 finite_attribute_error,
#                 error_percentile,
#             )
#         ),
#         1.0e-8,
#     )


def save_gt_prediction_comparison(
    path: Path,
    *,
    gt_density: np.ndarray,
    predicted_density: np.ndarray,
    gt_attribute: np.ndarray,
    predicted_attribute: np.ndarray,
    gt_valid_mask: np.ndarray,
    predicted_valid_mask: np.ndarray,
    density_upper_percentile: float,
    attribute_lower_percentile: float,
    attribute_upper_percentile: float,
    error_percentile: float,
    attribute_name: str,
    density_psnr: float,
    density_ssim: float,
    attribute_psnr: float,
    attribute_ssim: float,
) -> None:
    """Save GT, prediction, and automatically selected zoomed regions."""

    # Kept in the signature for compatibility with the existing caller.
    del error_percentile

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        gt_density_normalized,
        predicted_density_normalized,
    ) = normalize_density_pair(
        gt_density,
        predicted_density,
        upper_percentile=(
            density_upper_percentile
        ),
    )

    common_mask = (
        gt_valid_mask
        & predicted_valid_mask
        & np.isfinite(gt_attribute)
        & np.isfinite(predicted_attribute)
    )

    if not np.any(common_mask):
        raise RuntimeError(
            "No common valid attribute pixels for comparison."
        )

    finite_gt_attribute = gt_attribute[
        gt_valid_mask
        & np.isfinite(gt_attribute)
    ]

    if finite_gt_attribute.size == 0:
        raise RuntimeError(
            "No finite GT attribute pixels are available."
        )

    attribute_lower = float(
        np.percentile(
            finite_gt_attribute,
            attribute_lower_percentile,
        )
    )

    attribute_upper = float(
        np.percentile(
            finite_gt_attribute,
            attribute_upper_percentile,
        )
    )

    if attribute_upper <= attribute_lower:
        attribute_upper = (
            attribute_lower + 1.0
        )

    gt_attribute_display = np.asarray(
        gt_attribute,
        dtype=np.float64,
    ).copy()

    predicted_attribute_display = np.asarray(
        predicted_attribute,
        dtype=np.float64,
    ).copy()

    gt_attribute_display[
        ~gt_valid_mask
    ] = np.nan

    predicted_attribute_display[
        ~predicted_valid_mask
    ] = np.nan

    # ---------------------------------------------------------------
    # Automatically choose one common ROI from the density ground truth.
    # The same spatial region is used for density and attribute images.
    # ---------------------------------------------------------------

    x_start, y_start, roi_size = (
        select_automatic_roi(
            gt_density_normalized,
            roi_fraction=0.18,
            minimum_size=128,
            maximum_size=512,
        )
    )

    x_stop = x_start + roi_size
    y_stop = y_start + roi_size

    density_gt_zoom = (
        gt_density_normalized[
            y_start:y_stop,
            x_start:x_stop,
        ]
    )

    density_prediction_zoom = (
        predicted_density_normalized[
            y_start:y_stop,
            x_start:x_stop,
        ]
    )

    attribute_gt_zoom = (
        gt_attribute_display[
            y_start:y_stop,
            x_start:x_stop,
        ]
    )

    attribute_prediction_zoom = (
        predicted_attribute_display[
            y_start:y_stop,
            x_start:x_stop,
        ]
    )

    # Wide paper-style layout:
    #
    # Density GT | Density prediction | GT zoom | Prediction zoom
    # Attr. GT   | Attr. prediction   | GT zoom | Prediction zoom
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True,
    )

    # ---------------------------------------------------------------
    # Density row
    # ---------------------------------------------------------------

    density_artist = axes[0, 0].imshow(
        gt_density_normalized,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 0].set_title(
        "Density GT"
    )

    axes[0, 1].imshow(
        predicted_density_normalized,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 1].set_title(
        "Density Prediction"
    )

    axes[0, 2].imshow(
        density_gt_zoom,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    axes[0, 2].set_title(
        "Density GT Zoom"
    )

    axes[0, 3].imshow(
        density_prediction_zoom,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    axes[0, 3].set_title(
        "Density Prediction Zoom"
    )

    # ---------------------------------------------------------------
    # Attribute row
    # ---------------------------------------------------------------

    attribute_artist = axes[1, 0].imshow(
        gt_attribute_display,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
    )

    axes[1, 0].set_title(
        f"{attribute_name} GT"
    )

    axes[1, 1].imshow(
        predicted_attribute_display,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
    )

    axes[1, 1].set_title(
        f"{attribute_name} Prediction"
    )

    axes[1, 2].imshow(
        attribute_gt_zoom,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
        interpolation="nearest",
    )

    axes[1, 2].set_title(
        f"{attribute_name} GT Zoom"
    )

    axes[1, 3].imshow(
        attribute_prediction_zoom,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
        interpolation="nearest",
    )

    axes[1, 3].set_title(
        f"{attribute_name} Prediction Zoom"
    )

    # ---------------------------------------------------------------
    # Mark the selected ROI on all full-resolution panels.
    # ---------------------------------------------------------------

    rectangle_style = {
        "linewidth": 3,
        "edgecolor": "cyan",
        "facecolor": "none",
    }

    for axis in (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    ):
        axis.add_patch(
            Rectangle(
                (
                    x_start,
                    y_start,
                ),
                roi_size,
                roi_size,
                **rectangle_style,
            )
        )

    # Add stronger borders around the zoom panels.
    for axis in (
        axes[0, 2],
        axes[0, 3],
        axes[1, 2],
        axes[1, 3],
    ):
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("None")

    # Shared color bars for each row.
    figure.colorbar(
        density_artist,
        ax=list(
            axes[0, :]
        ),
        fraction=0.018,
        pad=0.015,
        label="Normalized log density",
    )

    figure.colorbar(
        attribute_artist,
        ax=list(
            axes[1, :]
        ),
        fraction=0.018,
        pad=0.015,
        label=attribute_name,
    )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        (
            f"GT versus GPU prediction — "
            f"{gt_density.shape[1]} × {gt_density.shape[0]}\n"
            f"Density: PSNR={density_psnr:.3f} dB, "
            f"SSIM={density_ssim:.4f} | "
            f"Attribute: PSNR={attribute_psnr:.3f} dB, "
            f"SSIM={attribute_ssim:.4f}"
        ),
        fontsize=15,
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Automatic ROI: "
        f"x={x_start}:{x_stop}, "
        f"y={y_start}:{y_stop}, "
        f"size={roi_size} x {roi_size}"
    )
    
    return

    # figure, axes = plt.subplots(
    #     2,
    #     3,
    #     figsize=(16, 10),
    #     constrained_layout=True,
    # )
    
    

    # density_artist = axes[0, 0].imshow(
    #     gt_density_normalized,
    #     origin="lower",
    #     cmap="inferno",
    #     vmin=0.0,
    #     vmax=1.0,
    # )

    # axes[0, 0].set_title(
    #     "Density GT"
    # )

    # axes[0, 1].imshow(
    #     predicted_density_normalized,
    #     origin="lower",
    #     cmap="inferno",
    #     vmin=0.0,
    #     vmax=1.0,
    # )

    # axes[0, 1].set_title(
    #     "Density prediction"
    # )

    # density_error_artist = axes[0, 2].imshow(
    #     density_error,
    #     origin="lower",
    #     cmap="magma",
    #     vmin=0.0,
    #     vmax=density_error_maximum,
    # )

    # axes[0, 2].set_title(
    #     "Density absolute error"
    # )

    # figure.colorbar(
    #     density_artist,
    #     ax=[
    #         axes[0, 0],
    #         axes[0, 1],
    #     ],
    #     fraction=0.025,
    #     pad=0.02,
    #     label="Normalized log density",
    # )

    # figure.colorbar(
    #     density_error_artist,
    #     ax=axes[0, 2],
    #     fraction=0.046,
    #     pad=0.04,
    #     label="Absolute normalized error",
    # )

    figure, axes = plt.subplots(
                2,
                2,
                figsize=(10, 10),
                constrained_layout=True,
    )
    
    
    density_artist = axes[0, 0].imshow(
    gt_density_normalized,
    origin="lower",
    cmap="inferno",
    vmin=0.0,
    vmax=1.0,
    )
    axes[0, 0].set_title("Density GT")

    axes[0, 1].imshow(
        predicted_density_normalized,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("Density Prediction")

    figure.colorbar(
        density_artist,
        ax=[axes[0, 0], axes[0, 1]],
        fraction=0.025,
        pad=0.02,
        label="Normalized log density",
    )
    
    # attribute_artist = axes[1, 0].imshow(
    #     gt_attribute_display,
    #     origin="lower",
    #     cmap="viridis",
    #     vmin=attribute_lower,
    #     vmax=attribute_upper,
    # )

    # axes[1, 0].set_title(
    #     f"{attribute_name} GT"
    # )

    # axes[1, 1].imshow(
    #     predicted_attribute_display,
    #     origin="lower",
    #     cmap="viridis",
    #     vmin=attribute_lower,
    #     vmax=attribute_upper,
    # )

    # axes[1, 1].set_title(
    #     f"{attribute_name} prediction"
    # )

    # attribute_error_artist = axes[1, 2].imshow(
    #     attribute_error,
    #     origin="lower",
    #     cmap="magma",
    #     vmin=0.0,
    #     vmax=attribute_error_maximum,
    # )

    # axes[1, 2].set_title(
    #     f"{attribute_name} absolute error"
    # )

    # figure.colorbar(
    #     attribute_artist,
    #     ax=[
    #         axes[1, 0],
    #         axes[1, 1],
    #     ],
    #     fraction=0.025,
    #     pad=0.02,
    #     label=attribute_name,
    # )

    # figure.colorbar(
    #     attribute_error_artist,
    #     ax=axes[1, 2],
    #     fraction=0.046,
    #     pad=0.04,
    #     label="Absolute attribute error",
    # )
    
    attribute_artist = axes[1, 0].imshow(
    gt_attribute_display,
    origin="lower",
    cmap="viridis",
    vmin=attribute_lower,
    vmax=attribute_upper,
    )
    axes[1, 0].set_title(f"{attribute_name} GT")

    axes[1, 1].imshow(
        predicted_attribute_display,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
    )
    axes[1, 1].set_title(f"{attribute_name} Prediction")

    figure.colorbar(
        attribute_artist,
        ax=[axes[1, 0], axes[1, 1]],
        fraction=0.025,
        pad=0.02,
        label=attribute_name,
    )

    for axis in axes.ravel():
        axis.set_axis_off()

    figure.suptitle(
        (
            f"GT versus GPU prediction — "
            f"{gt_density.shape[1]} × {gt_density.shape[0]}\n"
            f"Density: PSNR={density_psnr:.3f} dB, "
            f"SSIM={density_ssim:.4f} | "
            f"Attribute: PSNR={attribute_psnr:.3f} dB, "
            f"SSIM={attribute_ssim:.4f}"
        ),
        fontsize=14,
    )

    figure.savefig(
        path,
        dpi=300,
    )

    plt.close(
        figure
    )


def save_resolution_arrays(
    output_directory: Path,
    *,
    density: np.ndarray,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_directory / "density.npy",
        density.astype(
            np.float32
        ),
    )

    np.save(
        output_directory / "attribute.npy",
        attribute.astype(
            np.float32
        ),
    )

    np.save(
        output_directory / "valid_mask.npy",
        valid_mask.astype(
            np.bool_
        ),
    )


def save_csv(
    path: Path,
    results: list[ResolutionResult],
) -> None:
    rows = [
        asdict(result)
        for result in results
    ]

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    path: Path,
    *,
    args: argparse.Namespace,
    model: GaussianModel,
    gpu_model: GpuGaussianModel,
    load_seconds: float,
    upload_seconds: float,
    results: list[ResolutionResult],
) -> None:
    payload: dict[str, Any] = {
        "model": str(
            args.model.resolve()
        ),
        "ground_truth_root": str(
            args.gt_root.resolve()
        ),
        "gaussians": model.n_gaussians,
        "particles": model.n_particles,
        "attribute": model.attribute_name,
        "gpu": torch.cuda.get_device_name(
            gpu_model.device
        ),
        "model_load_seconds": load_seconds,
        "gpu_upload_seconds": upload_seconds,
        "gpu_model_memory_megabytes": (
            gpu_model.memory_megabytes()
        ),
        "tile_size": args.tile_size,
        "minimum_eigenvalue": (
            args.minimum_eigenvalue
        ),
        "minimum_pixel_variance": (
            args.minimum_pixel_variance
        ),
        "sigma_extent": args.sigma_extent,
        "view_padding": args.view_padding,
        "beta": args.beta,
        "blob_sigma_pixels": args.blob,
        "relative_density_threshold": (
            args.relative_density_threshold
        ),
        "warmup_frames": args.warmup_frames,
        "timed_frames": args.timed_frames,
        "density_upper_percentile": (
            args.density_upper_percentile
        ),
        "attribute_lower_percentile": (
            args.attribute_lower_percentile
        ),
        "attribute_upper_percentile": (
            args.attribute_upper_percentile
        ),
        "results": [
            asdict(result)
            for result in results
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_markdown_table(
    path: Path,
    results: list[ResolutionResult],
) -> None:
    lines = [
        "| Resolution | Density PSNR ↑ | Density SSIM ↑ | "
        "Attribute PSNR ↑ | Attribute SSIM ↑ | "
        "Attribute MAE ↓ | Time (ms) ↓ | FPS ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.resolution_label} | "
            f"{result.density_psnr:.3f} | "
            f"{result.density_ssim:.4f} | "
            f"{result.attribute_psnr:.3f} | "
            f"{result.attribute_ssim:.4f} | "
            f"{result.attribute_mae:.3f} | "
            f"{result.total_frame_ms:.3f} | "
            f"{result.fps:.2f} |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_latex_quality_table(
    path: Path,
    results: list[ResolutionResult],
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Rendering quality and performance at different output resolutions.}",
        r"\label{tab:resolution_quality_performance}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"Resolution & D-PSNR $\uparrow$ & D-SSIM $\uparrow$ & "
            r"A-PSNR $\uparrow$ & A-SSIM $\uparrow$ & "
            r"A-MAE $\downarrow$ & Time (ms) $\downarrow$ & FPS $\uparrow$ \\"
        ),
        r"\midrule",
    ]

    for result in results:
        lines.append(
            f"{result.resolution_label} & "
            f"{result.density_psnr:.3f} & "
            f"{result.density_ssim:.4f} & "
            f"{result.attribute_psnr:.3f} & "
            f"{result.attribute_ssim:.4f} & "
            f"{result.attribute_mae:.3f} & "
            f"{result.total_frame_ms:.3f} & "
            f"{result.fps:.2f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_latex_performance_table(
    path: Path,
    results: list[ResolutionResult],
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-stage rendering performance at different output resolutions.}",
        r"\label{tab:resolution_stage_performance}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Resolution & Projection & Conditional & "
            r"Intersections & Rendering & Total & FPS \\"
        ),
        r"\midrule",
    ]

    for result in results:
        lines.append(
            f"{result.resolution_label} & "
            f"{result.projection_ms:.3f} & "
            f"{result.conditional_projection_ms:.3f} & "
            f"{result.intersections_ms:.3f} & "
            f"{result.rendering_ms:.3f} & "
            f"{result.total_frame_ms:.3f} & "
            f"{result.fps:.2f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_result(
    result: ResolutionResult,
) -> None:
    print()
    print("-" * 80)
    print(
        f"Resolution: {result.width} x {result.height}"
    )
    print("-" * 80)

    print(
        f"Density PSNR             : "
        f"{result.density_psnr:.6f} dB"
    )

    print(
        f"Density SSIM             : "
        f"{result.density_ssim:.6f}"
    )

    print(
        f"Attribute PSNR           : "
        f"{result.attribute_psnr:.6f} dB"
    )

    print(
        f"Attribute SSIM           : "
        f"{result.attribute_ssim:.6f}"
    )

    print(
        f"Attribute MAE            : "
        f"{result.attribute_mae:.6f}"
    )

    print(
        f"Attribute RMSE           : "
        f"{result.attribute_rmse:.6f}"
    )

    print(
        f"Common attribute pixels : "
        f"{result.attribute_common_pixels:,}"
    )

    print()

    print(
        f"Projection               : "
        f"{result.projection_ms:.3f} ms"
    )

    print(
        f"Conditional projection   : "
        f"{result.conditional_projection_ms:.3f} ms"
    )

    print(
        f"Tile intersections       : "
        f"{result.intersections_ms:.3f} ms"
    )

    print(
        f"Conditional rendering    : "
        f"{result.rendering_ms:.3f} ms"
    )

    print(
        f"Complete frame           : "
        f"{result.total_frame_ms:.3f} ms"
    )

    print(
        f"Frames per second        : "
        f"{result.fps:.2f}"
    )

    print(
        f"Density retained         : "
        f"{100.0 * result.density_retained_fraction:.6f}%"
    )


def print_summary(
    results: list[ResolutionResult],
) -> None:
    print()
    print("=" * 116)
    print(
        "Resolution benchmark summary"
    )
    print("=" * 116)

    header = (
        f"{'Resolution':>12} "
        f"{'D-PSNR':>10} "
        f"{'D-SSIM':>10} "
        f"{'A-PSNR':>10} "
        f"{'A-SSIM':>10} "
        f"{'Proj ms':>10} "
        f"{'Render ms':>11} "
        f"{'Total ms':>10} "
        f"{'FPS':>10}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.resolution_label:>12} "
            f"{result.density_psnr:>10.3f} "
            f"{result.density_ssim:>10.4f} "
            f"{result.attribute_psnr:>10.3f} "
            f"{result.attribute_ssim:>10.4f} "
            f"{result.projection_ms:>10.3f} "
            f"{result.rendering_ms:>11.3f} "
            f"{result.total_frame_ms:>10.3f} "
            f"{result.fps:>10.2f}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "Resolution quality and performance benchmark"
    )
    print("=" * 80)

    print(
        f"Model                  : {args.model}"
    )

    print(
        f"Ground truth root      : {args.gt_root.resolve()}"
    )

    print(
        f"Output                 : {args.output.resolve()}"
    )

    print(
        f"Resolutions            : {args.resolutions}"
    )

    print(
        f"View padding           : {args.view_padding}"
    )

    print(
        f"Beta smoothing         : {args.beta}"
    )

    print(
        f"Blob sigma             : {args.blob} pixels"
    )

    print(
        f"Sigma extent           : {args.sigma_extent}"
    )

    print(
        f"Warm-up frames         : {args.warmup_frames}"
    )

    print(
        f"Timed frames           : {args.timed_frames}"
    )

    print(
        f"GPU                    : "
        f"{torch.cuda.get_device_name(device)}"
    )

    print()

    load_start = perf_counter()

    cpu_model = GaussianModel.load(
        args.model
    )

    load_seconds = (
        perf_counter()
        - load_start
    )

    (
        _minimum,
        _maximum,
        center,
        view_width,
        camera_distance,
    ) = model_bounds(
        cpu_model,
        view_padding=args.view_padding,
    )

    torch.cuda.synchronize(
        device
    )

    upload_start = perf_counter()

    gpu_model = GpuGaussianModel.from_cpu(
        cpu_model,
        device=device,
        minimum_eigenvalue=(
            args.minimum_eigenvalue
        ),
    )

    gpu_model.synchronize()

    upload_seconds = (
        perf_counter()
        - upload_start
    )

    print(
        f"Gaussians              : "
        f"{gpu_model.n_gaussians:,}"
    )

    print(
        f"Particles represented  : "
        f"{gpu_model.n_particles:,}"
    )

    print(
        f"Attribute              : "
        f"{gpu_model.attribute_name}"
    )

    print(
        f"Model load             : "
        f"{load_seconds:.6f} s"
    )

    print(
        f"One-time GPU upload    : "
        f"{upload_seconds:.6f} s"
    )

    print(
        f"GPU model memory       : "
        f"{gpu_model.memory_megabytes():.3f} MiB"
    )

    results: list[ResolutionResult] = []

    for resolution in args.resolutions:
        width = int(
            resolution
        )

        height = int(
            resolution
        )

        print()
        print("=" * 80)
        print(
            f"Benchmarking {width} x {height}"
        )
        print("=" * 80)

        camera = build_camera(
            center=center,
            view_width=view_width,
            camera_distance=camera_distance,
            width=width,
            height=height,
        )

        (
            timing,
            projected,
            intersections,
            render_result,
        ) = benchmark_resolution(
            gpu_model,
            camera,
            width=width,
            height=height,
            tile_size=args.tile_size,
            minimum_eigenvalue=(
                args.minimum_eigenvalue
            ),
            minimum_pixel_variance=(
                args.minimum_pixel_variance
            ),
            sigma_extent=args.sigma_extent,
            beta=args.beta,
            blob_sigma_pixels=args.blob,
            normalize_gaussian_mass=(
                not args.unnormalized_mass
            ),
            relative_density_threshold=(
                args.relative_density_threshold
            ),
            warmup_frames=args.warmup_frames,
            timed_frames=args.timed_frames,
        )

        predicted_density = np.asarray(
            render_result.density_numpy(),
            dtype=np.float64,
        )

        predicted_attribute = np.asarray(
            render_result.attribute_numpy(),
            dtype=np.float64,
        )

        predicted_valid_mask = np.asarray(
            render_result.valid_mask_numpy(),
            dtype=np.bool_,
        )

        (
            gt_density,
            gt_attribute,
            gt_valid_mask,
        ) = load_ground_truth(
            args.gt_root,
            resolution,
        )

        quality = calculate_quality_metrics(
            gt_density=gt_density,
            predicted_density=predicted_density,
            gt_attribute=gt_attribute,
            predicted_attribute=predicted_attribute,
            gt_valid_mask=gt_valid_mask,
            predicted_valid_mask=(
                predicted_valid_mask
            ),
            density_upper_percentile=(
                args.density_upper_percentile
            ),
            attribute_lower_percentile=(
                args.attribute_lower_percentile
            ),
            attribute_upper_percentile=(
                args.attribute_upper_percentile
            ),
        )

        valid_attribute_values = (
            predicted_attribute[
                predicted_valid_mask
                & np.isfinite(
                    predicted_attribute
                )
            ]
        )

        if valid_attribute_values.size == 0:
            raise RuntimeError(
                "Rendered attribute has no finite valid values."
            )

        gt_density_sum = float(
            gt_density.sum()
        )

        predicted_density_sum = float(
            predicted_density.sum()
        )

        density_retained_fraction = (
            predicted_density_sum
            / gt_density_sum
            if gt_density_sum > 0.0
            else float("nan")
        )

        resolution_result = ResolutionResult(
            resolution_label=(
                f"{resolution // 1024}K"
                if resolution % 1024 == 0
                else str(resolution)
            ),
            width=width,
            height=height,
            valid_gaussians=(
                projected.n_valid
            ),
            tile_intersections=(
                intersections.n_intersections
            ),
            density_psnr=(
                quality.density_psnr
            ),
            density_ssim=(
                quality.density_ssim
            ),
            attribute_psnr=(
                quality.attribute_psnr
            ),
            attribute_ssim=(
                quality.attribute_ssim
            ),
            attribute_mae=(
                quality.attribute_mae
            ),
            attribute_rmse=(
                quality.attribute_rmse
            ),
            attribute_common_pixels=(
                quality.attribute_common_pixels
            ),
            projection_ms=(
                timing.projection_ms
            ),
            conditional_projection_ms=(
                timing.conditional_projection_ms
            ),
            intersections_ms=(
                timing.intersections_ms
            ),
            rendering_ms=(
                timing.rendering_ms
            ),
            total_frame_ms=(
                timing.total_frame_ms
            ),
            fps=timing.fps,
            density_sum=(
                predicted_density_sum
            ),
            density_minimum=float(
                predicted_density.min()
            ),
            density_maximum=float(
                predicted_density.max()
            ),
            density_retained_fraction=(
                density_retained_fraction
            ),
            attribute_minimum=float(
                valid_attribute_values.min()
            ),
            attribute_maximum=float(
                valid_attribute_values.max()
            ),
            attribute_mean=float(
                valid_attribute_values.mean()
            ),
        )

        results.append(
            resolution_result
        )

        resolution_output_directory = (
            args.output / str(resolution)
        )

        save_resolution_arrays(
            resolution_output_directory,
            density=predicted_density,
            attribute=predicted_attribute,
            valid_mask=predicted_valid_mask,
        )

        save_gt_prediction_comparison(
            resolution_output_directory
            / "gt_vs_prediction.png",
            gt_density=gt_density,
            predicted_density=predicted_density,
            gt_attribute=gt_attribute,
            predicted_attribute=predicted_attribute,
            gt_valid_mask=gt_valid_mask,
            predicted_valid_mask=(
                predicted_valid_mask
            ),
            density_upper_percentile=(
                args.density_upper_percentile
            ),
            attribute_lower_percentile=(
                args.attribute_lower_percentile
            ),
            attribute_upper_percentile=(
                args.attribute_upper_percentile
            ),
            error_percentile=(
                args.comparison_error_percentile
            ),
            attribute_name=(
                gpu_model.attribute_name
                if gpu_model.attribute_name is not None
                else "Conditional attribute"
            ),
            density_psnr=(
                quality.density_psnr
            ),
            density_ssim=(
                quality.density_ssim
            ),
            attribute_psnr=(
                quality.attribute_psnr
            ),
            attribute_ssim=(
                quality.attribute_ssim
            ),
        )

        print_result(
            resolution_result
        )

        print(
            "Comparison image          : "
            f"{resolution_output_directory / 'gt_vs_prediction.png'}"
        )

        del predicted_density
        del predicted_attribute
        del predicted_valid_mask
        del gt_density
        del gt_attribute
        del gt_valid_mask
        del render_result
        del intersections
        del projected

        torch.cuda.empty_cache()

    print_summary(
        results
    )

    save_csv(
        args.output / "results.csv",
        results,
    )

    save_json(
        args.output / "results.json",
        args=args,
        model=cpu_model,
        gpu_model=gpu_model,
        load_seconds=load_seconds,
        upload_seconds=upload_seconds,
        results=results,
    )

    save_markdown_table(
        args.output / "results_table.md",
        results,
    )

    save_latex_quality_table(
        args.output / "quality_table.tex",
        results,
    )

    save_latex_performance_table(
        args.output / "performance_table.tex",
        results,
    )

    print()
    print("Saved benchmark files")
    print("-" * 80)

    for path in (
        args.output / "results.csv",
        args.output / "results.json",
        args.output / "results_table.md",
        args.output / "quality_table.tex",
        args.output / "performance_table.tex",
    ):
        print(path)


if __name__ == "__main__":
    main()