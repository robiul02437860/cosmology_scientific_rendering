from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)


FloatArray = NDArray[np.floating]


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
    # "/home/robiul/Particle_flow/HACC_project/output/hacc_process_parallel_v2/full_hacc_1pct_components_4/simple_model.npz"
)

DEFAULT_DATA_DIR = Path(
    "/home/robiul/Particle_flow/HACC_project/datasets/prepared_illustris3_dm"
    # "/home/robiul/Particle_flow/HACC_dataset/prepared_hacc_mpicosmo"
)

DEFAULT_OUTPUT = Path(
    "ground_truth/resolution_benchmark/Illustris"
)

DEFAULT_RESOLUTIONS = (
    1024,
    2048,
    4096,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate single-plane orthographic particle ground truth "
            "using histogram projection followed by Gaussian blob filtering."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=(
            "Saved Gaussian model used only to reproduce exactly the same "
            "camera center and view width as the GPU renderer."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing positions.npy and the attribute array."
        ),
    )

    parser.add_argument(
        "--positions-file",
        type=str,
        default="positions.npy",
    )

    parser.add_argument(
        "--attribute-file",
        type=str,
        default="velocity_dispersion.npy",
        # default="speed.npy",
        help=(
            "Scalar particle-attribute filename inside --data-dir. "
            "For example velocity_dispersion.npy or subfind_density.npy."
        ),
    )

    parser.add_argument(
        "--attribute-name",
        type=str,
        default="SubfindVelDisp",
        # default="Velocity speed"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help=(
            "Number of particles processed in each memory-mapped chunk."
        ),
    )

    parser.add_argument(
        "--blob",
        type=float,
        default=2.0,
        help=(
            "Gaussian blob sigma measured in pixels. This value remains "
            "fixed at every image resolution, matching the base code."
        ),
    )

    parser.add_argument(
        "--blob-truncate",
        type=float,
        default=3.0,
        help=(
            "Gaussian-filter support radius in multiples of sigma."
        ),
    )

    parser.add_argument(
        "--view-padding",
        type=float,
        default=0.02,
        help=(
            "Fractional view padding. The default 0.02 reproduces the "
            "GPU renderer's view_width = max_xy_span * 1.02."
        ),
    )

    parser.add_argument(
        "--relative-density-threshold",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--minimum-density-threshold",
        type=float,
        default=1.0e-12,
    )

    parser.add_argument(
        "--accumulator-dtype",
        choices=("float32", "float64"),
        default="float64",
        help=(
            "Accumulation dtype before filtering. float64 is safer for "
            "large particle counts."
        ),
    )

    parser.add_argument(
        "--flip-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flip image Y to match the GPU orthographic projection, where "
            "camera-space positive Y maps toward smaller image rows."
        ),
    )

    parser.add_argument(
        "--save-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Gaussian model does not exist: {args.model}"
        )

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {args.data_dir}"
        )

    positions_path = (
        args.data_dir / args.positions_file
    )

    attribute_path = (
        args.data_dir / args.attribute_file
    )

    if not positions_path.exists():
        raise FileNotFoundError(
            f"Position array does not exist: {positions_path}"
        )

    if not attribute_path.exists():
        raise FileNotFoundError(
            f"Attribute array does not exist: {attribute_path}"
        )

    if not args.resolutions:
        raise ValueError(
            "At least one resolution is required."
        )

    for resolution in args.resolutions:
        if resolution <= 0:
            raise ValueError(
                "Each resolution must be positive, "
                f"got {resolution}."
            )

    if args.chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    if args.blob < 0.0:
        raise ValueError(
            "blob must be nonnegative."
        )

    if args.blob_truncate <= 0.0:
        raise ValueError(
            "blob_truncate must be positive."
        )

    if args.view_padding < 0.0:
        raise ValueError(
            "view_padding must be nonnegative."
        )

    if args.relative_density_threshold < 0.0:
        raise ValueError(
            "relative_density_threshold must be nonnegative."
        )

    if args.minimum_density_threshold <= 0.0:
        raise ValueError(
            "minimum_density_threshold must be positive."
        )


def load_particle_arrays(
    *,
    data_dir: Path,
    positions_file: str,
    attribute_file: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Path,
    Path,
]:
    positions_path = (
        data_dir / positions_file
    )

    attribute_path = (
        data_dir / attribute_file
    )

    positions = np.load(
        positions_path,
        mmap_mode="r",
    )

    attribute = np.load(
        attribute_path,
        mmap_mode="r",
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            "positions must have shape (N,3), "
            f"got {positions.shape}."
        )

    if attribute.ndim == 2:
        if attribute.shape[1] != 1:
            raise ValueError(
                "A two-dimensional scalar attribute must have "
                f"shape (N,1), got {attribute.shape}."
            )

        attribute = attribute[:, 0]

    if attribute.ndim != 1:
        raise ValueError(
            "attribute must have shape (N,) or (N,1), "
            f"got {attribute.shape}."
        )

    if len(positions) != len(attribute):
        raise ValueError(
            "Particle and attribute counts differ: "
            f"{len(positions):,} versus "
            f"{len(attribute):,}."
        )

    return (
        positions,
        attribute,
        positions_path,
        attribute_path,
    )


def derive_camera_from_model(
    model: GaussianModel,
    *,
    view_padding: float,
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
]:
    """Derive the same XY view used by the GPU renderer."""

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

    if view_width <= 0.0:
        raise ValueError(
            "Derived camera view width is not positive."
        )

    x_min = float(
        center[0]
        - 0.5 * view_width
    )

    x_max = float(
        center[0]
        + 0.5 * view_width
    )

    y_min = float(
        center[1]
        - 0.5 * view_width
    )

    y_max = float(
        center[1]
        + 0.5 * view_width
    )

    return (
        center,
        view_width,
        x_min,
        x_max,
        y_min,
        y_max,
    )


def project_particle_chunk(
    positions: np.ndarray,
    attributes: np.ndarray,
    *,
    resolution: int,
    x_min: float,
    y_min: float,
    view_width: float,
    flip_y: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Project one particle chunk and return flat indices and values."""

    finite = (
        np.isfinite(
            positions[:, 0]
        )
        & np.isfinite(
            positions[:, 1]
        )
        & np.isfinite(
            attributes
        )
    )

    if not np.any(finite):
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.float64,
            ),
        )

    x = np.asarray(
        positions[
            finite,
            0,
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        positions[
            finite,
            1,
        ],
        dtype=np.float64,
    )

    values = np.asarray(
        attributes[
            finite
        ],
        dtype=np.float64,
    )

    pixels_per_world_unit = (
        float(resolution)
        / view_width
    )

    pixel_x_float = (
        (x - x_min)
        * pixels_per_world_unit
    )

    # The GPU renderer uses:
    #
    # pixel_y = H/2 - camera_y * scale
    #
    # which corresponds to flipping the world-space Y histogram.
    pixel_y_up_float = (
        (y - y_min)
        * pixels_per_world_unit
    )

    pixel_x = np.floor(
        pixel_x_float
    ).astype(
        np.int64,
        copy=False,
    )

    pixel_y_up = np.floor(
        pixel_y_up_float
    ).astype(
        np.int64,
        copy=False,
    )

    if flip_y:
        pixel_y = (
            resolution
            - 1
            - pixel_y_up
        )
    else:
        pixel_y = pixel_y_up

    inside = (
        (pixel_x >= 0)
        & (pixel_x < resolution)
        & (pixel_y >= 0)
        & (pixel_y < resolution)
    )

    if not np.any(inside):
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.float64,
            ),
        )

    pixel_x = pixel_x[
        inside
    ]

    pixel_y = pixel_y[
        inside
    ]

    values = values[
        inside
    ]

    flat_indices = (
        pixel_y * resolution
        + pixel_x
    )

    return (
        flat_indices,
        values,
    )


def accumulate_histograms(
    positions: np.ndarray,
    attribute: np.ndarray,
    *,
    resolution: int,
    chunk_size: int,
    x_min: float,
    y_min: float,
    view_width: float,
    flip_y: bool,
    accumulator_dtype: np.dtype,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Accumulate density and attribute numerator before blob filtering."""

    shape = (
        resolution,
        resolution,
    )

    density = np.zeros(
        shape,
        dtype=accumulator_dtype,
    )

    attribute_numerator = np.zeros(
        shape,
        dtype=accumulator_dtype,
    )

    density_flat = density.reshape(-1)

    numerator_flat = (
        attribute_numerator.reshape(-1)
    )

    number_of_pixels = (
        resolution * resolution
    )

    particle_count = len(
        positions
    )

    accepted_particles = 0
    rejected_particles = 0

    for start in range(
        0,
        particle_count,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            particle_count,
        )

        position_chunk = np.asarray(
            positions[start:end]
        )

        attribute_chunk = np.asarray(
            attribute[start:end]
        ).reshape(-1)

        (
            flat_indices,
            values,
        ) = project_particle_chunk(
            position_chunk,
            attribute_chunk,
            resolution=resolution,
            x_min=x_min,
            y_min=y_min,
            view_width=view_width,
            flip_y=flip_y,
        )

        accepted_in_chunk = int(
            flat_indices.size
        )

        accepted_particles += (
            accepted_in_chunk
        )

        rejected_particles += (
            end
            - start
            - accepted_in_chunk
        )

        if accepted_in_chunk > 0:
            density_increment = np.bincount(
                flat_indices,
                minlength=number_of_pixels,
            )

            numerator_increment = np.bincount(
                flat_indices,
                weights=values,
                minlength=number_of_pixels,
            )

            density_flat += (
                density_increment.astype(
                    accumulator_dtype,
                    copy=False,
                )
            )

            numerator_flat += (
                numerator_increment.astype(
                    accumulator_dtype,
                    copy=False,
                )
            )

        percentage = (
            100.0
            * float(end)
            / float(particle_count)
        )

        print(
            f"\r  Projecting particles: "
            f"{end:,}/{particle_count:,} "
            f"({percentage:6.2f}%)",
            end="",
            flush=True,
        )

    print()

    return (
        density,
        attribute_numerator,
        accepted_particles,
        rejected_particles,
    )


def apply_blob_filter(
    density: np.ndarray,
    attribute_numerator: np.ndarray,
    *,
    blob_sigma_pixels: float,
    truncate: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Apply the same isotropic screen-space Gaussian blob to both moments."""

    if blob_sigma_pixels <= 0.0:
        return (
            density,
            attribute_numerator,
        )

    filtered_density = gaussian_filter(
        density,
        sigma=blob_sigma_pixels,
        mode="constant",
        cval=0.0,
        truncate=truncate,
        output=density.dtype,
    )

    filtered_numerator = gaussian_filter(
        attribute_numerator,
        sigma=blob_sigma_pixels,
        mode="constant",
        cval=0.0,
        truncate=truncate,
        output=attribute_numerator.dtype,
    )

    return (
        filtered_density,
        filtered_numerator,
    )


def compute_attribute_field(
    density: np.ndarray,
    attribute_numerator: np.ndarray,
    *,
    relative_density_threshold: float,
    minimum_density_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    maximum_density = float(
        density.max(initial=0.0)
    )

    density_threshold = max(
        minimum_density_threshold,
        maximum_density
        * relative_density_threshold,
    )

    valid_mask = (
        np.isfinite(density)
        & np.isfinite(
            attribute_numerator
        )
        & (
            density
            > density_threshold
        )
    )

    attribute = np.full(
        density.shape,
        np.nan,
        dtype=np.float64,
    )

    attribute[valid_mask] = (
        attribute_numerator[
            valid_mask
        ]
        / density[
            valid_mask
        ]
    )

    return (
        attribute,
        valid_mask,
        density_threshold,
    )


def save_density_preview(
    path: Path,
    density: np.ndarray,
) -> None:
    positive = density[
        np.isfinite(density)
        & (density > 0.0)
    ]

    if positive.size == 0:
        image = np.zeros_like(
            density,
            dtype=np.float64,
        )

    else:
        upper = float(
            np.percentile(
                positive,
                99.9,
            )
        )

        image = np.log1p(
            np.maximum(
                density,
                0.0,
            )
        )

        image /= max(
            np.log1p(upper),
            1.0e-12,
        )

        image = np.clip(
            image,
            0.0,
            1.0,
        )

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="inferno",
    )


def save_attribute_preview(
    path: Path,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[
    float,
    float,
]:
    finite_values = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if finite_values.size == 0:
        raise RuntimeError(
            "No finite attribute values are available."
        )

    lower = float(
        np.percentile(
            finite_values,
            1.0,
        )
    )

    upper = float(
        np.percentile(
            finite_values,
            99.0,
        )
    )

    if upper <= lower:
        upper = lower + 1.0

    image = np.asarray(
        attribute,
        dtype=np.float64,
    ).copy()

    image[
        ~valid_mask
    ] = np.nan

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )

    return (
        lower,
        upper,
    )


def save_combined_preview(
    path: Path,
    *,
    density: np.ndarray,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
    attribute_name: str,
) -> None:
    positive_density = density[
        np.isfinite(density)
        & (density > 0.0)
    ]

    density_upper = (
        float(
            np.percentile(
                positive_density,
                99.9,
            )
        )
        if positive_density.size > 0
        else 1.0
    )

    valid_attribute = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if valid_attribute.size == 0:
        raise RuntimeError(
            "No finite valid attribute values are available."
        )

    attribute_lower = float(
        np.percentile(
            valid_attribute,
            1.0,
        )
    )

    attribute_upper = float(
        np.percentile(
            valid_attribute,
            99.0,
        )
    )

    if attribute_upper <= attribute_lower:
        attribute_upper = (
            attribute_lower + 1.0
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    density_artist = axes[0].imshow(
        np.log1p(
            np.maximum(
                density,
                0.0,
            )
        ),
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=np.log1p(
            density_upper
        ),
    )

    axes[0].set_title(
        "Particle density GT"
    )

    axes[0].set_axis_off()

    figure.colorbar(
        density_artist,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="log(1 + density)",
    )

    masked_attribute = np.ma.masked_where(
        ~valid_mask,
        attribute,
    )

    attribute_artist = axes[1].imshow(
        masked_attribute,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
    )

    axes[1].set_title(
        f"{attribute_name} GT"
    )

    axes[1].set_axis_off()

    figure.colorbar(
        attribute_artist,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label=attribute_name,
    )

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(
        figure
    )


def generate_resolution(
    positions: np.ndarray,
    particle_attribute: np.ndarray,
    *,
    resolution: int,
    output_root: Path,
    chunk_size: int,
    x_min: float,
    y_min: float,
    view_width: float,
    flip_y: bool,
    blob_sigma_pixels: float,
    blob_truncate: float,
    relative_density_threshold: float,
    minimum_density_threshold: float,
    accumulator_dtype: np.dtype,
    attribute_name: str,
    save_previews: bool,
) -> dict[str, object]:
    print()
    print("=" * 80)
    print(
        f"Generating {resolution} x {resolution} blob particle GT"
    )
    print("=" * 80)

    total_start = perf_counter()

    accumulation_start = perf_counter()

    (
        density,
        attribute_numerator,
        accepted_particles,
        rejected_particles,
    ) = accumulate_histograms(
        positions,
        particle_attribute,
        resolution=resolution,
        chunk_size=chunk_size,
        x_min=x_min,
        y_min=y_min,
        view_width=view_width,
        flip_y=flip_y,
        accumulator_dtype=(
            accumulator_dtype
        ),
    )

    accumulation_seconds = (
        perf_counter()
        - accumulation_start
    )

    print(
        f"  Applying Gaussian blob: "
        f"sigma={blob_sigma_pixels} pixels"
    )

    filter_start = perf_counter()

    (
        density,
        attribute_numerator,
    ) = apply_blob_filter(
        density,
        attribute_numerator,
        blob_sigma_pixels=(
            blob_sigma_pixels
        ),
        truncate=blob_truncate,
    )

    filter_seconds = (
        perf_counter()
        - filter_start
    )

    (
        attribute,
        valid_mask,
        density_threshold,
    ) = compute_attribute_field(
        density,
        attribute_numerator,
        relative_density_threshold=(
            relative_density_threshold
        ),
        minimum_density_threshold=(
            minimum_density_threshold
        ),
    )

    output_directory = (
        output_root / str(resolution)
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_directory
        / "density.npy",
        density.astype(
            np.float32
        ),
    )

    np.save(
        output_directory
        / "attribute_numerator.npy",
        attribute_numerator.astype(
            np.float32
        ),
    )

    np.save(
        output_directory
        / "attribute.npy",
        attribute.astype(
            np.float32
        ),
    )

    np.save(
        output_directory
        / "valid_mask.npy",
        valid_mask.astype(
            np.bool_
        ),
    )

    display_lower = float("nan")
    display_upper = float("nan")

    if save_previews:
        save_density_preview(
            output_directory
            / "density.png",
            density,
        )

        (
            display_lower,
            display_upper,
        ) = save_attribute_preview(
            output_directory
            / "attribute.png",
            attribute,
            valid_mask,
        )

        save_combined_preview(
            output_directory
            / "comparison.png",
            density=density,
            attribute=attribute,
            valid_mask=valid_mask,
            attribute_name=attribute_name,
        )

    finite_attribute_values = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if finite_attribute_values.size == 0:
        raise RuntimeError(
            "Generated GT has no valid attribute values."
        )

    total_seconds = (
        perf_counter()
        - total_start
    )

    statistics: dict[str, object] = {
        "resolution": resolution,
        "width": resolution,
        "height": resolution,
        "accepted_particles": (
            accepted_particles
        ),
        "rejected_particles": (
            rejected_particles
        ),
        "blob_sigma_pixels": (
            blob_sigma_pixels
        ),
        "blob_truncate": (
            blob_truncate
        ),
        "view_width": view_width,
        "x_min": x_min,
        "y_min": y_min,
        "flip_y": flip_y,
        "density_threshold": (
            density_threshold
        ),
        "density_sum": float(
            density.sum()
        ),
        "density_minimum": float(
            density.min()
        ),
        "density_maximum": float(
            density.max()
        ),
        "density_mean": float(
            density.mean()
        ),
        "valid_attribute_pixels": int(
            valid_mask.sum()
        ),
        "attribute_minimum": float(
            finite_attribute_values.min()
        ),
        "attribute_maximum": float(
            finite_attribute_values.max()
        ),
        "attribute_mean": float(
            finite_attribute_values.mean()
        ),
        "attribute_display_lower": (
            display_lower
        ),
        "attribute_display_upper": (
            display_upper
        ),
        "histogram_accumulation_seconds": (
            accumulation_seconds
        ),
        "blob_filter_seconds": (
            filter_seconds
        ),
        "total_seconds": (
            total_seconds
        ),
        "output_directory": str(
            output_directory.resolve()
        ),
    }

    (
        output_directory
        / "statistics.json"
    ).write_text(
        json.dumps(
            statistics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Accepted particles       : "
        f"{accepted_particles:,}"
    )

    print(
        f"Rejected particles       : "
        f"{rejected_particles:,}"
    )

    print(
        f"Blob sigma               : "
        f"{blob_sigma_pixels:.6f} pixels"
    )

    print(
        f"Histogram accumulation   : "
        f"{accumulation_seconds:.3f} s"
    )

    print(
        f"Blob filtering           : "
        f"{filter_seconds:.3f} s"
    )

    print(
        f"Density sum              : "
        f"{density.sum():.12g}"
    )

    print(
        f"Density threshold        : "
        f"{density_threshold:.12g}"
    )

    print(
        f"Valid attribute pixels   : "
        f"{valid_mask.sum():,}"
    )

    print(
        f"Attribute range          : "
        f"[{finite_attribute_values.min():.9g}, "
        f"{finite_attribute_values.max():.9g}]"
    )

    print(
        f"Total GT time            : "
        f"{total_seconds:.3f} s"
    )

    print(
        f"Output                    : "
        f"{output_directory.resolve()}"
    )

    return statistics


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_root = args.output

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    accumulator_dtype = (
        np.float64
        if args.accumulator_dtype == "float64"
        else np.float32
    )

    (
        positions,
        particle_attribute,
        positions_path,
        attribute_path,
    ) = load_particle_arrays(
        data_dir=args.data_dir,
        positions_file=args.positions_file,
        attribute_file=args.attribute_file,
    )

    print("=" * 80)
    print(
        "Single-plane blob particle ground-truth generation"
    )
    print("=" * 80)

    print(
        f"Model                   : "
        f"{args.model.resolve()}"
    )

    print(
        f"Positions               : "
        f"{positions_path.resolve()}"
    )

    print(
        f"Attribute               : "
        f"{attribute_path.resolve()}"
    )

    print(
        f"Attribute name          : "
        f"{args.attribute_name}"
    )

    print(
        f"Particle count          : "
        f"{len(positions):,}"
    )

    print(
        f"Resolutions             : "
        f"{args.resolutions}"
    )

    print(
        f"Chunk size              : "
        f"{args.chunk_size:,}"
    )

    print(
        f"Blob sigma              : "
        f"{args.blob} pixels at every resolution"
    )

    print(
        f"View padding            : "
        f"{args.view_padding}"
    )

    print(
        f"Accumulator dtype       : "
        f"{args.accumulator_dtype}"
    )

    print(
        f"Output root             : "
        f"{output_root.resolve()}"
    )

    model = GaussianModel.load(
        args.model
    )

    (
        camera_center,
        view_width,
        x_min,
        x_max,
        y_min,
        y_max,
    ) = derive_camera_from_model(
        model,
        view_padding=args.view_padding,
    )

    print()
    print("Camera")
    print("-" * 80)

    print(
        f"Center                  : "
        f"{camera_center}"
    )

    print(
        f"View width              : "
        f"{view_width:.12g}"
    )

    print(
        f"X range                 : "
        f"[{x_min:.12g}, {x_max:.12g}]"
    )

    print(
        f"Y range                 : "
        f"[{y_min:.12g}, {y_max:.12g}]"
    )

    results: list[
        dict[str, object]
    ] = []

    for resolution in args.resolutions:
        result = generate_resolution(
            positions,
            particle_attribute,
            resolution=int(resolution),
            output_root=output_root,
            chunk_size=args.chunk_size,
            x_min=x_min,
            y_min=y_min,
            view_width=view_width,
            flip_y=args.flip_y,
            blob_sigma_pixels=args.blob,
            blob_truncate=args.blob_truncate,
            relative_density_threshold=(
                args.relative_density_threshold
            ),
            minimum_density_threshold=(
                args.minimum_density_threshold
            ),
            accumulator_dtype=(
                accumulator_dtype
            ),
            attribute_name=(
                args.attribute_name
            ),
            save_previews=(
                args.save_previews
            ),
        )

        results.append(
            result
        )

    metadata: dict[str, object] = {
        "description": (
            "Single-plane orthographic particle GT generated by histogram "
            "projection followed by fixed screen-space Gaussian filtering."
        ),
        "model": str(
            args.model.resolve()
        ),
        "positions": str(
            positions_path.resolve()
        ),
        "attribute": str(
            attribute_path.resolve()
        ),
        "attribute_name": (
            args.attribute_name
        ),
        "particle_count": int(
            len(positions)
        ),
        "camera_center": (
            camera_center.tolist()
        ),
        "view_width": (
            view_width
        ),
        "x_range": [
            x_min,
            x_max,
        ],
        "y_range": [
            y_min,
            y_max,
        ],
        "view_padding": (
            args.view_padding
        ),
        "flip_y": (
            args.flip_y
        ),
        "blob_sigma_pixels": (
            args.blob
        ),
        "blob_sigma_policy": (
            "fixed screen-space sigma at every resolution"
        ),
        "blob_truncate": (
            args.blob_truncate
        ),
        "relative_density_threshold": (
            args.relative_density_threshold
        ),
        "minimum_density_threshold": (
            args.minimum_density_threshold
        ),
        "accumulator_dtype": (
            args.accumulator_dtype
        ),
        "resolutions": results,
    }

    (
        output_root
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "Ground-truth generation complete"
    )
    print("=" * 80)

    for result in results:
        print(
            f"{result['resolution']} x "
            f"{result['resolution']}  "
            f"{result['total_seconds']:.3f} s"
        )

    print()
    print(
        f"Metadata                : "
        f"{(output_root / 'metadata.json').resolve()}"
    )


if __name__ == "__main__":
    main()



# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from time import perf_counter

# import matplotlib.pyplot as plt
# from matplotlib import colormaps
# import numpy as np
# from numpy.typing import NDArray
# from scipy.ndimage import gaussian_filter

# from scientific_gsplat_renderer.data.gaussian_model import (
#     GaussianModel,
# )


# FloatArray = NDArray[np.floating]


# DEFAULT_MODEL = Path(
#     "/home/robiul/Particle_flow/HACC_project/output/"
#     "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
#     # "/home/robiul/Particle_flow/HACC_project/output/hacc_process_parallel_v2/full_hacc_1pct_components_4/simple_model.npz"
# )

# DEFAULT_DATA_DIR = Path(
#     "/home/robiul/Particle_flow/HACC_project/datasets/prepared_illustris3_dm"
#     # "/home/robiul/Particle_flow/HACC_dataset/prepared_hacc_mpicosmo"
# )

# DEFAULT_OUTPUT = Path(
#     "ground_truth/resolution_benchmark/Illustris"
#     # "ground_truth/resolution_benchmark/HACC"
    
# )

# DEFAULT_RESOLUTIONS = (
#     1024,
#     2048,
#     4096,
# )


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Generate single-plane orthographic particle ground truth "
#             "using histogram projection followed by Gaussian blob filtering."
#         )
#     )

#     parser.add_argument(
#         "--model",
#         type=Path,
#         default=DEFAULT_MODEL,
#         help=(
#             "Saved Gaussian model used only to reproduce exactly the same "
#             "camera center and view width as the GPU renderer."
#         ),
#     )

#     parser.add_argument(
#         "--data-dir",
#         type=Path,
#         default=DEFAULT_DATA_DIR,
#         help=(
#             "Directory containing positions.npy and the attribute array."
#         ),
#     )

#     parser.add_argument(
#         "--positions-file",
#         type=str,
#         default="positions.npy",
#     )

#     parser.add_argument(
#         "--attribute-file",
#         type=str,
#         default="velocity_dispersion.npy",
#         # default="speed.npy",
#         help=(
#             "Scalar particle-attribute filename inside --data-dir. "
#             "For example velocity_dispersion.npy or subfind_density.npy."
#         ),
#     )

#     parser.add_argument(
#         "--attribute-name",
#         type=str,
#         default="SubfindVelDisp",
#         # default="Velocity speed"
#     )

#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=DEFAULT_OUTPUT,
#     )

#     parser.add_argument(
#         "--resolutions",
#         type=int,
#         nargs="+",
#         default=list(DEFAULT_RESOLUTIONS),
#     )

#     parser.add_argument(
#         "--chunk-size",
#         type=int,
#         default=2_000_000,
#         help=(
#             "Number of particles processed in each memory-mapped chunk."
#         ),
#     )

#     parser.add_argument(
#         "--blob",
#         type=float,
#         default=2.0,
#         help=(
#             "Gaussian blob sigma measured in pixels. This value remains "
#             "fixed at every image resolution, matching the base code."
#         ),
#     )

#     parser.add_argument(
#         "--blob-truncate",
#         type=float,
#         default=3.0,
#         help=(
#             "Gaussian-filter support radius in multiples of sigma."
#         ),
#     )

#     parser.add_argument(
#         "--view-padding",
#         type=float,
#         default=0.0,
#         help=(
#             "Fractional view padding. The default 0.02 reproduces the "
#             "GPU renderer's view_width = max_xy_span * 1.02."
#         ),
#     )

#     parser.add_argument(
#         "--relative-density-threshold",
#         type=float,
#         default=1.0e-6,
#     )

#     parser.add_argument(
#         "--minimum-density-threshold",
#         type=float,
#         default=1.0e-12,
#     )

#     parser.add_argument(
#         "--accumulator-dtype",
#         choices=("float32", "float64"),
#         default="float64",
#         help=(
#             "Accumulation dtype before filtering. float64 is safer for "
#             "large particle counts."
#         ),
#     )

#     parser.add_argument(
#         "--flip-y",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help=(
#             "Flip image Y to match the GPU orthographic projection, where "
#             "camera-space positive Y maps toward smaller image rows."
#         ),
#     )

#     parser.add_argument(
#         "--density-percentile",
#         type=float,
#         default=99.9,
#         help=(
#             "Positive-density percentile used as the shared preview "
#             "luminance maximum."
#         ),
#     )

#     parser.add_argument(
#         "--attribute-lower-percentile",
#         type=float,
#         default=1.0,
#     )

#     parser.add_argument(
#         "--attribute-upper-percentile",
#         type=float,
#         default=99.0,
#     )

#     parser.add_argument(
#         "--density-colormap",
#         type=str,
#         default="inferno",
#     )

#     parser.add_argument(
#         "--attribute-colormap",
#         type=str,
#         default="viridis",
#     )

#     parser.add_argument(
#         "--save-previews",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )

#     return parser.parse_args()


# def validate_args(
#     args: argparse.Namespace,
# ) -> None:
#     if not args.model.exists():
#         raise FileNotFoundError(
#             f"Gaussian model does not exist: {args.model}"
#         )

#     if not args.data_dir.exists():
#         raise FileNotFoundError(
#             f"Data directory does not exist: {args.data_dir}"
#         )

#     positions_path = (
#         args.data_dir / args.positions_file
#     )

#     attribute_path = (
#         args.data_dir / args.attribute_file
#     )

#     if not positions_path.exists():
#         raise FileNotFoundError(
#             f"Position array does not exist: {positions_path}"
#         )

#     if not attribute_path.exists():
#         raise FileNotFoundError(
#             f"Attribute array does not exist: {attribute_path}"
#         )

#     if not args.resolutions:
#         raise ValueError(
#             "At least one resolution is required."
#         )

#     for resolution in args.resolutions:
#         if resolution <= 0:
#             raise ValueError(
#                 "Each resolution must be positive, "
#                 f"got {resolution}."
#             )

#     if args.chunk_size <= 0:
#         raise ValueError(
#             "chunk_size must be positive."
#         )

#     if args.blob < 0.0:
#         raise ValueError(
#             "blob must be nonnegative."
#         )

#     if args.blob_truncate <= 0.0:
#         raise ValueError(
#             "blob_truncate must be positive."
#         )

#     if args.view_padding < 0.0:
#         raise ValueError(
#             "view_padding must be nonnegative."
#         )

#     if args.relative_density_threshold < 0.0:
#         raise ValueError(
#             "relative_density_threshold must be nonnegative."
#         )

#     if args.minimum_density_threshold <= 0.0:
#         raise ValueError(
#             "minimum_density_threshold must be positive."
#         )

#     if not (
#         0.0
#         < args.density_percentile
#         <= 100.0
#     ):
#         raise ValueError(
#             "density_percentile must be in (0, 100]."
#         )

#     if not (
#         0.0
#         <= args.attribute_lower_percentile
#         < args.attribute_upper_percentile
#         <= 100.0
#     ):
#         raise ValueError(
#             "Attribute percentiles must satisfy "
#             "0 <= lower < upper <= 100."
#         )

#     for colormap_name in (
#         args.density_colormap,
#         args.attribute_colormap,
#     ):
#         if colormap_name not in colormaps:
#             raise ValueError(
#                 f"Unknown Matplotlib colormap: {colormap_name!r}"
#             )


# def load_particle_arrays(
#     *,
#     data_dir: Path,
#     positions_file: str,
#     attribute_file: str,
# ) -> tuple[
#     np.ndarray,
#     np.ndarray,
#     Path,
#     Path,
# ]:
#     positions_path = (
#         data_dir / positions_file
#     )

#     attribute_path = (
#         data_dir / attribute_file
#     )

#     positions = np.load(
#         positions_path,
#         mmap_mode="r",
#     )

#     attribute = np.load(
#         attribute_path,
#         mmap_mode="r",
#     )

#     if (
#         positions.ndim != 2
#         or positions.shape[1] != 3
#     ):
#         raise ValueError(
#             "positions must have shape (N,3), "
#             f"got {positions.shape}."
#         )

#     if attribute.ndim == 2:
#         if attribute.shape[1] != 1:
#             raise ValueError(
#                 "A two-dimensional scalar attribute must have "
#                 f"shape (N,1), got {attribute.shape}."
#             )

#         attribute = attribute[:, 0]

#     if attribute.ndim != 1:
#         raise ValueError(
#             "attribute must have shape (N,) or (N,1), "
#             f"got {attribute.shape}."
#         )

#     if len(positions) != len(attribute):
#         raise ValueError(
#             "Particle and attribute counts differ: "
#             f"{len(positions):,} versus "
#             f"{len(attribute):,}."
#         )

#     return (
#         positions,
#         attribute,
#         positions_path,
#         attribute_path,
#     )


# def derive_camera_from_model(
#     model: GaussianModel,
#     *,
#     view_padding: float,
# ) -> tuple[
#     np.ndarray,
#     float,
#     float,
#     float,
#     float,
#     float,
# ]:
#     """Derive the same XY view used by the GPU renderer."""

#     minimum = np.asarray(
#         model.means.min(axis=0),
#         dtype=np.float64,
#     )

#     maximum = np.asarray(
#         model.means.max(axis=0),
#         dtype=np.float64,
#     )

#     center = 0.5 * (
#         minimum + maximum
#     )

#     span = maximum - minimum

#     view_width = float(
#         max(
#             float(span[0]),
#             float(span[1]),
#         )
#         * (1.0 + view_padding)
#     )

#     if view_width <= 0.0:
#         raise ValueError(
#             "Derived camera view width is not positive."
#         )

#     x_min = float(
#         center[0]
#         - 0.5 * view_width
#     )

#     x_max = float(
#         center[0]
#         + 0.5 * view_width
#     )

#     y_min = float(
#         center[1]
#         - 0.5 * view_width
#     )

#     y_max = float(
#         center[1]
#         + 0.5 * view_width
#     )

#     return (
#         center,
#         view_width,
#         x_min,
#         x_max,
#         y_min,
#         y_max,
#     )


# def project_particle_chunk(
#     positions: np.ndarray,
#     attributes: np.ndarray,
#     *,
#     resolution: int,
#     x_min: float,
#     y_min: float,
#     view_width: float,
#     flip_y: bool,
# ) -> tuple[
#     np.ndarray,
#     np.ndarray,
# ]:
#     """Project one particle chunk and return flat indices and values."""

#     finite = (
#         np.isfinite(
#             positions[:, 0]
#         )
#         & np.isfinite(
#             positions[:, 1]
#         )
#         & np.isfinite(
#             attributes
#         )
#     )

#     if not np.any(finite):
#         return (
#             np.empty(
#                 0,
#                 dtype=np.int64,
#             ),
#             np.empty(
#                 0,
#                 dtype=np.float64,
#             ),
#         )

#     x = np.asarray(
#         positions[
#             finite,
#             0,
#         ],
#         dtype=np.float64,
#     )

#     y = np.asarray(
#         positions[
#             finite,
#             1,
#         ],
#         dtype=np.float64,
#     )

#     values = np.asarray(
#         attributes[
#             finite
#         ],
#         dtype=np.float64,
#     )

#     pixels_per_world_unit = (
#         float(resolution)
#         / view_width
#     )

#     pixel_x_float = (
#         (x - x_min)
#         * pixels_per_world_unit
#     )

#     # The GPU renderer uses:
#     #
#     # pixel_y = H/2 - camera_y * scale
#     #
#     # which corresponds to flipping the world-space Y histogram.
#     pixel_y_up_float = (
#         (y - y_min)
#         * pixels_per_world_unit
#     )

#     pixel_x = np.floor(
#         pixel_x_float
#     ).astype(
#         np.int64,
#         copy=False,
#     )

#     pixel_y_up = np.floor(
#         pixel_y_up_float
#     ).astype(
#         np.int64,
#         copy=False,
#     )

#     if flip_y:
#         pixel_y = (
#             resolution
#             - 1
#             - pixel_y_up
#         )
#     else:
#         pixel_y = pixel_y_up

#     inside = (
#         (pixel_x >= 0)
#         & (pixel_x < resolution)
#         & (pixel_y >= 0)
#         & (pixel_y < resolution)
#     )

#     if not np.any(inside):
#         return (
#             np.empty(
#                 0,
#                 dtype=np.int64,
#             ),
#             np.empty(
#                 0,
#                 dtype=np.float64,
#             ),
#         )

#     pixel_x = pixel_x[
#         inside
#     ]

#     pixel_y = pixel_y[
#         inside
#     ]

#     values = values[
#         inside
#     ]

#     flat_indices = (
#         pixel_y * resolution
#         + pixel_x
#     )

#     return (
#         flat_indices,
#         values,
#     )


# def accumulate_histograms(
#     positions: np.ndarray,
#     attribute: np.ndarray,
#     *,
#     resolution: int,
#     chunk_size: int,
#     x_min: float,
#     y_min: float,
#     view_width: float,
#     flip_y: bool,
#     accumulator_dtype: np.dtype,
# ) -> tuple[
#     np.ndarray,
#     np.ndarray,
#     int,
#     int,
# ]:
#     """Accumulate density and attribute numerator before blob filtering."""

#     shape = (
#         resolution,
#         resolution,
#     )

#     density = np.zeros(
#         shape,
#         dtype=accumulator_dtype,
#     )

#     attribute_numerator = np.zeros(
#         shape,
#         dtype=accumulator_dtype,
#     )

#     density_flat = density.reshape(-1)

#     numerator_flat = (
#         attribute_numerator.reshape(-1)
#     )

#     number_of_pixels = (
#         resolution * resolution
#     )

#     particle_count = len(
#         positions
#     )

#     accepted_particles = 0
#     rejected_particles = 0

#     for start in range(
#         0,
#         particle_count,
#         chunk_size,
#     ):
#         end = min(
#             start + chunk_size,
#             particle_count,
#         )

#         position_chunk = np.asarray(
#             positions[start:end]
#         )

#         attribute_chunk = np.asarray(
#             attribute[start:end]
#         ).reshape(-1)

#         (
#             flat_indices,
#             values,
#         ) = project_particle_chunk(
#             position_chunk,
#             attribute_chunk,
#             resolution=resolution,
#             x_min=x_min,
#             y_min=y_min,
#             view_width=view_width,
#             flip_y=flip_y,
#         )

#         accepted_in_chunk = int(
#             flat_indices.size
#         )

#         accepted_particles += (
#             accepted_in_chunk
#         )

#         rejected_particles += (
#             end
#             - start
#             - accepted_in_chunk
#         )

#         if accepted_in_chunk > 0:
#             density_increment = np.bincount(
#                 flat_indices,
#                 minlength=number_of_pixels,
#             )

#             numerator_increment = np.bincount(
#                 flat_indices,
#                 weights=values,
#                 minlength=number_of_pixels,
#             )

#             density_flat += (
#                 density_increment.astype(
#                     accumulator_dtype,
#                     copy=False,
#                 )
#             )

#             numerator_flat += (
#                 numerator_increment.astype(
#                     accumulator_dtype,
#                     copy=False,
#                 )
#             )

#         percentage = (
#             100.0
#             * float(end)
#             / float(particle_count)
#         )

#         print(
#             f"\r  Projecting particles: "
#             f"{end:,}/{particle_count:,} "
#             f"({percentage:6.2f}%)",
#             end="",
#             flush=True,
#         )

#     print()

#     return (
#         density,
#         attribute_numerator,
#         accepted_particles,
#         rejected_particles,
#     )


# def apply_blob_filter(
#     density: np.ndarray,
#     attribute_numerator: np.ndarray,
#     *,
#     blob_sigma_pixels: float,
#     truncate: float,
# ) -> tuple[
#     np.ndarray,
#     np.ndarray,
# ]:
#     """Apply the same isotropic screen-space Gaussian blob to both moments."""

#     if blob_sigma_pixels <= 0.0:
#         return (
#             density,
#             attribute_numerator,
#         )

#     filtered_density = gaussian_filter(
#         density,
#         sigma=blob_sigma_pixels,
#         mode="constant",
#         cval=0.0,
#         truncate=truncate,
#         output=density.dtype,
#     )

#     filtered_numerator = gaussian_filter(
#         attribute_numerator,
#         sigma=blob_sigma_pixels,
#         mode="constant",
#         cval=0.0,
#         truncate=truncate,
#         output=attribute_numerator.dtype,
#     )

#     return (
#         filtered_density,
#         filtered_numerator,
#     )


# def compute_attribute_field(
#     density: np.ndarray,
#     attribute_numerator: np.ndarray,
#     *,
#     relative_density_threshold: float,
#     minimum_density_threshold: float,
# ) -> tuple[
#     np.ndarray,
#     np.ndarray,
#     float,
# ]:
#     maximum_density = float(
#         density.max(initial=0.0)
#     )

#     density_threshold = max(
#         minimum_density_threshold,
#         maximum_density
#         * relative_density_threshold,
#     )

#     valid_mask = (
#         np.isfinite(density)
#         & np.isfinite(
#             attribute_numerator
#         )
#         & (
#             density
#             > density_threshold
#         )
#     )

#     attribute = np.full(
#         density.shape,
#         np.nan,
#         dtype=np.float64,
#     )

#     attribute[valid_mask] = (
#         attribute_numerator[
#             valid_mask
#         ]
#         / density[
#             valid_mask
#         ]
#     )

#     return (
#         attribute,
#         valid_mask,
#         density_threshold,
#     )


# def compute_preview_ranges(
#     density: np.ndarray,
#     attribute: np.ndarray,
#     valid_mask: np.ndarray,
#     *,
#     density_percentile: float,
#     attribute_lower_percentile: float,
#     attribute_upper_percentile: float,
# ) -> tuple[
#     float,
#     float,
#     float,
# ]:
#     """Compute one density range and one attribute range per resolution.

#     These exact ranges are reused by every preview generated for the
#     resolution, so density-only, attribute-only, and combined figures have
#     consistent color interpretation.
#     """

#     positive_density = density[
#         np.isfinite(density)
#         & (density > 0.0)
#     ]

#     if positive_density.size == 0:
#         density_upper = 1.0
#     else:
#         density_upper = float(
#             np.percentile(
#                 positive_density,
#                 density_percentile,
#             )
#         )

#     density_upper = max(
#         density_upper,
#         1.0e-12,
#     )

#     finite_attribute = attribute[
#         valid_mask
#         & np.isfinite(attribute)
#     ]

#     if finite_attribute.size == 0:
#         raise RuntimeError(
#             "No finite valid attribute values are available."
#         )

#     attribute_lower = float(
#         np.percentile(
#             finite_attribute,
#             attribute_lower_percentile,
#         )
#     )

#     attribute_upper = float(
#         np.percentile(
#             finite_attribute,
#             attribute_upper_percentile,
#         )
#     )

#     if attribute_upper <= attribute_lower:
#         attribute_upper = (
#             attribute_lower + 1.0
#         )

#     return (
#         density_upper,
#         attribute_lower,
#         attribute_upper,
#     )


# def normalize_density_luminance(
#     density: np.ndarray,
#     *,
#     density_upper: float,
# ) -> np.ndarray:
#     """Paper-style log-density luminance in [0,1]."""

#     luminance = np.log1p(
#         np.maximum(
#             density,
#             0.0,
#         )
#     )

#     luminance /= max(
#         np.log1p(density_upper),
#         1.0e-12,
#     )

#     return np.clip(
#         np.nan_to_num(
#             luminance,
#             nan=0.0,
#             posinf=1.0,
#             neginf=0.0,
#         ),
#         0.0,
#         1.0,
#     )


# def paper_style_attribute_rgb(
#     attribute: np.ndarray,
#     density: np.ndarray,
#     valid_mask: np.ndarray,
#     *,
#     attribute_lower: float,
#     attribute_upper: float,
#     density_upper: float,
#     colormap_name: str,
# ) -> np.ndarray:
#     """Create the original paper-style attribute visualization.

#     Hue is determined by the normalized mean scalar attribute, while
#     luminance is determined by normalized projected density:

#         RGB = colormap(attribute_normalized) * density_luminance
#     """

#     attribute_normalized = np.clip(
#         (
#             np.nan_to_num(
#                 attribute,
#                 nan=attribute_lower,
#                 posinf=attribute_upper,
#                 neginf=attribute_lower,
#             )
#             - attribute_lower
#         )
#         / max(
#             attribute_upper
#             - attribute_lower,
#             1.0e-12,
#         ),
#         0.0,
#         1.0,
#     )

#     density_luminance = (
#         normalize_density_luminance(
#             density,
#             density_upper=density_upper,
#         )
#     )

#     color = colormaps.get_cmap(
#         colormap_name
#     )(
#         attribute_normalized
#     )[
#         ...,
#         :3,
#     ]

#     image = np.clip(
#         color
#         * density_luminance[
#             ...,
#             None,
#         ],
#         0.0,
#         1.0,
#     )

#     image[
#         ~valid_mask
#     ] = 0.0

#     return image


# def save_density_preview(
#     path: Path,
#     density: np.ndarray,
#     *,
#     density_upper: float,
#     colormap_name: str,
# ) -> None:
#     """Save single-plane log-density preview."""

#     image = normalize_density_luminance(
#         density,
#         density_upper=density_upper,
#     )

#     plt.imsave(
#         path,
#         image,
#         origin="lower",
#         cmap=colormap_name,
#         vmin=0.0,
#         vmax=1.0,
#     )


# def save_attribute_preview(
#     path: Path,
#     *,
#     attribute: np.ndarray,
#     density: np.ndarray,
#     valid_mask: np.ndarray,
#     attribute_lower: float,
#     attribute_upper: float,
#     density_upper: float,
#     colormap_name: str,
# ) -> None:
#     """Save hue=attribute and luminance=density preview."""

#     image = paper_style_attribute_rgb(
#         attribute,
#         density,
#         valid_mask,
#         attribute_lower=attribute_lower,
#         attribute_upper=attribute_upper,
#         density_upper=density_upper,
#         colormap_name=colormap_name,
#     )

#     plt.imsave(
#         path,
#         image,
#         origin="lower",
#     )


# def save_combined_preview(
#     path: Path,
#     *,
#     density: np.ndarray,
#     attribute: np.ndarray,
#     valid_mask: np.ndarray,
#     attribute_name: str,
#     density_upper: float,
#     attribute_lower: float,
#     attribute_upper: float,
#     density_colormap_name: str,
#     attribute_colormap_name: str,
# ) -> None:
#     """Save density and paper-style attribute previews side by side."""

#     density_luminance = (
#         normalize_density_luminance(
#             density,
#             density_upper=density_upper,
#         )
#     )

#     density_rgb = colormaps.get_cmap(
#         density_colormap_name
#     )(
#         density_luminance
#     )[
#         ...,
#         :3,
#     ]

#     attribute_rgb = (
#         paper_style_attribute_rgb(
#             attribute,
#             density,
#             valid_mask,
#             attribute_lower=attribute_lower,
#             attribute_upper=attribute_upper,
#             density_upper=density_upper,
#             colormap_name=(
#                 attribute_colormap_name
#             ),
#         )
#     )

#     figure, axes = plt.subplots(
#         1,
#         2,
#         figsize=(12, 5),
#         constrained_layout=True,
#     )

#     axes[0].imshow(
#         density_rgb,
#         origin="lower",
#     )

#     axes[0].set_title(
#         "Particle density GT"
#     )

#     axes[0].set_axis_off()

#     axes[1].imshow(
#         attribute_rgb,
#         origin="lower",
#     )

#     axes[1].set_title(
#         f"{attribute_name} + density GT"
#     )

#     axes[1].set_axis_off()

#     figure.suptitle(
#         "Hue = mean attribute; luminance = particle density"
#     )

#     figure.savefig(
#         path,
#         dpi=180,
#     )

#     plt.close(
#         figure
#     )

# def generate_resolution(
#     positions: np.ndarray,
#     particle_attribute: np.ndarray,
#     *,
#     resolution: int,
#     output_root: Path,
#     chunk_size: int,
#     x_min: float,
#     y_min: float,
#     view_width: float,
#     flip_y: bool,
#     blob_sigma_pixels: float,
#     blob_truncate: float,
#     relative_density_threshold: float,
#     minimum_density_threshold: float,
#     accumulator_dtype: np.dtype,
#     attribute_name: str,
#     density_percentile: float,
#     attribute_lower_percentile: float,
#     attribute_upper_percentile: float,
#     density_colormap_name: str,
#     attribute_colormap_name: str,
#     save_previews: bool,
# ) -> dict[str, object]:
#     print()
#     print("=" * 80)
#     print(
#         f"Generating {resolution} x {resolution} blob particle GT"
#     )
#     print("=" * 80)

#     total_start = perf_counter()

#     accumulation_start = perf_counter()

#     (
#         density,
#         attribute_numerator,
#         accepted_particles,
#         rejected_particles,
#     ) = accumulate_histograms(
#         positions,
#         particle_attribute,
#         resolution=resolution,
#         chunk_size=chunk_size,
#         x_min=x_min,
#         y_min=y_min,
#         view_width=view_width,
#         flip_y=flip_y,
#         accumulator_dtype=(
#             accumulator_dtype
#         ),
#     )

#     accumulation_seconds = (
#         perf_counter()
#         - accumulation_start
#     )

#     print(
#         f"  Applying Gaussian blob: "
#         f"sigma={blob_sigma_pixels} pixels"
#     )

#     filter_start = perf_counter()

#     (
#         density,
#         attribute_numerator,
#     ) = apply_blob_filter(
#         density,
#         attribute_numerator,
#         blob_sigma_pixels=(
#             blob_sigma_pixels
#         ),
#         truncate=blob_truncate,
#     )

#     filter_seconds = (
#         perf_counter()
#         - filter_start
#     )

#     (
#         attribute,
#         valid_mask,
#         density_threshold,
#     ) = compute_attribute_field(
#         density,
#         attribute_numerator,
#         relative_density_threshold=(
#             relative_density_threshold
#         ),
#         minimum_density_threshold=(
#             minimum_density_threshold
#         ),
#     )

#     output_directory = (
#         output_root / str(resolution)
#     )

#     output_directory.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     np.save(
#         output_directory
#         / "density.npy",
#         density.astype(
#             np.float32
#         ),
#     )

#     np.save(
#         output_directory
#         / "attribute_numerator.npy",
#         attribute_numerator.astype(
#             np.float32
#         ),
#     )

#     np.save(
#         output_directory
#         / "attribute.npy",
#         attribute.astype(
#             np.float32
#         ),
#     )

#     np.save(
#         output_directory
#         / "valid_mask.npy",
#         valid_mask.astype(
#             np.bool_
#         ),
#     )

#     (
#         density_display_upper,
#         display_lower,
#         display_upper,
#     ) = compute_preview_ranges(
#         density,
#         attribute,
#         valid_mask,
#         density_percentile=(
#             density_percentile
#         ),
#         attribute_lower_percentile=(
#             attribute_lower_percentile
#         ),
#         attribute_upper_percentile=(
#             attribute_upper_percentile
#         ),
#     )

#     if save_previews:
#         save_density_preview(
#             output_directory
#             / "density.png",
#             density,
#             density_upper=(
#                 density_display_upper
#             ),
#             colormap_name=(
#                 density_colormap_name
#             ),
#         )

#         save_attribute_preview(
#             output_directory
#             / "attribute.png",
#             attribute=attribute,
#             density=density,
#             valid_mask=valid_mask,
#             attribute_lower=display_lower,
#             attribute_upper=display_upper,
#             density_upper=(
#                 density_display_upper
#             ),
#             colormap_name=(
#                 attribute_colormap_name
#             ),
#         )

#         save_combined_preview(
#             output_directory
#             / "comparison.png",
#             density=density,
#             attribute=attribute,
#             valid_mask=valid_mask,
#             attribute_name=attribute_name,
#             density_upper=(
#                 density_display_upper
#             ),
#             attribute_lower=display_lower,
#             attribute_upper=display_upper,
#             density_colormap_name=(
#                 density_colormap_name
#             ),
#             attribute_colormap_name=(
#                 attribute_colormap_name
#             ),
#         )

#     finite_attribute_values = attribute[
#         valid_mask
#         & np.isfinite(attribute)
#     ]

#     if finite_attribute_values.size == 0:
#         raise RuntimeError(
#             "Generated GT has no valid attribute values."
#         )

#     total_seconds = (
#         perf_counter()
#         - total_start
#     )

#     statistics: dict[str, object] = {
#         "resolution": resolution,
#         "width": resolution,
#         "height": resolution,
#         "accepted_particles": (
#             accepted_particles
#         ),
#         "rejected_particles": (
#             rejected_particles
#         ),
#         "blob_sigma_pixels": (
#             blob_sigma_pixels
#         ),
#         "blob_truncate": (
#             blob_truncate
#         ),
#         "view_width": view_width,
#         "x_min": x_min,
#         "y_min": y_min,
#         "flip_y": flip_y,
#         "density_threshold": (
#             density_threshold
#         ),
#         "density_sum": float(
#             density.sum()
#         ),
#         "density_minimum": float(
#             density.min()
#         ),
#         "density_maximum": float(
#             density.max()
#         ),
#         "density_mean": float(
#             density.mean()
#         ),
#         "density_display_percentile": (
#             density_percentile
#         ),
#         "density_display_upper": (
#             density_display_upper
#         ),
#         "density_colormap": (
#             density_colormap_name
#         ),
#         "valid_attribute_pixels": int(
#             valid_mask.sum()
#         ),
#         "attribute_minimum": float(
#             finite_attribute_values.min()
#         ),
#         "attribute_maximum": float(
#             finite_attribute_values.max()
#         ),
#         "attribute_mean": float(
#             finite_attribute_values.mean()
#         ),
#         "attribute_display_lower": (
#             display_lower
#         ),
#         "attribute_display_upper": (
#             display_upper
#         ),
#         "attribute_lower_percentile": (
#             attribute_lower_percentile
#         ),
#         "attribute_upper_percentile": (
#             attribute_upper_percentile
#         ),
#         "attribute_colormap": (
#             attribute_colormap_name
#         ),
#         "attribute_preview_definition": (
#             "RGB = colormap(normalized attribute) * "
#             "normalized log density"
#         ),
#         "histogram_accumulation_seconds": (
#             accumulation_seconds
#         ),
#         "blob_filter_seconds": (
#             filter_seconds
#         ),
#         "total_seconds": (
#             total_seconds
#         ),
#         "output_directory": str(
#             output_directory.resolve()
#         ),
#     }

#     (
#         output_directory
#         / "statistics.json"
#     ).write_text(
#         json.dumps(
#             statistics,
#             indent=2,
#         ),
#         encoding="utf-8",
#     )

#     print()
#     print(
#         f"Accepted particles       : "
#         f"{accepted_particles:,}"
#     )

#     print(
#         f"Rejected particles       : "
#         f"{rejected_particles:,}"
#     )

#     print(
#         f"Blob sigma               : "
#         f"{blob_sigma_pixels:.6f} pixels"
#     )

#     print(
#         f"Histogram accumulation   : "
#         f"{accumulation_seconds:.3f} s"
#     )

#     print(
#         f"Blob filtering           : "
#         f"{filter_seconds:.3f} s"
#     )

#     print(
#         f"Density sum              : "
#         f"{density.sum():.12g}"
#     )

#     print(
#         f"Density threshold        : "
#         f"{density_threshold:.12g}"
#     )

#     print(
#         f"Valid attribute pixels   : "
#         f"{valid_mask.sum():,}"
#     )

#     print(
#         f"Attribute range          : "
#         f"[{finite_attribute_values.min():.9g}, "
#         f"{finite_attribute_values.max():.9g}]"
#     )

#     print(
#         f"Total GT time            : "
#         f"{total_seconds:.3f} s"
#     )

#     print(
#         f"Output                    : "
#         f"{output_directory.resolve()}"
#     )

#     return statistics


# def main() -> None:
#     args = parse_args()
#     validate_args(args)

#     output_root = args.output

#     output_root.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     accumulator_dtype = (
#         np.float64
#         if args.accumulator_dtype == "float64"
#         else np.float32
#     )

#     (
#         positions,
#         particle_attribute,
#         positions_path,
#         attribute_path,
#     ) = load_particle_arrays(
#         data_dir=args.data_dir,
#         positions_file=args.positions_file,
#         attribute_file=args.attribute_file,
#     )

#     print("=" * 80)
#     print(
#         "Single-plane blob particle ground-truth generation"
#     )
#     print("=" * 80)

#     print(
#         f"Model                   : "
#         f"{args.model.resolve()}"
#     )

#     print(
#         f"Positions               : "
#         f"{positions_path.resolve()}"
#     )

#     print(
#         f"Attribute               : "
#         f"{attribute_path.resolve()}"
#     )

#     print(
#         f"Attribute name          : "
#         f"{args.attribute_name}"
#     )

#     print(
#         f"Particle count          : "
#         f"{len(positions):,}"
#     )

#     print(
#         f"Resolutions             : "
#         f"{args.resolutions}"
#     )

#     print(
#         f"Chunk size              : "
#         f"{args.chunk_size:,}"
#     )

#     print(
#         f"Blob sigma              : "
#         f"{args.blob} pixels at every resolution"
#     )

#     print(
#         f"View padding            : "
#         f"{args.view_padding}"
#     )

#     print(
#         f"Accumulator dtype       : "
#         f"{args.accumulator_dtype}"
#     )

#     print(
#         f"Density percentile      : "
#         f"{args.density_percentile}"
#     )

#     print(
#         f"Attribute percentiles   : "
#         f"[{args.attribute_lower_percentile}, "
#         f"{args.attribute_upper_percentile}]"
#     )

#     print(
#         f"Density colormap        : "
#         f"{args.density_colormap}"
#     )

#     print(
#         f"Attribute colormap      : "
#         f"{args.attribute_colormap}"
#     )

#     print(
#         f"Output root             : "
#         f"{output_root.resolve()}"
#     )

#     model = GaussianModel.load(
#         args.model
#     )

#     (
#         camera_center,
#         view_width,
#         x_min,
#         x_max,
#         y_min,
#         y_max,
#     ) = derive_camera_from_model(
#         model,
#         view_padding=args.view_padding,
#     )

#     print()
#     print("Camera")
#     print("-" * 80)

#     print(
#         f"Center                  : "
#         f"{camera_center}"
#     )

#     print(
#         f"View width              : "
#         f"{view_width:.12g}"
#     )

#     print(
#         f"X range                 : "
#         f"[{x_min:.12g}, {x_max:.12g}]"
#     )

#     print(
#         f"Y range                 : "
#         f"[{y_min:.12g}, {y_max:.12g}]"
#     )

#     results: list[
#         dict[str, object]
#     ] = []

#     for resolution in args.resolutions:
#         result = generate_resolution(
#             positions,
#             particle_attribute,
#             resolution=int(resolution),
#             output_root=output_root,
#             chunk_size=args.chunk_size,
#             x_min=x_min,
#             y_min=y_min,
#             view_width=view_width,
#             flip_y=args.flip_y,
#             blob_sigma_pixels=args.blob,
#             blob_truncate=args.blob_truncate,
#             relative_density_threshold=(
#                 args.relative_density_threshold
#             ),
#             minimum_density_threshold=(
#                 args.minimum_density_threshold
#             ),
#             accumulator_dtype=(
#                 accumulator_dtype
#             ),
#             attribute_name=(
#                 args.attribute_name
#             ),
#             density_percentile=(
#                 args.density_percentile
#             ),
#             attribute_lower_percentile=(
#                 args.attribute_lower_percentile
#             ),
#             attribute_upper_percentile=(
#                 args.attribute_upper_percentile
#             ),
#             density_colormap_name=(
#                 args.density_colormap
#             ),
#             attribute_colormap_name=(
#                 args.attribute_colormap
#             ),
#             save_previews=(
#                 args.save_previews
#             ),
#         )

#         results.append(
#             result
#         )

#     metadata: dict[str, object] = {
#         "description": (
#             "Single-plane orthographic particle GT generated by histogram "
#             "projection followed by fixed screen-space Gaussian filtering."
#         ),
#         "model": str(
#             args.model.resolve()
#         ),
#         "positions": str(
#             positions_path.resolve()
#         ),
#         "attribute": str(
#             attribute_path.resolve()
#         ),
#         "attribute_name": (
#             args.attribute_name
#         ),
#         "particle_count": int(
#             len(positions)
#         ),
#         "camera_center": (
#             camera_center.tolist()
#         ),
#         "view_width": (
#             view_width
#         ),
#         "x_range": [
#             x_min,
#             x_max,
#         ],
#         "y_range": [
#             y_min,
#             y_max,
#         ],
#         "view_padding": (
#             args.view_padding
#         ),
#         "flip_y": (
#             args.flip_y
#         ),
#         "blob_sigma_pixels": (
#             args.blob
#         ),
#         "blob_sigma_policy": (
#             "fixed screen-space sigma at every resolution"
#         ),
#         "blob_truncate": (
#             args.blob_truncate
#         ),
#         "relative_density_threshold": (
#             args.relative_density_threshold
#         ),
#         "minimum_density_threshold": (
#             args.minimum_density_threshold
#         ),
#         "accumulator_dtype": (
#             args.accumulator_dtype
#         ),
#         "preview_transfer_function": {
#             "density": (
#                 "log1p density normalized by the selected "
#                 "positive-density percentile"
#             ),
#             "attribute": (
#                 "colormap(normalized mean attribute) multiplied "
#                 "by normalized log-density luminance"
#             ),
#             "density_percentile": (
#                 args.density_percentile
#             ),
#             "attribute_lower_percentile": (
#                 args.attribute_lower_percentile
#             ),
#             "attribute_upper_percentile": (
#                 args.attribute_upper_percentile
#             ),
#             "density_colormap": (
#                 args.density_colormap
#             ),
#             "attribute_colormap": (
#                 args.attribute_colormap
#             ),
#         },
#         "resolutions": results,
#     }

#     (
#         output_root
#         / "metadata.json"
#     ).write_text(
#         json.dumps(
#             metadata,
#             indent=2,
#         ),
#         encoding="utf-8",
#     )

#     print()
#     print("=" * 80)
#     print(
#         "Ground-truth generation complete"
#     )
#     print("=" * 80)

#     for result in results:
#         print(
#             f"{result['resolution']} x "
#             f"{result['resolution']}  "
#             f"{result['total_seconds']:.3f} s"
#         )

#     print()
#     print(
#         f"Metadata                : "
#         f"{(output_root / 'metadata.json').resolve()}"
#     )


# if __name__ == "__main__":
#     main()