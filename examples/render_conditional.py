from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.projection.conditional_attribute import (
    project_conditional_attributes_orthographic,
)
from scientific_gsplat_renderer.projection.orthographic import (
    project_gaussians_orthographic,
)
from scientific_gsplat_renderer.rasterization.conditional_cpu import (
    rasterize_conditional_attribute_cpu,
)
from scientific_gsplat_renderer.rendering.scientific_cuda import (
    render_projected_gaussians_cuda,
)


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render additive density and a conditional scalar field "
            "from a scientific Gaussian model."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/conditional"),
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Rendering backend. CUDA is the default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--maximum-gaussians",
        type=int,
        default=0,
        help=(
            "Maximum number of Gaussians for the CPU backend. "
            "Zero means all Gaussians."
        ),
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
        "--exponent-cutoff",
        type=float,
        default=9.0,
    )
    parser.add_argument(
        "--relative-density-threshold",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--no-normalize-gaussian-mass",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError(
            "Image width and height must be positive"
        )

    if args.maximum_gaussians < 0:
        raise ValueError(
            "--maximum-gaussians must be non-negative"
        )

    if args.minimum_pixel_variance < 0.0:
        raise ValueError(
            "--minimum-pixel-variance must be non-negative"
        )

    if args.sigma_extent <= 0.0:
        raise ValueError(
            "--sigma-extent must be positive"
        )

    if args.exponent_cutoff <= 0.0:
        raise ValueError(
            "--exponent-cutoff must be positive"
        )

    if args.relative_density_threshold < 0.0:
        raise ValueError(
            "--relative-density-threshold must be non-negative"
        )

    if args.epsilon <= 0.0:
        raise ValueError(
            "--epsilon must be positive"
        )

    if args.tile_size <= 0:
        raise ValueError(
            "--tile-size must be positive"
        )


def build_camera(
    model: GaussianModel,
    *,
    image_width: int,
    image_height: int,
) -> OrthographicCamera:
    minimum = np.asarray(
        model.means.min(axis=0),
        dtype=np.float64,
    )
    maximum = np.asarray(
        model.means.max(axis=0),
        dtype=np.float64,
    )

    center = 0.5 * (minimum + maximum)
    span = maximum - minimum

    view_width = float(
        max(span[0], span[1]) * 1.02
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
        image_width=image_width,
        image_height=image_height,
        near=0.0,
        far=2.0 * camera_distance,
    )


def save_density(
    path: Path,
    density: np.ndarray,
) -> None:
    density = np.asarray(
        density,
        dtype=np.float64,
    )

    positive = density[
        np.isfinite(density)
        & (density > 0.0)
    ]

    if positive.size == 0:
        image = np.zeros_like(density)
    else:
        upper = float(
            np.percentile(positive, 99.9)
        )

        denominator = max(
            np.log1p(upper),
            1.0e-12,
        )

        image = np.log1p(
            np.maximum(density, 0.0)
        )
        image /= denominator
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
        vmin=0.0,
        vmax=1.0,
    )


def save_attribute(
    path: Path,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    attribute = np.asarray(
        attribute,
        dtype=np.float64,
    )
    valid_mask = np.asarray(
        valid_mask,
        dtype=np.bool_,
    )

    finite = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if finite.size == 0:
        raise ValueError(
            "No finite conditional attribute values were rendered"
        )

    lower = float(
        np.percentile(finite, 1.0)
    )
    upper = float(
        np.percentile(finite, 99.0)
    )

    if upper <= lower:
        upper = lower + 1.0

    image = np.nan_to_num(
        attribute,
        nan=lower,
        posinf=upper,
        neginf=lower,
    )
    image = np.clip(
        image,
        lower,
        upper,
    )

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )


def render_cpu(
    projected,
    conditional,
    *,
    args: argparse.Namespace,
    normalize_gaussian_mass: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    maximum_gaussians = (
        None
        if args.maximum_gaussians == 0
        else args.maximum_gaussians
    )

    result = rasterize_conditional_attribute_cpu(
        projected,
        conditional,
        image_width=args.width,
        image_height=args.height,
        maximum_gaussians=maximum_gaussians,
        exponent_cutoff=args.exponent_cutoff,
        epsilon=args.epsilon,
        relative_density_threshold=(
            args.relative_density_threshold
        ),
        normalize_gaussian_mass=(
            normalize_gaussian_mass
        ),
    )

    statistics: dict[str, float | int] = {
        "render_seconds": result.render_seconds,
        "rendered_gaussians": result.rendered_gaussians,
        "image_mass": result.image_mass,
        "density_threshold": result.density_threshold,
    }

    return (
        np.asarray(result.density),
        np.asarray(result.attribute_numerator),
        np.asarray(result.attribute),
        np.asarray(result.valid_mask),
        statistics,
    )


def render_cuda(
    projected,
    conditional,
    *,
    args: argparse.Namespace,
    normalize_gaussian_mass: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    result = render_projected_gaussians_cuda(
        projected,
        conditional=conditional,
        image_width=args.width,
        image_height=args.height,
        tile_size=args.tile_size,
        epsilon=args.epsilon,
        relative_density_threshold=(
            args.relative_density_threshold
        ),
        normalize_gaussian_mass=(
            normalize_gaussian_mass
        ),
        device=args.device,
    )

    density = result.density_numpy()
    numerator = (
        result.attribute_numerator_numpy()
    )
    attribute = result.attribute_numpy()
    valid_mask = result.valid_mask_numpy()

    if numerator is None:
        raise RuntimeError(
            "CUDA renderer did not return an attribute numerator"
        )

    if attribute is None:
        raise RuntimeError(
            "CUDA renderer did not return a conditional attribute"
        )

    statistics: dict[str, float | int] = {
        "preparation_seconds": result.preparation_seconds,
        "intersection_seconds": result.intersection_seconds,
        "rasterization_seconds": result.rasterization_seconds,
        "render_seconds": result.total_seconds,
        "rendered_gaussians": projected.n_gaussians,
        "intersections": result.n_intersections,
        "image_mass": float(
            np.asarray(density).sum()
        ),
    }

    return (
        np.asarray(density),
        np.asarray(numerator),
        np.asarray(attribute),
        np.asarray(valid_mask),
        statistics,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model does not exist: {args.model}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalize_gaussian_mass = (
        not args.no_normalize_gaussian_mass
    )

    print("=" * 72)
    print("Scientific conditional Gaussian rendering")
    print("=" * 72)
    print(f"Model                  : {args.model}")
    print(f"Backend                : {args.backend}")
    print(
        f"Image                  : "
        f"{args.width} x {args.height}"
    )
    print(
        f"Normalize mass         : "
        f"{normalize_gaussian_mass}"
    )

    print("\nLoading model...")
    load_start = perf_counter()

    model = GaussianModel.load(
        args.model
    )

    load_seconds = (
        perf_counter() - load_start
    )

    print(
        f"Gaussians              : "
        f"{model.n_gaussians:,}"
    )
    print(
        f"Particles represented  : "
        f"{model.n_particles:,}"
    )
    print(
        f"Attribute              : "
        f"{model.attribute_name}"
    )
    print(
        f"Load time              : "
        f"{load_seconds:.6f} seconds"
    )

    camera = build_camera(
        model,
        image_width=args.width,
        image_height=args.height,
    )

    print("\nProjecting Gaussians...")
    projection_start = perf_counter()

    projected = project_gaussians_orthographic(
        model,
        camera,
        minimum_pixel_variance=(
            args.minimum_pixel_variance
        ),
        sigma_extent=args.sigma_extent,
    )

    conditional = (
        project_conditional_attributes_orthographic(
            model,
            projected,
            camera,
        )
    )

    projection_seconds = (
        perf_counter() - projection_start
    )

    common_valid = (
        np.asarray(projected.valid)
        & np.asarray(conditional.valid)
    )

    print(
        f"Projection time        : "
        f"{projection_seconds:.6f} seconds"
    )
    print(
        f"Valid Gaussians        : "
        f"{common_valid.sum():,}"
    )

    print(
        f"\nRendering with {args.backend.upper()}..."
    )

    if args.backend == "cuda":
        (
            density,
            attribute_numerator,
            attribute,
            valid_mask,
            statistics,
        ) = render_cuda(
            projected,
            conditional,
            args=args,
            normalize_gaussian_mass=(
                normalize_gaussian_mass
            ),
        )
    else:
        (
            density,
            attribute_numerator,
            attribute,
            valid_mask,
            statistics,
        ) = render_cpu(
            projected,
            conditional,
            args=args,
            normalize_gaussian_mass=(
                normalize_gaussian_mass
            ),
        )

    np.save(
        args.output / "density.npy",
        density.astype(np.float32),
    )
    np.save(
        args.output / "attribute_numerator.npy",
        attribute_numerator.astype(np.float32),
    )
    np.save(
        args.output / "conditional_attribute.npy",
        attribute.astype(np.float32),
    )
    np.save(
        args.output / "valid_mask.npy",
        valid_mask,
    )

    save_density(
        args.output / "density.png",
        density,
    )
    save_attribute(
        args.output / "conditional_attribute.png",
        attribute,
        valid_mask,
    )

    values = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    print("\n" + "=" * 72)
    print("Rendering result")
    print("=" * 72)
    print(
        f"Backend                : "
        f"{args.backend}"
    )
    print(
        f"Attribute              : "
        f"{model.attribute_name}"
    )
    print(
        f"Total render time      : "
        f"{float(statistics['render_seconds']):.6f} seconds"
    )

    if args.backend == "cuda":
        print(
            f"CUDA preparation       : "
            f"{float(statistics['preparation_seconds']):.6f} seconds"
        )
        print(
            f"CUDA intersections     : "
            f"{float(statistics['intersection_seconds']):.6f} seconds"
        )
        print(
            f"CUDA rasterization     : "
            f"{float(statistics['rasterization_seconds']):.6f} seconds"
        )
        print(
            f"Tile intersections     : "
            f"{int(statistics['intersections']):,}"
        )

    print(
        f"Image density sum      : "
        f"{float(statistics['image_mass']):.12g}"
    )
    print(
        f"Valid attribute pixels : "
        f"{valid_mask.sum():,}"
    )
    print(
        f"Attribute minimum      : "
        f"{values.min():.9g}"
    )
    print(
        f"Attribute maximum      : "
        f"{values.max():.9g}"
    )
    print(
        f"Attribute mean         : "
        f"{values.mean():.9g}"
    )
    print(
        f"Outputs                : "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()