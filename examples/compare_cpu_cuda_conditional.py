from __future__ import annotations

import argparse
from dataclasses import replace
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
from scientific_gsplat_renderer.projection import (
    ProjectedConditionalAttributes,
    ProjectedGaussians,
    project_conditional_attributes_orthographic,
    project_gaussians_orthographic,
)
from scientific_gsplat_renderer.rasterization.conditional_cpu import (
    ConditionalAttributeRenderResult,
    rasterize_conditional_attribute_cpu,
)
from scientific_gsplat_renderer.rendering.scientific_cuda import (
    ScientificCudaRenderResult,
    render_projected_gaussians_cuda,
)


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU and CUDA conditional-attribute rendering "
            "using identical projected Gaussians."
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
        default=Path("outputs/compare_cpu_cuda_conditional"),
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
        default=262_144,
        help=(
            "Maximum number of common valid Gaussians rendered by "
            "both CPU and CUDA. Use 0 to render every valid Gaussian."
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
        help=(
            "CPU Mahalanobis-squared cutoff. For sigma-extent 3, "
            "the matching value is 9."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument(
        "--relative-density-threshold",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--no-normalize-gaussian-mass",
        action="store_true",
        help="Use Gaussian mass directly as peak amplitude.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0:
        raise ValueError(
            f"--width must be positive, got {args.width}"
        )

    if args.height <= 0:
        raise ValueError(
            f"--height must be positive, got {args.height}"
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

    if args.tile_size <= 0:
        raise ValueError(
            "--tile-size must be positive"
        )

    if args.epsilon <= 0.0:
        raise ValueError(
            "--epsilon must be positive"
        )

    if args.relative_density_threshold < 0.0:
        raise ValueError(
            "--relative-density-threshold must be non-negative"
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


def common_valid_indices(
    projected: ProjectedGaussians,
    conditional: ProjectedConditionalAttributes,
    *,
    maximum_gaussians: int | None,
) -> np.ndarray:
    if projected.n_gaussians != conditional.n_gaussians:
        raise ValueError(
            "Projected and conditional Gaussian counts differ: "
            f"{projected.n_gaussians} versus "
            f"{conditional.n_gaussians}"
        )

    valid = np.asarray(
        projected.valid,
        dtype=np.bool_,
    ).copy()

    valid &= np.asarray(
        conditional.valid,
        dtype=np.bool_,
    )

    valid &= np.isfinite(
        projected.means_pixel
    ).all(axis=1)

    valid &= np.isfinite(
        projected.covariances_pixel
    ).all(axis=(1, 2))

    valid &= np.isfinite(
        projected.inverse_covariances_pixel
    ).all(axis=(1, 2))

    valid &= np.isfinite(
        projected.radii_pixel
    )

    valid &= (
        np.asarray(projected.radii_pixel) > 0.0
    )

    valid &= np.isfinite(
        projected.depths
    )

    valid &= np.isfinite(
        projected.masses
    )

    valid &= (
        np.asarray(projected.masses) > 0.0
    )

    valid &= np.isfinite(
        conditional.means
    )

    valid &= np.isfinite(
        conditional.cross_covariances_pixel
    ).all(axis=1)

    valid &= np.isfinite(
        conditional.slopes_pixel
    ).all(axis=1)

    indices = np.flatnonzero(valid)

    if indices.size == 0:
        raise ValueError(
            "No common valid Gaussians remain"
        )

    if (
        maximum_gaussians is not None
        and indices.size > maximum_gaussians
    ):
        positions = np.linspace(
            0,
            indices.size - 1,
            num=maximum_gaussians,
            dtype=np.int64,
        )

        indices = indices[positions]

    return indices


def select_projected_gaussians(
    projected: ProjectedGaussians,
    indices: np.ndarray,
) -> ProjectedGaussians:
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    n_selected = int(indices.size)

    return replace(
        projected,
        means_camera=np.asarray(
            projected.means_camera[indices],
            dtype=np.float32,
        ),
        means_pixel=np.asarray(
            projected.means_pixel[indices],
            dtype=np.float32,
        ),
        covariances_camera=np.asarray(
            projected.covariances_camera[indices],
            dtype=np.float32,
        ),
        covariances_pixel=np.asarray(
            projected.covariances_pixel[indices],
            dtype=np.float32,
        ),
        inverse_covariances_pixel=np.asarray(
            projected.inverse_covariances_pixel[indices],
            dtype=np.float32,
        ),
        radii_pixel=np.asarray(
            projected.radii_pixel[indices],
            dtype=np.float32,
        ),
        depths=np.asarray(
            projected.depths[indices],
            dtype=np.float32,
        ),
        masses=np.asarray(
            projected.masses[indices],
            dtype=np.float32,
        ),
        valid=np.ones(
            n_selected,
            dtype=np.bool_,
        ),
    )


def select_conditional_attributes(
    conditional: ProjectedConditionalAttributes,
    indices: np.ndarray,
) -> ProjectedConditionalAttributes:
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    n_selected = int(indices.size)

    return ProjectedConditionalAttributes(
        means=np.asarray(
            conditional.means[indices],
            dtype=np.float32,
        ),
        cross_covariances_pixel=np.asarray(
            conditional.cross_covariances_pixel[indices],
            dtype=np.float32,
        ),
        slopes_pixel=np.asarray(
            conditional.slopes_pixel[indices],
            dtype=np.float32,
        ),
        valid=np.ones(
            n_selected,
            dtype=np.bool_,
        ),
    )


def compute_error_statistics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate,
        dtype=np.float64,
    )

    if reference.shape != candidate.shape:
        raise ValueError(
            "Comparison arrays must have equal shapes: "
            f"{reference.shape} versus {candidate.shape}"
        )

    valid = (
        np.isfinite(reference)
        & np.isfinite(candidate)
    )

    if mask is not None:
        mask_array = np.asarray(
            mask,
            dtype=np.bool_,
        )

        if mask_array.shape != reference.shape:
            raise ValueError(
                "Comparison mask must match image shape: "
                f"{mask_array.shape} versus {reference.shape}"
            )

        valid &= mask_array

    number_valid = int(valid.sum())

    if number_valid == 0:
        return {
            "count": 0,
            "max_absolute_error": float("nan"),
            "mean_absolute_error": float("nan"),
            "rmse": float("nan"),
            "maximum_relative_error": float("nan"),
            "mean_relative_error": float("nan"),
        }

    reference_values = reference[valid]
    candidate_values = candidate[valid]

    absolute_error = np.abs(
        candidate_values - reference_values
    )

    relative_denominator = np.maximum(
        np.abs(reference_values),
        1.0e-12,
    )

    relative_error = (
        absolute_error / relative_denominator
    )

    return {
        "count": number_valid,
        "max_absolute_error": float(
            absolute_error.max()
        ),
        "mean_absolute_error": float(
            absolute_error.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    (
                        candidate_values
                        - reference_values
                    )
                    ** 2
                )
            )
        ),
        "maximum_relative_error": float(
            relative_error.max()
        ),
        "mean_relative_error": float(
            relative_error.mean()
        ),
    }


def print_statistics(
    title: str,
    statistics: dict[str, float | int],
) -> None:
    print(title)
    print("-" * 72)
    print(
        f"Compared values        : "
        f"{int(statistics['count']):,}"
    )
    print(
        f"Maximum absolute error : "
        f"{statistics['max_absolute_error']:.9g}"
    )
    print(
        f"Mean absolute error    : "
        f"{statistics['mean_absolute_error']:.9g}"
    )
    print(
        f"RMSE                   : "
        f"{statistics['rmse']:.9g}"
    )
    print(
        f"Maximum relative error : "
        f"{statistics['maximum_relative_error']:.9g}"
    )
    print(
        f"Mean relative error    : "
        f"{statistics['mean_relative_error']:.9g}"
    )


def save_scalar_image(
    path: Path,
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    cmap: str,
) -> None:
    image = np.asarray(
        values,
        dtype=np.float64,
    )

    image = np.nan_to_num(
        image,
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
        cmap=cmap,
        vmin=lower,
        vmax=upper,
    )


def save_density_pair(
    output: Path,
    cpu_density: np.ndarray,
    cuda_density: np.ndarray,
) -> None:
    combined = np.concatenate(
        [
            cpu_density.ravel(),
            cuda_density.ravel(),
        ]
    )

    positive = combined[
        np.isfinite(combined)
        & (combined > 0.0)
    ]

    if positive.size == 0:
        upper = 1.0
    else:
        upper = float(
            np.percentile(
                positive,
                99.9,
            )
        )

    denominator = max(
        np.log1p(upper),
        1.0e-12,
    )

    cpu_image = np.clip(
        np.log1p(
            np.maximum(cpu_density, 0.0)
        )
        / denominator,
        0.0,
        1.0,
    )

    cuda_image = np.clip(
        np.log1p(
            np.maximum(cuda_density, 0.0)
        )
        / denominator,
        0.0,
        1.0,
    )

    plt.imsave(
        output / "density_cpu.png",
        cpu_image,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )

    plt.imsave(
        output / "density_cuda.png",
        cuda_image,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )


def save_attribute_pair(
    output: Path,
    cpu_attribute: np.ndarray,
    cuda_attribute: np.ndarray,
    common_mask: np.ndarray,
) -> None:
    values = np.concatenate(
        [
            cpu_attribute[
                common_mask
                & np.isfinite(cpu_attribute)
            ],
            cuda_attribute[
                common_mask
                & np.isfinite(cuda_attribute)
            ],
        ]
    )

    if values.size == 0:
        raise ValueError(
            "No common finite attribute values are available"
        )

    lower = float(
        np.percentile(values, 1.0)
    )

    upper = float(
        np.percentile(values, 99.0)
    )

    if upper <= lower:
        upper = lower + 1.0

    save_scalar_image(
        output / "attribute_cpu.png",
        cpu_attribute,
        lower=lower,
        upper=upper,
        cmap="viridis",
    )

    save_scalar_image(
        output / "attribute_cuda.png",
        cuda_attribute,
        lower=lower,
        upper=upper,
        cmap="viridis",
    )


def save_error_image(
    path: Path,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> None:
    error = np.abs(
        np.asarray(candidate, dtype=np.float64)
        - np.asarray(reference, dtype=np.float64)
    )

    valid = np.isfinite(error)

    if mask is not None:
        valid &= np.asarray(
            mask,
            dtype=np.bool_,
        )

    finite_error = error[valid]

    if finite_error.size == 0:
        upper = 1.0
    else:
        upper = float(
            np.percentile(
                finite_error,
                99.9,
            )
        )

        upper = max(
            upper,
            float(finite_error.max()) * 1.0e-6,
            1.0e-12,
        )

    image = np.zeros_like(
        error,
        dtype=np.float64,
    )

    image[valid] = np.clip(
        error[valid] / upper,
        0.0,
        1.0,
    )

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )


def save_arrays(
    output: Path,
    *,
    cpu: ConditionalAttributeRenderResult,
    cuda: ScientificCudaRenderResult,
    cuda_density: np.ndarray,
    cuda_numerator: np.ndarray,
    cuda_attribute: np.ndarray,
    common_mask: np.ndarray,
) -> None:
    np.save(
        output / "density_cpu.npy",
        cpu.density.astype(
            np.float32,
        ),
    )

    np.save(
        output / "density_cuda.npy",
        cuda_density.astype(
            np.float32,
        ),
    )

    np.save(
        output / "attribute_numerator_cpu.npy",
        cpu.attribute_numerator.astype(
            np.float32,
        ),
    )

    np.save(
        output / "attribute_numerator_cuda.npy",
        cuda_numerator.astype(
            np.float32,
        ),
    )

    np.save(
        output / "attribute_cpu.npy",
        cpu.attribute.astype(
            np.float32,
        ),
    )

    np.save(
        output / "attribute_cuda.npy",
        cuda_attribute.astype(
            np.float32,
        ),
    )

    np.save(
        output / "valid_mask_cpu.npy",
        cpu.valid_mask,
    )

    np.save(
        output / "valid_mask_cuda.npy",
        cuda.valid_mask_numpy(),
    )

    np.save(
        output / "valid_mask_common.npy",
        common_mask,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model does not exist: {args.model}"
        )

    maximum_gaussians = (
        None
        if args.maximum_gaussians == 0
        else args.maximum_gaussians
    )

    normalize_gaussian_mass = (
        not args.no_normalize_gaussian_mass
    )

    print("=" * 72)
    print("CPU versus CUDA conditional rendering")
    print("=" * 72)
    print(f"Model                  : {args.model}")
    print(
        f"Image                  : "
        f"{args.width} x {args.height}"
    )
    print(
        f"Maximum Gaussians      : "
        f"{maximum_gaussians}"
    )
    print(
        f"Sigma extent           : "
        f"{args.sigma_extent}"
    )
    print(
        f"Exponent cutoff        : "
        f"{args.exponent_cutoff}"
    )
    print(
        f"Normalize mass         : "
        f"{normalize_gaussian_mass}"
    )
    print(f"CUDA device            : {args.device}")

    print("\nLoading model...")
    load_start = perf_counter()

    model = GaussianModel.load(
        args.model
    )

    load_seconds = (
        perf_counter() - load_start
    )

    print(
        f"Model Gaussians        : "
        f"{model.n_gaussians:,}"
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

    projected_all = project_gaussians_orthographic(
        model,
        camera,
        minimum_pixel_variance=(
            args.minimum_pixel_variance
        ),
        sigma_extent=args.sigma_extent,
    )

    conditional_all = (
        project_conditional_attributes_orthographic(
            model,
            projected_all,
            camera,
        )
    )

    projection_seconds = (
        perf_counter() - projection_start
    )

    indices = common_valid_indices(
        projected_all,
        conditional_all,
        maximum_gaussians=maximum_gaussians,
    )

    projected = select_projected_gaussians(
        projected_all,
        indices,
    )

    conditional = select_conditional_attributes(
        conditional_all,
        indices,
    )

    print(
        f"Projection time        : "
        f"{projection_seconds:.6f} seconds"
    )
    print(
        f"Common Gaussians       : "
        f"{indices.size:,}"
    )

    print("\nRendering CPU reference...")

    cpu_result = rasterize_conditional_attribute_cpu(
        projected,
        conditional,
        image_width=args.width,
        image_height=args.height,
        maximum_gaussians=None,
        exponent_cutoff=args.exponent_cutoff,
        epsilon=args.epsilon,
        relative_density_threshold=(
            args.relative_density_threshold
        ),
        normalize_gaussian_mass=(
            normalize_gaussian_mass
        ),
    )

    print(
        f"CPU render time        : "
        f"{cpu_result.render_seconds:.6f} seconds"
    )
    print(
        f"CPU rendered Gaussians : "
        f"{cpu_result.rendered_gaussians:,}"
    )

    print("\nRendering CUDA result...")

    cuda_result = render_projected_gaussians_cuda(
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

    if cuda_result.attribute_numerator is None:
        raise RuntimeError(
            "CUDA renderer did not return an attribute numerator"
        )

    if cuda_result.attribute is None:
        raise RuntimeError(
            "CUDA renderer did not return a normalized attribute"
        )

    cuda_density = (
        cuda_result.density_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    cuda_numerator = (
        cuda_result.attribute_numerator_numpy()
    )

    if cuda_numerator is None:
        raise RuntimeError(
            "CUDA attribute numerator conversion returned None"
        )

    cuda_numerator = cuda_numerator.astype(
        np.float64,
        copy=False,
    )

    cuda_attribute = (
        cuda_result.attribute_numpy()
    )

    if cuda_attribute is None:
        raise RuntimeError(
            "CUDA attribute conversion returned None"
        )

    cuda_attribute = cuda_attribute.astype(
        np.float64,
        copy=False,
    )

    cuda_valid_mask = (
        cuda_result.valid_mask_numpy()
    )

    common_mask = (
        cpu_result.valid_mask
        & cuda_valid_mask
        & np.isfinite(cpu_result.attribute)
        & np.isfinite(cuda_attribute)
    )

    print(
        f"CUDA preparation       : "
        f"{cuda_result.preparation_seconds:.6f} seconds"
    )
    print(
        f"CUDA intersections     : "
        f"{cuda_result.intersection_seconds:.6f} seconds"
    )
    print(
        f"CUDA rasterization     : "
        f"{cuda_result.rasterization_seconds:.6f} seconds"
    )
    print(
        f"CUDA total             : "
        f"{cuda_result.total_seconds:.6f} seconds"
    )
    print(
        f"CUDA intersections     : "
        f"{cuda_result.n_intersections:,}"
    )
    print(
        f"Common valid pixels    : "
        f"{common_mask.sum():,}"
    )

    density_statistics = compute_error_statistics(
        cpu_result.density,
        cuda_density,
    )

    numerator_statistics = compute_error_statistics(
        cpu_result.attribute_numerator,
        cuda_numerator,
    )

    attribute_statistics = compute_error_statistics(
        cpu_result.attribute,
        cuda_attribute,
        mask=common_mask,
    )

    print("\n" + "=" * 72)
    print("Numerical comparison")
    print("=" * 72)

    print_statistics(
        "Density",
        density_statistics,
    )

    print()

    print_statistics(
        "Attribute numerator",
        numerator_statistics,
    )

    print()

    print_statistics(
        "Normalized conditional attribute",
        attribute_statistics,
    )

    cpu_image_mass = float(
        cpu_result.density.sum()
    )

    cuda_image_mass = float(
        cuda_density.sum()
    )

    mass_difference = (
        cuda_image_mass - cpu_image_mass
    )

    mass_relative_error = (
        abs(mass_difference)
        / max(
            abs(cpu_image_mass),
            1.0e-12,
        )
    )

    print("\nMass comparison")
    print("-" * 72)
    print(
        f"CPU image mass         : "
        f"{cpu_image_mass:.12g}"
    )
    print(
        f"CUDA image mass        : "
        f"{cuda_image_mass:.12g}"
    )
    print(
        f"Mass difference        : "
        f"{mass_difference:.12g}"
    )
    print(
        f"Mass relative error    : "
        f"{mass_relative_error:.12g}"
    )

    print("\nSaving arrays and images...")

    save_arrays(
        args.output,
        cpu=cpu_result,
        cuda=cuda_result,
        cuda_density=cuda_density,
        cuda_numerator=cuda_numerator,
        cuda_attribute=cuda_attribute,
        common_mask=common_mask,
    )

    save_density_pair(
        args.output,
        cpu_result.density,
        cuda_density,
    )

    save_attribute_pair(
        args.output,
        cpu_result.attribute,
        cuda_attribute,
        common_mask,
    )

    save_error_image(
        args.output / "density_error.png",
        cpu_result.density,
        cuda_density,
    )

    save_error_image(
        args.output / "attribute_numerator_error.png",
        cpu_result.attribute_numerator,
        cuda_numerator,
    )

    save_error_image(
        args.output / "attribute_error.png",
        cpu_result.attribute,
        cuda_attribute,
        mask=common_mask,
    )

    print(
        f"Outputs                : "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()