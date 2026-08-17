from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from scientific_gsplat_renderer.camera import OrthographicCamera
from scientific_gsplat_renderer.data import GaussianModel
from scientific_gsplat_renderer.projection import (
    project_gaussians_orthographic,
)
from scientific_gsplat_renderer.rendering import (
    render_projected_gaussians_cuda,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a saved scientific Gaussian model using the "
            "custom additive gsplat CUDA renderer."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default="/home/robiul/Particle_flow/HACC_project/output/illustris3_missing_tests/full_94m_0_5pct/simple_model.npz",
        help="Path to the saved Gaussian model (.npz).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cuda_render"),
        help="Output directory.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=600,
        help="Output image width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Output image height.",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="CUDA tile size.",
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.01,
        help=(
            "Fractional padding around the model. "
            "For example, 0.05 adds 5 percent padding."
        ),
    )

    parser.add_argument(
        "--view",
        choices=[
            "positive-z",
            "negative-z",
            "positive-x",
            "negative-x",
            "positive-y",
            "negative-y",
            "isometric",
        ],
        default="positive-z",
        help="Orthographic camera direction.",
    )

    parser.add_argument(
        "--minimum-eigenvalue",
        type=float,
        default=1.0e-6,
        help="Minimum stabilized world-space covariance eigenvalue.",
    )

    parser.add_argument(
        "--minimum-pixel-variance",
        type=float,
        default=0.25,
        help=(
            "Minimum projected covariance eigenvalue in squared pixels. "
            "The default 0.25 enforces a minimum projected standard "
            "deviation of 0.5 pixel for screen-space antialiasing."
        ),
    )

    parser.add_argument(
        "--sigma-extent",
        type=float,
        default=3.0,
        help="Gaussian footprint radius in standard deviations.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-12,
        help="Minimum density for attribute normalization.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device, such as cuda or cuda:0.",
    )

    parser.add_argument(
        "--raw-mass-amplitude",
        action="store_true",
        help=(
            "Use mass directly as the Gaussian peak amplitude. "
            "By default, projected Gaussians are normalized so their "
            "continuous integral equals their mass."
        ),
    )

    parser.add_argument(
        "--density-percentile",
        type=float,
        default=99.9,
        help="Upper percentile used for density visualization.",
    )

    parser.add_argument(
        "--attribute-percentile-low",
        type=float,
        default=1.0,
        help="Lower percentile used for attribute visualization.",
    )

    parser.add_argument(
        "--attribute-percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile used for attribute visualization.",
    )

    return parser.parse_args()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)

    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(
            f"Cannot normalize vector with norm {norm}"
        )

    return vector / norm


def camera_direction_and_up(
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-to-target direction and approximate up vector."""

    if view == "positive-z":
        # Camera is on the positive-Z side, looking toward negative Z.
        return (
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )

    if view == "negative-z":
        return (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )

    if view == "positive-x":
        return (
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

    if view == "negative-x":
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

    if view == "positive-y":
        return (
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )

    if view == "negative-y":
        return (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

    if view == "isometric":
        return (
            normalize_vector(
                np.array([-1.0, -1.0, -1.0], dtype=np.float64)
            ),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

    raise ValueError(f"Unsupported view: {view}")


def make_camera(
    model: GaussianModel,
    *,
    image_width: int,
    image_height: int,
    padding: float,
    view: str,
) -> OrthographicCamera:
    """Construct a camera that contains the complete Gaussian model."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "Image dimensions must be positive"
        )

    if padding < 0.0:
        raise ValueError(
            f"padding must be non-negative, got {padding}"
        )

    means = np.asarray(
        model.means,
        dtype=np.float64,
    )

    bounds_min = np.min(means, axis=0)
    bounds_max = np.max(means, axis=0)

    center = 0.5 * (bounds_min + bounds_max)
    extent = bounds_max - bounds_min

    maximum_extent = float(np.max(extent))

    if not np.isfinite(maximum_extent) or maximum_extent <= 0.0:
        raise ValueError(
            "The model has invalid or degenerate spatial bounds"
        )

    forward, approximate_up = camera_direction_and_up(view)

    forward = normalize_vector(forward)
    approximate_up = normalize_vector(approximate_up)

    # Construct the camera-space right and up axes so we can determine the
    # projected extent of the model for arbitrary view directions.
    right = np.cross(forward, approximate_up)

    if np.linalg.norm(right) < 1.0e-12:
        raise ValueError(
            "Camera forward and up vectors are parallel"
        )

    right = normalize_vector(right)
    camera_up = normalize_vector(
        np.cross(right, forward)
    )

    # Use all eight corners of the model's axis-aligned bounding box.
    corners = np.array(
        [
            [x, y, z]
            for x in (bounds_min[0], bounds_max[0])
            for y in (bounds_min[1], bounds_max[1])
            for z in (bounds_min[2], bounds_max[2])
        ],
        dtype=np.float64,
    )

    centered_corners = corners - center

    projected_x = centered_corners @ right
    projected_y = centered_corners @ camera_up
    projected_depth = centered_corners @ forward

    projected_width = float(
        projected_x.max() - projected_x.min()
    )

    projected_height = float(
        projected_y.max() - projected_y.min()
    )

    image_aspect = image_width / image_height

    # OrthographicCamera receives view_width and derives view height from
    # the image aspect ratio. The width must therefore fit both dimensions.
    width_required_by_x = projected_width
    width_required_by_y = projected_height * image_aspect

    view_width = max(
        width_required_by_x,
        width_required_by_y,
    )

    view_width *= 1.0 + 2.0 * padding

    if not np.isfinite(view_width) or view_width <= 0.0:
        raise ValueError(
            f"Computed invalid camera view width: {view_width}"
        )

    # Position the camera far enough behind the nearest bounding-box corner
    # so that every point has positive camera-space depth.
    depth_extent = float(
        projected_depth.max() - projected_depth.min()
    )

    camera_distance = (
        maximum_extent
        + depth_extent
        + 1.0
    )

    position = center - forward * camera_distance
    target = center

    near = 0.0
    far = (
        camera_distance
        + depth_extent
        + maximum_extent
        + 1.0
    )

    return OrthographicCamera(
        position=position,
        target=target,
        up=camera_up,
        view_width=view_width,
        image_width=image_width,
        image_height=image_height,
        near=near,
        far=far,
    )


def finite_statistics(
    array: np.ndarray,
) -> dict[str, float | int]:
    finite = np.isfinite(array)

    if not finite.any():
        return {
            "finite_values": 0,
            "minimum": float("nan"),
            "maximum": float("nan"),
            "mean": float("nan"),
            "sum": float("nan"),
        }

    values = array[finite]

    return {
        "finite_values": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "sum": float(values.sum()),
    }


def save_density_image(
    density: np.ndarray,
    output_path: Path,
    *,
    upper_percentile: float,
) -> None:
    finite_positive = density[
        np.isfinite(density) & (density > 0.0)
    ]

    if finite_positive.size == 0:
        display = np.zeros_like(
            density,
            dtype=np.float64,
        )
        vmax = 1.0
    else:
        # Scientific density fields commonly span many orders of magnitude.
        display = np.log1p(
            np.maximum(density, 0.0)
        )

        vmax = float(
            np.percentile(
                np.log1p(finite_positive),
                upper_percentile,
            )
        )

        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.max(display))

        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

    figure, axis = plt.subplots(
        figsize=(8, 8),
        constrained_layout=True,
    )

    image = axis.imshow(
        display,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )

    axis.set_title("Scientific CUDA density: log(1 + density)")
    axis.set_xlabel("Pixel x")
    axis.set_ylabel("Pixel y")

    figure.colorbar(
        image,
        ax=axis,
        label="log(1 + density)",
    )

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def save_attribute_image(
    attribute: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
    *,
    attribute_name: str,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    finite_valid = (
        valid_mask
        & np.isfinite(attribute)
    )

    values = attribute[finite_valid]

    if values.size == 0:
        display = np.full(
            attribute.shape,
            np.nan,
            dtype=np.float64,
        )
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = float(
            np.percentile(
                values,
                lower_percentile,
            )
        )

        vmax = float(
            np.percentile(
                values,
                upper_percentile,
            )
        )

        if not np.isfinite(vmin):
            vmin = float(np.min(values))

        if not np.isfinite(vmax):
            vmax = float(np.max(values))

        if vmax <= vmin:
            difference = max(
                abs(vmin) * 1.0e-6,
                1.0e-6,
            )
            vmax = vmin + difference

        display = np.where(
            finite_valid,
            attribute,
            np.nan,
        )

    figure, axis = plt.subplots(
        figsize=(8, 8),
        constrained_layout=True,
    )

    image = axis.imshow(
        display,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    axis.set_title(
        f"Scientific CUDA attribute: {attribute_name}"
    )
    axis.set_xlabel("Pixel x")
    axis.set_ylabel("Pixel y")

    figure.colorbar(
        image,
        ax=axis,
        label=attribute_name,
    )

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def print_model_information(
    model: GaussianModel,
) -> None:
    bounds_min = np.min(model.means, axis=0)
    bounds_max = np.max(model.means, axis=0)

    print("Model information")
    print("-" * 72)
    print(f"Gaussians             : {model.n_gaussians:,}")
    print(f"Particles represented : {model.n_particles:,}")
    print(f"Attribute             : {model.attribute_name}")
    print(f"Box size              : {model.box_size}")
    print(f"Weight sum            : {float(np.sum(model.weights)):.12g}")
    print(f"Model bounds minimum  : {bounds_min}")
    print(f"Model bounds maximum  : {bounds_max}")
    print()


def print_camera_information(
    camera: OrthographicCamera,
) -> None:
    print("Camera")
    print("-" * 72)
    print(f"Position     : {camera.position}")
    print(f"Target       : {camera.target}")
    print(f"Up           : {camera.up}")
    print(f"View width   : {camera.view_width:.8g}")
    print(
        f"Image size   : "
        f"{camera.image_width} x {camera.image_height}"
    )
    print(f"Near / far   : {camera.near} / {camera.far}")
    print()


def main() -> None:
    args = parse_arguments()

    if not 0.0 < args.density_percentile <= 100.0:
        raise ValueError(
            "--density-percentile must be in (0, 100]"
        )

    if not (
        0.0
        <= args.attribute_percentile_low
        < args.attribute_percentile_high
        <= 100.0
    ):
        raise ValueError(
            "Attribute percentiles must satisfy "
            "0 <= low < high <= 100"
        )

    model_path = args.model.expanduser().resolve()
    output_directory = args.output.expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Scientific Gaussian CUDA rendering")
    print("=" * 72)
    print(f"Model path      : {model_path}")
    print(f"Output directory: {output_directory}")
    print(f"CUDA device     : {args.device}")
    print()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current PyTorch environment"
        )

    device = torch.device(args.device)

    if device.type != "cuda":
        raise ValueError(
            f"This example requires a CUDA device, got {device}"
        )

    print(
        f"GPU              : "
        f"{torch.cuda.get_device_name(device)}"
    )
    print(
        f"PyTorch version  : {torch.__version__}"
    )
    print(
        f"PyTorch CUDA     : {torch.version.cuda}"
    )
    print()

    load_start = perf_counter()
    model = GaussianModel.load(model_path)
    load_seconds = perf_counter() - load_start

    print_model_information(model)
    print(f"Model load time: {load_seconds:.6f} seconds")
    print()

    camera = make_camera(
        model,
        image_width=args.width,
        image_height=args.height,
        padding=args.padding,
        view=args.view,
    )

    print_camera_information(camera)

    print("Projecting Gaussians...")
    projection_start = perf_counter()

    projected = project_gaussians_orthographic(
        model,
        camera,
        minimum_eigenvalue=args.minimum_eigenvalue,
        minimum_pixel_variance=args.minimum_pixel_variance,
        sigma_extent=args.sigma_extent,
    )
    
    valid_covariances = projected.covariances_pixel[
    projected.valid
    ]

    eigenvalues = np.linalg.eigvalsh(
        valid_covariances.astype(np.float64)
    )

    minimum_eigenvalues = eigenvalues[:, 0]
    maximum_eigenvalues = eigenvalues[:, 1]

    aspect_ratios = np.sqrt(
        maximum_eigenvalues
        / np.maximum(minimum_eigenvalues, 1.0e-20)
    )

    print()
    print("Projected covariance statistics")
    print("-" * 72)

    print(
        "Minimum eigenvalue percentiles:",
        np.percentile(
            minimum_eigenvalues,
            [0, 1, 10, 50, 90, 99, 100],
        ),
    )

    print(
        "Maximum eigenvalue percentiles:",
        np.percentile(
            maximum_eigenvalues,
            [0, 1, 10, 50, 90, 99, 100],
        ),
    )

    print(
        "Footprint aspect-ratio percentiles:",
        np.percentile(
            aspect_ratios,
            [0, 50, 90, 95, 99, 99.9, 100],
        ),
    )

    print(
        f"Aspect ratio > 5 : "
        f"{np.count_nonzero(aspect_ratios > 5.0):,}"
    )

    print(
        f"Aspect ratio > 10: "
        f"{np.count_nonzero(aspect_ratios > 10.0):,}"
    )

    print(
        f"Aspect ratio > 20: "
        f"{np.count_nonzero(aspect_ratios > 20.0):,}"
    )

    print(
        f"Aspect ratio > 50: "
        f"{np.count_nonzero(aspect_ratios > 50.0):,}"
    )

    print()

    projection_seconds = perf_counter() - projection_start

    valid_projected = int(
        np.count_nonzero(projected.valid)
    )

    print(
        f"Projection time          : "
        f"{projection_seconds:.6f} seconds"
    )
    print(
        f"Projected Gaussians      : "
        f"{projected.n_gaussians:,}"
    )
    print(
        f"Valid projected Gaussians: "
        f"{valid_projected:,}"
    )
    print()

    print("Rendering on CUDA...")

    # result = render_projected_gaussians_cuda(
    #     projected,
    #     image_width=args.width,
    #     image_height=args.height,
    #     attribute_means=model.attribute_means,
    #     tile_size=args.tile_size,
    #     epsilon=args.epsilon,
    #     normalize_gaussian_mass=(
    #         not args.raw_mass_amplitude
    #     ),
    #     device=device,
    # )
    
    result = render_projected_gaussians_cuda(
    projected,
    image_width=args.width,
    image_height=args.height,
    attribute_means=model.attribute_means,
    tile_size=args.tile_size,
    epsilon=args.epsilon,
    relative_density_threshold=1.0e-7,
    normalize_gaussian_mass=(
        not args.raw_mass_amplitude
    ),
    device=device,
    )

    density = result.density_numpy()
    attribute = result.attribute_numpy()

    valid_mask = (
        result.valid_mask
        .detach()
        .cpu()
        .numpy()
    )

    density_path = output_directory / "density.npy"

    np.save(
        density_path,
        density,
    )

    density_png_path = output_directory / "density.png"

    save_density_image(
        density,
        density_png_path,
        upper_percentile=args.density_percentile,
    )

    attribute_path: Path | None = None
    attribute_png_path: Path | None = None

    if attribute is not None:
        attribute_path = output_directory / "attribute.npy"

        np.save(
            attribute_path,
            attribute,
        )

        attribute_png_path = (
            output_directory / "attribute.png"
        )

        attribute_name = (
            model.attribute_name
            if model.attribute_name is not None
            else "attribute"
        )

        save_attribute_image(
            attribute,
            valid_mask,
            attribute_png_path,
            attribute_name=attribute_name,
            lower_percentile=(
                args.attribute_percentile_low
            ),
            upper_percentile=(
                args.attribute_percentile_high
            ),
        )

    np.save(
        output_directory / "valid_mask.npy",
        valid_mask,
    )

    density_stats = finite_statistics(density)

    print()
    print("CUDA rendering results")
    print("-" * 72)
    print(
        f"Input Gaussians       : "
        f"{result.n_input_gaussians:,}"
    )
    print(
        f"Rendered Gaussians    : "
        f"{result.n_rendered_gaussians:,}"
    )
    print(
        f"Tile intersections    : "
        f"{result.n_intersections:,}"
    )
    print(
        f"Preparation time      : "
        f"{result.preparation_seconds:.6f} seconds"
    )
    print(
        f"Intersection time     : "
        f"{result.intersection_seconds:.6f} seconds"
    )
    print(
        f"Rasterization time    : "
        f"{result.rasterization_seconds:.6f} seconds"
    )
    print(
        f"CUDA renderer total   : "
        f"{result.total_seconds:.6f} seconds"
    )
    print(
    f"Density threshold     : "
    f"{result.density_threshold:.12g}"
    )
    print(
        f"Projection + rendering: "
        f"{projection_seconds + result.total_seconds:.6f} seconds"
    )
    print()

    print("Density statistics")
    print("-" * 72)
    print(
        f"Shape         : {density.shape}"
    )
    print(
        f"Finite values : "
        f"{density_stats['finite_values']:,}"
    )
    print(
        f"Minimum       : "
        f"{density_stats['minimum']:.12g}"
    )
    print(
        f"Maximum       : "
        f"{density_stats['maximum']:.12g}"
    )
    print(
        f"Mean          : "
        f"{density_stats['mean']:.12g}"
    )
    print(
        f"Pixel sum     : "
        f"{density_stats['sum']:.12g}"
    )
    print(
        f"Valid pixels  : "
        f"{int(np.count_nonzero(valid_mask)):,}"
    )

    if attribute is not None:
        valid_attribute_values = attribute[
            valid_mask & np.isfinite(attribute)
        ]

        print()
        print("Attribute statistics")
        print("-" * 72)
        print(
            f"Attribute name: {model.attribute_name}"
        )
        print(
            f"Shape         : {attribute.shape}"
        )
        print(
            f"Valid values  : "
            f"{valid_attribute_values.size:,}"
        )

        if valid_attribute_values.size > 0:
            print(
                f"Minimum       : "
                f"{float(valid_attribute_values.min()):.12g}"
            )
            print(
                f"Maximum       : "
                f"{float(valid_attribute_values.max()):.12g}"
            )
            print(
                f"Mean          : "
                f"{float(valid_attribute_values.mean()):.12g}"
            )

    print()
    print("Saved outputs")
    print("-" * 72)
    print(f"Density array : {density_path}")
    print(f"Density image : {density_png_path}")
    print(
        f"Valid mask    : "
        f"{output_directory / 'valid_mask.npy'}"
    )

    if attribute_path is not None:
        print(f"Attribute array: {attribute_path}")

    if attribute_png_path is not None:
        print(f"Attribute image: {attribute_png_path}")

    print()
    print("Rendering completed successfully.")


if __name__ == "__main__":
    main()