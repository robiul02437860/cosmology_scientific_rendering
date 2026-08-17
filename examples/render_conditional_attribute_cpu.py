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


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/conditional_cpu"),
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
        default=262144,
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

    return parser.parse_args()


def save_density(path: Path, density: np.ndarray) -> None:
    positive = density[density > 0.0]

    if positive.size == 0:
        image = np.zeros_like(density)
    else:
        upper = float(np.percentile(positive, 99.9))
        image = np.log1p(density)
        image /= max(np.log1p(upper), 1.0e-12)
        image = np.clip(image, 0.0, 1.0)

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="inferno",
    )


def save_attribute(
    path: Path,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    finite = attribute[
        valid_mask & np.isfinite(attribute)
    ]

    if finite.size == 0:
        raise ValueError(
            "No finite conditional attribute values were rendered"
        )

    lower = float(np.percentile(finite, 1.0))
    upper = float(np.percentile(finite, 99.0))

    image = np.nan_to_num(
        attribute,
        nan=lower,
        posinf=upper,
        neginf=lower,
    )
    image = np.clip(image, lower, upper)

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = GaussianModel.load(args.model)

    minimum = model.means.min(axis=0)
    maximum = model.means.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = maximum - minimum

    view_width = float(span[:2].max() * 1.02)
    camera_distance = float(span.max() * 2.0)

    camera = OrthographicCamera(
        position=np.array(
            [
                center[0],
                center[1],
                center[2] + camera_distance,
            ],
            dtype=np.float64,
        ),
        target=center.astype(np.float64),
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=view_width,
        image_width=args.width,
        image_height=args.height,
        near=0.0,
        far=2.0 * camera_distance,
    )

    print("Projecting spatial Gaussians...")
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

    print(
        f"Projection time: "
        f"{perf_counter() - projection_start:.6f} seconds"
    )

    print(
        f"Valid conditional Gaussians: "
        f"{conditional.valid.sum():,}"
    )

    print(
        f"Rendering {args.maximum_gaussians:,} Gaussians "
        "with the CPU reference..."
    )

    result = rasterize_conditional_attribute_cpu(
        projected,
        conditional,
        image_width=args.width,
        image_height=args.height,
        maximum_gaussians=args.maximum_gaussians,
        exponent_cutoff=9.0,
        relative_density_threshold=1.0e-6,
    )

    np.save(
        args.output / "density.npy",
        result.density.astype(np.float32),
    )
    np.save(
        args.output / "attribute_numerator.npy",
        result.attribute_numerator.astype(np.float32),
    )
    np.save(
        args.output / "conditional_attribute.npy",
        result.attribute.astype(np.float32),
    )
    np.save(
        args.output / "valid_mask.npy",
        result.valid_mask,
    )

    save_density(
        args.output / "density.png",
        result.density,
    )
    save_attribute(
        args.output / "conditional_attribute.png",
        result.attribute,
        result.valid_mask,
    )

    values = result.attribute[result.valid_mask]

    print("=" * 72)
    print("Conditional attribute CPU rendering")
    print("=" * 72)
    print(
        f"Attribute              : {model.attribute_name}"
    )
    print(
        f"Rendered Gaussians     : "
        f"{result.rendered_gaussians:,}"
    )
    print(
        f"Rendering time         : "
        f"{result.render_seconds:.6f} seconds"
    )
    print(
        f"Density sum            : "
        f"{result.image_mass:.9g}"
    )
    print(
        f"Density threshold      : "
        f"{result.density_threshold:.9g}"
    )
    print(
        f"Valid attribute pixels : "
        f"{result.valid_mask.sum():,}"
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
        f"Outputs                : {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
