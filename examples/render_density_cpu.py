from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from scientific_gsplat_renderer import (
    GaussianModel,
    OrthographicCamera,
    project_gaussians_orthographic,
    rasterize_density_cpu,
)


MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/"
    "simple_model.npz"
)

OUTPUT_PATH = Path(
    "outputs/cpu_density.png"
)

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600

# Use an integer for a preview, for example:
# MAXIMUM_GAUSSIANS = 10_000
#
# Use None to render every valid Gaussian.
MAXIMUM_GAUSSIANS: int | None = None


def main() -> None:
    print("Loading model...")

    model = GaussianModel.load(
        MODEL_PATH
    )

    minimum = np.min(
        model.means,
        axis=0,
    )

    maximum = np.max(
        model.means,
        axis=0,
    )

    center = 0.5 * (
        minimum + maximum
    )

    extent = maximum - minimum

    view_width = float(
        max(
            extent[0],
            extent[1],
        )
        * 1.02
    )

    camera_distance = float(
        max(extent)
    )

    camera = OrthographicCamera(
        position=np.array(
            [
                center[0],
                center[1],
                maximum[2] + camera_distance,
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
        far=3.0 * camera_distance,
    )

    print("Projecting Gaussians...")

    projection_start = perf_counter()

    projected = project_gaussians_orthographic(
        model,
        camera,
        minimum_eigenvalue=1e-6,
        minimum_pixel_variance=1e-4,
        sigma_extent=4.0,
    )

    projection_seconds = (
        perf_counter()
        - projection_start
    )

    print(
        "Projection time: "
        f"{projection_seconds:.6f} seconds"
    )

    print(
        f"Valid Gaussians: {projected.n_valid:,}"
    )

    if MAXIMUM_GAUSSIANS is None:
        print(
            "Rasterizing all "
            f"{projected.n_valid:,} valid Gaussians "
            "on CPU..."
        )
    else:
        requested_gaussians = min(
            MAXIMUM_GAUSSIANS,
            projected.n_valid,
        )

        print(
            "Rasterizing "
            f"{requested_gaussians:,} Gaussians "
            "on CPU..."
        )

    rasterization_start = perf_counter()

    result = rasterize_density_cpu(
        projected,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        maximum_gaussians=MAXIMUM_GAUSSIANS,
    )

    rasterization_seconds = (
        perf_counter()
        - rasterization_start
    )

    density_minimum = float(
        np.min(result.density)
    )

    density_maximum = float(
        np.max(result.density)
    )

    density_mean = float(
        np.mean(result.density)
    )

    print("=" * 72)
    print("Reference CPU Density Rendering")
    print("=" * 72)

    print(
        "Rendered Gaussians : "
        f"{result.rendered_gaussians:,}"
    )

    print(
        "Skipped Gaussians  : "
        f"{result.skipped_gaussians:,}"
    )

    print(
        "Projection time    : "
        f"{projection_seconds:.6f} seconds"
    )

    print(
        "Rasterization time : "
        f"{rasterization_seconds:.6f} seconds"
    )

    print(
        "Total render time  : "
        f"{projection_seconds + rasterization_seconds:.6f} seconds"
    )

    print(
        "Density minimum    : "
        f"{density_minimum:.12g}"
    )

    print(
        "Density maximum    : "
        f"{density_maximum:.12g}"
    )

    print(
        "Density mean       : "
        f"{density_mean:.12g}"
    )

    print(
        "Input mass         : "
        f"{result.input_mass:.12g}"
    )

    print(
        "Image mass         : "
        f"{result.image_mass:.12g}"
    )

    print(
        "Retained fraction  : "
        f"{result.retained_mass_fraction:.12g}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display_image = np.log1p(
        result.density
    )

    figure, axis = plt.subplots(
        figsize=(8, 8),
        constrained_layout=True,
    )

    image = axis.imshow(
        display_image,
        origin="upper",
        cmap="magma",
        interpolation="nearest",
    )

    axis.set_title(
        "Reference CPU Gaussian Density\n"
        f"{result.rendered_gaussians:,} Gaussians"
    )

    axis.set_xlabel("Pixel x")
    axis.set_ylabel("Pixel y")

    figure.colorbar(
        image,
        ax=axis,
        label="log(1 + density)",
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved image         : {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()