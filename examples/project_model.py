from __future__ import annotations

from pathlib import Path

import numpy as np

from scientific_gsplat_renderer import (
    GaussianModel,
    OrthographicCamera,
    project_gaussians_orthographic,
)


MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/"
    "simple_model.npz"
)


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
        max(extent[0], extent[1])
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
        image_width=600,
        image_height=600,
        near=0.0,
        far=3.0 * camera_distance,
    )

    print("Projecting Gaussians...")

    projected = project_gaussians_orthographic(
        model,
        camera,
        minimum_eigenvalue=1e-6,
        minimum_pixel_variance=1e-4,
        sigma_extent=3.0,
    )

    print("=" * 72)
    print("Orthographic Gaussian Projection")
    print("=" * 72)

    print(f"Model              : {MODEL_PATH}")
    print(f"Gaussians          : {projected.n_gaussians:,}")
    print(f"Valid Gaussians    : {projected.n_valid:,}")
    print(
        "Valid fraction     : "
        f"{projected.n_valid / projected.n_gaussians:.6f}"
    )

    print()
    print("Camera")
    print("------")
    print(f"Position           : {camera.position}")
    print(f"Target             : {camera.target}")
    print(f"View width         : {camera.view_width}")
    print(f"View height        : {camera.view_height}")
    print(f"Image size         : {camera.image_size}")

    print()
    print("Pixel means")
    print("-----------")
    print(
        "Minimum            : "
        f"{np.min(projected.means_pixel, axis=0)}"
    )
    print(
        "Maximum            : "
        f"{np.max(projected.means_pixel, axis=0)}"
    )

    print()
    print("Depths")
    print("------")
    print(
        f"Minimum            : {np.min(projected.depths)}"
    )
    print(
        f"Maximum            : {np.max(projected.depths)}"
    )

    print()
    print("Pixel radii")
    print("-----------")
    print(
        f"Minimum            : {np.min(projected.radii_pixel)}"
    )
    print(
        f"Mean               : {np.mean(projected.radii_pixel)}"
    )
    print(
        f"Maximum            : {np.max(projected.radii_pixel)}"
    )

    print()
    print("First valid Gaussian")
    print("--------------------")

    valid_indices = np.flatnonzero(
        projected.valid
    )

    if len(valid_indices) == 0:
        print("No valid Gaussians")
        return

    index = int(valid_indices[0])

    print(f"Index              : {index}")
    print(
        "Camera mean        : "
        f"{projected.means_camera[index]}"
    )
    print(
        "Pixel mean         : "
        f"{projected.means_pixel[index]}"
    )
    print(
        "Pixel covariance   :"
    )
    print(
        projected.covariances_pixel[index]
    )
    print(
        "Inverse covariance :"
    )
    print(
        projected.inverse_covariances_pixel[index]
    )
    print(
        f"Radius             : {projected.radii_pixel[index]}"
    )
    print(
        f"Depth              : {projected.depths[index]}"
    )
    print(
        f"Mass               : {projected.masses[index]}"
    )


if __name__ == "__main__":
    main()