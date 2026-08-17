from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, pi
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ..projection.conditional_attribute import (
    ProjectedConditionalAttributes,
)
from ..projection.orthographic import ProjectedGaussians


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ConditionalAttributeRenderResult:
    density: FloatArray
    attribute_numerator: FloatArray
    attribute: FloatArray
    valid_mask: NDArray[np.bool_]

    input_gaussians: int
    rendered_gaussians: int
    skipped_gaussians: int

    input_mass: float
    image_mass: float
    render_seconds: float
    density_threshold: float


def rasterize_conditional_attribute_cpu(
    projected: ProjectedGaussians,
    conditional: ProjectedConditionalAttributes,
    *,
    image_width: int,
    image_height: int,
    maximum_gaussians: int | None = None,
    exponent_cutoff: float = 9.0,
    epsilon: float = 1.0e-12,
    relative_density_threshold: float = 1.0e-7,
    normalize_gaussian_mass: bool = True,
) -> ConditionalAttributeRenderResult:
    """Render density and a conditional scalar attribute on the CPU.

    For Gaussian ``i`` and pixel-center position ``p``:

        a_i(p) = mu_a,i + slope_i dot (p - mu_pixel,i)

    The final intensive scalar field is:

        A(p) =
            sum_i weight_i(p) a_i(p)
            --------------------------------
                  sum_i weight_i(p)
    """

    start_time = perf_counter()

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width and image_height must be positive"
        )

    if exponent_cutoff <= 0.0:
        raise ValueError(
            f"exponent_cutoff must be positive, got {exponent_cutoff}"
        )

    if epsilon <= 0.0:
        raise ValueError(
            f"epsilon must be positive, got {epsilon}"
        )

    if relative_density_threshold < 0.0:
        raise ValueError(
            "relative_density_threshold must be non-negative"
        )

    n_gaussians = projected.n_gaussians

    if conditional.n_gaussians != n_gaussians:
        raise ValueError(
            "Projected Gaussian and conditional-attribute counts differ: "
            f"{n_gaussians} versus {conditional.n_gaussians}"
        )

    valid = np.asarray(projected.valid, dtype=np.bool_).copy()
    valid &= np.asarray(conditional.valid, dtype=np.bool_)

    valid &= np.isfinite(projected.means_pixel).all(axis=1)
    valid &= np.isfinite(
        projected.covariances_pixel
    ).all(axis=(1, 2))
    valid &= np.isfinite(
        projected.inverse_covariances_pixel
    ).all(axis=(1, 2))
    valid &= np.isfinite(projected.radii_pixel)
    valid &= np.asarray(projected.radii_pixel) > 0.0
    valid &= np.isfinite(projected.masses)
    valid &= np.asarray(projected.masses) > 0.0

    valid_indices = np.flatnonzero(valid)

    if maximum_gaussians is not None:
        if maximum_gaussians <= 0:
            raise ValueError(
                "maximum_gaussians must be positive when provided"
            )

        if len(valid_indices) > maximum_gaussians:
            selection_positions = np.linspace(
                0,
                len(valid_indices) - 1,
                num=maximum_gaussians,
                dtype=np.int64,
            )
            valid_indices = valid_indices[
                selection_positions
            ]

    density = np.zeros(
        (image_height, image_width),
        dtype=np.float64,
    )
    attribute_numerator = np.zeros_like(density)

    input_mass = 0.0
    rendered_gaussians = 0

    for gaussian_index in valid_indices:
        mean_x, mean_y = np.asarray(
            projected.means_pixel[gaussian_index],
            dtype=np.float64,
        )

        covariance = np.asarray(
            projected.covariances_pixel[gaussian_index],
            dtype=np.float64,
        )
        inverse_covariance = np.asarray(
            projected.inverse_covariances_pixel[gaussian_index],
            dtype=np.float64,
        )

        radius = float(
            projected.radii_pixel[gaussian_index]
        )
        mass = float(projected.masses[gaussian_index])

        attribute_mean = float(
            conditional.means[gaussian_index]
        )
        slope_x, slope_y = np.asarray(
            conditional.slopes_pixel[gaussian_index],
            dtype=np.float64,
        )

        determinant = float(np.linalg.det(covariance))

        if (
            not np.isfinite(determinant)
            or determinant <= 0.0
        ):
            continue

        if normalize_gaussian_mass:
            amplitude = mass / (
                2.0 * pi * np.sqrt(determinant)
            )
        else:
            amplitude = mass

        if not np.isfinite(amplitude) or amplitude <= 0.0:
            continue

        x_min = max(
            0,
            int(floor(mean_x - radius)),
        )
        x_max = min(
            image_width - 1,
            int(ceil(mean_x + radius)),
        )
        y_min = max(
            0,
            int(floor(mean_y - radius)),
        )
        y_max = min(
            image_height - 1,
            int(ceil(mean_y + radius)),
        )

        if x_min > x_max or y_min > y_max:
            continue

        pixel_x = (
            np.arange(x_min, x_max + 1, dtype=np.float64)
            + 0.5
        )
        pixel_y = (
            np.arange(y_min, y_max + 1, dtype=np.float64)
            + 0.5
        )

        grid_x, grid_y = np.meshgrid(
            pixel_x,
            pixel_y,
            indexing="xy",
        )

        delta_x = grid_x - mean_x
        delta_y = grid_y - mean_y

        mahalanobis_squared = (
            inverse_covariance[0, 0]
            * delta_x
            * delta_x
            + 2.0
            * inverse_covariance[0, 1]
            * delta_x
            * delta_y
            + inverse_covariance[1, 1]
            * delta_y
            * delta_y
        )

        support = (
            np.isfinite(mahalanobis_squared)
            & (mahalanobis_squared >= 0.0)
            & (mahalanobis_squared <= exponent_cutoff)
        )

        gaussian_weight = np.zeros_like(
            mahalanobis_squared
        )

        gaussian_weight[support] = (
            amplitude
            * np.exp(
                -0.5
                * mahalanobis_squared[support]
            )
        )

        conditional_attribute = (
            attribute_mean
            + slope_x * delta_x
            + slope_y * delta_y
        )

        conditional_attribute = np.where(
            np.isfinite(conditional_attribute),
            conditional_attribute,
            attribute_mean,
        )

        image_slice = np.s_[
            y_min : y_max + 1,
            x_min : x_max + 1,
        ]

        density[image_slice] += gaussian_weight

        attribute_numerator[image_slice] += (
            gaussian_weight * conditional_attribute
        )

        input_mass += mass
        rendered_gaussians += 1

    maximum_density = float(
        density.max(initial=0.0)
    )

    density_threshold = max(
        epsilon,
        maximum_density * relative_density_threshold,
    )

    valid_mask = (
        np.isfinite(density)
        & (density > density_threshold)
    )

    attribute = np.full(
        density.shape,
        np.nan,
        dtype=np.float64,
    )

    attribute[valid_mask] = (
        attribute_numerator[valid_mask]
        / density[valid_mask]
    )

    render_seconds = perf_counter() - start_time

    return ConditionalAttributeRenderResult(
        density=density,
        attribute_numerator=attribute_numerator,
        attribute=attribute,
        valid_mask=valid_mask,
        input_gaussians=n_gaussians,
        rendered_gaussians=rendered_gaussians,
        skipped_gaussians=(
            n_gaussians - rendered_gaussians
        ),
        input_mass=float(input_mass),
        image_mass=float(density.sum()),
        render_seconds=float(render_seconds),
        density_threshold=float(density_threshold),
    )
