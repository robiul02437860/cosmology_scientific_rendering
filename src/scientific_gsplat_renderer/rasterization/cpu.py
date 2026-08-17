from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scientific_gsplat_renderer.projection import (
    ProjectedGaussians,
)


FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class DensityRenderResult:
    """Result produced by the reference CPU density rasterizer.

    Attributes
    ----------
    density
        Rasterized density image with shape ``(height, width)``.
    rendered_gaussians
        Number of Gaussian components processed.
    skipped_gaussians
        Number of invalid or excluded Gaussian components.
    input_mass
        Total mass of all Gaussian components processed.
    image_mass
        Sum of all density pixels.
    """

    density: FloatArray
    rendered_gaussians: int
    skipped_gaussians: int
    input_mass: float
    image_mass: float

    def __post_init__(self) -> None:
        if self.density.ndim != 2:
            raise ValueError(
                "density must have shape (height, width), "
                f"got {self.density.shape}"
            )

        if not np.all(np.isfinite(self.density)):
            raise ValueError(
                "density must contain only finite values"
            )

        if np.any(self.density < 0.0):
            raise ValueError(
                "density must contain only nonnegative values"
            )

        if self.rendered_gaussians < 0:
            raise ValueError(
                "rendered_gaussians must be nonnegative"
            )

        if self.skipped_gaussians < 0:
            raise ValueError(
                "skipped_gaussians must be nonnegative"
            )

        if not np.isfinite(self.input_mass):
            raise ValueError(
                "input_mass must be finite"
            )

        if self.input_mass < 0.0:
            raise ValueError(
                "input_mass must be nonnegative"
            )

        if not np.isfinite(self.image_mass):
            raise ValueError(
                "image_mass must be finite"
            )

        if self.image_mass < 0.0:
            raise ValueError(
                "image_mass must be nonnegative"
            )

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return the image shape as ``(height, width)``."""

        return (
            int(self.density.shape[0]),
            int(self.density.shape[1]),
        )

    @property
    def retained_mass_fraction(self) -> float:
        """Return the fraction of processed mass present in the image."""

        if self.input_mass == 0.0:
            return 0.0

        return self.image_mass / self.input_mass


def rasterize_density_cpu(
    projected: ProjectedGaussians,
    *,
    image_width: int,
    image_height: int,
    maximum_gaussians: int | None = None,
    exponent_cutoff: float | None = None,
) -> DensityRenderResult:
    """Rasterize projected Gaussians into a density image on the CPU.

    This function is a reference implementation. It prioritizes clarity
    and correctness over speed.

    Each projected Gaussian is evaluated at pixel-center coordinates:

    ``(x + 0.5, y + 0.5)``

    The Gaussian contribution is

    ``mass / (2 pi sqrt(det(Sigma))) * exp(-0.5 d.T P d)``

    where

    - ``Sigma`` is the projected 2D covariance,
    - ``P`` is its inverse,
    - ``d`` is the displacement from the Gaussian mean.

    Parameters
    ----------
    projected
        Projected Gaussian parameters.
    image_width
        Output image width in pixels.
    image_height
        Output image height in pixels.
    maximum_gaussians
        Optional maximum number of valid Gaussians to process. This is
        useful for reference tests on large models.
    exponent_cutoff
        Optional maximum Mahalanobis-squared distance. Pixels beyond
        this distance receive no contribution. When omitted, the
        projected Gaussian radius determines the evaluated region.

    Returns
    -------
    DensityRenderResult
        Density image and rendering statistics.
    """

    _validate_rasterization_parameters(
        image_width=image_width,
        image_height=image_height,
        maximum_gaussians=maximum_gaussians,
        exponent_cutoff=exponent_cutoff,
    )

    density = np.zeros(
        (image_height, image_width),
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(
        projected.valid
    )

    # if maximum_gaussians is not None:
    #     valid_indices = valid_indices[
    #         :maximum_gaussians
    #     ]
    
    if (
        maximum_gaussians is not None
        and len(valid_indices) > maximum_gaussians
    ):
        selection_positions = np.linspace(
            0,
            len(valid_indices) - 1,
            num=maximum_gaussians,
            dtype=np.int64,
        )

        valid_indices = valid_indices[
            selection_positions
        ]

    input_mass = float(
        np.sum(
            projected.masses[valid_indices],
            dtype=np.float64,
        )
    )

    for index_value in valid_indices:
        index = int(index_value)

        _rasterize_one_gaussian(
            density=density,
            mean=projected.means_pixel[index],
            covariance=projected.covariances_pixel[index],
            inverse_covariance=(
                projected.inverse_covariances_pixel[index]
            ),
            radius=float(
                projected.radii_pixel[index]
            ),
            mass=float(
                projected.masses[index]
            ),
            exponent_cutoff=exponent_cutoff,
        )

    rendered_gaussians = int(
        len(valid_indices)
    )

    skipped_gaussians = (
        projected.n_gaussians
        - rendered_gaussians
    )

    image_mass = float(
        np.sum(
            density,
            dtype=np.float64,
        )
    )

    return DensityRenderResult(
        density=density,
        rendered_gaussians=rendered_gaussians,
        skipped_gaussians=skipped_gaussians,
        input_mass=input_mass,
        image_mass=image_mass,
    )


def _rasterize_one_gaussian(
    *,
    density: FloatArray,
    mean: FloatArray,
    covariance: FloatArray,
    inverse_covariance: FloatArray,
    radius: float,
    mass: float,
    exponent_cutoff: float | None,
) -> None:
    """Add one projected Gaussian to an existing density image."""

    if mass == 0.0:
        return

    image_height, image_width = density.shape

    center_x = float(mean[0])
    center_y = float(mean[1])

    minimum_x = max(
        0,
        int(np.floor(center_x - radius)),
    )

    maximum_x = min(
        image_width - 1,
        int(np.ceil(center_x + radius)),
    )

    minimum_y = max(
        0,
        int(np.floor(center_y - radius)),
    )

    maximum_y = min(
        image_height - 1,
        int(np.ceil(center_y + radius)),
    )

    if minimum_x > maximum_x:
        return

    if minimum_y > maximum_y:
        return

    determinant = float(
        np.linalg.det(covariance)
    )

    if not np.isfinite(determinant):
        raise ValueError(
            "Projected covariance determinant must be finite"
        )

    if determinant <= 0.0:
        raise ValueError(
            "Projected covariance determinant must be positive, "
            f"got {determinant}"
        )

    normalization = (
        mass
        / (
            2.0
            * np.pi
            * np.sqrt(determinant)
        )
    )

    pixel_x = (
        np.arange(
            minimum_x,
            maximum_x + 1,
            dtype=np.float64,
        )
        + 0.5
    )

    pixel_y = (
        np.arange(
            minimum_y,
            maximum_y + 1,
            dtype=np.float64,
        )
        + 0.5
    )

    grid_x, grid_y = np.meshgrid(
        pixel_x,
        pixel_y,
        indexing="xy",
    )

    delta_x = grid_x - center_x
    delta_y = grid_y - center_y

    precision_xx = float(
        inverse_covariance[0, 0]
    )

    precision_xy = float(
        inverse_covariance[0, 1]
    )

    precision_yy = float(
        inverse_covariance[1, 1]
    )

    mahalanobis_squared = (
        precision_xx * delta_x * delta_x
        + 2.0
        * precision_xy
        * delta_x
        * delta_y
        + precision_yy
        * delta_y
        * delta_y
    )

    exponent = -0.5 * mahalanobis_squared

    contribution = (
        normalization
        * np.exp(exponent)
    )

    if exponent_cutoff is not None:
        contribution = np.where(
            mahalanobis_squared
            <= exponent_cutoff,
            contribution,
            0.0,
        )

    density[
        minimum_y : maximum_y + 1,
        minimum_x : maximum_x + 1,
    ] += contribution


def _validate_rasterization_parameters(
    *,
    image_width: int,
    image_height: int,
    maximum_gaussians: int | None,
    exponent_cutoff: float | None,
) -> None:
    """Validate CPU rasterization parameters."""

    if isinstance(image_width, bool) or not isinstance(
        image_width,
        int,
    ):
        raise TypeError(
            "image_width must be an integer"
        )

    if image_width <= 0:
        raise ValueError(
            "image_width must be positive, "
            f"got {image_width}"
        )

    if isinstance(image_height, bool) or not isinstance(
        image_height,
        int,
    ):
        raise TypeError(
            "image_height must be an integer"
        )

    if image_height <= 0:
        raise ValueError(
            "image_height must be positive, "
            f"got {image_height}"
        )

    if maximum_gaussians is not None:
        if isinstance(
            maximum_gaussians,
            bool,
        ) or not isinstance(
            maximum_gaussians,
            int,
        ):
            raise TypeError(
                "maximum_gaussians must be an integer or None"
            )

        if maximum_gaussians <= 0:
            raise ValueError(
                "maximum_gaussians must be positive, "
                f"got {maximum_gaussians}"
            )

    if exponent_cutoff is not None:
        if not np.isfinite(
            exponent_cutoff
        ):
            raise ValueError(
                "exponent_cutoff must be finite"
            )

        if exponent_cutoff <= 0.0:
            raise ValueError(
                "exponent_cutoff must be positive, "
                f"got {exponent_cutoff}"
            )