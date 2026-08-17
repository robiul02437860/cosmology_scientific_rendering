from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..camera.orthographic import OrthographicCamera
from ..data.gaussian_model import GaussianModel
from .orthographic import ProjectedGaussians


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ProjectedConditionalAttributes:
    """
    Conditional scalar-attribute information in image coordinates.

    Attributes
    ----------
    means
        Per-Gaussian scalar attribute means with shape ``(N,)``.

    cross_covariances_pixel
        Projected position-attribute cross covariances

        ``Cov(pixel_position, attribute)``

        with shape ``(N, 2)``.

    slopes_pixel
        Per-Gaussian conditional slopes in attribute units per pixel,
        with shape ``(N, 2)``.

        For each Gaussian, the conditional attribute is

        ``a(p) = mean + slopes_pixel @ (p - mean_pixel)``.

    valid
        Boolean mask indicating Gaussians with usable conditional
        parameters, with shape ``(N,)``.
    """

    means: FloatArray
    cross_covariances_pixel: FloatArray
    slopes_pixel: FloatArray
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        means = np.asarray(self.means)
        cross_covariances = np.asarray(
            self.cross_covariances_pixel
        )
        slopes = np.asarray(self.slopes_pixel)
        valid = np.asarray(self.valid)

        if means.ndim != 1:
            raise ValueError(
                f"means must have shape (N,), got {means.shape}"
            )

        n_gaussians = int(means.shape[0])

        if cross_covariances.shape != (
            n_gaussians,
            2,
        ):
            raise ValueError(
                "cross_covariances_pixel must have shape "
                f"({n_gaussians}, 2), "
                f"got {cross_covariances.shape}"
            )

        if slopes.shape != (
            n_gaussians,
            2,
        ):
            raise ValueError(
                "slopes_pixel must have shape "
                f"({n_gaussians}, 2), "
                f"got {slopes.shape}"
            )

        if valid.shape != (n_gaussians,):
            raise ValueError(
                "valid must have shape "
                f"({n_gaussians},), "
                f"got {valid.shape}"
            )

    @property
    def n_gaussians(self) -> int:
        """Return the number of projected Gaussians."""
        return int(self.means.shape[0])

    @property
    def conditional_parameters(self) -> FloatArray:
        """
        Return parameters in the CUDA channel layout.

        Returns
        -------
        ndarray
            Float32 array with shape ``(N, 3)`` where:

            - channel 0 is the attribute mean
            - channel 1 is the x slope in attribute units per pixel
            - channel 2 is the y slope in attribute units per pixel
        """
        parameters = np.empty(
            (self.n_gaussians, 3),
            dtype=np.float32,
        )

        parameters[:, 0] = np.asarray(
            self.means,
            dtype=np.float32,
        )

        parameters[:, 1:3] = np.asarray(
            self.slopes_pixel,
            dtype=np.float32,
        )

        return parameters


def _normalize_attribute_means(
    attribute_means: FloatArray,
) -> NDArray[np.float64]:
    """Normalize attribute means to a one-dimensional float64 array."""
    means = np.asarray(
        attribute_means,
        dtype=np.float64,
    )

    if means.ndim == 2:
        if means.shape[1] != 1:
            raise ValueError(
                "attribute_means must have shape (N,) or (N, 1), "
                f"got {means.shape}"
            )

        means = means[:, 0]

    if means.ndim != 1:
        raise ValueError(
            "attribute_means must have shape (N,) or (N, 1), "
            f"got {means.shape}"
        )

    return means


def _validate_pixel_projection_inputs(
    *,
    n_gaussians: int,
    position_attribute_cross_covariances: NDArray[np.float64],
    covariances_pixel: NDArray[np.float64],
    world_to_camera_rotation: NDArray[np.float64],
    pixel_scale_x: float,
    pixel_scale_y: float,
    regularization: float,
) -> None:
    """Validate arrays and scalar values used by conditional projection."""
    if position_attribute_cross_covariances.shape != (
        n_gaussians,
        3,
    ):
        raise ValueError(
            "position_attribute_cross_covariances must have shape "
            f"({n_gaussians}, 3), got "
            f"{position_attribute_cross_covariances.shape}"
        )

    if covariances_pixel.shape != (
        n_gaussians,
        2,
        2,
    ):
        raise ValueError(
            "covariances_pixel must have shape "
            f"({n_gaussians}, 2, 2), "
            f"got {covariances_pixel.shape}"
        )

    if world_to_camera_rotation.shape != (3, 3):
        raise ValueError(
            "world_to_camera_rotation must have shape (3, 3), "
            f"got {world_to_camera_rotation.shape}"
        )

    if (
        not np.isfinite(pixel_scale_x)
        or pixel_scale_x <= 0.0
    ):
        raise ValueError(
            "pixel_scale_x must be positive and finite, "
            f"got {pixel_scale_x}"
        )

    if (
        not np.isfinite(pixel_scale_y)
        or pixel_scale_y <= 0.0
    ):
        raise ValueError(
            "pixel_scale_y must be positive and finite, "
            f"got {pixel_scale_y}"
        )

    if (
        not np.isfinite(regularization)
        or regularization < 0.0
    ):
        raise ValueError(
            "regularization must be finite and non-negative, "
            f"got {regularization}"
        )


def _project_cross_covariances_to_pixel(
    *,
    position_attribute_cross_covariances: FloatArray,
    world_to_camera_rotation: FloatArray,
    pixel_scale_x: float,
    pixel_scale_y: float,
    flip_y: bool,
) -> NDArray[np.float64]:
    """
    Project world-space position-attribute cross covariance to pixels.

    The stored cross covariance is interpreted as

    ``Cov(world_position, attribute)``

    with shape ``(N, 3)``.

    First it is rotated into camera coordinates, then transformed into
    pixel coordinates using the orthographic camera-to-pixel Jacobian.
    """
    cross_world = np.asarray(
        position_attribute_cross_covariances,
        dtype=np.float64,
    )

    rotation = np.asarray(
        world_to_camera_rotation,
        dtype=np.float64,
    )

    if cross_world.ndim != 2 or cross_world.shape[1] != 3:
        raise ValueError(
            "position_attribute_cross_covariances must have "
            f"shape (N, 3), got {cross_world.shape}"
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            "world_to_camera_rotation must have shape (3, 3), "
            f"got {rotation.shape}"
        )

    if (
        not np.isfinite(pixel_scale_x)
        or pixel_scale_x <= 0.0
    ):
        raise ValueError(
            "pixel_scale_x must be positive and finite, "
            f"got {pixel_scale_x}"
        )

    if (
        not np.isfinite(pixel_scale_y)
        or pixel_scale_y <= 0.0
    ):
        raise ValueError(
            "pixel_scale_y must be positive and finite, "
            f"got {pixel_scale_y}"
        )

    # Cov(camera_position, attribute)
    #
    # The cross covariance is stored as a row vector for each Gaussian,
    # so the equivalent batched transformation is:
    #
    #     cross_camera = cross_world @ R.T
    cross_camera = cross_world @ rotation.T

    signed_pixel_scale_y = (
        -pixel_scale_y
        if flip_y
        else pixel_scale_y
    )

    # Orthographic camera-to-pixel Jacobian:
    #
    #     pixel_x = sx * camera_x + offset_x
    #     pixel_y = sy * camera_y + offset_y
    #
    # The y scale is negative when image coordinates increase downward.
    pixel_jacobian = np.array(
        [
            [pixel_scale_x, 0.0, 0.0],
            [0.0, signed_pixel_scale_y, 0.0],
        ],
        dtype=np.float64,
    )

    # Cov(pixel_position, attribute)
    return cross_camera @ pixel_jacobian.T


def compute_pixel_conditional_parameters(
    *,
    attribute_means: FloatArray,
    position_attribute_cross_covariances: FloatArray,
    covariances_pixel: FloatArray,
    world_to_camera_rotation: FloatArray,
    pixel_scale_x: float,
    pixel_scale_y: float,
    flip_y: bool = True,
    regularization: float = 1.0e-8,
) -> FloatArray:
    """
    Compute conditional attribute parameters in pixel coordinates.

    The returned array has shape ``(N, 3)``:

    ``parameters[:, 0]``
        Attribute mean.

    ``parameters[:, 1]``
        Conditional x slope in attribute units per pixel.

    ``parameters[:, 2]``
        Conditional y slope in attribute units per pixel.

    For Gaussian ``i``:

    ``a_i(p) = mu_a_i + b_i.T @ (p - mu_pixel_i)``

    where

    ``b_i = Sigma_pp_i^{-1} Sigma_pa_i``.

    Here:

    - ``Sigma_pp`` is the projected positional covariance.
    - ``Sigma_pa`` is the projected position-attribute cross covariance.
    """
    means = _normalize_attribute_means(
        attribute_means
    )

    cross_world = np.asarray(
        position_attribute_cross_covariances,
        dtype=np.float64,
    )

    covariances_pixel_array = np.asarray(
        covariances_pixel,
        dtype=np.float64,
    )

    rotation = np.asarray(
        world_to_camera_rotation,
        dtype=np.float64,
    )

    n_gaussians = int(means.shape[0])

    _validate_pixel_projection_inputs(
        n_gaussians=n_gaussians,
        position_attribute_cross_covariances=(
            cross_world
        ),
        covariances_pixel=covariances_pixel_array,
        world_to_camera_rotation=rotation,
        pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y,
        regularization=regularization,
    )

    cross_pixel = _project_cross_covariances_to_pixel(
        position_attribute_cross_covariances=(
            cross_world
        ),
        world_to_camera_rotation=rotation,
        pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y,
        flip_y=flip_y,
    )

    regularized_covariances = (
        covariances_pixel_array.copy()
    )

    regularized_covariances[:, 0, 0] += (
        regularization
    )

    regularized_covariances[:, 1, 1] += (
        regularization
    )

    parameters = np.full(
        (n_gaussians, 3),
        np.nan,
        dtype=np.float32,
    )

    parameters[:, 0] = means.astype(
        np.float32,
        copy=False,
    )

    finite_inputs = (
        np.isfinite(means)
        & np.isfinite(cross_pixel).all(axis=1)
        & np.isfinite(
            regularized_covariances
        ).all(axis=(1, 2))
    )

    determinants = (
        regularized_covariances[:, 0, 0]
        * regularized_covariances[:, 1, 1]
        - regularized_covariances[:, 0, 1]
        * regularized_covariances[:, 1, 0]
    )

    solvable = (
        finite_inputs
        & np.isfinite(determinants)
        & (determinants > 0.0)
    )

    solvable_indices = np.flatnonzero(
        solvable
    )

    if solvable_indices.size > 0:
        slopes = np.linalg.solve(
            regularized_covariances[
                solvable_indices
            ],
            cross_pixel[
                solvable_indices,
                :,
                None,
            ],
        )[..., 0]

        parameters[
            solvable_indices,
            1:3,
        ] = slopes.astype(
            np.float32,
            copy=False,
        )

    return parameters


def project_conditional_attributes_orthographic(
    model: GaussianModel,
    projected: ProjectedGaussians,
    camera: OrthographicCamera,
    *,
    regularization: float = 1.0e-8,
) -> ProjectedConditionalAttributes:
    """
    Project a Gaussian model's conditional scalar attributes into pixels.

    This is the canonical conditional-attribute projection used by both
    the CPU and CUDA renderers.
    """
    if model.attribute_means is None:
        raise ValueError(
            "GaussianModel does not contain attribute_means"
        )

    if (
        model.position_attribute_cross_covariances
        is None
    ):
        raise ValueError(
            "GaussianModel does not contain "
            "position_attribute_cross_covariances"
        )

    n_gaussians = model.n_gaussians

    if projected.n_gaussians != n_gaussians:
        raise ValueError(
            "Model and projected Gaussian counts differ: "
            f"{n_gaussians} versus "
            f"{projected.n_gaussians}"
        )

    image_width, image_height = camera.image_size

    if image_width <= 0:
        raise ValueError(
            f"camera image width must be positive, got {image_width}"
        )

    if image_height <= 0:
        raise ValueError(
            f"camera image height must be positive, got {image_height}"
        )

    if (
        not np.isfinite(camera.view_width)
        or camera.view_width <= 0.0
    ):
        raise ValueError(
            "camera.view_width must be positive and finite"
        )

    if (
        not np.isfinite(camera.view_height)
        or camera.view_height <= 0.0
    ):
        raise ValueError(
            "camera.view_height must be positive and finite"
        )

    if (
        not np.isfinite(regularization)
        or regularization < 0.0
    ):
        raise ValueError(
            "regularization must be finite and non-negative, "
            f"got {regularization}"
        )

    # Match the exact camera basis used by the orthographic positional
    # projection:
    #
    # camera_x = dot(world_offset, camera.right)
    # camera_y = dot(world_offset, camera.up)
    # camera_z = dot(world_offset, camera.forward)
    world_to_camera_rotation = np.stack(
        [
            camera.right,
            camera.up,
            camera.forward,
        ],
        axis=0,
    ).astype(
        np.float64,
        copy=False,
    )

    pixel_scale_x = (
        float(image_width)
        / float(camera.view_width)
    )

    pixel_scale_y = (
        float(image_height)
        / float(camera.view_height)
    )

    cross_covariances_pixel = (
        _project_cross_covariances_to_pixel(
            position_attribute_cross_covariances=(
                model.position_attribute_cross_covariances
            ),
            world_to_camera_rotation=(
                world_to_camera_rotation
            ),
            pixel_scale_x=pixel_scale_x,
            pixel_scale_y=pixel_scale_y,
            flip_y=True,
        )
    )

    parameters = compute_pixel_conditional_parameters(
        attribute_means=model.attribute_means,
        position_attribute_cross_covariances=(
            model.position_attribute_cross_covariances
        ),
        covariances_pixel=(
            projected.covariances_pixel
        ),
        world_to_camera_rotation=(
            world_to_camera_rotation
        ),
        pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y,
        flip_y=True,
        regularization=regularization,
    )

    means = parameters[:, 0]

    slopes_pixel = parameters[:, 1:3]

    cross_covariances_pixel_float32 = (
        cross_covariances_pixel.astype(
            np.float32,
            copy=False,
        )
    )

    valid = np.asarray(
        projected.valid,
        dtype=np.bool_,
    ).copy()

    valid &= np.isfinite(means)

    valid &= np.isfinite(
        cross_covariances_pixel_float32
    ).all(axis=1)

    valid &= np.isfinite(
        slopes_pixel
    ).all(axis=1)

    return ProjectedConditionalAttributes(
        means=means,
        cross_covariances_pixel=(
            cross_covariances_pixel_float32
        ),
        slopes_pixel=slopes_pixel,
        valid=valid,
    )