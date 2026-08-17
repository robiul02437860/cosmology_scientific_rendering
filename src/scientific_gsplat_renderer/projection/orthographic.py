from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ProjectedGaussians:
    """Gaussian parameters after orthographic projection.

    All Gaussian arrays retain the same first dimension as the input
    Gaussian model. The ``valid`` mask determines which Gaussians intersect
    the image and camera depth range.
    """

    means_camera: FloatArray
    means_pixel: FloatArray
    covariances_camera: FloatArray
    covariances_pixel: FloatArray
    inverse_covariances_pixel: FloatArray
    radii_pixel: FloatArray
    depths: FloatArray
    masses: FloatArray
    valid: NDArray[np.bool_]

    @property
    def n_gaussians(self) -> int:
        """Number of projected Gaussians."""

        return int(self.means_pixel.shape[0])

    @property
    def n_valid(self) -> int:
        """Number of projected Gaussians marked as valid."""

        return int(np.count_nonzero(self.valid))
    


def _validate_projection_parameters(
    *,
    minimum_eigenvalue: float,
    minimum_pixel_variance: float,
    sigma_extent: float,
) -> None:
    """Validate numerical projection parameters."""

    if not np.isfinite(minimum_eigenvalue):
        raise ValueError(
            "minimum_eigenvalue must be finite"
        )

    if minimum_eigenvalue <= 0.0:
        raise ValueError(
            "minimum_eigenvalue must be positive, "
            f"got {minimum_eigenvalue}"
        )

    if not np.isfinite(minimum_pixel_variance):
        raise ValueError(
            "minimum_pixel_variance must be finite"
        )

    if minimum_pixel_variance <= 0.0:
        raise ValueError(
            "minimum_pixel_variance must be positive, "
            f"got {minimum_pixel_variance}"
        )

    if not np.isfinite(sigma_extent):
        raise ValueError(
            "sigma_extent must be finite"
        )

    if sigma_extent <= 0.0:
        raise ValueError(
            "sigma_extent must be positive, "
            f"got {sigma_extent}"
        )


def _stabilize_symmetric_matrices(
    matrices: FloatArray,
    *,
    minimum_eigenvalue: float,
) -> FloatArray:
    """Clamp eigenvalues of batched symmetric matrices.

    Parameters
    ----------
    matrices
        Array with shape ``(K, D, D)``.
    minimum_eigenvalue
        Smallest allowed eigenvalue.

    Returns
    -------
    FloatArray
        Stabilized matrices with the same shape.
    """

    symmetric = 0.5 * (
        matrices
        + np.swapaxes(matrices, 1, 2)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        symmetric
    )

    clamped_eigenvalues = np.maximum(
        eigenvalues,
        minimum_eigenvalue,
    )

    stabilized = np.einsum(
        "kij,kj,klj->kil",
        eigenvectors,
        clamped_eigenvalues,
        eigenvectors,
        optimize=True,
    )

    stabilized = 0.5 * (
        stabilized
        + np.swapaxes(stabilized, 1, 2)
    )

    return np.asarray(
        stabilized,
        dtype=np.float64,
    )


def _project_means_to_pixels(
    means_camera: FloatArray,
    camera: OrthographicCamera,
) -> FloatArray:
    """Map camera-space means to pixel-center coordinates."""

    scale_x = float(
        camera.pixels_per_world_unit_x
    )

    scale_y = float(
        camera.pixels_per_world_unit_y
    )

    pixel_x = (
        means_camera[:, 0] * scale_x
        + 0.5 * float(camera.image_width)
    )

    # Camera-space positive y points upward. Image row coordinates
    # increase downward, so the y scale has the opposite sign.
    pixel_y = (
        0.5 * float(camera.image_height)
        - means_camera[:, 1] * scale_y
    )

    return np.stack(
        [pixel_x, pixel_y],
        axis=1,
    )


def _project_covariances_to_pixels(
    covariances_camera: FloatArray,
    camera: OrthographicCamera,
    *,
    minimum_pixel_variance: float,
) -> FloatArray:
    """Project camera-space covariances into pixel coordinates.

    For an orthographic camera, the 3D-to-2D Jacobian is constant:

    ``J = [[scale_x, 0, 0], [0, -scale_y, 0]]``.

    Therefore, the projected covariance is:

    ``Sigma_pixel = J Sigma_camera J^T``.
    """

    scale = np.array(
        [
            float(
                camera.pixels_per_world_unit_x
            ),
            -float(
                camera.pixels_per_world_unit_y
            ),
        ],
        dtype=np.float64,
    )

    covariance_xy = np.asarray(
        covariances_camera[:, :2, :2],
        dtype=np.float64,
    )

    covariances_pixel = (
        covariance_xy
        * scale[None, :, None]
        * scale[None, None, :]
    )

    # Clamp projected covariance eigenvalues rather than adding a fixed
    # value to the diagonal. This preserves already-valid eigenvalues and
    # matches the CPU reference stabilization semantics.
    return _stabilize_symmetric_matrices(
        covariances_pixel,
        minimum_eigenvalue=minimum_pixel_variance,
    )


def _compute_valid_mask(
    *,
    means_pixel: FloatArray,
    radii_pixel: FloatArray,
    depths: FloatArray,
    camera: OrthographicCamera,
) -> NDArray[np.bool_]:
    """Determine which projected Gaussians can affect the image."""

    x = means_pixel[:, 0]
    y = means_pixel[:, 1]
    radius = radii_pixel

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(radius)
        & np.isfinite(depths)
    )

    positive_radius = radius > 0.0

    intersects_image = (
        (x + radius >= 0.0)
        & (x - radius < float(camera.image_width))
        & (y + radius >= 0.0)
        & (y - radius < float(camera.image_height))
    )

    in_front_of_near = (
        depths >= float(camera.near)
    )

    if np.isfinite(camera.far):
        before_far = (
            depths <= float(camera.far)
        )
    else:
        before_far = np.ones_like(
            depths,
            dtype=np.bool_,
        )

    return np.asarray(
        finite
        & positive_radius
        & intersects_image
        & in_front_of_near
        & before_far,
        dtype=np.bool_,
    )


def project_gaussians_orthographic(
    model: GaussianModel,
    camera: OrthographicCamera,
    *,
    minimum_eigenvalue: float = 1e-6,
    minimum_pixel_variance: float = 1e-4,
    sigma_extent: float = 3.0,
) -> ProjectedGaussians:
    """Project a 3D Gaussian model through an orthographic camera.

    Parameters
    ----------
    model
        Scientific 3D Gaussian model.
    camera
        Orthographic camera defining the view and image resolution.
    minimum_eigenvalue
        Minimum allowed eigenvalue for world-space covariance
        stabilization.
    minimum_pixel_variance
        Minimum allowed eigenvalue for projected 2D covariances,
        measured in squared pixels.
    sigma_extent
        Number of standard deviations defining the projected radius.

    Returns
    -------
    ProjectedGaussians
        Camera-space and pixel-space Gaussian parameters.
    """

    _validate_projection_parameters(
        minimum_eigenvalue=minimum_eigenvalue,
        minimum_pixel_variance=minimum_pixel_variance,
        sigma_extent=sigma_extent,
    )

    stabilized_world_covariances = (
        model.stabilized_covariances(
            minimum_eigenvalue=minimum_eigenvalue
        )
    )

    means_camera = np.asarray(
        camera.world_to_camera(
            model.means
        ),
        dtype=np.float64,
    )

    covariances_camera = np.asarray(
        camera.rotate_covariances_to_camera(
            stabilized_world_covariances
        ),
        dtype=np.float64,
    )

    means_pixel = _project_means_to_pixels(
        means_camera,
        camera,
    )

    covariances_pixel = (
        _project_covariances_to_pixels(
            covariances_camera,
            camera,
            minimum_pixel_variance=(
                minimum_pixel_variance
            ),
        )
    )

    inverse_covariances_pixel = np.linalg.inv(
        covariances_pixel
    )

    eigenvalues_pixel = np.linalg.eigvalsh(
        covariances_pixel
    )

    maximum_variances = eigenvalues_pixel[:, -1]

    radii_pixel = (
        sigma_extent
        * np.sqrt(maximum_variances)
    )

    depths = np.asarray(
        means_camera[:, 2],
        dtype=np.float64,
    )

    masses = np.asarray(
        model.masses,
        dtype=np.float64,
    )

    valid = _compute_valid_mask(
        means_pixel=means_pixel,
        radii_pixel=radii_pixel,
        depths=depths,
        camera=camera,
    )

    return ProjectedGaussians(
        means_camera=means_camera,
        means_pixel=np.asarray(
            means_pixel,
            dtype=np.float64,
        ),
        covariances_camera=covariances_camera,
        covariances_pixel=np.asarray(
            covariances_pixel,
            dtype=np.float64,
        ),
        inverse_covariances_pixel=np.asarray(
            inverse_covariances_pixel,
            dtype=np.float64,
        ),
        radii_pixel=np.asarray(
            radii_pixel,
            dtype=np.float64,
        ),
        depths=depths,
        masses=masses,
        valid=valid,
    )