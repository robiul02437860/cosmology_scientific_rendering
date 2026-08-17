# from __future__ import annotations

# from dataclasses import dataclass
# import math

# import torch
# from torch import Tensor

# from ..camera.orthographic import OrthographicCamera
# from .model import GpuGaussianModel


# @dataclass(slots=True)
# class GpuProjectedGaussians:
#     """Orthographically projected Gaussians stored on the GPU.

#     Shapes
#     ------
#     means_camera
#         ``(N, 3)`` camera-space Gaussian means.

#     means_pixel
#         ``(N, 2)`` pixel-center Gaussian means.

#     covariances_camera
#         ``(N, 3, 3)`` camera-space covariance matrices.

#     covariances_pixel
#         ``(N, 2, 2)`` pixel-space covariance matrices.

#     inverse_covariances_pixel
#         ``(N, 2, 2)`` inverse pixel-space covariance matrices.

#     radii_pixel
#         ``(N,)`` floating-point support radii.

#     radii_xy
#         ``(N, 2)`` integer support radii used by gsplat tile
#         intersections.

#     depths
#         ``(N,)`` camera-space depth values.

#     masses
#         ``(N,)`` Gaussian masses.

#     valid
#         ``(N,)`` Boolean validity mask.
#     """

#     means_camera: Tensor
#     means_pixel: Tensor

#     covariances_camera: Tensor
#     covariances_pixel: Tensor
#     inverse_covariances_pixel: Tensor

#     radii_pixel: Tensor
#     radii_xy: Tensor

#     depths: Tensor
#     masses: Tensor
#     valid: Tensor

#     @property
#     def device(self) -> torch.device:
#         """Device containing the projected tensors."""

#         return self.means_pixel.device

#     @property
#     def dtype(self) -> torch.dtype:
#         """Floating-point dtype of the projected tensors."""

#         return self.means_pixel.dtype

#     @property
#     def n_gaussians(self) -> int:
#         """Total number of projected Gaussians."""

#         return int(self.means_pixel.shape[0])

#     @property
#     def n_valid(self) -> int:
#         """Number of Gaussians marked as valid."""

#         return int(self.valid.sum().item())


# def _validate_projection_parameters(
#     *,
#     minimum_eigenvalue: float,
#     minimum_pixel_variance: float,
#     sigma_extent: float,
# ) -> None:
#     """Validate numerical projection parameters."""

#     if not math.isfinite(minimum_eigenvalue):
#         raise ValueError(
#             "minimum_eigenvalue must be finite."
#         )

#     if minimum_eigenvalue <= 0.0:
#         raise ValueError(
#             "minimum_eigenvalue must be positive, "
#             f"got {minimum_eigenvalue}."
#         )

#     if not math.isfinite(minimum_pixel_variance):
#         raise ValueError(
#             "minimum_pixel_variance must be finite."
#         )

#     if minimum_pixel_variance <= 0.0:
#         raise ValueError(
#             "minimum_pixel_variance must be positive, "
#             f"got {minimum_pixel_variance}."
#         )

#     if not math.isfinite(sigma_extent):
#         raise ValueError(
#             "sigma_extent must be finite."
#         )

#     if sigma_extent <= 0.0:
#         raise ValueError(
#             "sigma_extent must be positive, "
#             f"got {sigma_extent}."
#         )




# def _camera_rotation(
#     camera: OrthographicCamera,
#     *,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> Tensor:
#     """Copy the camera rotation matrix to the GPU."""

#     return torch.as_tensor(
#         camera.rotation_matrix,
#         dtype=dtype,
#         device=device,
#     ).contiguous()


# def _camera_position(
#     camera: OrthographicCamera,
#     *,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> Tensor:
#     """Copy the camera position to the GPU."""

#     return torch.as_tensor(
#         camera.position,
#         dtype=dtype,
#         device=device,
#     ).contiguous()


# def _stabilize_symmetric_matrices(
#     matrices: Tensor,
#     *,
#     minimum_eigenvalue: float,
# ) -> Tensor:
#     """Clamp eigenvalues of batched symmetric matrices.

#     Parameters
#     ----------
#     matrices
#         Tensor with shape ``(N, D, D)``.

#     minimum_eigenvalue
#         Smallest allowed eigenvalue.

#     Returns
#     -------
#     Tensor
#         Stabilized symmetric matrices with the same shape.
#     """

#     symmetric = 0.5 * (
#         matrices
#         + matrices.transpose(-1, -2)
#     )

#     eigenvalues, eigenvectors = torch.linalg.eigh(
#         symmetric
#     )

#     clamped_eigenvalues = torch.clamp(
#         eigenvalues,
#         min=float(minimum_eigenvalue),
#     )

#     stabilized = torch.matmul(
#         torch.matmul(
#             eigenvectors,
#             torch.diag_embed(clamped_eigenvalues),
#         ),
#         eigenvectors.transpose(-1, -2),
#     )

#     # Remove small asymmetry introduced by floating-point reconstruction.
#     stabilized = 0.5 * (
#         stabilized
#         + stabilized.transpose(-1, -2)
#     )

#     return stabilized

# def _stabilize_invert_symmetric_2x2(
#     matrices: Tensor,
#     *,
#     minimum_eigenvalue: float,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Stabilize and invert symmetric 2x2 matrices analytically.

#     Parameters
#     ----------
#     matrices
#         Symmetric matrices with shape ``(N, 2, 2)``.

#     minimum_eigenvalue
#         Lower bound applied to both eigenvalues.

#     Returns
#     -------
#     stabilized
#         Stabilized matrices with shape ``(N, 2, 2)``.

#     inverse
#         Analytic inverse of each stabilized matrix.

#     maximum_eigenvalue
#         Largest stabilized eigenvalue for each matrix, shape ``(N,)``.

#     Notes
#     -----
#     For

#         C = [[a, b],
#              [b, c]]

#     the eigenvalues are

#         lambda_min/max =
#             0.5 * (
#                 a + c
#                 -/+ sqrt((a-c)^2 + 4b^2)
#             )

#     When only the smaller eigenvalue needs clamping, the matrix is
#     reconstructed using its minimum-eigenvalue spectral projector:

#         P_min = (lambda_max I - C)
#                 / (lambda_max - lambda_min)

#         C_stable =
#             C + (minimum - lambda_min) P_min

#     This preserves the CPU eigenvalue-clamping semantics without calling
#     general eigendecomposition or matrix-inverse routines.
#     """

#     if matrices.ndim != 3:
#         raise ValueError(
#             "matrices must have shape (N, 2, 2), "
#             f"got {tuple(matrices.shape)}."
#         )

#     if tuple(matrices.shape[-2:]) != (2, 2):
#         raise ValueError(
#             "matrices must have shape (N, 2, 2), "
#             f"got {tuple(matrices.shape)}."
#         )

#     # Explicitly symmetrize.
#     a = matrices[:, 0, 0]
#     b = 0.5 * (
#         matrices[:, 0, 1]
#         + matrices[:, 1, 0]
#     )
#     c = matrices[:, 1, 1]

#     trace = a + c

#     discriminant = torch.sqrt(
#         torch.clamp(
#             (a - c).square()
#             + 4.0 * b.square(),
#             min=0.0,
#         )
#     )

#     lambda_min = 0.5 * (
#         trace - discriminant
#     )

#     lambda_max = 0.5 * (
#         trace + discriminant
#     )

#     minimum = float(
#         minimum_eigenvalue
#     )

#     # Start from the original symmetric matrix.
#     stable_a = a
#     stable_b = b
#     stable_c = c

#     both_below = (
#         lambda_max < minimum
#     )

#     only_minimum_below = (
#         (lambda_min < minimum)
#         & ~both_below
#     )

#     # When both eigenvalues are below the threshold, the clamped matrix is
#     # simply minimum * identity.
#     stable_a = torch.where(
#         both_below,
#         torch.full_like(
#             stable_a,
#             minimum,
#         ),
#         stable_a,
#     )

#     stable_b = torch.where(
#         both_below,
#         torch.zeros_like(
#             stable_b,
#         ),
#         stable_b,
#     )

#     stable_c = torch.where(
#         both_below,
#         torch.full_like(
#             stable_c,
#             minimum,
#         ),
#         stable_c,
#     )

#     # Clamp only the smaller eigenvalue.
#     #
#     # P_min = (lambda_max I - C) / (lambda_max - lambda_min)
#     spectral_gap = (
#         lambda_max - lambda_min
#     )

#     safe_gap = torch.clamp(
#         spectral_gap,
#         min=torch.finfo(
#             matrices.dtype
#         ).eps,
#     )

#     correction = (
#         minimum - lambda_min
#     )

#     projector_min_00 = (
#         lambda_max - a
#     ) / safe_gap

#     projector_min_01 = (
#         -b / safe_gap
#     )

#     projector_min_11 = (
#         lambda_max - c
#     ) / safe_gap

#     corrected_a = (
#         a
#         + correction
#         * projector_min_00
#     )

#     corrected_b = (
#         b
#         + correction
#         * projector_min_01
#     )

#     corrected_c = (
#         c
#         + correction
#         * projector_min_11
#     )

#     stable_a = torch.where(
#         only_minimum_below,
#         corrected_a,
#         stable_a,
#     )

#     stable_b = torch.where(
#         only_minimum_below,
#         corrected_b,
#         stable_b,
#     )

#     stable_c = torch.where(
#         only_minimum_below,
#         corrected_c,
#         stable_c,
#     )

#     stabilized = torch.stack(
#         (
#             stable_a,
#             stable_b,
#             stable_b,
#             stable_c,
#         ),
#         dim=-1,
#     ).reshape(-1, 2, 2)

#     # Analytic inverse.
#     determinant = (
#         stable_a * stable_c
#         - stable_b.square()
#     )

#     determinant_floor = max(
#         minimum * minimum,
#         torch.finfo(
#             matrices.dtype
#         ).tiny,
#     )

#     safe_determinant = torch.clamp(
#         determinant,
#         min=determinant_floor,
#     )

#     inverse_a = (
#         stable_c / safe_determinant
#     )

#     inverse_b = (
#         -stable_b / safe_determinant
#     )

#     inverse_c = (
#         stable_a / safe_determinant
#     )

#     inverse = torch.stack(
#         (
#             inverse_a,
#             inverse_b,
#             inverse_b,
#             inverse_c,
#         ),
#         dim=-1,
#     ).reshape(-1, 2, 2)

#     maximum_eigenvalue = torch.clamp(
#         lambda_max,
#         min=minimum,
#     )

#     return (
#         stabilized.contiguous(),
#         inverse.contiguous(),
#         maximum_eigenvalue.contiguous(),
#     )

# def _compute_valid_mask(
#     *,
#     means_camera: Tensor,
#     means_pixel: Tensor,
#     covariances_pixel: Tensor,
#     inverse_covariances_pixel: Tensor,
#     radii_pixel: Tensor,
#     radii_integer: Tensor,
#     depths: Tensor,
#     camera: OrthographicCamera,
#     width: int,
#     height: int,
#     cull_to_image: bool,
# ) -> Tensor:
#     """Compute the GPU projection validity mask."""

#     valid = (
#         torch.isfinite(means_camera).all(dim=-1)
#         & torch.isfinite(means_pixel).all(dim=-1)
#         & torch.isfinite(covariances_pixel).all(
#             dim=(-2, -1)
#         )
#         & torch.isfinite(
#             inverse_covariances_pixel
#         ).all(dim=(-2, -1))
#         & torch.isfinite(radii_pixel)
#         & torch.isfinite(depths)
#         & (radii_pixel > 0.0)
#         & (radii_integer > 0)
#     )

#     near = float(camera.near)

#     valid = valid & (
#         depths >= near
#     )

#     far = float(camera.far)

#     if math.isfinite(far):
#         valid = valid & (
#             depths <= far
#         )

#     if cull_to_image:
#         x = means_pixel[:, 0]
#         y = means_pixel[:, 1]

#         intersects_image = (
#             (x + radii_pixel >= 0.0)
#             & (x - radii_pixel < float(width))
#             & (y + radii_pixel >= 0.0)
#             & (y - radii_pixel < float(height))
#         )

#         valid = valid & intersects_image

#     return valid


# def project_gaussians_orthographic_gpu(
#     model: GpuGaussianModel,
#     camera: OrthographicCamera,
#     *,
#     minimum_eigenvalue: float = 1.0e-6,
#     minimum_pixel_variance: float = 1.0e-4,
#     sigma_extent: float = 3.0,
#     cull_to_image: bool = True,
#     support_sigma: float | None = None,
#     covariance_regularization: float | None = None,
#     determinant_epsilon: float | None = None,
# ) -> GpuProjectedGaussians:
#     """Project GPU-resident Gaussians with an orthographic camera.

#     This implementation follows the same mathematical operations as the
#     validated CPU projector:

#     1. Clamp world-space covariance eigenvalues.
#     2. Rotate means and covariances into camera space.
#     3. Project means and covariances into pixel coordinates.
#     4. Clamp projected 2D covariance eigenvalues.
#     5. Invert the stabilized pixel covariances.
#     6. Compute projected support radii and validity.

#     Parameters
#     ----------
#     model
#         Gaussian model already resident on the GPU.

#     camera
#         Orthographic camera.

#     minimum_eigenvalue
#         Minimum world-space covariance eigenvalue.

#     minimum_pixel_variance
#         Minimum projected covariance eigenvalue in squared pixels.

#     sigma_extent
#         Number of standard deviations used for the support radius.

#     cull_to_image
#         Mark Gaussians that do not intersect the image as invalid.

#     support_sigma
#         Deprecated alias for ``sigma_extent``. It is retained for
#         compatibility with earlier code.

#     covariance_regularization
#         Deprecated alias for ``minimum_pixel_variance``. Earlier code added
#         this value to the covariance diagonal. The updated implementation
#         interprets it as an eigenvalue lower bound.

#     determinant_epsilon
#         Deprecated compatibility argument. Stabilized covariance matrices
#         are positive definite, so determinant clamping is no longer needed.

#     Returns
#     -------
#     GpuProjectedGaussians
#         GPU-resident projected Gaussian parameters.
#     """

#     # ------------------------------------------------------------------
#     # Backward-compatible aliases
#     # ------------------------------------------------------------------

#     if support_sigma is not None:
#         sigma_extent = float(
#             support_sigma
#         )

#     # ------------------------------------------------------------------
#     # Backward compatibility with the previous API.
#     #
#     # Previously:
#     #
#     #     covariance_regularization = 0.0
#     #
#     # meant "do not regularize".
#     #
#     # In the new implementation the equivalent behaviour is to keep the
#     # default minimum_pixel_variance unchanged.
#     # ------------------------------------------------------------------

#     if covariance_regularization is not None:
#         value = float(covariance_regularization)

#         if value < 0.0:
#             raise ValueError(
#                 "covariance_regularization must be nonnegative."
#             )

#         if value > 0.0:
#             minimum_pixel_variance = value

#     # Kept only so older callers do not fail. The value is not required
#     # after eigenvalue-based covariance stabilization.
#     if determinant_epsilon is not None:
#         if not math.isfinite(determinant_epsilon):
#             raise ValueError(
#                 "determinant_epsilon must be finite."
#             )

#         if determinant_epsilon < 0.0:
#             raise ValueError(
#                 "determinant_epsilon must be nonnegative, "
#                 f"got {determinant_epsilon}."
#             )

#     _validate_projection_parameters(
#         minimum_eigenvalue=minimum_eigenvalue,
#         minimum_pixel_variance=minimum_pixel_variance,
#         sigma_extent=sigma_extent,
#     )

#     device = model.device
#     dtype = model.dtype

#     width, height = camera.image_size

#     width = int(width)
#     height = int(height)

#     if width <= 0 or height <= 0:
#         raise ValueError(
#             f"Invalid image size: {(width, height)}."
#         )

#     view_width = float(
#         camera.view_width
#     )

#     view_height = float(
#         camera.view_height
#     )

#     if not math.isfinite(view_width):
#         raise ValueError(
#             "camera.view_width must be finite."
#         )

#     if not math.isfinite(view_height):
#         raise ValueError(
#             "camera.view_height must be finite."
#         )

#     if view_width <= 0.0 or view_height <= 0.0:
#         raise ValueError(
#             "Camera view dimensions must be positive, "
#             f"got {(view_width, view_height)}."
#         )

#     rotation = _camera_rotation(
#         camera,
#         device=device,
#         dtype=dtype,
#     )

#     camera_position = _camera_position(
#         camera,
#         device=device,
#         dtype=dtype,
#     )

#     # ------------------------------------------------------------------
#     # 1. Stabilize world-space covariance matrices.
#     #
#     # This matches:
#     #
#     #     model.stabilized_covariances(
#     #         minimum_eigenvalue=minimum_eigenvalue
#     #     )
#     #
#     # in the CPU projector.
#     # ------------------------------------------------------------------

#     # stabilized_world_covariances = (
#     #     _stabilize_symmetric_matrices(
#     #         model.covariances,
#     #         minimum_eigenvalue=minimum_eigenvalue,
#     #     )
#     # )

#     model.require_stabilization(
#     minimum_eigenvalue
#     )

#     stabilized_world_covariances = (
#         model.stabilized_covariances
#     )

#     # ------------------------------------------------------------------
#     # 2. Transform means from world space into camera space.
#     #
#     # Row-vector representation:
#     #
#     #     mean_camera = (mean_world - camera_position) @ R^T
#     # ------------------------------------------------------------------

#     centered_means = (
#         model.means
#         - camera_position
#     )

#     rotation_t = rotation.transpose(
#         0,
#         1,
#     )

#     means_camera = torch.matmul(
#         centered_means,
#         rotation_t,
#     )

#     # ------------------------------------------------------------------
#     # 3. Rotate covariance matrices into camera space.
#     #
#     #     covariance_camera = R covariance_world R^T
#     # ------------------------------------------------------------------

#     covariances_camera = torch.matmul(
#         torch.matmul(
#             rotation.unsqueeze(0),
#             stabilized_world_covariances,
#         ),
#         rotation_t.unsqueeze(0),
#     )

#     covariances_camera = 0.5 * (
#         covariances_camera
#         + covariances_camera.transpose(-1, -2)
#     )

#     # ------------------------------------------------------------------
#     # 4. Project camera-space means into pixel-center coordinates.
#     #
#     #     pixel_x = camera_x * scale_x + width / 2
#     #
#     #     pixel_y = height / 2 - camera_y * scale_y
#     #
#     # Image row coordinates increase downward, producing the negative
#     # camera-Y scale.
#     # ------------------------------------------------------------------

#     scale_x = float(
#         camera.pixels_per_world_unit_x
#     )

#     scale_y = float(
#         camera.pixels_per_world_unit_y
#     )

#     means_pixel_x = (
#         means_camera[:, 0] * scale_x
#         + 0.5 * float(width)
#     )

#     means_pixel_y = (
#         0.5 * float(height)
#         - means_camera[:, 1] * scale_y
#     )

#     means_pixel = torch.stack(
#         (
#             means_pixel_x,
#             means_pixel_y,
#         ),
#         dim=-1,
#     )

#     # ------------------------------------------------------------------
#     # 5. Project camera-space covariance into pixel coordinates.
#     #
#     # The orthographic Jacobian is:
#     #
#     #     J = [[scale_x,       0, 0],
#     #          [      0, -scale_y, 0]]
#     #
#     # The projected covariance is:
#     #
#     #     covariance_pixel = J covariance_camera J^T
#     # ------------------------------------------------------------------

#     scale = torch.tensor(
#         [
#             scale_x,
#             -scale_y,
#         ],
#         dtype=dtype,
#         device=device,
#     )

#     covariance_xy_camera = (
#         covariances_camera[:, :2, :2]
#     )

#     covariances_pixel_unstabilized = (
#         covariance_xy_camera
#         * scale.view(1, 2, 1)
#         * scale.view(1, 1, 2)
#     )

#     # ------------------------------------------------------------------
#     # 6. Clamp projected covariance eigenvalues.
#     #
#     # This replaces the old behavior that added a fixed value to both
#     # diagonal entries. Eigenvalue clamping matches the CPU projector.
#     # ------------------------------------------------------------------

#     # covariances_pixel = (
#     #     _stabilize_symmetric_matrices(
#     #         covariances_pixel_unstabilized,
#     #         minimum_eigenvalue=minimum_pixel_variance,
#     #     )
#     # )

#     # # ------------------------------------------------------------------
#     # # 7. Invert stabilized projected covariances.
#     # # ------------------------------------------------------------------

#     # inverse_covariances_pixel = (
#     #     torch.linalg.inv(
#     #         covariances_pixel
#     #     )
#     # )

#     # # ------------------------------------------------------------------
#     # # 8. Compute Gaussian radii.
#     # #
#     # # The largest eigenvalue is already available conceptually from the
#     # # stabilized covariance, but computing eigvalsh here closely matches
#     # # the CPU implementation and keeps the operation explicit.
#     # # ------------------------------------------------------------------

#     # eigenvalues_pixel = torch.linalg.eigvalsh(
#     #     covariances_pixel
#     # )

#     # maximum_variances = (
#     #     eigenvalues_pixel[:, -1]
#     # )

#     # radii_pixel = (
#     #     float(sigma_extent)
#     #     * torch.sqrt(maximum_variances)
#     # )

#     (
#     covariances_pixel,
#     inverse_covariances_pixel,
#     maximum_variances,
#     ) = _stabilize_invert_symmetric_2x2(
#         covariances_pixel_unstabilized,
#         minimum_eigenvalue=minimum_pixel_variance,
#     )

#     radii_pixel = (
#         float(sigma_extent)
#         * torch.sqrt(maximum_variances)
#     )

#     # gsplat tile intersections expect integer radii. Use the same
#     # conservative enclosing radius along both image axes.
#     radii_integer = torch.ceil(
#         radii_pixel
#     ).to(torch.int32)

#     radii_xy = torch.stack(
#         (
#             radii_integer,
#             radii_integer,
#         ),
#         dim=-1,
#     )

#     # ------------------------------------------------------------------
#     # 9. Camera-space depth and mass.
#     # ------------------------------------------------------------------

#     depths = means_camera[:, 2]

#     masses = model.masses

#     # ------------------------------------------------------------------
#     # 10. Validity and image-intersection mask.
#     # ------------------------------------------------------------------

#     valid = _compute_valid_mask(
#         means_camera=means_camera,
#         means_pixel=means_pixel,
#         covariances_pixel=covariances_pixel,
#         inverse_covariances_pixel=(
#             inverse_covariances_pixel
#         ),
#         radii_pixel=radii_pixel,
#         radii_integer=radii_integer,
#         depths=depths,
#         camera=camera,
#         width=width,
#         height=height,
#         cull_to_image=cull_to_image,
#     )

#     return GpuProjectedGaussians(
#         means_camera=means_camera.contiguous(),
#         means_pixel=means_pixel.contiguous(),
#         covariances_camera=(
#             covariances_camera.contiguous()
#         ),
#         covariances_pixel=(
#             covariances_pixel.contiguous()
#         ),
#         inverse_covariances_pixel=(
#             inverse_covariances_pixel.contiguous()
#         ),
#         radii_pixel=radii_pixel.contiguous(),
#         radii_xy=radii_xy.contiguous(),
#         depths=depths.contiguous(),
#         masses=masses.contiguous(),
#         valid=valid.contiguous(),
#     )



from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..camera.orthographic import OrthographicCamera
from .model import GpuGaussianModel


@dataclass(slots=True)
class GpuProjectedGaussians:
    """Orthographically projected Gaussians stored on the GPU.

    Shapes
    ------
    means_camera
        ``(N, 3)`` camera-space Gaussian means.

    means_pixel
        ``(N, 2)`` pixel-center Gaussian means.

    covariances_camera
        ``(N, 3, 3)`` camera-space covariance matrices.

    covariances_pixel
        ``(N, 2, 2)`` pixel-space covariance matrices.

    inverse_covariances_pixel
        ``(N, 2, 2)`` inverse pixel-space covariance matrices.

    radii_pixel
        ``(N,)`` floating-point support radii.

    radii_xy
        ``(N, 2)`` integer support radii used by gsplat tile
        intersections.

    depths
        ``(N,)`` camera-space depth values.

    masses
        ``(N,)`` Gaussian masses.

    valid
        ``(N,)`` Boolean validity mask.
    """

    means_camera: Tensor
    means_pixel: Tensor

    covariances_camera: Tensor
    covariances_pixel: Tensor
    inverse_covariances_pixel: Tensor

    radii_pixel: Tensor
    radii_xy: Tensor

    depths: Tensor
    masses: Tensor
    valid: Tensor

    @property
    def device(self) -> torch.device:
        """Device containing the projected tensors."""

        return self.means_pixel.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype of the projected tensors."""

        return self.means_pixel.dtype

    @property
    def n_gaussians(self) -> int:
        """Total number of projected Gaussians."""

        return int(self.means_pixel.shape[0])

    @property
    def n_valid(self) -> int:
        """Number of Gaussians marked as valid."""

        return int(self.valid.sum().item())


def _validate_projection_parameters(
    *,
    minimum_eigenvalue: float,
    minimum_pixel_variance: float,
    sigma_extent: float,
) -> None:
    """Validate numerical projection parameters."""

    if not math.isfinite(minimum_eigenvalue):
        raise ValueError(
            "minimum_eigenvalue must be finite."
        )

    if minimum_eigenvalue <= 0.0:
        raise ValueError(
            "minimum_eigenvalue must be positive, "
            f"got {minimum_eigenvalue}."
        )

    if not math.isfinite(minimum_pixel_variance):
        raise ValueError(
            "minimum_pixel_variance must be finite."
        )

    if minimum_pixel_variance <= 0.0:
        raise ValueError(
            "minimum_pixel_variance must be positive, "
            f"got {minimum_pixel_variance}."
        )

    if not math.isfinite(sigma_extent):
        raise ValueError(
            "sigma_extent must be finite."
        )

    if sigma_extent <= 0.0:
        raise ValueError(
            "sigma_extent must be positive, "
            f"got {sigma_extent}."
        )


def _validate_smoothing_parameters(
    *,
    beta: float,
    blob_sigma_pixels: float,
) -> None:
    """Validate paper-style covariance and screen-space blob smoothing."""

    if not math.isfinite(beta):
        raise ValueError(
            "beta must be finite."
        )

    if beta < 0.0:
        raise ValueError(
            "beta must be nonnegative, "
            f"got {beta}."
        )

    if not math.isfinite(blob_sigma_pixels):
        raise ValueError(
            "blob_sigma_pixels must be finite."
        )

    if blob_sigma_pixels < 0.0:
        raise ValueError(
            "blob_sigma_pixels must be nonnegative, "
            f"got {blob_sigma_pixels}."
        )


def _camera_rotation(
    camera: OrthographicCamera,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Copy the camera rotation matrix to the GPU."""

    return torch.as_tensor(
        camera.rotation_matrix,
        dtype=dtype,
        device=device,
    ).contiguous()


def _camera_position(
    camera: OrthographicCamera,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Copy the camera position to the GPU."""

    return torch.as_tensor(
        camera.position,
        dtype=dtype,
        device=device,
    ).contiguous()


def _stabilize_symmetric_matrices(
    matrices: Tensor,
    *,
    minimum_eigenvalue: float,
) -> Tensor:
    """Clamp eigenvalues of batched symmetric matrices.

    Parameters
    ----------
    matrices
        Tensor with shape ``(N, D, D)``.

    minimum_eigenvalue
        Smallest allowed eigenvalue.

    Returns
    -------
    Tensor
        Stabilized symmetric matrices with the same shape.
    """

    symmetric = 0.5 * (
        matrices
        + matrices.transpose(-1, -2)
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(
        symmetric
    )

    clamped_eigenvalues = torch.clamp(
        eigenvalues,
        min=float(minimum_eigenvalue),
    )

    stabilized = torch.matmul(
        torch.matmul(
            eigenvectors,
            torch.diag_embed(clamped_eigenvalues),
        ),
        eigenvectors.transpose(-1, -2),
    )

    # Remove small asymmetry introduced by floating-point reconstruction.
    stabilized = 0.5 * (
        stabilized
        + stabilized.transpose(-1, -2)
    )

    return stabilized

def _stabilize_invert_symmetric_2x2(
    matrices: Tensor,
    *,
    minimum_eigenvalue: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stabilize and invert symmetric 2x2 matrices analytically.

    Parameters
    ----------
    matrices
        Symmetric matrices with shape ``(N, 2, 2)``.

    minimum_eigenvalue
        Lower bound applied to both eigenvalues.

    Returns
    -------
    stabilized
        Stabilized matrices with shape ``(N, 2, 2)``.

    inverse
        Analytic inverse of each stabilized matrix.

    maximum_eigenvalue
        Largest stabilized eigenvalue for each matrix, shape ``(N,)``.

    Notes
    -----
    For

        C = [[a, b],
             [b, c]]

    the eigenvalues are

        lambda_min/max =
            0.5 * (
                a + c
                -/+ sqrt((a-c)^2 + 4b^2)
            )

    When only the smaller eigenvalue needs clamping, the matrix is
    reconstructed using its minimum-eigenvalue spectral projector:

        P_min = (lambda_max I - C)
                / (lambda_max - lambda_min)

        C_stable =
            C + (minimum - lambda_min) P_min

    This preserves the CPU eigenvalue-clamping semantics without calling
    general eigendecomposition or matrix-inverse routines.
    """

    if matrices.ndim != 3:
        raise ValueError(
            "matrices must have shape (N, 2, 2), "
            f"got {tuple(matrices.shape)}."
        )

    if tuple(matrices.shape[-2:]) != (2, 2):
        raise ValueError(
            "matrices must have shape (N, 2, 2), "
            f"got {tuple(matrices.shape)}."
        )

    # Explicitly symmetrize.
    a = matrices[:, 0, 0]
    b = 0.5 * (
        matrices[:, 0, 1]
        + matrices[:, 1, 0]
    )
    c = matrices[:, 1, 1]

    trace = a + c

    discriminant = torch.sqrt(
        torch.clamp(
            (a - c).square()
            + 4.0 * b.square(),
            min=0.0,
        )
    )

    lambda_min = 0.5 * (
        trace - discriminant
    )

    lambda_max = 0.5 * (
        trace + discriminant
    )

    minimum = float(
        minimum_eigenvalue
    )

    # Start from the original symmetric matrix.
    stable_a = a
    stable_b = b
    stable_c = c

    both_below = (
        lambda_max < minimum
    )

    only_minimum_below = (
        (lambda_min < minimum)
        & ~both_below
    )

    # When both eigenvalues are below the threshold, the clamped matrix is
    # simply minimum * identity.
    stable_a = torch.where(
        both_below,
        torch.full_like(
            stable_a,
            minimum,
        ),
        stable_a,
    )

    stable_b = torch.where(
        both_below,
        torch.zeros_like(
            stable_b,
        ),
        stable_b,
    )

    stable_c = torch.where(
        both_below,
        torch.full_like(
            stable_c,
            minimum,
        ),
        stable_c,
    )

    # Clamp only the smaller eigenvalue.
    #
    # P_min = (lambda_max I - C) / (lambda_max - lambda_min)
    spectral_gap = (
        lambda_max - lambda_min
    )

    safe_gap = torch.clamp(
        spectral_gap,
        min=torch.finfo(
            matrices.dtype
        ).eps,
    )

    correction = (
        minimum - lambda_min
    )

    projector_min_00 = (
        lambda_max - a
    ) / safe_gap

    projector_min_01 = (
        -b / safe_gap
    )

    projector_min_11 = (
        lambda_max - c
    ) / safe_gap

    corrected_a = (
        a
        + correction
        * projector_min_00
    )

    corrected_b = (
        b
        + correction
        * projector_min_01
    )

    corrected_c = (
        c
        + correction
        * projector_min_11
    )

    stable_a = torch.where(
        only_minimum_below,
        corrected_a,
        stable_a,
    )

    stable_b = torch.where(
        only_minimum_below,
        corrected_b,
        stable_b,
    )

    stable_c = torch.where(
        only_minimum_below,
        corrected_c,
        stable_c,
    )

    stabilized = torch.stack(
        (
            stable_a,
            stable_b,
            stable_b,
            stable_c,
        ),
        dim=-1,
    ).reshape(-1, 2, 2)

    # Analytic inverse.
    determinant = (
        stable_a * stable_c
        - stable_b.square()
    )

    determinant_floor = max(
        minimum * minimum,
        torch.finfo(
            matrices.dtype
        ).tiny,
    )

    safe_determinant = torch.clamp(
        determinant,
        min=determinant_floor,
    )

    inverse_a = (
        stable_c / safe_determinant
    )

    inverse_b = (
        -stable_b / safe_determinant
    )

    inverse_c = (
        stable_a / safe_determinant
    )

    inverse = torch.stack(
        (
            inverse_a,
            inverse_b,
            inverse_b,
            inverse_c,
        ),
        dim=-1,
    ).reshape(-1, 2, 2)

    maximum_eigenvalue = torch.clamp(
        lambda_max,
        min=minimum,
    )

    return (
        stabilized.contiguous(),
        inverse.contiguous(),
        maximum_eigenvalue.contiguous(),
    )

def _compute_valid_mask(
    *,
    means_camera: Tensor,
    means_pixel: Tensor,
    covariances_pixel: Tensor,
    inverse_covariances_pixel: Tensor,
    radii_pixel: Tensor,
    radii_integer: Tensor,
    depths: Tensor,
    camera: OrthographicCamera,
    width: int,
    height: int,
    cull_to_image: bool,
) -> Tensor:
    """Compute the GPU projection validity mask."""

    valid = (
        torch.isfinite(means_camera).all(dim=-1)
        & torch.isfinite(means_pixel).all(dim=-1)
        & torch.isfinite(covariances_pixel).all(
            dim=(-2, -1)
        )
        & torch.isfinite(
            inverse_covariances_pixel
        ).all(dim=(-2, -1))
        & torch.isfinite(radii_pixel)
        & torch.isfinite(depths)
        & (radii_pixel > 0.0)
        & (radii_integer > 0)
    )

    near = float(camera.near)

    valid = valid & (
        depths >= near
    )

    far = float(camera.far)

    if math.isfinite(far):
        valid = valid & (
            depths <= far
        )

    if cull_to_image:
        x = means_pixel[:, 0]
        y = means_pixel[:, 1]

        intersects_image = (
            (x + radii_pixel >= 0.0)
            & (x - radii_pixel < float(width))
            & (y + radii_pixel >= 0.0)
            & (y - radii_pixel < float(height))
        )

        valid = valid & intersects_image

    return valid


def project_gaussians_orthographic_gpu(
    model: GpuGaussianModel,
    camera: OrthographicCamera,
    *,
    minimum_eigenvalue: float = 1.0e-6,
    minimum_pixel_variance: float = 1.0e-4,
    sigma_extent: float = 3.0,
    beta: float = 0.0,
    blob_sigma_pixels: float = 0.0,
    cull_to_image: bool = True,
    support_sigma: float | None = None,
    covariance_regularization: float | None = None,
    determinant_epsilon: float | None = None,
) -> GpuProjectedGaussians:
    """Project GPU-resident Gaussians with an orthographic camera.

    This implementation follows the same mathematical operations as the
    validated CPU projector:

    1. Clamp world-space covariance eigenvalues.
    2. Rotate means and covariances into camera space.
    3. Project means and covariances into pixel coordinates.
    4. Clamp projected 2D covariance eigenvalues.
    5. Invert the stabilized pixel covariances.
    6. Compute projected support radii and validity.

    Parameters
    ----------
    model
        Gaussian model already resident on the GPU.

    camera
        Orthographic camera.

    minimum_eigenvalue
        Minimum world-space covariance eigenvalue.

    minimum_pixel_variance
        Minimum projected covariance eigenvalue in squared pixels.

    sigma_extent
        Number of standard deviations used for the support radius.

    beta
        Paper-style covariance smoothing parameter. The projected model uses
        ``(1 + beta) * Sigma`` before screen-space projection.

    blob_sigma_pixels
        Standard deviation of an isotropic screen-space Gaussian blob,
        measured in pixels. Its variance is added to the projected covariance
        diagonal before stabilization, inversion, and radius computation.

    cull_to_image
        Mark Gaussians that do not intersect the image as invalid.

    support_sigma
        Deprecated alias for ``sigma_extent``. It is retained for
        compatibility with earlier code.

    covariance_regularization
        Deprecated alias for ``minimum_pixel_variance``. Earlier code added
        this value to the covariance diagonal. The updated implementation
        interprets it as an eigenvalue lower bound.

    determinant_epsilon
        Deprecated compatibility argument. Stabilized covariance matrices
        are positive definite, so determinant clamping is no longer needed.

    Returns
    -------
    GpuProjectedGaussians
        GPU-resident projected Gaussian parameters.
    """

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    if support_sigma is not None:
        sigma_extent = float(
            support_sigma
        )

    # ------------------------------------------------------------------
    # Backward compatibility with the previous API.
    #
    # Previously:
    #
    #     covariance_regularization = 0.0
    #
    # meant "do not regularize".
    #
    # In the new implementation the equivalent behaviour is to keep the
    # default minimum_pixel_variance unchanged.
    # ------------------------------------------------------------------

    if covariance_regularization is not None:
        value = float(covariance_regularization)

        if value < 0.0:
            raise ValueError(
                "covariance_regularization must be nonnegative."
            )

        if value > 0.0:
            minimum_pixel_variance = value

    # Kept only so older callers do not fail. The value is not required
    # after eigenvalue-based covariance stabilization.
    if determinant_epsilon is not None:
        if not math.isfinite(determinant_epsilon):
            raise ValueError(
                "determinant_epsilon must be finite."
            )

        if determinant_epsilon < 0.0:
            raise ValueError(
                "determinant_epsilon must be nonnegative, "
                f"got {determinant_epsilon}."
            )

    _validate_projection_parameters(
        minimum_eigenvalue=minimum_eigenvalue,
        minimum_pixel_variance=minimum_pixel_variance,
        sigma_extent=sigma_extent,
    )

    _validate_smoothing_parameters(
        beta=beta,
        blob_sigma_pixels=blob_sigma_pixels,
    )

    device = model.device
    dtype = model.dtype

    width, height = camera.image_size

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image size: {(width, height)}."
        )

    view_width = float(
        camera.view_width
    )

    view_height = float(
        camera.view_height
    )

    if not math.isfinite(view_width):
        raise ValueError(
            "camera.view_width must be finite."
        )

    if not math.isfinite(view_height):
        raise ValueError(
            "camera.view_height must be finite."
        )

    if view_width <= 0.0 or view_height <= 0.0:
        raise ValueError(
            "Camera view dimensions must be positive, "
            f"got {(view_width, view_height)}."
        )

    rotation = _camera_rotation(
        camera,
        device=device,
        dtype=dtype,
    )

    camera_position = _camera_position(
        camera,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------------
    # 1. Stabilize world-space covariance matrices.
    #
    # This matches:
    #
    #     model.stabilized_covariances(
    #         minimum_eigenvalue=minimum_eigenvalue
    #     )
    #
    # in the CPU projector.
    # ------------------------------------------------------------------

    # stabilized_world_covariances = (
    #     _stabilize_symmetric_matrices(
    #         model.covariances,
    #         minimum_eigenvalue=minimum_eigenvalue,
    #     )
    # )

    model.require_stabilization(
    minimum_eigenvalue
    )

    stabilized_world_covariances = (
        model.stabilized_covariances
    )

    # ------------------------------------------------------------------
    # 2. Transform means from world space into camera space.
    #
    # Row-vector representation:
    #
    #     mean_camera = (mean_world - camera_position) @ R^T
    # ------------------------------------------------------------------

    centered_means = (
        model.means
        - camera_position
    )

    rotation_t = rotation.transpose(
        0,
        1,
    )

    means_camera = torch.matmul(
        centered_means,
        rotation_t,
    )

    # ------------------------------------------------------------------
    # 3. Rotate covariance matrices into camera space.
    #
    #     covariance_camera = R covariance_world R^T
    # ------------------------------------------------------------------

    covariances_camera = torch.matmul(
        torch.matmul(
            rotation.unsqueeze(0),
            stabilized_world_covariances,
        ),
        rotation_t.unsqueeze(0),
    )

    covariances_camera = 0.5 * (
        covariances_camera
        + covariances_camera.transpose(-1, -2)
    )

    # Paper-style model smoothing:
    #
    #     Sigma_beta = (1 + beta) * Sigma
    #
    # Because beta is a scalar, applying it after camera rotation is exactly
    # equivalent to applying it in world space before rotation.
    if beta > 0.0:
        covariances_camera = (
            covariances_camera
            * (1.0 + float(beta))
        )

    # ------------------------------------------------------------------
    # 4. Project camera-space means into pixel-center coordinates.
    #
    #     pixel_x = camera_x * scale_x + width / 2
    #
    #     pixel_y = height / 2 - camera_y * scale_y
    #
    # Image row coordinates increase downward, producing the negative
    # camera-Y scale.
    # ------------------------------------------------------------------

    scale_x = float(
        camera.pixels_per_world_unit_x
    )

    scale_y = float(
        camera.pixels_per_world_unit_y
    )

    means_pixel_x = (
        means_camera[:, 0] * scale_x
        + 0.5 * float(width)
    )

    means_pixel_y = (
        0.5 * float(height)
        - means_camera[:, 1] * scale_y
    )

    means_pixel = torch.stack(
        (
            means_pixel_x,
            means_pixel_y,
        ),
        dim=-1,
    )

    # ------------------------------------------------------------------
    # 5. Project camera-space covariance into pixel coordinates.
    #
    # The orthographic Jacobian is:
    #
    #     J = [[scale_x,       0, 0],
    #          [      0, -scale_y, 0]]
    #
    # The projected covariance is:
    #
    #     covariance_pixel = J covariance_camera J^T
    # ------------------------------------------------------------------

    scale = torch.tensor(
        [
            scale_x,
            -scale_y,
        ],
        dtype=dtype,
        device=device,
    )

    covariance_xy_camera = (
        covariances_camera[:, :2, :2]
    )

    covariances_pixel_unstabilized = (
        covariance_xy_camera
        * scale.view(1, 2, 1)
        * scale.view(1, 1, 2)
    )

    # Match the particle-GT Gaussian blob by convolving every projected
    # Gaussian with an isotropic screen-space Gaussian:
    #
    #     Sigma_pixel <- Sigma_pixel + blob_sigma_pixels^2 I
    #
    # This must happen before covariance stabilization, inversion, and support
    # radius computation.
    if blob_sigma_pixels > 0.0:
        blob_variance = (
            float(blob_sigma_pixels) ** 2
        )

        covariances_pixel_unstabilized = (
            covariances_pixel_unstabilized.clone()
        )

        covariances_pixel_unstabilized[
            :,
            0,
            0,
        ] += blob_variance

        covariances_pixel_unstabilized[
            :,
            1,
            1,
        ] += blob_variance

    # ------------------------------------------------------------------
    # 6. Clamp projected covariance eigenvalues.
    #
    # This replaces the old behavior that added a fixed value to both
    # diagonal entries. Eigenvalue clamping matches the CPU projector.
    # ------------------------------------------------------------------

    # covariances_pixel = (
    #     _stabilize_symmetric_matrices(
    #         covariances_pixel_unstabilized,
    #         minimum_eigenvalue=minimum_pixel_variance,
    #     )
    # )

    # # ------------------------------------------------------------------
    # # 7. Invert stabilized projected covariances.
    # # ------------------------------------------------------------------

    # inverse_covariances_pixel = (
    #     torch.linalg.inv(
    #         covariances_pixel
    #     )
    # )

    # # ------------------------------------------------------------------
    # # 8. Compute Gaussian radii.
    # #
    # # The largest eigenvalue is already available conceptually from the
    # # stabilized covariance, but computing eigvalsh here closely matches
    # # the CPU implementation and keeps the operation explicit.
    # # ------------------------------------------------------------------

    # eigenvalues_pixel = torch.linalg.eigvalsh(
    #     covariances_pixel
    # )

    # maximum_variances = (
    #     eigenvalues_pixel[:, -1]
    # )

    # radii_pixel = (
    #     float(sigma_extent)
    #     * torch.sqrt(maximum_variances)
    # )

    (
    covariances_pixel,
    inverse_covariances_pixel,
    maximum_variances,
    ) = _stabilize_invert_symmetric_2x2(
        covariances_pixel_unstabilized,
        minimum_eigenvalue=minimum_pixel_variance,
    )

    radii_pixel = (
        float(sigma_extent)
        * torch.sqrt(maximum_variances)
    )

    # gsplat tile intersections expect integer radii. Use the same
    # conservative enclosing radius along both image axes.
    radii_integer = torch.ceil(
        radii_pixel
    ).to(torch.int32)

    radii_xy = torch.stack(
        (
            radii_integer,
            radii_integer,
        ),
        dim=-1,
    )

    # ------------------------------------------------------------------
    # 9. Camera-space depth and mass.
    # ------------------------------------------------------------------

    depths = means_camera[:, 2]

    masses = model.masses

    # ------------------------------------------------------------------
    # 10. Validity and image-intersection mask.
    # ------------------------------------------------------------------

    valid = _compute_valid_mask(
        means_camera=means_camera,
        means_pixel=means_pixel,
        covariances_pixel=covariances_pixel,
        inverse_covariances_pixel=(
            inverse_covariances_pixel
        ),
        radii_pixel=radii_pixel,
        radii_integer=radii_integer,
        depths=depths,
        camera=camera,
        width=width,
        height=height,
        cull_to_image=cull_to_image,
    )

    return GpuProjectedGaussians(
        means_camera=means_camera.contiguous(),
        means_pixel=means_pixel.contiguous(),
        covariances_camera=(
            covariances_camera.contiguous()
        ),
        covariances_pixel=(
            covariances_pixel.contiguous()
        ),
        inverse_covariances_pixel=(
            inverse_covariances_pixel.contiguous()
        ),
        radii_pixel=radii_pixel.contiguous(),
        radii_xy=radii_xy.contiguous(),
        depths=depths.contiguous(),
        masses=masses.contiguous(),
        valid=valid.contiguous(),
    )