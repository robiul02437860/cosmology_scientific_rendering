from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..camera.orthographic import OrthographicCamera
from .model import GpuGaussianModel
from .orthographic_projection import (
    GpuProjectedGaussians,
)


@dataclass(slots=True)
class GpuProjectedConditionalAttributes:
    """Conditional scalar-attribute parameters stored on the GPU.

    For Gaussian ``i`` and pixel-center position ``p``:

    ``a_i(p) = mean_i + slope_i dot (p - mean_pixel_i)``

    Shapes
    ------
    means
        ``(N,)`` Gaussian attribute means.

    cross_covariances_camera
        ``(N, 3)`` position-attribute cross covariances expressed in
        camera coordinates.

    cross_covariances_pixel
        ``(N, 2)`` projected position-attribute cross covariances in
        pixel coordinates.

    slopes_pixel
        ``(N, 2)`` conditional linear slopes in attribute units per pixel.

    valid
        ``(N,)`` Boolean mask indicating finite, usable conditional
        parameters.
    """

    means: Tensor
    cross_covariances_camera: Tensor
    cross_covariances_pixel: Tensor
    slopes_pixel: Tensor
    valid: Tensor

    @property
    def device(self) -> torch.device:
        """Return the device containing the tensors."""

        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype."""

        return self.means.dtype

    @property
    def n_gaussians(self) -> int:
        """Return the number of Gaussian components."""

        return int(self.means.shape[0])

    @property
    def n_valid(self) -> int:
        """Return the number of valid conditional components."""

        return int(self.valid.sum().item())

    @property
    def conditional_parameters(self) -> Tensor:
        """Return CUDA rasterizer parameters with shape ``(N, 3)``.

        Channel layout:

        - channel 0: attribute mean
        - channel 1: pixel-space x slope
        - channel 2: pixel-space y slope
        """

        return torch.cat(
            (
                self.means.unsqueeze(-1),
                self.slopes_pixel,
            ),
            dim=-1,
        ).contiguous()


def _validate_inputs(
    model: GpuGaussianModel,
    projected: GpuProjectedGaussians,
) -> None:
    """Validate model and projected-Gaussian compatibility."""

    if model.n_gaussians != projected.n_gaussians:
        raise ValueError(
            "Model and projected Gaussian counts differ: "
            f"{model.n_gaussians} versus "
            f"{projected.n_gaussians}."
        )

    if model.device != projected.device:
        raise ValueError(
            "Model and projected Gaussians must be on the same "
            f"device, got {model.device} and {projected.device}."
        )

    if model.dtype != projected.dtype:
        raise ValueError(
            "Model and projected Gaussians must use the same dtype, "
            f"got {model.dtype} and {projected.dtype}."
        )

    n = model.n_gaussians

    if tuple(
        model.position_attribute_cross_covariances.shape
    ) != (n, 3):
        raise ValueError(
            "position_attribute_cross_covariances must have shape "
            f"({n}, 3), got "
            f"{tuple(model.position_attribute_cross_covariances.shape)}."
        )

    if tuple(
        projected.inverse_covariances_pixel.shape
    ) != (n, 2, 2):
        raise ValueError(
            "inverse_covariances_pixel must have shape "
            f"({n}, 2, 2), got "
            f"{tuple(projected.inverse_covariances_pixel.shape)}."
        )

    if tuple(projected.valid.shape) != (n,):
        raise ValueError(
            f"projected.valid must have shape ({n},), got "
            f"{tuple(projected.valid.shape)}."
        )


def _camera_rotation(
    camera: OrthographicCamera,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Copy the world-to-camera rotation matrix to the tensor device."""

    return torch.as_tensor(
        camera.rotation_matrix,
        dtype=dtype,
        device=device,
    ).contiguous()


def project_conditional_attributes_orthographic_gpu(
    model: GpuGaussianModel,
    projected: GpuProjectedGaussians,
    camera: OrthographicCamera,
    *,
    slope_limit: float | None = None,
) -> GpuProjectedConditionalAttributes:
    """Project conditional scalar-attribute parameters on the GPU.

    The model stores the world-space position-attribute cross covariance

    ``c_world = Cov(X_world, A)``.

    It is first rotated into camera coordinates:

    ``c_camera = R c_world``.

    The image-space position transform has Jacobian

    ``J = diag(scale_x, -scale_y)``.

    Therefore:

    ``c_pixel = J c_camera_xy``.

    The conditional slope is

    ``slope_pixel = covariance_pixel^-1 c_pixel``.

    Since the covariance is symmetric, this is equivalent to the row-vector
    expression commonly written as:

    ``c_pixel^T covariance_pixel^-1``.

    Parameters
    ----------
    model
        Persistent GPU-resident Gaussian model.

    projected
        GPU orthographic projection of the same model.

    camera
        Camera used to create ``projected``.

    slope_limit
        Optional absolute slope clamp. ``None`` preserves the unconstrained
        conditional model. This should normally remain ``None`` during
        scientific validation.

    Returns
    -------
    GpuProjectedConditionalAttributes
        Attribute means, projected cross covariances, pixel-space slopes,
        and validity mask.
    """

    _validate_inputs(
        model,
        projected,
    )

    if slope_limit is not None:
        if not math.isfinite(slope_limit):
            raise ValueError(
                "slope_limit must be finite when provided."
            )

        if slope_limit <= 0.0:
            raise ValueError(
                "slope_limit must be positive when provided, "
                f"got {slope_limit}."
            )

    device = model.device
    dtype = model.dtype

    rotation = _camera_rotation(
        camera,
        device=device,
        dtype=dtype,
    )

    # --------------------------------------------------------------
    # 1. Rotate world-space position-attribute cross covariance.
    #
    # The cross covariance is treated as a column vector:
    #
    #     c_camera = R c_world
    #
    # With batched row storage this becomes:
    #
    #     c_camera_rows = c_world_rows R^T
    # --------------------------------------------------------------

    cross_covariances_camera = torch.matmul(
        model.position_attribute_cross_covariances,
        rotation.transpose(0, 1),
    )

    # --------------------------------------------------------------
    # 2. Convert camera-space cross covariance to pixel coordinates.
    #
    # Pixel x grows with camera x.
    # Pixel y grows opposite camera y.
    # --------------------------------------------------------------

    scale_x = float(
        camera.pixels_per_world_unit_x
    )

    scale_y = float(
        camera.pixels_per_world_unit_y
    )

    cross_covariance_pixel_x = (
        cross_covariances_camera[:, 0]
        * scale_x
    )

    cross_covariance_pixel_y = (
        cross_covariances_camera[:, 1]
        * (-scale_y)
    )

    cross_covariances_pixel = torch.stack(
        (
            cross_covariance_pixel_x,
            cross_covariance_pixel_y,
        ),
        dim=-1,
    )

    # --------------------------------------------------------------
    # 3. Compute conditional slopes.
    #
    # For each Gaussian:
    #
    #     slope = inverse_covariance_pixel @ cross_covariance_pixel
    # --------------------------------------------------------------

    slopes_pixel = torch.matmul(
        projected.inverse_covariances_pixel,
        cross_covariances_pixel.unsqueeze(-1),
    ).squeeze(-1)

    if slope_limit is not None:
        slopes_pixel = torch.clamp(
            slopes_pixel,
            min=-float(slope_limit),
            max=float(slope_limit),
        )

    means = model.attribute_means

    valid = (
        projected.valid
        & torch.isfinite(means)
        & torch.isfinite(
            cross_covariances_camera
        ).all(dim=-1)
        & torch.isfinite(
            cross_covariances_pixel
        ).all(dim=-1)
        & torch.isfinite(
            slopes_pixel
        ).all(dim=-1)
    )

    return GpuProjectedConditionalAttributes(
        means=means.contiguous(),
        cross_covariances_camera=(
            cross_covariances_camera.contiguous()
        ),
        cross_covariances_pixel=(
            cross_covariances_pixel.contiguous()
        ),
        slopes_pixel=slopes_pixel.contiguous(),
        valid=valid.contiguous(),
    )