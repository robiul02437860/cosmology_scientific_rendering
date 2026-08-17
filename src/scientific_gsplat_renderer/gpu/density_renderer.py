from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from gsplat.cuda import _wrapper

from .orthographic_projection import (
    GpuProjectedGaussians,
)
from .tile_intersections import (
    GpuTileIntersections,
)


@dataclass(slots=True)
class GpuDensityRenderResult:
    """Result of additive scientific density rendering on the GPU."""

    density: Tensor

    input_gaussians: int
    valid_gaussians: int
    intersections: int

    preparation_seconds: float
    rasterization_seconds: float
    total_seconds: float

    @property
    def device(self) -> torch.device:
        """Return the device containing the rendered density."""

        return self.density.device

    @property
    def image_height(self) -> int:
        """Return the rendered image height."""

        return int(self.density.shape[0])

    @property
    def image_width(self) -> int:
        """Return the rendered image width."""

        return int(self.density.shape[1])

    @property
    def image_mass(self) -> float:
        """Return the sum of all rendered density values."""

        return float(
            self.density.sum().item()
        )

    def density_numpy(self):
        """Copy the density image to a NumPy array."""

        return (
            self.density
            .detach()
            .cpu()
            .numpy()
        )


def _validate_inputs(
    projected: GpuProjectedGaussians,
    intersections: GpuTileIntersections,
) -> None:
    """Validate projected tensors and tile intersections."""

    if projected.device.type != "cuda":
        raise ValueError(
            "GPU density rendering requires CUDA tensors."
        )

    if intersections.device != projected.device:
        raise ValueError(
            "Projected Gaussians and tile intersections must be "
            "on the same CUDA device."
        )

    n_gaussians = projected.n_gaussians

    expected_shapes = {
        "means_pixel": (
            n_gaussians,
            2,
        ),
        "covariances_pixel": (
            n_gaussians,
            2,
            2,
        ),
        "inverse_covariances_pixel": (
            n_gaussians,
            2,
            2,
        ),
        "masses": (
            n_gaussians,
        ),
        "valid": (
            n_gaussians,
        ),
    }

    tensors = {
        "means_pixel": projected.means_pixel,
        "covariances_pixel": (
            projected.covariances_pixel
        ),
        "inverse_covariances_pixel": (
            projected.inverse_covariances_pixel
        ),
        "masses": projected.masses,
        "valid": projected.valid,
    }

    for name, tensor in tensors.items():
        expected = expected_shapes[name]

        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, "
                f"got {tuple(tensor.shape)}."
            )

        if tensor.device != projected.device:
            raise ValueError(
                f"{name} must be on {projected.device}, "
                f"got {tensor.device}."
            )

    expected_offset_shape = (
        1,
        intersections.tile_height,
        intersections.tile_width,
    )

    if tuple(
        intersections.isect_offsets.shape
    ) != expected_offset_shape:
        raise ValueError(
            "Intersection offsets must have shape "
            f"{expected_offset_shape}, got "
            f"{tuple(intersections.isect_offsets.shape)}."
        )

    if (
        intersections.flatten_ids.ndim != 1
    ):
        raise ValueError(
            "flatten_ids must be one-dimensional."
        )

    if (
        intersections.flatten_ids.numel()
        != intersections.isect_ids.numel()
    ):
        raise ValueError(
            "isect_ids and flatten_ids must contain the same "
            "number of intersections."
        )


def _build_conics(
    projected: GpuProjectedGaussians,
) -> Tensor:
    """Pack inverse covariance matrices into gsplat conic form.

    The expected conic layout is:

    ``[inverse_xx, inverse_xy, inverse_yy]``.
    """

    inverse = (
        projected.inverse_covariances_pixel
    )

    conics = torch.stack(
        (
            inverse[:, 0, 0],
            inverse[:, 0, 1],
            inverse[:, 1, 1],
        ),
        dim=-1,
    )

    return conics.contiguous()


def _build_amplitudes(
    projected: GpuProjectedGaussians,
    *,
    normalize_gaussian_mass: bool,
) -> Tensor:
    """Compute the additive peak amplitude of every Gaussian.

    When normalization is enabled:

    ``amplitude = mass / (2*pi*sqrt(det(covariance_pixel)))``

    so each complete two-dimensional Gaussian integrates to its mass.
    """

    masses = projected.masses

    if not normalize_gaussian_mass:
        return masses.contiguous()

    covariance = projected.covariances_pixel

    determinant = (
        covariance[:, 0, 0]
        * covariance[:, 1, 1]
        - covariance[:, 0, 1]
        * covariance[:, 1, 0]
    )

    determinant = torch.clamp(
        determinant,
        min=torch.finfo(
            covariance.dtype
        ).tiny,
    )

    normalization = (
        2.0
        * math.pi
        * torch.sqrt(determinant)
    )

    amplitudes = (
        masses / normalization
    )

    return amplitudes.contiguous()


def render_density_gpu(
    projected: GpuProjectedGaussians,
    intersections: GpuTileIntersections,
    *,
    normalize_gaussian_mass: bool = True,
) -> GpuDensityRenderResult:
    """Render additive density using the custom scientific CUDA kernel.

    The rendered field is

    ``density(p) = sum_i amplitude_i exp(-0.5 q_i(p))``

    where ``q_i`` is the projected Gaussian Mahalanobis distance.

    Parameters
    ----------
    projected
        GPU-resident projected Gaussian parameters.

    intersections
        gsplat tile intersections built from the same projected data.

    normalize_gaussian_mass
        If true, normalize every projected Gaussian so its integral over
        the complete image plane equals its represented mass.

    Returns
    -------
    GpuDensityRenderResult
        GPU density image and timing information.
    """

    _validate_inputs(
        projected,
        intersections,
    )

    device = projected.device

    preparation_start = torch.cuda.Event(
        enable_timing=True
    )
    preparation_end = torch.cuda.Event(
        enable_timing=True
    )

    rasterization_start = torch.cuda.Event(
        enable_timing=True
    )
    rasterization_end = torch.cuda.Event(
        enable_timing=True
    )

    preparation_start.record()

    means2d = (
        projected.means_pixel
        .unsqueeze(0)
        .contiguous()
    )

    conics = (
        _build_conics(projected)
        .unsqueeze(0)
        .contiguous()
    )

    amplitudes = (
        _build_amplitudes(
            projected,
            normalize_gaussian_mass=(
                normalize_gaussian_mass
            ),
        )
        .unsqueeze(0)
        .contiguous()
    )

    # The conditional kernel requires three channels:
    #
    # channel 0: conditional mean
    # channel 1: x slope
    # channel 2: y slope
    #
    # Density is returned independently through `alphas`, so these values
    # are zero for a density-only render.
    conditional_parameters = torch.zeros(
        (
            1,
            projected.n_gaussians,
            3,
        ),
        dtype=projected.dtype,
        device=device,
    )

    preparation_end.record()

    rasterization_start.record()

    (
        _renders,
        alphas,
        _last_ids,
    ) = (
        _wrapper
        .rasterize_to_pixels_scientific_conditional(
            means2d=means2d,
            conics=conics,
            conditional_params=(
                conditional_parameters
            ),
            opacities=amplitudes,
            image_width=(
                intersections.image_width
            ),
            image_height=(
                intersections.image_height
            ),
            tile_size=(
                intersections.tile_size
            ),
            isect_offsets=(
                intersections.isect_offsets
            ),
            flatten_ids=(
                intersections.flatten_ids
            ),
            backgrounds=None,
            masks=None,
        )
    )

    rasterization_end.record()
    rasterization_end.synchronize()

    preparation_seconds = (
        preparation_start.elapsed_time(
            preparation_end
        )
        / 1000.0
    )

    rasterization_seconds = (
        rasterization_start.elapsed_time(
            rasterization_end
        )
        / 1000.0
    )

    if alphas.ndim == 4:
        if alphas.shape[-1] != 1:
            raise RuntimeError(
                "Expected alphas with one final channel, "
                f"got shape {tuple(alphas.shape)}."
            )

        density = alphas[0, :, :, 0]

    elif alphas.ndim == 3:
        density = alphas[0]

    else:
        raise RuntimeError(
            "Unexpected alpha/density tensor shape: "
            f"{tuple(alphas.shape)}."
        )

    density = density.contiguous()

    if tuple(density.shape) != (
        intersections.image_height,
        intersections.image_width,
    ):
        raise RuntimeError(
            "Unexpected rendered density shape: "
            f"{tuple(density.shape)}."
        )

    if not torch.isfinite(density).all():
        raise RuntimeError(
            "Rendered density contains NaN or infinite values."
        )

    total_seconds = (
        preparation_seconds
        + rasterization_seconds
    )

    return GpuDensityRenderResult(
        density=density,
        input_gaussians=(
            projected.n_gaussians
        ),
        valid_gaussians=(
            intersections.n_valid_gaussians
        ),
        intersections=(
            intersections.n_intersections
        ),
        preparation_seconds=float(
            preparation_seconds
        ),
        rasterization_seconds=float(
            rasterization_seconds
        ),
        total_seconds=float(
            total_seconds
        ),
    )