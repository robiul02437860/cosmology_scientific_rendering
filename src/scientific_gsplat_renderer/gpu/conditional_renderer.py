from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from gsplat.cuda import _wrapper

from .conditional_projection import (
    GpuProjectedConditionalAttributes,
)
from .orthographic_projection import (
    GpuProjectedGaussians,
)
from .tile_intersections import (
    GpuTileIntersections,
)


@dataclass(slots=True)
class GpuConditionalRenderResult:
    """GPU result for density and a conditional scalar attribute."""

    density: Tensor
    attribute_numerator: Tensor
    attribute: Tensor
    valid_mask: Tensor

    input_gaussians: int
    valid_gaussians: int
    intersections: int

    preparation_seconds: float
    rasterization_seconds: float
    normalization_seconds: float
    total_seconds: float

    density_threshold: float

    @property
    def device(self) -> torch.device:
        return self.density.device

    @property
    def image_height(self) -> int:
        return int(self.density.shape[0])

    @property
    def image_width(self) -> int:
        return int(self.density.shape[1])

    @property
    def image_mass(self) -> float:
        return float(
            self.density.sum().item()
        )

    def density_numpy(self):
        return (
            self.density
            .detach()
            .cpu()
            .numpy()
        )

    def attribute_numerator_numpy(self):
        return (
            self.attribute_numerator
            .detach()
            .cpu()
            .numpy()
        )

    def attribute_numpy(self):
        return (
            self.attribute
            .detach()
            .cpu()
            .numpy()
        )

    def valid_mask_numpy(self):
        return (
            self.valid_mask
            .detach()
            .cpu()
            .numpy()
        )


def _validate_inputs(
    projected: GpuProjectedGaussians,
    conditional: GpuProjectedConditionalAttributes,
    intersections: GpuTileIntersections,
) -> None:
    """Validate compatibility among GPU rendering inputs."""

    if projected.device.type != "cuda":
        raise ValueError(
            "GPU conditional rendering requires CUDA tensors."
        )

    if conditional.device != projected.device:
        raise ValueError(
            "Conditional parameters and projected Gaussians must "
            "be on the same CUDA device."
        )

    if intersections.device != projected.device:
        raise ValueError(
            "Tile intersections and projected Gaussians must "
            "be on the same CUDA device."
        )

    if projected.n_gaussians != conditional.n_gaussians:
        raise ValueError(
            "Projected Gaussian and conditional parameter counts differ: "
            f"{projected.n_gaussians} versus "
            f"{conditional.n_gaussians}."
        )

    n_gaussians = projected.n_gaussians

    expected_shapes = {
        "means_pixel": (n_gaussians, 2),
        "covariances_pixel": (n_gaussians, 2, 2),
        "inverse_covariances_pixel": (n_gaussians, 2, 2),
        "masses": (n_gaussians,),
        "projected_valid": (n_gaussians,),
        "conditional_means": (n_gaussians,),
        "conditional_slopes": (n_gaussians, 2),
        "conditional_valid": (n_gaussians,),
    }

    actual_shapes = {
        "means_pixel": tuple(
            projected.means_pixel.shape
        ),
        "covariances_pixel": tuple(
            projected.covariances_pixel.shape
        ),
        "inverse_covariances_pixel": tuple(
            projected.inverse_covariances_pixel.shape
        ),
        "masses": tuple(
            projected.masses.shape
        ),
        "projected_valid": tuple(
            projected.valid.shape
        ),
        "conditional_means": tuple(
            conditional.means.shape
        ),
        "conditional_slopes": tuple(
            conditional.slopes_pixel.shape
        ),
        "conditional_valid": tuple(
            conditional.valid.shape
        ),
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[name]

        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {actual_shape}."
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
            "isect_offsets must have shape "
            f"{expected_offset_shape}, got "
            f"{tuple(intersections.isect_offsets.shape)}."
        )

    if intersections.flatten_ids.ndim != 1:
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
    """Pack inverse covariance matrices as ``[xx, xy, yy]``."""

    inverse = (
        projected.inverse_covariances_pixel
    )

    return torch.stack(
        (
            inverse[:, 0, 0],
            inverse[:, 0, 1],
            inverse[:, 1, 1],
        ),
        dim=-1,
    ).contiguous()


def _build_amplitudes(
    projected: GpuProjectedGaussians,
    *,
    normalize_gaussian_mass: bool,
) -> Tensor:
    """Compute Gaussian peak amplitudes."""

    masses = projected.masses

    if not normalize_gaussian_mass:
        return masses.contiguous()

    covariance = (
        projected.covariances_pixel
    )

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

    return (
        masses / normalization
    ).contiguous()


def render_conditional_attribute_gpu(
    projected: GpuProjectedGaussians,
    conditional: GpuProjectedConditionalAttributes,
    intersections: GpuTileIntersections,
    *,
    normalize_gaussian_mass: bool = True,
    relative_density_threshold: float = 1.0e-6,
    epsilon: float = 1.0e-12,
) -> GpuConditionalRenderResult:
    """Render density and a conditional scalar attribute on the GPU."""

    _validate_inputs(
        projected,
        conditional,
        intersections,
    )

    if not math.isfinite(
        relative_density_threshold
    ):
        raise ValueError(
            "relative_density_threshold must be finite."
        )

    if relative_density_threshold < 0.0:
        raise ValueError(
            "relative_density_threshold must be nonnegative."
        )

    if not math.isfinite(epsilon):
        raise ValueError(
            "epsilon must be finite."
        )

    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be positive."
        )

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

    normalization_start = torch.cuda.Event(
        enable_timing=True
    )
    normalization_end = torch.cuda.Event(
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

    conditional_parameters = (
        conditional
        .conditional_parameters
        .unsqueeze(0)
        .contiguous()
    )

    preparation_end.record()

    rasterization_start.record()

    (
        renders,
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

    normalization_start.record()

    if alphas.ndim == 4:
        if alphas.shape[-1] != 1:
            raise RuntimeError(
                "Expected one density channel, got "
                f"{tuple(alphas.shape)}."
            )

        density = alphas[
            0,
            :,
            :,
            0,
        ]

    elif alphas.ndim == 3:
        density = alphas[0]

    else:
        raise RuntimeError(
            "Unexpected density tensor shape: "
            f"{tuple(alphas.shape)}."
        )

    if renders.ndim != 4:
        raise RuntimeError(
            "Expected renders with shape (1,H,W,C), got "
            f"{tuple(renders.shape)}."
        )

    if renders.shape[-1] < 1:
        raise RuntimeError(
            "Conditional renderer returned no output channels."
        )

    attribute_numerator = renders[
        0,
        :,
        :,
        0,
    ]

    maximum_density = density.max()

    density_threshold_tensor = torch.maximum(
        torch.as_tensor(
            epsilon,
            dtype=density.dtype,
            device=density.device,
        ),
        maximum_density
        * float(
            relative_density_threshold
        ),
    )

    valid_mask = (
        torch.isfinite(density)
        & torch.isfinite(
            attribute_numerator
        )
        & (
            density
            > density_threshold_tensor
        )
    )

    safe_density = torch.clamp(
        density,
        min=float(epsilon),
    )

    normalized_attribute = (
        attribute_numerator
        / safe_density
    )

    attribute = torch.full_like(
        density,
        float("nan"),
    )

    attribute = torch.where(
        valid_mask,
        normalized_attribute,
        attribute,
    )

    density = density.contiguous()
    attribute_numerator = (
        attribute_numerator.contiguous()
    )
    attribute = attribute.contiguous()
    valid_mask = valid_mask.contiguous()

    normalization_end.record()
    normalization_end.synchronize()

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

    normalization_seconds = (
        normalization_start.elapsed_time(
            normalization_end
        )
        / 1000.0
    )

    total_seconds = (
        preparation_seconds
        + rasterization_seconds
        + normalization_seconds
    )

    expected_image_shape = (
        intersections.image_height,
        intersections.image_width,
    )

    if tuple(density.shape) != expected_image_shape:
        raise RuntimeError(
            "Unexpected density shape: "
            f"{tuple(density.shape)}."
        )

    if tuple(attribute.shape) != expected_image_shape:
        raise RuntimeError(
            "Unexpected attribute shape: "
            f"{tuple(attribute.shape)}."
        )

    if not torch.isfinite(
        density
    ).all():
        raise RuntimeError(
            "Rendered density contains NaN or infinite values."
        )

    valid_attribute_values = attribute[
        valid_mask
    ]

    if (
        valid_attribute_values.numel() > 0
        and not torch.isfinite(
            valid_attribute_values
        ).all()
    ):
        raise RuntimeError(
            "Valid rendered attributes contain NaN or infinite values."
        )

    density_threshold = float(
        density_threshold_tensor.item()
    )

    return GpuConditionalRenderResult(
        density=density,
        attribute_numerator=(
            attribute_numerator
        ),
        attribute=attribute,
        valid_mask=valid_mask,
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
        normalization_seconds=float(
            normalization_seconds
        ),
        total_seconds=float(
            total_seconds
        ),
        density_threshold=(
            density_threshold
        ),
    )