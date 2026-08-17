from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from gsplat.cuda._wrapper import (
    rasterize_to_pixels_scientific_conditional,
)

from scientific_gsplat_renderer.projection import (
    ProjectedConditionalAttributes,
    ProjectedGaussians,
)
from scientific_gsplat_renderer.rasterization import (
    inverse_covariances_to_conics,
    mass_to_normalized_amplitude,
    scientific_rasterize_to_pixels,
)
from scientific_gsplat_renderer.tile import (
    build_gsplat_intersections,
)


@dataclass(frozen=True, slots=True)
class ScientificCudaRenderResult:
    """Result produced by the scientific CUDA renderer.

    Attributes
    ----------
    density
        Additively accumulated density field with shape ``[H, W]``.

    attribute
        Density-weighted intensive attribute field with shape ``[H, W]``.
        This is ``None`` when no attribute input was supplied.

    attribute_numerator
        Additive attribute numerator with shape ``[H, W]``:

        ``sum_i attribute_i(pixel) * weight_i(pixel)``

        This is ``None`` when no attribute input was supplied.

    accumulated_weights
        Additive Gaussian weights with shape ``[H, W]``. This is currently
        the same tensor as ``density``.

    valid_mask
        Boolean image with shape ``[H, W]``. A pixel is considered valid
        when its density exceeds the selected absolute/relative threshold.

    density_threshold
        Scalar threshold used to create ``valid_mask``.
    """

    density: Tensor
    attribute: Tensor | None
    attribute_numerator: Tensor | None
    accumulated_weights: Tensor
    valid_mask: Tensor
    density_threshold: float

    n_input_gaussians: int
    n_rendered_gaussians: int
    n_intersections: int

    preparation_seconds: float
    intersection_seconds: float
    rasterization_seconds: float
    total_seconds: float

    @property
    def device(self) -> torch.device:
        """Return the device containing the rendered fields."""
        return self.density.device

    def density_numpy(self) -> np.ndarray:
        """Copy the density image to a NumPy array."""
        return self.density.detach().cpu().numpy()

    def attribute_numpy(self) -> np.ndarray | None:
        """Copy the normalized attribute image to NumPy."""
        if self.attribute is None:
            return None

        return self.attribute.detach().cpu().numpy()

    def attribute_numerator_numpy(self) -> np.ndarray | None:
        """Copy the attribute numerator image to NumPy."""
        if self.attribute_numerator is None:
            return None

        return self.attribute_numerator.detach().cpu().numpy()

    def valid_mask_numpy(self) -> np.ndarray:
        """Copy the valid-pixel mask to NumPy."""
        return self.valid_mask.detach().cpu().numpy()


def _as_cuda_float32(
    array: np.ndarray | Tensor,
    *,
    device: torch.device,
    name: str,
) -> Tensor:
    """Convert an array to a contiguous CUDA float32 tensor."""
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    elif isinstance(array, Tensor):
        tensor = array
    else:
        raise TypeError(
            f"{name} must be a NumPy array or torch.Tensor, "
            f"got {type(array)!r}"
        )

    return tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).contiguous()


def _as_cuda_bool(
    array: np.ndarray | Tensor,
    *,
    device: torch.device,
    name: str,
) -> Tensor:
    """Convert an array to a contiguous CUDA Boolean tensor."""
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    elif isinstance(array, Tensor):
        tensor = array
    else:
        raise TypeError(
            f"{name} must be a NumPy array or torch.Tensor, "
            f"got {type(array)!r}"
        )

    return tensor.to(
        device=device,
        dtype=torch.bool,
        non_blocking=True,
    ).contiguous()


def _validate_projected_gaussians(
    projected: ProjectedGaussians,
) -> None:
    """Validate the shapes stored in ProjectedGaussians."""
    n_gaussians = projected.n_gaussians

    expected_shapes = {
        "means_camera": (n_gaussians, 3),
        "means_pixel": (n_gaussians, 2),
        "covariances_camera": (n_gaussians, 3, 3),
        "covariances_pixel": (n_gaussians, 2, 2),
        "inverse_covariances_pixel": (n_gaussians, 2, 2),
        "radii_pixel": (n_gaussians,),
        "depths": (n_gaussians,),
        "masses": (n_gaussians,),
        "valid": (n_gaussians,),
    }

    for name, expected_shape in expected_shapes.items():
        value = getattr(projected, name)
        actual_shape = tuple(value.shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"projected.{name} must have shape {expected_shape}, "
                f"got {actual_shape}"
            )


def _validate_conditional_attributes(
    conditional: ProjectedConditionalAttributes,
    *,
    n_gaussians: int,
) -> None:
    """Validate projected conditional attribute arrays."""
    if conditional.n_gaussians != n_gaussians:
        raise ValueError(
            "conditional must contain one entry for every projected "
            f"Gaussian: expected {n_gaussians}, "
            f"got {conditional.n_gaussians}"
        )

    if tuple(conditional.means.shape) != (n_gaussians,):
        raise ValueError(
            "conditional.means must have shape "
            f"({n_gaussians},), got "
            f"{tuple(conditional.means.shape)}"
        )

    if tuple(conditional.slopes_pixel.shape) != (
        n_gaussians,
        2,
    ):
        raise ValueError(
            "conditional.slopes_pixel must have shape "
            f"({n_gaussians}, 2), got "
            f"{tuple(conditional.slopes_pixel.shape)}"
        )

    if tuple(conditional.valid.shape) != (n_gaussians,):
        raise ValueError(
            "conditional.valid must have shape "
            f"({n_gaussians},), got "
            f"{tuple(conditional.valid.shape)}"
        )


def _filter_projected_gaussians(
    projected: ProjectedGaussians,
    valid_indices: np.ndarray,
) -> ProjectedGaussians:
    """Create a compact projection containing selected Gaussians.

    The tile builder and rasterizer must receive Gaussians in exactly the
    same order because ``flatten_ids`` contains indices into the compact
    rasterization tensors.
    """
    valid_indices = np.asarray(
        valid_indices,
        dtype=np.int64,
    )

    if valid_indices.ndim != 1:
        raise ValueError(
            "valid_indices must be one-dimensional, "
            f"got shape {valid_indices.shape}"
        )

    n_selected = int(valid_indices.size)

    return replace(
        projected,
        means_camera=np.asarray(
            projected.means_camera[valid_indices],
            dtype=np.float32,
        ),
        means_pixel=np.asarray(
            projected.means_pixel[valid_indices],
            dtype=np.float32,
        ),
        covariances_camera=np.asarray(
            projected.covariances_camera[valid_indices],
            dtype=np.float32,
        ),
        covariances_pixel=np.asarray(
            projected.covariances_pixel[valid_indices],
            dtype=np.float32,
        ),
        inverse_covariances_pixel=np.asarray(
            projected.inverse_covariances_pixel[valid_indices],
            dtype=np.float32,
        ),
        radii_pixel=np.asarray(
            projected.radii_pixel[valid_indices],
            dtype=np.float32,
        ),
        depths=np.asarray(
            projected.depths[valid_indices],
            dtype=np.float32,
        ),
        masses=np.asarray(
            projected.masses[valid_indices],
            dtype=np.float32,
        ),
        valid=np.ones(
            n_selected,
            dtype=np.bool_,
        ),
    )


def _compute_density_threshold(
    density: Tensor,
    *,
    epsilon: float,
    relative_density_threshold: float,
) -> Tensor:
    """Compute the scalar density threshold used for attribute validity.

    The final threshold is

    ``max(epsilon, relative_density_threshold * max(density))``.

    The absolute epsilon protects division from zero. The relative term
    removes pixels whose Gaussian support is negligible relative to the
    strongest pixel in the image.
    """
    absolute_threshold = torch.as_tensor(
        epsilon,
        dtype=density.dtype,
        device=density.device,
    )

    maximum_density = torch.max(density)

    relative_threshold = (
        maximum_density * relative_density_threshold
    )

    return torch.maximum(
        absolute_threshold,
        relative_threshold,
    )


def render_projected_gaussians_cuda(
    projected: ProjectedGaussians,
    *,
    image_width: int,
    image_height: int,
    conditional: ProjectedConditionalAttributes | None = None,
    attribute_means: np.ndarray | Tensor | None = None,
    tile_size: int = 16,
    epsilon: float = 1.0e-12,
    relative_density_threshold: float = 1.0e-6,
    normalize_gaussian_mass: bool = True,
    device: str | torch.device = "cuda",
) -> ScientificCudaRenderResult:
    """Render projected scientific Gaussians using the CUDA backend.

    The rendering pipeline is:

    1. Validate and filter projected Gaussians.
    2. Convert inverse covariance matrices to compact gsplat conics.
    3. Build Gaussian-to-tile intersections.
    4. Additively rasterize density.
    5. Optionally rasterize an attribute numerator.
    6. Normalize the attribute only in sufficiently supported pixels.

    Parameters
    ----------
    projected
        Orthographically projected Gaussian model.

    image_width
        Output image width in pixels.

    image_height
        Output image height in pixels.

    conditional
        Projected conditional attribute data for every Gaussian.

        Each Gaussian contains the CUDA parameters:

        ``[attribute_mean, slope_x_pixel, slope_y_pixel]``.

        At pixel position ``p``, the custom CUDA kernel evaluates

        ``attribute_mean + slope_x * delta_x + slope_y * delta_y``

        where ``delta = p - mean_pixel``.

        ``conditional`` and ``attribute_means`` are mutually exclusive.

    attribute_means
        Optional legacy mean-only scalar attribute for every original
        Gaussian. Accepted shapes are ``[N]`` and ``[N, 1]``.

        This path does not evaluate spatial conditional slopes. Use
        ``conditional`` for full conditional rendering.

    tile_size
        Width and height of each gsplat tile.

    epsilon
        Absolute minimum density threshold and denominator clamp.

    relative_density_threshold
        Relative validity threshold measured as a fraction of maximum image
        density. The final threshold is

        ``max(epsilon, relative_density_threshold * density.max())``.

    normalize_gaussian_mass
        If true, convert each Gaussian mass into the normalized projected
        Gaussian amplitude. This makes the continuous projected Gaussian
        integrate to its mass.

        If false, raw mass is used as the Gaussian peak multiplier.

    device
        CUDA device such as ``"cuda"`` or ``"cuda:0"``.
    """
    total_start = perf_counter()

    if image_width <= 0:
        raise ValueError(
            f"image_width must be positive, got {image_width}"
        )

    if image_height <= 0:
        raise ValueError(
            f"image_height must be positive, got {image_height}"
        )

    if tile_size <= 0:
        raise ValueError(
            f"tile_size must be positive, got {tile_size}"
        )

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            f"epsilon must be finite and positive, got {epsilon}"
        )

    if (
        not np.isfinite(relative_density_threshold)
        or relative_density_threshold < 0.0
    ):
        raise ValueError(
            "relative_density_threshold must be finite and "
            f"non-negative, got {relative_density_threshold}"
        )

    if (
        conditional is not None
        and attribute_means is not None
    ):
        raise ValueError(
            "Provide either conditional or attribute_means, not both"
        )

    cuda_device = torch.device(device)

    if cuda_device.type != "cuda":
        raise ValueError(
            "Scientific CUDA rendering requires a CUDA device, "
            f"got {cuda_device}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch reports that CUDA is unavailable"
        )

    _validate_projected_gaussians(projected)

    n_input_gaussians = projected.n_gaussians

    if conditional is not None:
        _validate_conditional_attributes(
            conditional,
            n_gaussians=n_input_gaussians,
        )

    preparation_start = perf_counter()

    means_pixel_all = _as_cuda_float32(
        projected.means_pixel,
        device=cuda_device,
        name="projected.means_pixel",
    )

    inverse_covariances_all = _as_cuda_float32(
        projected.inverse_covariances_pixel,
        device=cuda_device,
        name="projected.inverse_covariances_pixel",
    )

    radii_all = _as_cuda_float32(
        projected.radii_pixel,
        device=cuda_device,
        name="projected.radii_pixel",
    )

    depths_all = _as_cuda_float32(
        projected.depths,
        device=cuda_device,
        name="projected.depths",
    )

    masses_all = _as_cuda_float32(
        projected.masses,
        device=cuda_device,
        name="projected.masses",
    )

    valid = _as_cuda_bool(
        projected.valid,
        device=cuda_device,
        name="projected.valid",
    ).clone()

    valid &= torch.isfinite(
        means_pixel_all
    ).all(dim=-1)

    valid &= torch.isfinite(
        inverse_covariances_all
    ).all(dim=(-2, -1))

    valid &= torch.isfinite(radii_all)
    valid &= radii_all > 0.0

    valid &= torch.isfinite(depths_all)

    valid &= torch.isfinite(masses_all)
    valid &= masses_all >= 0.0

    conditional_parameters_all: Tensor | None = None

    if conditional is not None:
        conditional_valid = _as_cuda_bool(
            conditional.valid,
            device=cuda_device,
            name="conditional.valid",
        )

        conditional_parameters_all = _as_cuda_float32(
            conditional.conditional_parameters,
            device=cuda_device,
            name="conditional.conditional_parameters",
        )

        expected_shape = (
            n_input_gaussians,
            3,
        )

        if (
            tuple(conditional_parameters_all.shape)
            != expected_shape
        ):
            raise ValueError(
                "conditional parameters must have shape "
                f"{expected_shape}, got "
                f"{tuple(conditional_parameters_all.shape)}"
            )

        valid &= conditional_valid

        valid &= torch.isfinite(
            conditional_parameters_all
        ).all(dim=-1)

    attributes_all: Tensor | None = None

    if attribute_means is not None:
        attributes_all = _as_cuda_float32(
            attribute_means,
            device=cuda_device,
            name="attribute_means",
        )

        if attributes_all.ndim == 2:
            if attributes_all.shape[1] != 1:
                raise ValueError(
                    "Two-dimensional attribute_means must have shape "
                    f"[N, 1], got {tuple(attributes_all.shape)}"
                )

            attributes_all = attributes_all[:, 0]

        if attributes_all.ndim != 1:
            raise ValueError(
                "attribute_means must have shape [N] or [N, 1], "
                f"got {tuple(attributes_all.shape)}"
            )

        if attributes_all.shape[0] != n_input_gaussians:
            raise ValueError(
                "attribute_means must contain one value for every input "
                f"Gaussian: expected {n_input_gaussians}, "
                f"got {attributes_all.shape[0]}"
            )

        valid &= torch.isfinite(attributes_all)

    valid_indices = torch.nonzero(
        valid,
        as_tuple=False,
    ).squeeze(-1)

    if valid_indices.numel() == 0:
        raise ValueError(
            "No valid projected Gaussians remain for rendering"
        )

    valid_indices_cpu = (
        valid_indices
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    filtered_projected = _filter_projected_gaussians(
        projected,
        valid_indices_cpu,
    )

    means2d = means_pixel_all[valid_indices]
    inverse_covariances = inverse_covariances_all[valid_indices]
    radii = radii_all[valid_indices]
    depths = depths_all[valid_indices]
    masses = masses_all[valid_indices]

    conics = inverse_covariances_to_conics(
        inverse_covariances
    )

    if normalize_gaussian_mass:
        amplitudes = mass_to_normalized_amplitude(
            masses,
            conics,
        )
    else:
        amplitudes = masses

    means2d = (
        means2d
        .unsqueeze(0)
        .contiguous()
    )

    conics = (
        conics
        .unsqueeze(0)
        .contiguous()
    )

    radii = (
        radii
        .unsqueeze(0)
        .contiguous()
    )

    depths = (
        depths
        .unsqueeze(0)
        .contiguous()
    )

    amplitudes = (
        amplitudes
        .unsqueeze(0)
        .contiguous()
    )

    filtered_conditional_parameters: Tensor | None = None

    if conditional_parameters_all is not None:
        filtered_conditional_parameters = (
            conditional_parameters_all[valid_indices]
            .unsqueeze(0)
            .contiguous()
        )

    filtered_attributes: Tensor | None = None

    if attributes_all is not None:
        filtered_attributes = (
            attributes_all[valid_indices]
            .unsqueeze(0)
            .unsqueeze(-1)
            .contiguous()
        )

    torch.cuda.synchronize(cuda_device)

    preparation_seconds = (
        perf_counter() - preparation_start
    )

    intersection_start = perf_counter()

    intersections = build_gsplat_intersections(
        filtered_projected,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        device=cuda_device,
    )

    torch.cuda.synchronize(cuda_device)

    intersection_seconds = (
        perf_counter() - intersection_start
    )

    rasterization_start = perf_counter()

    n_rendered_gaussians = int(
        valid_indices.numel()
    )

    attribute: Tensor | None = None
    attribute_numerator: Tensor | None = None

    if filtered_conditional_parameters is not None:
        (
            conditional_render,
            conditional_weights,
            _last_ids,
        ) = rasterize_to_pixels_scientific_conditional(
            means2d=means2d,
            conics=conics,
            conditional_params=(
                filtered_conditional_parameters
            ),
            opacities=amplitudes,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            isect_offsets=intersections.isect_offsets,
            flatten_ids=intersections.flatten_ids,
            backgrounds=None,
            masks=None,
        )

        attribute_numerator = (
            conditional_render[0, ..., 0]
            .contiguous()
        )

        density = (
            conditional_weights[0, ..., 0]
            .contiguous()
        )

    else:
        density_values = torch.ones(
            (
                1,
                n_rendered_gaussians,
                1,
            ),
            dtype=torch.float32,
            device=cuda_device,
        )

        density_result = scientific_rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            values=density_values,
            amplitudes=amplitudes,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            tile_offsets=intersections.isect_offsets,
            flatten_ids=intersections.flatten_ids,
        )

        density = (
            density_result
            .accumulated_weights[0, ..., 0]
            .contiguous()
        )

    accumulated_weights = density

    density_threshold_tensor = _compute_density_threshold(
        density,
        epsilon=epsilon,
        relative_density_threshold=(
            relative_density_threshold
        ),
    )

    valid_mask = (
        density > density_threshold_tensor
    )

    if attribute_numerator is not None:
        safe_density = density.clamp_min(
            epsilon
        )

        normalized_attribute = (
            attribute_numerator / safe_density
        )

        attribute = torch.where(
            valid_mask,
            normalized_attribute,
            torch.zeros_like(
                normalized_attribute
            ),
        )

    elif filtered_attributes is not None:
        attribute_result = scientific_rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            values=filtered_attributes,
            amplitudes=amplitudes,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            tile_offsets=intersections.isect_offsets,
            flatten_ids=intersections.flatten_ids,
        )

        attribute_numerator = (
            attribute_result
            .accumulated_values[0, ..., 0]
            .contiguous()
        )

        safe_density = density.clamp_min(
            epsilon
        )

        normalized_attribute = (
            attribute_numerator / safe_density
        )

        attribute = torch.where(
            valid_mask,
            normalized_attribute,
            torch.zeros_like(
                normalized_attribute
            ),
        )

    torch.cuda.synchronize(cuda_device)

    rasterization_seconds = (
        perf_counter() - rasterization_start
    )

    total_seconds = (
        perf_counter() - total_start
    )

    density_threshold = float(
        density_threshold_tensor
        .detach()
        .cpu()
        .item()
    )

    return ScientificCudaRenderResult(
        density=density,
        attribute=attribute,
        attribute_numerator=attribute_numerator,
        accumulated_weights=accumulated_weights,
        valid_mask=valid_mask,
        density_threshold=density_threshold,
        n_input_gaussians=n_input_gaussians,
        n_rendered_gaussians=n_rendered_gaussians,
        n_intersections=(
            intersections.n_intersections
        ),
        preparation_seconds=preparation_seconds,
        intersection_seconds=intersection_seconds,
        rasterization_seconds=rasterization_seconds,
        total_seconds=total_seconds,
    )