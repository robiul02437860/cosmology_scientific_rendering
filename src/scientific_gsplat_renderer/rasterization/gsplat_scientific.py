from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import pi

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ScientificRasterizationResult:
    """Outputs produced by the additive scientific CUDA rasterizer.

    Shapes
    ------
    accumulated_values:
        [I, H, W, C]

    accumulated_weights:
        [I, H, W, 1]

    last_intersection_ids:
        [I, H, W]
    """

    accumulated_values: Tensor
    accumulated_weights: Tensor
    last_intersection_ids: Tensor

    @property
    def density(self) -> Tensor:
        """Return the scalar accumulated-weight image as [I, H, W]."""
        return self.accumulated_weights[..., 0]

    def normalized_values(self, epsilon: float = 1.0e-12) -> Tensor:
        """Return density-weighted intensive values.

        For an attribute channel, the CUDA kernel computes

            numerator = sum_i weight_i * attribute_i

        while accumulated_weights contains

            denominator = sum_i weight_i.

        This method returns numerator / denominator.
        """
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        denominator = self.accumulated_weights

        return torch.where(
            denominator > epsilon,
            self.accumulated_values / denominator,
            torch.zeros_like(self.accumulated_values),
        )


@lru_cache(maxsize=1)
def _scientific_forward():
    """Resolve the custom compiled CUDA function lazily."""
    from gsplat.cuda import _wrapper

    return _wrapper._make_lazy_cuda_func(
        "rasterize_to_pixels_scientific_fwd"
    )


def gaussian_normalization_from_conics(
    conics: Tensor,
    *,
    minimum_determinant: float = 1.0e-20,
) -> Tensor:
    """Compute the normalization of a 2D Gaussian from its inverse covariance.

    The conic is stored as

        [a, b, c]

    corresponding to

        inverse_covariance = [[a, b],
                              [b, c]].

    For a normalized two-dimensional Gaussian,

        G(x) = sqrt(det(inverse_covariance)) / (2*pi)
               * exp(-0.5 * x^T inverse_covariance x).

    Parameters
    ----------
    conics:
        Tensor with final dimension 3.

    minimum_determinant:
        Lower bound applied to the inverse-covariance determinant.

    Returns
    -------
    Tensor
        Normalization factors with shape conics.shape[:-1].
    """
    if conics.ndim < 2 or conics.shape[-1] != 3:
        raise ValueError(
            "conics must have shape [..., 3], "
            f"got {tuple(conics.shape)}"
        )

    if minimum_determinant <= 0.0:
        raise ValueError(
            "minimum_determinant must be positive, "
            f"got {minimum_determinant}"
        )

    a = conics[..., 0]
    b = conics[..., 1]
    c = conics[..., 2]

    determinant_inverse = a * c - b * b
    determinant_inverse = determinant_inverse.clamp_min(
        minimum_determinant
    )

    return torch.sqrt(determinant_inverse) / (2.0 * pi)


def mass_to_normalized_amplitude(
    masses: Tensor,
    conics: Tensor,
    *,
    minimum_determinant: float = 1.0e-20,
) -> Tensor:
    """Convert Gaussian masses into normalized 2D amplitudes.

    The returned value is suitable for the CUDA `opacities` argument:

        amplitude_i =
            mass_i / (2*pi*sqrt(det(covariance_i)))

    which is equivalent to

        mass_i * sqrt(det(inverse_covariance_i)) / (2*pi).
    """
    if masses.shape != conics.shape[:-1]:
        raise ValueError(
            "masses and conics have incompatible shapes: "
            f"{tuple(masses.shape)} and {tuple(conics.shape)}"
        )

    normalization = gaussian_normalization_from_conics(
        conics,
        minimum_determinant=minimum_determinant,
    )

    return masses * normalization


def _require_cuda_float32(name: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if not tensor.is_cuda:
        raise ValueError(
            f"{name} must be on a CUDA device, got {tensor.device}"
        )

    if tensor.dtype != torch.float32:
        raise TypeError(
            f"{name} must have dtype torch.float32, got {tensor.dtype}"
        )

    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_cuda_int32(name: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if not tensor.is_cuda:
        raise ValueError(
            f"{name} must be on a CUDA device, got {tensor.device}"
        )

    if tensor.dtype != torch.int32:
        raise TypeError(
            f"{name} must have dtype torch.int32, got {tensor.dtype}"
        )

    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def scientific_rasterize_to_pixels(
    *,
    means2d: Tensor,
    conics: Tensor,
    values: Tensor,
    amplitudes: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    masks: Tensor | None = None,
) -> ScientificRasterizationResult:
    """Rasterize additive scientific Gaussian fields using CUDA.

    Parameters
    ----------
    means2d:
        Projected Gaussian means with shape [I, N, 2] or [nnz, 2].

    conics:
        Inverse 2D covariances in compact form with shape
        [I, N, 3] or [nnz, 3].

    values:
        Per-Gaussian values with shape [I, N, C] or [nnz, C].

        For density-only rendering, use values equal to one.

        For an intensive attribute, use the Gaussian attribute means.

    amplitudes:
        Per-Gaussian additive coefficients with shape [I, N] or [nnz].

        For mass-preserving rendering, use the result of
        `mass_to_normalized_amplitude(masses, conics)`.

    tile_offsets:
        Encoded tile offsets with shape [I, tile_height, tile_width].

    flatten_ids:
        Flattened Gaussian intersection IDs with shape [n_intersections].

    Returns
    -------
    ScientificRasterizationResult
        Additive values, accumulated weights, and last intersection IDs.
    """
    if image_width <= 0:
        raise ValueError(
            f"image_width must be positive, got {image_width}"
        )

    if image_height <= 0:
        raise ValueError(
            f"image_height must be positive, got {image_height}"
        )

    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    _require_cuda_float32("means2d", means2d)
    _require_cuda_float32("conics", conics)
    _require_cuda_float32("values", values)
    _require_cuda_float32("amplitudes", amplitudes)
    _require_cuda_int32("tile_offsets", tile_offsets)
    _require_cuda_int32("flatten_ids", flatten_ids)

    if means2d.shape[-1] != 2:
        raise ValueError(
            "means2d must have final dimension 2, "
            f"got {tuple(means2d.shape)}"
        )

    if conics.shape[-1] != 3:
        raise ValueError(
            "conics must have final dimension 3, "
            f"got {tuple(conics.shape)}"
        )

    if values.ndim not in (2, 3):
        raise ValueError(
            "values must have shape [nnz, C] or [I, N, C], "
            f"got {tuple(values.shape)}"
        )

    if values.shape[-1] <= 0:
        raise ValueError("values must contain at least one channel")

    packed = means2d.ndim == 2

    if packed:
        if conics.ndim != 2 or amplitudes.ndim != 1:
            raise ValueError(
                "For packed inputs, expected means2d [nnz,2], "
                "conics [nnz,3], amplitudes [nnz], and values [nnz,C]."
            )

        count = means2d.shape[0]

        if conics.shape[0] != count:
            raise ValueError("means2d and conics counts do not match")

        if values.shape[0] != count:
            raise ValueError("means2d and values counts do not match")

        if amplitudes.shape[0] != count:
            raise ValueError("means2d and amplitudes counts do not match")

    else:
        if means2d.ndim != 3:
            raise ValueError(
                "Unpacked means2d must have shape [I, N, 2], "
                f"got {tuple(means2d.shape)}"
            )

        if conics.shape[:-1] != means2d.shape[:-1]:
            raise ValueError(
                "means2d and conics leading dimensions do not match"
            )

        if values.shape[:-1] != means2d.shape[:-1]:
            raise ValueError(
                "means2d and values leading dimensions do not match"
            )

        if amplitudes.shape != means2d.shape[:-1]:
            raise ValueError(
                "amplitudes must have shape [I, N], "
                f"got {tuple(amplitudes.shape)}"
            )

    if tile_offsets.ndim != 3:
        raise ValueError(
            "tile_offsets must have shape [I, tile_height, tile_width], "
            f"got {tuple(tile_offsets.shape)}"
        )

    if flatten_ids.ndim != 1:
        raise ValueError(
            "flatten_ids must be one-dimensional, "
            f"got {tuple(flatten_ids.shape)}"
        )

    device = means2d.device

    tensors = {
        "conics": conics,
        "values": values,
        "amplitudes": amplitudes,
        "tile_offsets": tile_offsets,
        "flatten_ids": flatten_ids,
    }

    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"{name} is on {tensor.device}, but means2d is on {device}"
            )

    if masks is not None:
        if not masks.is_cuda:
            raise ValueError("masks must be on a CUDA device")

        if masks.device != device:
            raise ValueError(
                f"masks is on {masks.device}, expected {device}"
            )

        if masks.dtype != torch.bool:
            raise TypeError(
                f"masks must have dtype torch.bool, got {masks.dtype}"
            )

        if not masks.is_contiguous():
            raise ValueError("masks must be contiguous")

    forward = _scientific_forward()

    accumulated_values, accumulated_weights, last_ids = forward(
        means2d,
        conics,
        values,
        amplitudes,
        None,  # Backgrounds have no scientific additive meaning.
        masks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
    )

    return ScientificRasterizationResult(
        accumulated_values=accumulated_values,
        accumulated_weights=accumulated_weights,
        last_intersection_ids=last_ids,
    )