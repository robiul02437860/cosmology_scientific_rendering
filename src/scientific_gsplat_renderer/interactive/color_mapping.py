from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor


ColormapName = Literal[
    "grayscale",
    "inferno",
    "magma",
    "viridis",
    "turbo",
]

ScaleMode = Literal[
    "linear",
    "log",
]


@dataclass(frozen=True, slots=True)
class ScalarDisplayRange:
    """Scalar range used by the GPU transfer function."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum):
            raise ValueError(
                "Display-range minimum must be finite."
            )

        if not math.isfinite(self.maximum):
            raise ValueError(
                "Display-range maximum must be finite."
            )

        if self.maximum <= self.minimum:
            raise ValueError(
                "Display-range maximum must be greater than minimum: "
                f"[{self.minimum}, {self.maximum}]."
            )

    @property
    def width(self) -> float:
        return self.maximum - self.minimum


@dataclass(slots=True)
class GpuColorMapper:
    """GPU-resident scalar-field color mapper."""

    colormap: ColormapName = "viridis"

    display_range: ScalarDisplayRange = ScalarDisplayRange(
        minimum=0.0,
        maximum=1.0,
    )

    invalid_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    )

    def __post_init__(self) -> None:
        _validate_colormap_name(
            self.colormap
        )

        _validate_rgb(
            self.invalid_rgb,
            name="invalid_rgb",
        )

    def set_colormap(
        self,
        colormap: ColormapName,
    ) -> None:
        _validate_colormap_name(
            colormap
        )

        self.colormap = colormap

    def set_display_range(
        self,
        minimum: float,
        maximum: float,
    ) -> None:
        self.display_range = (
            ScalarDisplayRange(
                minimum=float(minimum),
                maximum=float(maximum),
            )
        )

    def map_linear(
        self,
        scalar: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return scalar_to_rgb_gpu(
            scalar,
            minimum=self.display_range.minimum,
            maximum=self.display_range.maximum,
            scale="linear",
            colormap=self.colormap,
            valid_mask=valid_mask,
            invalid_rgb=self.invalid_rgb,
        )

    def map_log_density(
        self,
        density: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return scalar_to_rgb_gpu(
            density,
            minimum=self.display_range.minimum,
            maximum=self.display_range.maximum,
            scale="log",
            colormap=self.colormap,
            valid_mask=valid_mask,
            invalid_rgb=self.invalid_rgb,
        )


# ============================================================================
# Normalization
# ============================================================================


def normalize_linear_gpu(
    scalar: Tensor,
    *,
    minimum: float,
    maximum: float,
) -> Tensor:
    """Normalize a CUDA scalar tensor to [0,1]."""

    _validate_scalar_tensor(
        scalar
    )

    minimum = float(
        minimum
    )

    maximum = float(
        maximum
    )

    if not math.isfinite(minimum):
        raise ValueError(
            "minimum must be finite."
        )

    if not math.isfinite(maximum):
        raise ValueError(
            "maximum must be finite."
        )

    if maximum <= minimum:
        raise ValueError(
            "maximum must exceed minimum."
        )

    normalized = (
        scalar - minimum
    ) / (
        maximum - minimum
    )

    normalized = torch.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return torch.clamp(
        normalized,
        min=0.0,
        max=1.0,
    )


def normalize_log_density_gpu(
    density: Tensor,
    *,
    minimum: float,
    maximum: float,
) -> Tensor:
    """Normalize a nonnegative scalar field after log(1+x).

    minimum and maximum remain in the original scientific units.
    """

    _validate_scalar_tensor(
        density
    )

    minimum = float(
        minimum
    )

    maximum = float(
        maximum
    )

    if not math.isfinite(minimum):
        raise ValueError(
            "minimum must be finite."
        )

    if not math.isfinite(maximum):
        raise ValueError(
            "maximum must be finite."
        )

    if minimum < 0.0:
        raise ValueError(
            "Logarithmic display minimum must be nonnegative."
        )

    if maximum <= minimum:
        raise ValueError(
            "Logarithmic display maximum must exceed minimum."
        )

    safe = torch.clamp(
        density,
        min=0.0,
    )

    log_scalar = torch.log1p(
        safe
    )

    log_minimum = math.log1p(
        minimum
    )

    log_maximum = math.log1p(
        maximum
    )

    normalized = (
        log_scalar - log_minimum
    ) / (
        log_maximum - log_minimum
    )

    normalized = torch.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return torch.clamp(
        normalized,
        min=0.0,
        max=1.0,
    )


def normalize_scalar_gpu(
    scalar: Tensor,
    *,
    minimum: float,
    maximum: float,
    scale: ScaleMode,
) -> Tensor:
    """Normalize a field using linear or logarithmic scaling."""

    if scale == "linear":
        return normalize_linear_gpu(
            scalar,
            minimum=minimum,
            maximum=maximum,
        )

    if scale == "log":
        return normalize_log_density_gpu(
            scalar,
            minimum=minimum,
            maximum=maximum,
        )

    raise ValueError(
        f"Unsupported scale: {scale!r}"
    )


# ============================================================================
# Colormapping
# ============================================================================


def apply_colormap_gpu(
    normalized: Tensor,
    *,
    colormap: ColormapName,
    valid_mask: Tensor | None = None,
    invalid_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
) -> Tensor:
    """Map normalized values to an RGB uint8 CUDA image."""

    _validate_scalar_tensor(
        normalized
    )

    _validate_colormap_name(
        colormap
    )

    _validate_rgb(
        invalid_rgb,
        name="invalid_rgb",
    )

    valid = _prepare_valid_mask(
        normalized,
        valid_mask,
    )

    x = torch.clamp(
        torch.nan_to_num(
            normalized,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        min=0.0,
        max=1.0,
    )

    if colormap == "grayscale":
        rgb = _grayscale_colormap(
            x
        )

    elif colormap == "viridis":
        rgb = _viridis_colormap(
            x
        )

    elif colormap == "inferno":
        rgb = _inferno_colormap(
            x
        )

    elif colormap == "magma":
        rgb = _magma_colormap(
            x
        )

    elif colormap == "turbo":
        rgb = _turbo_colormap(
            x
        )

    else:
        raise RuntimeError(
            f"Unsupported colormap: {colormap}"
        )

    rgb = torch.clamp(
        rgb,
        min=0.0,
        max=1.0,
    )

    invalid_color = torch.tensor(
        invalid_rgb,
        dtype=rgb.dtype,
        device=rgb.device,
    ) / 255.0

    rgb = torch.where(
        valid.unsqueeze(-1),
        rgb,
        invalid_color.view(
            1,
            1,
            3,
        ),
    )

    return (
        torch.round(
            rgb * 255.0
        )
        .to(
            dtype=torch.uint8
        )
        .contiguous()
    )


def scalar_to_rgb_gpu(
    scalar: Tensor,
    *,
    minimum: float,
    maximum: float,
    scale: ScaleMode,
    colormap: ColormapName,
    valid_mask: Tensor | None = None,
    invalid_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
) -> Tensor:
    """Map a scientific scalar field to RGB entirely on CUDA."""

    normalized = normalize_scalar_gpu(
        scalar,
        minimum=minimum,
        maximum=maximum,
        scale=scale,
    )

    valid = _prepare_valid_mask(
        scalar,
        valid_mask,
    )

    if scale == "log":
        valid = (
            valid
            & (scalar >= 0.0)
        )

    return apply_colormap_gpu(
        normalized,
        colormap=colormap,
        valid_mask=valid,
        invalid_rgb=invalid_rgb,
    )


# ============================================================================
# Opacity transfer function
# ============================================================================


def scalar_to_opacity_gpu(
    scalar: Tensor,
    *,
    minimum: float,
    maximum: float,
    scale: ScaleMode,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Generate an opacity field in [0,1].

    minimum
        Scalar value at which opacity becomes zero.

    maximum
        Scalar value at which opacity becomes one.

    Between the two thresholds opacity changes continuously.
    """

    _validate_scalar_tensor(
        scalar
    )

    minimum = float(
        minimum
    )

    maximum = float(
        maximum
    )

    if not math.isfinite(minimum):
        raise ValueError(
            "Opacity minimum must be finite."
        )

    if not math.isfinite(maximum):
        raise ValueError(
            "Opacity maximum must be finite."
        )

    if maximum <= minimum:
        raise ValueError(
            "Opacity maximum must exceed opacity minimum."
        )

    opacity = normalize_scalar_gpu(
        scalar,
        minimum=minimum,
        maximum=maximum,
        scale=scale,
    )

    valid = _prepare_valid_mask(
        scalar,
        valid_mask,
    )

    if scale == "log":
        valid = (
            valid
            & (scalar >= 0.0)
        )

    opacity = torch.where(
        valid,
        opacity,
        torch.zeros_like(
            opacity
        ),
    )

    return torch.clamp(
        opacity,
        min=0.0,
        max=1.0,
    )


def composite_rgb_with_opacity_gpu(
    rgb: Tensor,
    opacity: Tensor,
    *,
    background_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
) -> Tensor:
    """Composite CUDA RGB against a constant background using opacity."""

    if not isinstance(
        rgb,
        Tensor,
    ):
        raise TypeError(
            "rgb must be a torch.Tensor."
        )

    if rgb.device.type != "cuda":
        raise ValueError(
            "rgb must be CUDA-resident."
        )

    if rgb.dtype != torch.uint8:
        raise ValueError(
            "rgb must have dtype torch.uint8."
        )

    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
    ):
        raise ValueError(
            "rgb must have shape (H,W,3)."
        )

    if not isinstance(
        opacity,
        Tensor,
    ):
        raise TypeError(
            "opacity must be a torch.Tensor."
        )

    if opacity.device != rgb.device:
        raise ValueError(
            "opacity and rgb must be on the same CUDA device."
        )

    if opacity.shape != rgb.shape[:2]:
        raise ValueError(
            "opacity must have shape (H,W)."
        )

    if not opacity.dtype.is_floating_point:
        raise ValueError(
            "opacity must be floating point."
        )

    _validate_rgb(
        background_rgb,
        name="background_rgb",
    )

    foreground = (
        rgb.to(
            dtype=torch.float32
        )
        / 255.0
    )

    alpha = torch.clamp(
        opacity.to(
            dtype=torch.float32
        ),
        min=0.0,
        max=1.0,
    ).unsqueeze(
        dim=-1
    )

    background = torch.tensor(
        background_rgb,
        dtype=torch.float32,
        device=rgb.device,
    ).view(
        1,
        1,
        3,
    ) / 255.0

    composited = (
        alpha * foreground
        + (
            1.0 - alpha
        ) * background
    )

    return (
        torch.round(
            torch.clamp(
                composited,
                min=0.0,
                max=1.0,
            )
            * 255.0
        )
        .to(
            dtype=torch.uint8
        )
        .contiguous()
    )


def scalar_to_rgb_with_opacity_gpu(
    scalar: Tensor,
    *,
    color_minimum: float,
    color_maximum: float,
    opacity_minimum: float,
    opacity_maximum: float,
    color_scale: ScaleMode,
    opacity_scale: ScaleMode,
    colormap: ColormapName,
    valid_mask: Tensor | None = None,
    invalid_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
    background_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
) -> Tensor:
    """Apply independent color and opacity transfer functions on CUDA."""

    rgb = scalar_to_rgb_gpu(
        scalar,
        minimum=color_minimum,
        maximum=color_maximum,
        scale=color_scale,
        colormap=colormap,
        valid_mask=valid_mask,
        invalid_rgb=invalid_rgb,
    )

    opacity = scalar_to_opacity_gpu(
        scalar,
        minimum=opacity_minimum,
        maximum=opacity_maximum,
        scale=opacity_scale,
        valid_mask=valid_mask,
    )

    return composite_rgb_with_opacity_gpu(
        rgb,
        opacity,
        background_rgb=background_rgb,
    )


# ============================================================================
# Compatibility helpers
# ============================================================================


def density_to_rgb_gpu(
    density: Tensor,
    *,
    density_minimum: float,
    density_maximum: float,
    colormap: ColormapName = "inferno",
    valid_mask: Tensor | None = None,
    invalid_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
) -> Tensor:
    """Existing convenience function for log-density visualization."""

    return scalar_to_rgb_gpu(
        density,
        minimum=density_minimum,
        maximum=density_maximum,
        scale="log",
        colormap=colormap,
        valid_mask=valid_mask,
        invalid_rgb=invalid_rgb,
    )


def attribute_to_rgb_gpu(
    attribute: Tensor,
    *,
    attribute_minimum: float,
    attribute_maximum: float,
    valid_mask: Tensor,
    colormap: ColormapName = "viridis",
    invalid_rgb: tuple[int, int, int] = (
        0,
        0,
        0,
    ),
) -> Tensor:
    """Existing convenience function for linear attribute visualization."""

    return scalar_to_rgb_gpu(
        attribute,
        minimum=attribute_minimum,
        maximum=attribute_maximum,
        scale="linear",
        colormap=colormap,
        valid_mask=valid_mask,
        invalid_rgb=invalid_rgb,
    )


def rgb_gpu_to_numpy(
    rgb: Tensor,
) -> object:
    """Copy the final RGB CUDA image to CPU NumPy."""

    if not isinstance(
        rgb,
        Tensor,
    ):
        raise TypeError(
            "rgb must be a torch.Tensor."
        )

    if rgb.device.type != "cuda":
        raise ValueError(
            "rgb_gpu_to_numpy expects a CUDA tensor."
        )

    if rgb.dtype != torch.uint8:
        raise ValueError(
            "RGB tensor must have dtype torch.uint8."
        )

    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
    ):
        raise ValueError(
            "RGB tensor must have shape (H,W,3)."
        )

    return (
        rgb
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================================
# Validation helpers
# ============================================================================


def _prepare_valid_mask(
    scalar: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    finite = torch.isfinite(
        scalar
    )

    if valid_mask is None:
        return finite

    if not isinstance(
        valid_mask,
        Tensor,
    ):
        raise TypeError(
            "valid_mask must be a torch.Tensor."
        )

    if valid_mask.device != scalar.device:
        raise ValueError(
            "valid_mask and scalar must be on the same device."
        )

    if valid_mask.dtype != torch.bool:
        raise ValueError(
            "valid_mask must have dtype torch.bool."
        )

    if valid_mask.shape != scalar.shape:
        raise ValueError(
            "valid_mask and scalar must have the same shape."
        )

    return (
        finite
        & valid_mask
    )


def _validate_scalar_tensor(
    scalar: Tensor,
) -> None:
    if not isinstance(
        scalar,
        Tensor,
    ):
        raise TypeError(
            "scalar must be a torch.Tensor."
        )

    if scalar.device.type != "cuda":
        raise ValueError(
            "GPU color mapping requires a CUDA tensor."
        )

    if scalar.ndim != 2:
        raise ValueError(
            "Scalar field must have shape (H,W), "
            f"got {tuple(scalar.shape)}."
        )

    if not scalar.dtype.is_floating_point:
        raise ValueError(
            "Scalar field must use floating-point dtype."
        )


def _validate_colormap_name(
    colormap: str,
) -> None:
    supported = {
        "grayscale",
        "inferno",
        "magma",
        "viridis",
        "turbo",
    }

    if colormap not in supported:
        raise ValueError(
            f"Unsupported colormap {colormap!r}. "
            f"Supported: {sorted(supported)}."
        )


def _validate_rgb(
    rgb: tuple[int, int, int],
    *,
    name: str,
) -> None:
    if len(rgb) != 3:
        raise ValueError(
            f"{name} must contain exactly 3 values."
        )

    for value in rgb:
        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} values must be integers."
            )

        if (
            value < 0
            or value > 255
        ):
            raise ValueError(
                f"{name} values must be in [0,255]."
            )


# ============================================================================
# Colormaps
# ============================================================================


def _grayscale_colormap(
    x: Tensor,
) -> Tensor:
    return torch.stack(
        (
            x,
            x,
            x,
        ),
        dim=-1,
    )


def _viridis_colormap(
    x: Tensor,
) -> Tensor:
    coefficients = torch.tensor(
        [
            [
                0.2777273272234177,
                0.1050930431085774,
                -0.3308618287255563,
                -4.634230498983486,
                6.228269936347081,
                4.776384997670288,
                -5.435455855934631,
            ],
            [
                0.005407344544966578,
                1.404613529898575,
                0.214847559468213,
                -5.799100973351585,
                14.17993336680509,
                -13.74514537774601,
                4.645852612178535,
            ],
            [
                0.3340998053353061,
                1.384590162594685,
                0.09509516302823659,
                -19.33244095627987,
                56.69055260068105,
                -65.35303263337234,
                26.3124352495832,
            ],
        ],
        dtype=x.dtype,
        device=x.device,
    )

    return _evaluate_polynomial_rgb(
        x,
        coefficients,
    )


def _inferno_colormap(
    x: Tensor,
) -> Tensor:
    red = torch.clamp(
        1.7 * x
        - 0.7 * x * x,
        0.0,
        1.0,
    )

    green = torch.clamp(
        2.4 * x * x
        - 1.4 * x * x * x,
        0.0,
        1.0,
    )

    blue = torch.clamp(
        0.55
        + 1.8 * x
        - 4.0 * x * x
        + 2.1 * x * x * x,
        0.0,
        1.0,
    )

    return torch.stack(
        (
            red,
            green,
            blue,
        ),
        dim=-1,
    )


def _magma_colormap(
    x: Tensor,
) -> Tensor:
    red = torch.clamp(
        1.5 * x
        + 0.4 * x * x,
        0.0,
        1.0,
    )

    green = torch.clamp(
        -0.1
        + 1.8 * x * x,
        0.0,
        1.0,
    )

    blue = torch.clamp(
        0.25
        + 1.3 * x
        - 1.4 * x * x,
        0.0,
        1.0,
    )

    return torch.stack(
        (
            red,
            green,
            blue,
        ),
        dim=-1,
    )


def _turbo_colormap(
    x: Tensor,
) -> Tensor:
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x

    red = (
        0.13572138
        + 4.61539260 * x
        - 42.66032258 * x2
        + 132.13108234 * x3
        - 152.94239396 * x4
        + 59.28637943 * x5
    )

    green = (
        0.09140261
        + 2.19418839 * x
        + 4.84296658 * x2
        - 14.18503333 * x3
        + 4.27729857 * x4
        + 2.82956604 * x5
    )

    blue = (
        0.10667330
        + 12.64194608 * x
        - 60.58204836 * x2
        + 110.36276771 * x3
        - 89.90310912 * x4
        + 27.34824973 * x5
    )

    return torch.stack(
        (
            red,
            green,
            blue,
        ),
        dim=-1,
    )


def _evaluate_polynomial_rgb(
    x: Tensor,
    coefficients: Tensor,
) -> Tensor:
    result = torch.zeros(
        (
            x.shape[0],
            x.shape[1],
            3,
        ),
        dtype=x.dtype,
        device=x.device,
    )

    for index in range(
        coefficients.shape[1] - 1,
        -1,
        -1,
    ):
        result = (
            result
            * x.unsqueeze(-1)
            + coefficients[
                :,
                index,
            ].view(
                1,
                1,
                3,
            )
        )

    return result