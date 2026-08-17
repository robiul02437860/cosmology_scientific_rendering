from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.gpu import (
    GpuConditionalRenderResult,
    GpuGaussianModel,
    GpuProjectedGaussians,
    GpuTileIntersections,
    build_gpu_tile_intersections,
    project_conditional_attributes_orthographic_gpu,
    project_gaussians_orthographic_gpu,
    render_conditional_attribute_gpu,
)


@dataclass(slots=True)
class InteractiveFrame:
    """One GPU-rendered scientific frame.

    The scientific output fields remain resident on the GPU so that
    display-only changes such as colormap, transfer-function range,
    scale mode, and opacity can reuse the same frame without repeating
    Gaussian projection, tile construction, or CUDA rasterization.
    """

    density: Tensor
    attribute: Tensor
    valid_mask: Tensor

    projected: GpuProjectedGaussians
    intersections: GpuTileIntersections
    render_result: GpuConditionalRenderResult

    projection_ms: float
    conditional_projection_ms: float
    intersections_ms: float
    rendering_ms: float
    total_ms: float

    @property
    def fps(self) -> float:
        """Equivalent renderer FPS for the scientific frame."""
        if self.total_ms <= 0.0:
            return float("inf")
        return 1000.0 / self.total_ms

    @property
    def width(self) -> int:
        """Rendered image width."""
        return int(self.density.shape[1])

    @property
    def height(self) -> int:
        """Rendered image height."""
        return int(self.density.shape[0])

    @property
    def image_size(self) -> tuple[int, int]:
        """Rendered image size as (width, height)."""
        return (
            self.width,
            self.height,
        )

    @property
    def device(self) -> torch.device:
        """CUDA device containing this frame."""
        return self.density.device


class InteractiveScientificRenderer:
    """Reusable GPU-resident scientific Gaussian renderer.

    Full scientific rendering is required when camera parameters or
    Gaussian-render parameters change.

    The most recently completed ``InteractiveFrame`` is cached so that
    display-only operations can reuse the GPU-resident density and
    conditional-attribute fields.
    """

    def __init__(
        self,
        gpu_model: GpuGaussianModel,
        *,
        image_width: int = 1024,
        image_height: int = 1024,
        tile_size: int = 16,
        minimum_eigenvalue: float = 1.0e-6,
        minimum_pixel_variance: float = 0.25,
        sigma_extent: float = 3.0,
        beta: float = 0.5,
        blob_sigma_pixels: float = 2.0,
        relative_density_threshold: float = 1.0e-6,
        normalize_gaussian_mass: bool = True,
    ) -> None:
        self.gpu_model = gpu_model

        self._image_width = int(image_width)
        self._image_height = int(image_height)

        self.tile_size = int(tile_size)

        self.minimum_eigenvalue = float(
            minimum_eigenvalue
        )

        self.minimum_pixel_variance = float(
            minimum_pixel_variance
        )

        self.sigma_extent = float(
            sigma_extent
        )

        self.beta = float(
            beta
        )

        self.blob_sigma_pixels = float(
            blob_sigma_pixels
        )

        self.relative_density_threshold = float(
            relative_density_threshold
        )

        self.normalize_gaussian_mass = bool(
            normalize_gaussian_mass
        )

        self._validate_configuration()

        self._last_frame: InteractiveFrame | None = None

        self._projection_start = self._make_event()
        self._projection_end = self._make_event()

        self._conditional_start = self._make_event()
        self._conditional_end = self._make_event()

        self._intersections_start = self._make_event()
        self._intersections_end = self._make_event()

        self._rendering_start = self._make_event()
        self._rendering_end = self._make_event()

        self._frame_start = self._make_event()
        self._frame_end = self._make_event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.gpu_model.device

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def image_size(self) -> tuple[int, int]:
        return (
            self._image_width,
            self._image_height,
        )

    @property
    def last_frame(self) -> InteractiveFrame | None:
        """Most recently completed scientific frame."""
        return self._last_frame

    @property
    def has_cached_frame(self) -> bool:
        return self._last_frame is not None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def require_last_frame(self) -> InteractiveFrame:
        """Return cached frame or raise if no frame has been rendered."""
        if self._last_frame is None:
            raise RuntimeError(
                "No interactive scientific frame has been rendered yet."
            )

        return self._last_frame

    def clear_cached_frame(self) -> None:
        """Drop the cached frame reference."""
        self._last_frame = None

    # ------------------------------------------------------------------
    # CUDA timing
    # ------------------------------------------------------------------

    @staticmethod
    def _make_event() -> torch.cuda.Event:
        return torch.cuda.Event(
            enable_timing=True,
        )

    @staticmethod
    def _elapsed_ms(
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> float:
        return float(
            start.elapsed_time(end)
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if self.device.type != "cuda":
            raise ValueError(
                "Interactive rendering requires a CUDA-resident model."
            )

        if self._image_width <= 0:
            raise ValueError(
                "image_width must be positive."
            )

        if self._image_height <= 0:
            raise ValueError(
                "image_height must be positive."
            )

        if self.tile_size <= 0:
            raise ValueError(
                "tile_size must be positive."
            )

        if (
            not math.isfinite(
                self.minimum_eigenvalue
            )
            or self.minimum_eigenvalue <= 0.0
        ):
            raise ValueError(
                "minimum_eigenvalue must be finite and positive."
            )

        if (
            not math.isfinite(
                self.minimum_pixel_variance
            )
            or self.minimum_pixel_variance <= 0.0
        ):
            raise ValueError(
                "minimum_pixel_variance must be finite and positive."
            )

        if (
            not math.isfinite(
                self.sigma_extent
            )
            or self.sigma_extent <= 0.0
        ):
            raise ValueError(
                "sigma_extent must be finite and positive."
            )

        if (
            not math.isfinite(
                self.beta
            )
            or self.beta < 0.0
        ):
            raise ValueError(
                "beta must be finite and nonnegative."
            )

        if (
            not math.isfinite(
                self.blob_sigma_pixels
            )
            or self.blob_sigma_pixels < 0.0
        ):
            raise ValueError(
                "blob_sigma_pixels must be finite and nonnegative."
            )

        if (
            not math.isfinite(
                self.relative_density_threshold
            )
            or self.relative_density_threshold < 0.0
        ):
            raise ValueError(
                "relative_density_threshold must be finite and nonnegative."
            )

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------

    def set_resolution(
        self,
        width: int,
        height: int,
    ) -> None:
        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            raise ValueError(
                "Rendering dimensions must be positive."
            )

        if (
            width == self._image_width
            and height == self._image_height
        ):
            return

        self._image_width = width
        self._image_height = height

        self.clear_cached_frame()

    def set_beta(
        self,
        beta: float,
    ) -> None:
        beta = float(beta)

        if (
            not math.isfinite(beta)
            or beta < 0.0
        ):
            raise ValueError(
                "beta must be finite and nonnegative."
            )

        self.beta = beta

    def set_blob_sigma(
        self,
        blob_sigma_pixels: float,
    ) -> None:
        blob_sigma_pixels = float(
            blob_sigma_pixels
        )

        if (
            not math.isfinite(
                blob_sigma_pixels
            )
            or blob_sigma_pixels < 0.0
        ):
            raise ValueError(
                "blob_sigma_pixels must be finite and nonnegative."
            )

        self.blob_sigma_pixels = (
            blob_sigma_pixels
        )

    def set_sigma_extent(
        self,
        sigma_extent: float,
    ) -> None:
        sigma_extent = float(
            sigma_extent
        )

        if (
            not math.isfinite(
                sigma_extent
            )
            or sigma_extent <= 0.0
        ):
            raise ValueError(
                "sigma_extent must be finite and positive."
            )

        self.sigma_extent = sigma_extent

    def set_relative_density_threshold(
        self,
        relative_density_threshold: float,
    ) -> None:
        relative_density_threshold = float(
            relative_density_threshold
        )

        if (
            not math.isfinite(
                relative_density_threshold
            )
            or relative_density_threshold < 0.0
        ):
            raise ValueError(
                "relative_density_threshold must be finite and nonnegative."
            )

        self.relative_density_threshold = (
            relative_density_threshold
        )

    # ------------------------------------------------------------------
    # Scientific rendering
    # ------------------------------------------------------------------

    def render(
        self,
        camera: OrthographicCamera,
    ) -> InteractiveFrame:
        """Render one complete density + conditional-attribute frame."""

        if (
            camera.image_width
            != self._image_width
            or camera.image_height
            != self._image_height
        ):
            raise ValueError(
                "Camera image size does not match renderer size: "
                f"camera={camera.image_size}, "
                f"renderer={self.image_size}."
            )

        self._frame_start.record()

        # --------------------------------------------------------------
        # 1. Gaussian projection
        # --------------------------------------------------------------

        self._projection_start.record()

        projected = (
            project_gaussians_orthographic_gpu(
                self.gpu_model,
                camera,
                minimum_eigenvalue=(
                    self.minimum_eigenvalue
                ),
                minimum_pixel_variance=(
                    self.minimum_pixel_variance
                ),
                sigma_extent=(
                    self.sigma_extent
                ),
                beta=(
                    self.beta
                ),
                blob_sigma_pixels=(
                    self.blob_sigma_pixels
                ),
            )
        )

        self._projection_end.record()

        # --------------------------------------------------------------
        # 2. Conditional attribute projection
        # --------------------------------------------------------------

        self._conditional_start.record()

        conditional = (
            project_conditional_attributes_orthographic_gpu(
                self.gpu_model,
                projected,
                camera,
            )
        )

        self._conditional_end.record()

        # --------------------------------------------------------------
        # 3. Screen-space tile intersections
        # --------------------------------------------------------------

        self._intersections_start.record()

        intersections = (
            build_gpu_tile_intersections(
                projected,
                image_width=(
                    self._image_width
                ),
                image_height=(
                    self._image_height
                ),
                tile_size=(
                    self.tile_size
                ),
            )
        )

        self._intersections_end.record()

        # --------------------------------------------------------------
        # 4. Scientific CUDA rasterization
        # --------------------------------------------------------------

        self._rendering_start.record()

        render_result = (
            render_conditional_attribute_gpu(
                projected,
                conditional,
                intersections,
                normalize_gaussian_mass=(
                    self.normalize_gaussian_mass
                ),
                relative_density_threshold=(
                    self.relative_density_threshold
                ),
            )
        )

        self._rendering_end.record()

        self._frame_end.record()
        self._frame_end.synchronize()

        projection_ms = self._elapsed_ms(
            self._projection_start,
            self._projection_end,
        )

        conditional_projection_ms = (
            self._elapsed_ms(
                self._conditional_start,
                self._conditional_end,
            )
        )

        intersections_ms = self._elapsed_ms(
            self._intersections_start,
            self._intersections_end,
        )

        rendering_ms = self._elapsed_ms(
            self._rendering_start,
            self._rendering_end,
        )

        total_ms = self._elapsed_ms(
            self._frame_start,
            self._frame_end,
        )

        frame = InteractiveFrame(
            density=(
                render_result.density
            ),
            attribute=(
                render_result.attribute
            ),
            valid_mask=(
                render_result.valid_mask
            ),
            projected=(
                projected
            ),
            intersections=(
                intersections
            ),
            render_result=(
                render_result
            ),
            projection_ms=(
                projection_ms
            ),
            conditional_projection_ms=(
                conditional_projection_ms
            ),
            intersections_ms=(
                intersections_ms
            ),
            rendering_ms=(
                rendering_ms
            ),
            total_ms=(
                total_ms
            ),
        )

        self._last_frame = frame

        return frame

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warm_up(
        self,
        camera: OrthographicCamera,
        *,
        frames: int = 5,
    ) -> InteractiveFrame:
        """Warm up CUDA kernels and return the final rendered frame."""

        frames = int(frames)

        if frames <= 0:
            raise ValueError(
                "frames must be positive."
            )

        final_frame: InteractiveFrame | None = None

        for _ in range(frames):
            final_frame = self.render(
                camera
            )

        assert final_frame is not None

        return final_frame