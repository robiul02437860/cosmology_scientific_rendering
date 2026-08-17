
from __future__ import annotations

import argparse
import math
import threading
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import viser

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu import (
    GpuGaussianModel,
)
from scientific_gsplat_renderer.interactive import (
    InteractiveFrame,
    InteractiveScientificRenderer,
    rgb_gpu_to_numpy,
    scalar_to_rgb_gpu,
    scalar_to_rgb_with_opacity_gpu,
)


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)

# DEFAULT_MODEL = Path(
#     "/home/robiul/Particle_flow/HACC_project/output/illustris3_parallel/full_94m_1pct_parallel/simple_model.npz"
# )


# DEFAULT_MODEL = Path(
#     "/home/robiul/Particle_flow/HACC_project/output/hacc_process_parallel_v2/full_hacc_2pct_components_4/simple_model.npz"
# )

ViewMode = Literal[
    "Density",
    "Attribute",
]

DisplayScale = Literal[
    "Linear",
    "Log",
]

BackgroundMode = Literal[
    "Black",
    "White",
]


# ============================================================================
# State
# ============================================================================


@dataclass(frozen=True, slots=True)
class CameraState:
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    distance: float


@dataclass(frozen=True, slots=True)
class ViewerState:
    camera: CameraState
    mode: ViewMode

    # Scientific rendering.
    beta: float
    blob_sigma_pixels: float
    sigma_extent: float

    # Density color transfer function.
    density_minimum: float
    density_maximum: float
    density_colormap: str
    density_scale: DisplayScale

    # Density opacity transfer function.
    density_opacity_enabled: bool
    density_opacity_scale: DisplayScale
    density_opacity_minimum: float
    density_opacity_maximum: float

    # Attribute color transfer function.
    attribute_minimum: float
    attribute_maximum: float
    attribute_colormap: str
    attribute_scale: DisplayScale

    # Attribute opacity transfer function.
    attribute_opacity_enabled: bool
    attribute_opacity_scale: DisplayScale
    attribute_opacity_minimum: float
    attribute_opacity_maximum: float

    background: BackgroundMode

    jpeg_quality: int
    vertical_flip: bool

    revision: int = 0


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    center: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    span: np.ndarray

    base_view_width: float
    base_camera_distance: float


@dataclass(frozen=True, slots=True)
class FieldStatistics:
    minimum: float
    p1: float
    p5: float
    median: float
    p99: float
    p999: float
    maximum: float


# ============================================================================
# Statistics
# ============================================================================


def valid_scalar_values(
    scalar: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    positive_only: bool = False,
) -> torch.Tensor:

    valid = torch.isfinite(
        scalar
    )

    if valid_mask is not None:

        if valid_mask.device != scalar.device:
            raise ValueError(
                "valid mask must be on same device."
            )

        if valid_mask.shape != scalar.shape:
            raise ValueError(
                "valid mask shape mismatch."
            )

        valid = (
            valid
            & valid_mask
        )

    if positive_only:
        valid = (
            valid
            & (scalar > 0.0)
        )

    values = scalar[
        valid
    ]

    if values.numel() == 0:
        raise RuntimeError(
            "No valid scalar values."
        )

    return values


def compute_field_statistics_gpu(
    scalar: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    positive_only: bool = False,
) -> FieldStatistics:

    values = valid_scalar_values(
        scalar,
        valid_mask=valid_mask,
        positive_only=positive_only,
    )

    q = torch.tensor(
        [
            0.01,
            0.05,
            0.50,
            0.99,
            0.999,
        ],
        dtype=scalar.dtype,
        device=scalar.device,
    )

    quantiles = torch.quantile(
        values,
        q,
    )

    return FieldStatistics(
        minimum=float(
            values.min().item()
        ),
        p1=float(
            quantiles[0].item()
        ),
        p5=float(
            quantiles[1].item()
        ),
        median=float(
            quantiles[2].item()
        ),
        p99=float(
            quantiles[3].item()
        ),
        p999=float(
            quantiles[4].item()
        ),
        maximum=float(
            values.max().item()
        ),
    )


def percentile_range_gpu(
    scalar: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    lower_percentile: float,
    upper_percentile: float,
    positive_only: bool = False,
) -> tuple[float, float]:

    if not (
        0.0
        <= lower_percentile
        < upper_percentile
        <= 100.0
    ):
        raise ValueError(
            "Invalid percentile range."
        )

    values = valid_scalar_values(
        scalar,
        valid_mask=valid_mask,
        positive_only=positive_only,
    )

    q = torch.tensor(
        [
            lower_percentile / 100.0,
            upper_percentile / 100.0,
        ],
        dtype=scalar.dtype,
        device=scalar.device,
    )

    result = torch.quantile(
        values,
        q,
    )

    minimum = float(
        result[0].item()
    )

    maximum = float(
        result[1].item()
    )

    if maximum <= minimum:
        maximum = (
            minimum
            + max(
                abs(minimum) * 1.0e-6,
                1.0e-6,
            )
        )

    return (
        minimum,
        maximum,
    )


# ============================================================================
# Worker
# ============================================================================


class LatestStateRenderWorker:
    """Background renderer with cached display-only updates."""

    def __init__(
        self,
        *,
        server: viser.ViserServer,
        renderer: InteractiveScientificRenderer,
        initial_state: ViewerState,
        geometry: SceneGeometry,
        status_handle: object,
        statistics_handle: object,
    ) -> None:

        self._server = server
        self._renderer = renderer
        self._state = initial_state
        self._geometry = geometry

        self._status_handle = (
            status_handle
        )

        self._statistics_handle = (
            statistics_handle
        )

        self._state_lock = (
            threading.Lock()
        )

        self._image_lock = (
            threading.Lock()
        )

        self._request_event = (
            threading.Event()
        )

        self._stop_event = (
            threading.Event()
        )

        self._render_requested = False
        self._display_requested = False

        self._last_rgb: (
            np.ndarray | None
        ) = None

        self._thread = threading.Thread(
            target=self._run,
            name="viser-scientific-render-worker",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
    ) -> None:
        self._thread.start()
        self.request_render()

    def stop(
        self,
    ) -> None:
        self._stop_event.set()
        self._request_event.set()

        self._thread.join(
            timeout=5.0
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(
        self,
    ) -> ViewerState:
        with self._state_lock:
            return self._state

    def update_render_state(
        self,
        **changes: object,
    ) -> ViewerState:

        with self._state_lock:

            self._state = replace(
                self._state,
                **changes,
                revision=(
                    self._state.revision
                    + 1
                ),
            )

            self._render_requested = True
            self._display_requested = True

            updated = self._state

        self._request_event.set()

        return updated

    def update_display_state(
        self,
        **changes: object,
    ) -> ViewerState:

        with self._state_lock:

            self._state = replace(
                self._state,
                **changes,
                revision=(
                    self._state.revision
                    + 1
                ),
            )

            self._display_requested = True

            updated = self._state

        self._request_event.set()

        return updated

    def request_render(
        self,
    ) -> None:

        with self._state_lock:

            self._render_requested = True
            self._display_requested = True

        self._request_event.set()

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def save_last_image(
        self,
        path: Path,
    ) -> bool:

        with self._image_lock:

            if self._last_rgb is None:
                return False

            image = (
                self._last_rgb.copy()
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            import imageio.v3 as iio

        except ImportError as error:
            raise RuntimeError(
                "Saving screenshots requires imageio."
            ) from error

        iio.imwrite(
            path,
            image,
        )

        return True

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(
        self,
    ) -> None:

        while (
            not self._stop_event.is_set()
        ):

            self._request_event.wait(
                timeout=0.25
            )

            if self._stop_event.is_set():
                return

            if not self._request_event.is_set():
                continue

            with self._state_lock:

                state = self._state

                render_requested = (
                    self._render_requested
                )

                display_requested = (
                    self._display_requested
                )

                self._render_requested = False
                self._display_requested = False

                self._request_event.clear()

            try:

                if render_requested:

                    frame = (
                        self._render_scientific_frame(
                            state
                        )
                    )

                    self._update_statistics(
                        frame
                    )

                    self._present_frame(
                        frame,
                        state,
                        scientific_rendered=True,
                    )

                elif display_requested:

                    frame = (
                        self._renderer.require_last_frame()
                    )

                    self._present_frame(
                        frame,
                        state,
                        scientific_rendered=False,
                    )

            except Exception:

                traceback.print_exc()

                self._set_status(
                    "### Render error\n\n"
                    "See terminal for traceback."
                )

    # ------------------------------------------------------------------
    # Scientific render
    # ------------------------------------------------------------------

    def _render_scientific_frame(
        self,
        state: ViewerState,
    ) -> InteractiveFrame:

        camera = (
            orthographic_camera_from_state(
                state.camera,
                geometry=self._geometry,
                image_width=(
                    self._renderer.image_width
                ),
                image_height=(
                    self._renderer.image_height
                ),
            )
        )

        self._renderer.set_beta(
            state.beta
        )

        self._renderer.set_blob_sigma(
            state.blob_sigma_pixels
        )

        self._renderer.set_sigma_extent(
            state.sigma_extent
        )

        return self._renderer.render(
            camera
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _update_statistics(
        self,
        frame: InteractiveFrame,
    ) -> None:

        density = (
            compute_field_statistics_gpu(
                frame.density,
                positive_only=True,
            )
        )

        attribute = (
            compute_field_statistics_gpu(
                frame.attribute,
                valid_mask=(
                    frame.valid_mask
                ),
            )
        )

        self._statistics_handle.content = (
            "### Rendered Field Statistics\n\n"

            "#### Density D(u)\n\n"
            f"Min: **{density.minimum:.6g}**  \n"
            f"P1: **{density.p1:.6g}**  \n"
            f"P5: **{density.p5:.6g}**  \n"
            f"Median: **{density.median:.6g}**  \n"
            f"P99: **{density.p99:.6g}**  \n"
            f"P99.9: **{density.p999:.6g}**  \n"
            f"Max: **{density.maximum:.6g}**  \n\n"

            "#### Attribute A(u)\n\n"
            f"Min: **{attribute.minimum:.6g}**  \n"
            f"P1: **{attribute.p1:.6g}**  \n"
            f"P5: **{attribute.p5:.6g}**  \n"
            f"Median: **{attribute.median:.6g}**  \n"
            f"P99: **{attribute.p99:.6g}**  \n"
            f"P99.9: **{attribute.p999:.6g}**  \n"
            f"Max: **{attribute.maximum:.6g}**"
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _present_frame(
        self,
        frame: InteractiveFrame,
        state: ViewerState,
        *,
        scientific_rendered: bool,
    ) -> None:

        display_start = (
            time.perf_counter()
        )

        if state.background == "White":
            background_rgb = (
                255,
                255,
                255,
            )
        else:
            background_rgb = (
                0,
                0,
                0,
            )

        color_start = (
            time.perf_counter()
        )

        # ==============================================================
        # Density
        # ==============================================================

        if state.mode == "Density":

            valid_mask = (
                torch.isfinite(
                    frame.density
                )
                & (
                    frame.density
                    >= 0.0
                )
            )

            if (
                state.density_opacity_enabled
            ):

                rgb_gpu = (
                    scalar_to_rgb_with_opacity_gpu(
                        frame.density,

                        color_minimum=(
                            state.density_minimum
                        ),

                        color_maximum=(
                            state.density_maximum
                        ),

                        opacity_minimum=(
                            state.density_opacity_minimum
                        ),

                        opacity_maximum=(
                            state.density_opacity_maximum
                        ),

                        color_scale=(
                            state.density_scale.lower()
                        ),

                        opacity_scale=(
                            state.density_opacity_scale.lower()
                        ),

                        colormap=(
                            state.density_colormap
                        ),

                        valid_mask=(
                            valid_mask
                        ),

                        invalid_rgb=(
                            background_rgb
                        ),

                        background_rgb=(
                            background_rgb
                        ),
                    )
                )

            else:

                rgb_gpu = (
                    scalar_to_rgb_gpu(
                        frame.density,

                        minimum=(
                            state.density_minimum
                        ),

                        maximum=(
                            state.density_maximum
                        ),

                        scale=(
                            state.density_scale.lower()
                        ),

                        colormap=(
                            state.density_colormap
                        ),

                        valid_mask=(
                            valid_mask
                        ),

                        invalid_rgb=(
                            background_rgb
                        ),
                    )
                )

        # ==============================================================
        # Attribute
        # ==============================================================

        else:

            if (
                state.attribute_scale
                == "Log"
                and
                state.attribute_minimum < 0.0
            ):
                raise ValueError(
                    "Log attribute scale requires "
                    "nonnegative display minimum."
                )

            if (
                state.attribute_opacity_enabled
                and state.attribute_opacity_scale == "Log"
                and state.attribute_opacity_minimum < 0.0
            ):
                raise ValueError(
                    "Log attribute opacity requires "
                    "a nonnegative opacity minimum."
                )

            if (
                state.attribute_opacity_enabled
            ):

                rgb_gpu = (
                    scalar_to_rgb_with_opacity_gpu(
                        frame.attribute,

                        color_minimum=(
                            state.attribute_minimum
                        ),

                        color_maximum=(
                            state.attribute_maximum
                        ),

                        opacity_minimum=(
                            state.attribute_opacity_minimum
                        ),

                        opacity_maximum=(
                            state.attribute_opacity_maximum
                        ),

                        color_scale=(
                            state.attribute_scale.lower()
                        ),

                        opacity_scale=(
                            state.attribute_opacity_scale.lower()
                        ),

                        colormap=(
                            state.attribute_colormap
                        ),

                        valid_mask=(
                            frame.valid_mask
                        ),

                        invalid_rgb=(
                            background_rgb
                        ),

                        background_rgb=(
                            background_rgb
                        ),
                    )
                )

            else:

                rgb_gpu = (
                    scalar_to_rgb_gpu(
                        frame.attribute,

                        minimum=(
                            state.attribute_minimum
                        ),

                        maximum=(
                            state.attribute_maximum
                        ),

                        scale=(
                            state.attribute_scale.lower()
                        ),

                        colormap=(
                            state.attribute_colormap
                        ),

                        valid_mask=(
                            frame.valid_mask
                        ),

                        invalid_rgb=(
                            background_rgb
                        ),
                    )
                )

        torch.cuda.synchronize(
            self._renderer.device
        )

        color_ms = (
            time.perf_counter()
            - color_start
        ) * 1000.0

        # ==============================================================
        # GPU -> CPU
        # ==============================================================

        copy_start = (
            time.perf_counter()
        )

        rgb = rgb_gpu_to_numpy(
            rgb_gpu
        )

        copy_ms = (
            time.perf_counter()
            - copy_start
        ) * 1000.0

        if state.vertical_flip:

            rgb = np.ascontiguousarray(
                rgb[::-1]
            )

        # ==============================================================
        # Browser
        # ==============================================================

        send_start = (
            time.perf_counter()
        )

        self._server.scene.set_background_image(
            rgb,
            format="jpeg",
            jpeg_quality=(
                state.jpeg_quality
            ),
        )

        send_ms = (
            time.perf_counter()
            - send_start
        ) * 1000.0

        display_ms = (
            time.perf_counter()
            - display_start
        ) * 1000.0

        with self._image_lock:
            self._last_rgb = (
                rgb
            )

        # ==============================================================
        # Timing
        # ==============================================================

        if scientific_rendered:

            update_name = (
                "Full scientific render"
            )

            end_to_end_ms = (
                frame.total_ms
                + display_ms
            )

        else:

            update_name = (
                "Cached display update"
            )

            end_to_end_ms = (
                display_ms
            )

        end_to_end_fps = (
            float("inf")
            if end_to_end_ms <= 0.0
            else
            1000.0 / end_to_end_ms
        )

        # ==============================================================
        # Status
        # ==============================================================

        if state.mode == "Density":

            opacity_text = (
                "Enabled"
                if state.density_opacity_enabled
                else "Disabled"
            )

            active_range = (
                state.density_minimum,
                state.density_maximum,
            )

            active_opacity_range = (
                state.density_opacity_minimum,
                state.density_opacity_maximum,
            )

        else:

            opacity_text = (
                "Enabled"
                if state.attribute_opacity_enabled
                else "Disabled"
            )

            active_range = (
                state.attribute_minimum,
                state.attribute_maximum,
            )

            active_opacity_range = (
                state.attribute_opacity_minimum,
                state.attribute_opacity_maximum,
            )

        self._set_status(
            "### Live status\n\n"

            f"Update: **{update_name}**  \n"
            f"Field: **{state.mode}**  \n"
            f"Resolution: "
            f"**{frame.width} × {frame.height}**  \n"
            f"Background: **{state.background}**  \n\n"

            "#### Transfer function\n\n"

            f"Color range: "
            f"**[{active_range[0]:.6g}, "
            f"{active_range[1]:.6g}]**  \n"

            f"Opacity: **{opacity_text}**  \n"

            f"Opacity range: "
            f"**[{active_opacity_range[0]:.6g}, "
            f"{active_opacity_range[1]:.6g}]**  \n\n"

            "#### Scientific rendering\n\n"

            f"Valid Gaussians: "
            f"**{frame.projected.n_valid:,}**  \n"

            f"Tile intersections: "
            f"**{frame.intersections.n_intersections:,}**  \n"

            f"Projection: "
            f"**{frame.projection_ms:.3f} ms**  \n"

            f"Conditional projection: "
            f"**{frame.conditional_projection_ms:.3f} ms**  \n"

            f"Intersections: "
            f"**{frame.intersections_ms:.3f} ms**  \n"

            f"CUDA rendering: "
            f"**{frame.rendering_ms:.3f} ms**  \n"

            f"Scientific frame: "
            f"**{frame.total_ms:.3f} ms "
            f"({frame.fps:.1f} FPS)**  \n\n"

            "#### Display\n\n"

            f"GPU color/opacity mapping: "
            f"**{color_ms:.3f} ms**  \n"

            f"GPU → CPU: "
            f"**{copy_ms:.3f} ms**  \n"

            f"JPEG/browser: "
            f"**{send_ms:.3f} ms**  \n"

            f"End-to-end update: "
            f"**{end_to_end_ms:.3f} ms "
            f"({end_to_end_fps:.1f} FPS)**"
        )

    def _set_status(
        self,
        content: str,
    ) -> None:

        self._status_handle.content = (
            content
        )


# ============================================================================
# Arguments
# ============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--minimum-eigenvalue",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--minimum-pixel-variance",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--sigma-extent",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--blob",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--relative-density-threshold",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--density-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--density-max",
        type=float,
        default=36_000.0,
    )

    parser.add_argument(
        "--attribute-min",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--attribute-max",
        type=float,
        default=425.0,
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path(
            "outputs/viser_screenshots"
        ),
    )

    parser.add_argument(
        "--vertical-flip",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=False,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    for name in (
        "width",
        "height",
        "tile_size",
        "port",
    ):

        if int(
            getattr(
                args,
                name,
            )
        ) <= 0:

            raise ValueError(
                f"{name} must be positive."
            )

    if (
        args.density_min < 0.0
        or
        args.density_max <= args.density_min
    ):
        raise ValueError(
            "Invalid density range."
        )

    if (
        args.attribute_max
        <= args.attribute_min
    ):
        raise ValueError(
            "Invalid attribute range."
        )


# ============================================================================
# Camera
# ============================================================================


def calculate_scene_geometry(
    model: GaussianModel,
) -> SceneGeometry:

    minimum = np.asarray(
        model.means.min(
            axis=0
        ),
        dtype=np.float64,
    )

    maximum = np.asarray(
        model.means.max(
            axis=0
        ),
        dtype=np.float64,
    )

    center = (
        0.5
        * (
            minimum
            + maximum
        )
    )

    span = (
        maximum
        - minimum
    )

    base_view_width = float(
        max(
            float(span[0]),
            float(span[1]),
        )
        * 1.02
    )

    base_camera_distance = float(
        max(
            float(
                span.max()
            )
            * 2.0,
            1.0,
        )
    )

    return SceneGeometry(
        center=center,
        minimum=minimum,
        maximum=maximum,
        span=span,
        base_view_width=(
            base_view_width
        ),
        base_camera_distance=(
            base_camera_distance
        ),
    )


def initial_camera_state(
    geometry: SceneGeometry,
) -> CameraState:

    position = np.array(
        [
            geometry.center[0],
            geometry.center[1],
            (
                geometry.center[2]
                + geometry.base_camera_distance
            ),
        ],
        dtype=np.float64,
    )

    return CameraState(
        position=position,
        look_at=(
            geometry.center.copy()
        ),
        up=np.array(
            [
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        ),
        distance=(
            geometry.base_camera_distance
        ),
    )


def camera_state_from_viser(
    camera: viser.CameraHandle,
) -> CameraState:

    position = np.asarray(
        camera.position,
        dtype=np.float64,
    ).copy()

    look_at = np.asarray(
        camera.look_at,
        dtype=np.float64,
    ).copy()

    up = np.asarray(
        camera.up_direction,
        dtype=np.float64,
    ).copy()

    distance = float(
        np.linalg.norm(
            position
            - look_at
        )
    )

    if (
        not np.isfinite(position).all()
        or not np.isfinite(look_at).all()
        or not np.isfinite(up).all()
        or not math.isfinite(distance)
        or distance <= 1.0e-9
    ):
        raise ValueError(
            "Invalid camera."
        )

    return CameraState(
        position=position,
        look_at=look_at,
        up=up,
        distance=distance,
    )


def orthographic_camera_from_state(
    camera_state: CameraState,
    *,
    geometry: SceneGeometry,
    image_width: int,
    image_height: int,
) -> OrthographicCamera:

    distance_ratio = (
        camera_state.distance
        / geometry.base_camera_distance
    )

    distance_ratio = float(
        np.clip(
            distance_ratio,
            0.01,
            100.0,
        )
    )

    view_width = (
        geometry.base_view_width
        * distance_ratio
    )

    camera_distance = max(
        camera_state.distance,
        1.0,
    )

    return OrthographicCamera(
        position=(
            camera_state.position
        ),
        target=(
            camera_state.look_at
        ),
        up=(
            camera_state.up
        ),
        view_width=(
            view_width
        ),
        image_width=(
            image_width
        ),
        image_height=(
            image_height
        ),
        near=0.0,
        far=(
            2.0
            * camera_distance
        ),
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:

    args = parse_args()

    validate_args(
        args
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    print(
        "=" * 88
    )

    print(
        "Viser interactive scientific Gaussian renderer"
    )

    print(
        "=" * 88
    )

    # ==================================================================
    # Model
    # ==================================================================

    load_start = (
        time.perf_counter()
    )

    cpu_model = GaussianModel.load(
        args.model
    )

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    geometry = (
        calculate_scene_geometry(
            cpu_model
        )
    )

    upload_start = (
        time.perf_counter()
    )

    gpu_model = GpuGaussianModel.from_cpu(
        cpu_model,
        device=device,
        minimum_eigenvalue=(
            args.minimum_eigenvalue
        ),
    )

    gpu_model.synchronize()

    upload_seconds = (
        time.perf_counter()
        - upload_start
    )

    renderer = (
        InteractiveScientificRenderer(
            gpu_model,
            image_width=args.width,
            image_height=args.height,
            tile_size=args.tile_size,
            minimum_eigenvalue=(
                args.minimum_eigenvalue
            ),
            minimum_pixel_variance=(
                args.minimum_pixel_variance
            ),
            sigma_extent=(
                args.sigma_extent
            ),
            beta=args.beta,
            blob_sigma_pixels=(
                args.blob
            ),
            relative_density_threshold=(
                args.relative_density_threshold
            ),
            normalize_gaussian_mass=True,
        )
    )

    # ==================================================================
    # Camera
    # ==================================================================

    initial_camera = (
        initial_camera_state(
            geometry
        )
    )

    initial_ortho_camera = (
        orthographic_camera_from_state(
            initial_camera,
            geometry=geometry,
            image_width=args.width,
            image_height=args.height,
        )
    )

    if args.warmup_frames > 0:

        print(
            f"Warming up "
            f"{args.warmup_frames} GPU frames..."
        )

        renderer.warm_up(
            initial_ortho_camera,
            frames=(
                args.warmup_frames
            ),
        )

    # ==================================================================
    # Server
    # ==================================================================

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
    )

    server.scene.set_up_direction(
        "+y"
    )

    server.scene.set_global_visibility(
        False
    )

    server.initial_camera.position = (
        initial_camera.position
    )

    server.initial_camera.look_at = (
        initial_camera.look_at
    )

    server.initial_camera.up = (
        initial_camera.up
    )

    server.initial_camera.near = (
        0.01
    )

    server.initial_camera.far = (
        4.0
        * geometry.base_camera_distance
    )

    minimum_orbit_distance = (
        0.01
        * geometry.base_camera_distance
    )

    maximum_orbit_distance = (
        100.0
        * geometry.base_camera_distance
    )

    try:

        server.gui.main_panel.dock_left()

        server.gui.main_panel.set_width(
            400
        )

    except Exception:
        pass

    server.gui.add_markdown(
        "# Scientific Gaussian Renderer\n\n"
        "Camera changes rerender the scientific fields. "
        "Color and opacity transfer functions operate on "
        "the cached GPU fields."
    )

    # ==================================================================
    # Field
    # ==================================================================

    field = server.gui.add_dropdown(
        "Field",
        options=(
            "Density",
            "Attribute",
        ),
        initial_value="Density",
    )

    # ==================================================================
    # Density TF
    # ==================================================================

    with server.gui.add_folder(
        "Density Transfer Function"
    ):

        density_colormap = (
            server.gui.add_dropdown(
                "Colormap",
                options=(
                    "inferno",
                    "magma",
                    "viridis",
                    "turbo",
                    "grayscale",
                ),
                initial_value="inferno",
            )
        )

        density_scale = (
            server.gui.add_dropdown(
                "Color scale",
                options=(
                    "Log",
                    "Linear",
                ),
                initial_value="Log",
            )
        )

        density_minimum = (
            server.gui.add_number(
                "Color minimum",
                initial_value=float(
                    args.density_min
                ),
                min=0.0,
                step=1.0,
            )
        )

        density_maximum = (
            server.gui.add_number(
                "Color maximum",
                initial_value=float(
                    args.density_max
                ),
                min=0.0,
                step=1.0,
            )
        )

        auto_density = (
            server.gui.add_button(
                "Auto color range 5–99.9%"
            )
        )

        density_opacity_enabled = (
            server.gui.add_checkbox(
                "Enable opacity transfer function",
                initial_value=False,
            )
        )

        density_opacity_scale = (
            server.gui.add_dropdown(
                "Opacity scale",
                options=(
                    "Log",
                    "Linear",
                ),
                initial_value="Log",
            )
        )

        density_opacity_minimum = (
            server.gui.add_number(
                "Opacity = 0 below",
                initial_value=float(
                    args.density_min
                ),
                min=0.0,
                step=1.0,
            )
        )

        density_opacity_maximum = (
            server.gui.add_number(
                "Opacity = 1 above",
                initial_value=float(
                    args.density_max
                ),
                min=0.0,
                step=1.0,
            )
        )

    # ==================================================================
    # Attribute TF
    # ==================================================================

    with server.gui.add_folder(
        "Attribute Transfer Function"
    ):

        attribute_colormap = (
            server.gui.add_dropdown(
                "Colormap",
                options=(
                    "viridis",
                    "turbo",
                    "magma",
                    "inferno",
                    "grayscale",
                ),
                initial_value="viridis",
            )
        )

        attribute_scale = (
            server.gui.add_dropdown(
                "Color scale",
                options=(
                    "Linear",
                    "Log",
                ),
                initial_value="Linear",
            )
        )

        attribute_minimum = (
            server.gui.add_number(
                "Color minimum",
                initial_value=float(
                    args.attribute_min
                ),
                step=1.0,
            )
        )

        attribute_maximum = (
            server.gui.add_number(
                "Color maximum",
                initial_value=float(
                    args.attribute_max
                ),
                step=1.0,
            )
        )

        auto_attribute = (
            server.gui.add_button(
                "Auto color range 1–99%"
            )
        )

        attribute_opacity_enabled = (
            server.gui.add_checkbox(
                "Enable opacity transfer function",
                initial_value=False,
            )
        )

        attribute_opacity_scale = (
            server.gui.add_dropdown(
                "Opacity scale",
                options=(
                    "Linear",
                    "Log",
                ),
                initial_value="Linear",
            )
        )

        attribute_opacity_minimum = (
            server.gui.add_number(
                "Opacity = 0 below",
                initial_value=float(
                    args.attribute_min
                ),
                step=1.0,
            )
        )

        attribute_opacity_maximum = (
            server.gui.add_number(
                "Opacity = 1 above",
                initial_value=float(
                    args.attribute_max
                ),
                step=1.0,
            )
        )

    # ==================================================================
    # Statistics
    # ==================================================================

    statistics = (
        server.gui.add_markdown(
            "### Rendered Field Statistics\n\n"
            "Waiting for first scientific frame..."
        )
    )

    # ==================================================================
    # Rendering parameters
    # ==================================================================

    with server.gui.add_folder(
        "Gaussian Rendering"
    ):

        beta = server.gui.add_slider(
            "Beta",
            min=0.0,
            max=1.5,
            step=0.05,
            initial_value=float(
                args.beta
            ),
        )

        blob = server.gui.add_slider(
            "Blob sigma (px)",
            min=0.0,
            max=8.0,
            step=0.1,
            initial_value=float(
                args.blob
            ),
        )

        sigma_extent = (
            server.gui.add_slider(
                "Sigma extent",
                min=1.0,
                max=5.0,
                step=0.25,
                initial_value=float(
                    args.sigma_extent
                ),
            )
        )

    # ==================================================================
    # Display
    # ==================================================================

    with server.gui.add_folder(
        "Display"
    ):

        background = (
            server.gui.add_dropdown(
                "Background",
                options=(
                    "Black",
                    "White",
                ),
                initial_value="Black",
            )
        )

        jpeg_quality = (
            server.gui.add_slider(
                "JPEG quality",
                min=30,
                max=100,
                step=1,
                initial_value=int(
                    args.jpeg_quality
                ),
            )
        )

        vertical_flip = (
            server.gui.add_checkbox(
                "Vertical flip",
                initial_value=bool(
                    args.vertical_flip
                ),
            )
        )

    render_button = (
        server.gui.add_button(
            "Render now"
        )
    )

    reset_button = (
        server.gui.add_button(
            "Reset view"
        )
    )

    screenshot_button = (
        server.gui.add_button(
            "Save screenshot"
        )
    )

    status = (
        server.gui.add_markdown(
            "### Live status\n\n"
            "Waiting for first frame..."
        )
    )

    # ==================================================================
    # State
    # ==================================================================

    initial_state = ViewerState(
        camera=initial_camera,
        mode="Density",

        beta=float(
            args.beta
        ),

        blob_sigma_pixels=float(
            args.blob
        ),

        sigma_extent=float(
            args.sigma_extent
        ),

        density_minimum=float(
            args.density_min
        ),

        density_maximum=float(
            args.density_max
        ),

        density_colormap="inferno",
        density_scale="Log",

        density_opacity_enabled=False,
        density_opacity_scale="Log",

        density_opacity_minimum=float(
            args.density_min
        ),

        density_opacity_maximum=float(
            args.density_max
        ),

        attribute_minimum=float(
            args.attribute_min
        ),

        attribute_maximum=float(
            args.attribute_max
        ),

        attribute_colormap="viridis",
        attribute_scale="Linear",

        attribute_opacity_enabled=False,
        attribute_opacity_scale="Linear",

        attribute_opacity_minimum=float(
            args.attribute_min
        ),

        attribute_opacity_maximum=float(
            args.attribute_max
        ),

        background="Black",

        jpeg_quality=int(
            args.jpeg_quality
        ),

        vertical_flip=bool(
            args.vertical_flip
        ),
    )

    worker = (
        LatestStateRenderWorker(
            server=server,
            renderer=renderer,
            initial_state=initial_state,
            geometry=geometry,
            status_handle=status,
            statistics_handle=(
                statistics
            ),
        )
    )

    connected_clients: dict[
        str,
        viser.ClientHandle,
    ] = {}

    # ==================================================================
    # Camera callbacks
    # ==================================================================

    @server.on_client_connect
    def _on_client_connect(
        client: viser.ClientHandle,
    ) -> None:

        connected_clients[
            client.client_id
        ] = client

        client.camera.min_orbit_distance = (
            minimum_orbit_distance
        )

        client.camera.max_orbit_distance = (
            maximum_orbit_distance
        )

        print(
            f"Client connected: "
            f"{client.client_id}"
        )

        @client.camera.on_update
        def _on_camera_update(
            camera: viser.CameraHandle,
        ) -> None:

            try:
                camera_state = (
                    camera_state_from_viser(
                        camera
                    )
                )

            except ValueError:
                return

            worker.update_render_state(
                camera=(
                    camera_state
                )
            )

    @server.on_client_disconnect
    def _on_client_disconnect(
        client: viser.ClientHandle,
    ) -> None:

        connected_clients.pop(
            client.client_id,
            None,
        )

    # ==================================================================
    # Field
    # ==================================================================

    @field.on_update
    def _(_) -> None:

        worker.update_display_state(
            mode=field.value
        )

    # ==================================================================
    # Density color
    # ==================================================================

    @density_colormap.on_update
    def _(_) -> None:

        worker.update_display_state(
            density_colormap=(
                density_colormap.value
            )
        )

    @density_scale.on_update
    def _(_) -> None:

        worker.update_display_state(
            density_scale=(
                density_scale.value
            )
        )

    @density_minimum.on_update
    def _(_) -> None:

        minimum = float(
            density_minimum.value
        )

        maximum = float(
            density_maximum.value
        )

        if (
            minimum < 0.0
            or maximum <= minimum
        ):
            return

        worker.update_display_state(
            density_minimum=minimum,
            density_maximum=maximum,
        )

    @density_maximum.on_update
    def _(_) -> None:

        minimum = float(
            density_minimum.value
        )

        maximum = float(
            density_maximum.value
        )

        if (
            minimum < 0.0
            or maximum <= minimum
        ):
            return

        worker.update_display_state(
            density_minimum=minimum,
            density_maximum=maximum,
        )

    @auto_density.on_click
    def _(_) -> None:

        try:

            frame = (
                renderer.require_last_frame()
            )

            minimum, maximum = (
                percentile_range_gpu(
                    frame.density,
                    lower_percentile=5.0,
                    upper_percentile=99.9,
                    positive_only=True,
                )
            )

            density_minimum.value = (
                minimum
            )

            density_maximum.value = (
                maximum
            )

            worker.update_display_state(
                density_minimum=minimum,
                density_maximum=maximum,
            )

        except Exception as error:

            status.content = (
                "### Auto range error\n\n"
                f"`{error}`"
            )

    # ==================================================================
    # Density opacity
    # ==================================================================

    @density_opacity_enabled.on_update
    def _(_) -> None:

        worker.update_display_state(
            density_opacity_enabled=bool(
                density_opacity_enabled.value
            )
        )

    @density_opacity_scale.on_update
    def _(_) -> None:

        worker.update_display_state(
            density_opacity_scale=(
                density_opacity_scale.value
            )
        )

    @density_opacity_minimum.on_update
    def _(_) -> None:

        minimum = float(
            density_opacity_minimum.value
        )

        maximum = float(
            density_opacity_maximum.value
        )

        if (
            minimum < 0.0
            or maximum <= minimum
        ):
            return

        worker.update_display_state(
            density_opacity_minimum=minimum,
            density_opacity_maximum=maximum,
        )

    @density_opacity_maximum.on_update
    def _(_) -> None:

        minimum = float(
            density_opacity_minimum.value
        )

        maximum = float(
            density_opacity_maximum.value
        )

        if (
            minimum < 0.0
            or maximum <= minimum
        ):
            return

        worker.update_display_state(
            density_opacity_minimum=minimum,
            density_opacity_maximum=maximum,
        )

    # ==================================================================
    # Attribute color
    # ==================================================================

    @attribute_colormap.on_update
    def _(_) -> None:

        worker.update_display_state(
            attribute_colormap=(
                attribute_colormap.value
            )
        )

    @attribute_scale.on_update
    def _(_) -> None:

        worker.update_display_state(
            attribute_scale=(
                attribute_scale.value
            )
        )

    @attribute_minimum.on_update
    def _(_) -> None:

        minimum = float(
            attribute_minimum.value
        )

        maximum = float(
            attribute_maximum.value
        )

        if maximum <= minimum:
            return

        worker.update_display_state(
            attribute_minimum=minimum,
            attribute_maximum=maximum,
        )

    @attribute_maximum.on_update
    def _(_) -> None:

        minimum = float(
            attribute_minimum.value
        )

        maximum = float(
            attribute_maximum.value
        )

        if maximum <= minimum:
            return

        worker.update_display_state(
            attribute_minimum=minimum,
            attribute_maximum=maximum,
        )

    @auto_attribute.on_click
    def _(_) -> None:

        try:

            frame = (
                renderer.require_last_frame()
            )

            minimum, maximum = (
                percentile_range_gpu(
                    frame.attribute,
                    valid_mask=(
                        frame.valid_mask
                    ),
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
            )

            attribute_minimum.value = (
                minimum
            )

            attribute_maximum.value = (
                maximum
            )

            worker.update_display_state(
                attribute_minimum=minimum,
                attribute_maximum=maximum,
            )

        except Exception as error:

            status.content = (
                "### Auto range error\n\n"
                f"`{error}`"
            )

    # ==================================================================
    # Attribute opacity
    # ==================================================================

    @attribute_opacity_enabled.on_update
    def _(_) -> None:

        worker.update_display_state(
            attribute_opacity_enabled=bool(
                attribute_opacity_enabled.value
            )
        )

    @attribute_opacity_scale.on_update
    def _(_) -> None:

        worker.update_display_state(
            attribute_opacity_scale=(
                attribute_opacity_scale.value
            )
        )

    @attribute_opacity_minimum.on_update
    def _(_) -> None:

        minimum = float(
            attribute_opacity_minimum.value
        )

        maximum = float(
            attribute_opacity_maximum.value
        )

        if maximum <= minimum:
            return

        worker.update_display_state(
            attribute_opacity_minimum=minimum,
            attribute_opacity_maximum=maximum,
        )

    @attribute_opacity_maximum.on_update
    def _(_) -> None:

        minimum = float(
            attribute_opacity_minimum.value
        )

        maximum = float(
            attribute_opacity_maximum.value
        )

        if maximum <= minimum:
            return

        worker.update_display_state(
            attribute_opacity_minimum=minimum,
            attribute_opacity_maximum=maximum,
        )

    # ==================================================================
    # Scientific rendering parameters
    # ==================================================================

    @beta.on_update
    def _(_) -> None:

        worker.update_render_state(
            beta=float(
                beta.value
            )
        )

    @blob.on_update
    def _(_) -> None:

        worker.update_render_state(
            blob_sigma_pixels=float(
                blob.value
            )
        )

    @sigma_extent.on_update
    def _(_) -> None:

        worker.update_render_state(
            sigma_extent=float(
                sigma_extent.value
            )
        )

    # ==================================================================
    # Display
    # ==================================================================

    @background.on_update
    def _(_) -> None:

        worker.update_display_state(
            background=(
                background.value
            )
        )

    @jpeg_quality.on_update
    def _(_) -> None:

        worker.update_display_state(
            jpeg_quality=int(
                jpeg_quality.value
            )
        )

    @vertical_flip.on_update
    def _(_) -> None:

        worker.update_display_state(
            vertical_flip=bool(
                vertical_flip.value
            )
        )

    # ==================================================================
    # Buttons
    # ==================================================================

    @render_button.on_click
    def _(_) -> None:

        worker.request_render()

    @reset_button.on_click
    def _(_) -> None:

        for client in tuple(
            connected_clients.values()
        ):

            client.camera.position = (
                initial_camera.position
            )

            client.camera.look_at = (
                initial_camera.look_at
            )

            client.camera.up_direction = (
                initial_camera.up
            )

        worker.update_render_state(
            camera=(
                initial_camera
            )
        )

    @screenshot_button.on_click
    def _(_) -> None:

        timestamp = (
            time.strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        path = (
            args.screenshot_dir
            / (
                f"scientific_view_"
                f"{timestamp}.png"
            )
        )

        try:

            saved = (
                worker.save_last_image(
                    path
                )
            )

        except Exception as error:

            status.content = (
                "### Screenshot error\n\n"
                f"`{error}`"
            )

            return

        if saved:

            status.content = (
                "### Screenshot saved\n\n"
                f"`{path.resolve()}`"
            )

    # ==================================================================
    # Console
    # ==================================================================

    print()

    print(
        f"Model                  : {args.model}"
    )

    print(
        f"Gaussians              : "
        f"{gpu_model.n_gaussians:,}"
    )

    print(
        f"Particles represented  : "
        f"{gpu_model.n_particles:,}"
    )

    print(
        f"Attribute              : "
        f"{gpu_model.attribute_name}"
    )

    print(
        f"GPU                    : "
        f"{torch.cuda.get_device_name(device)}"
    )

    print(
        f"Model load             : "
        f"{load_seconds:.6f} s"
    )

    print(
        f"One-time GPU upload    : "
        f"{upload_seconds:.6f} s"
    )

    print(
        f"GPU model memory       : "
        f"{gpu_model.memory_megabytes():.3f} MiB"
    )

    print(
        f"Viewer address         : "
        f"http://{args.host}:{args.port}"
    )

    print()

    print(
        "SSH tunnel:"
    )

    print(
        f"ssh -N -L "
        f"{args.port}:127.0.0.1:{args.port} "
        f"robiul@129.123.10.47"
    )

    print()

    print(
        f"Open: http://localhost:{args.port}"
    )

    print(
        "Press Ctrl+C to stop."
    )

    # ==================================================================
    # Run
    # ==================================================================

    worker.start()

    try:

        while True:
            time.sleep(
                1.0
            )

    except KeyboardInterrupt:

        print(
            "\nStopping viewer..."
        )

    finally:

        worker.stop()


if __name__ == "__main__":
    main()