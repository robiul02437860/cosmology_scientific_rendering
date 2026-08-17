# from __future__ import annotations

# import argparse
# from pathlib import Path
# from time import perf_counter

# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# from scientific_gsplat_renderer.camera.orthographic import (
#     OrthographicCamera,
# )
# from scientific_gsplat_renderer.data.gaussian_model import (
#     GaussianModel,
# )
# from scientific_gsplat_renderer.gpu import (
#     GpuConditionalRenderResult,
#     GpuGaussianModel,
#     GpuProjectedConditionalAttributes,
#     GpuProjectedGaussians,
#     GpuTileIntersections,
#     build_gpu_tile_intersections,
#     project_conditional_attributes_orthographic_gpu,
#     project_gaussians_orthographic_gpu,
#     render_conditional_attribute_gpu,
# )


# DEFAULT_MODEL = Path(
#     "/home/robiul/Particle_flow/HACC_project/output/"
#     "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
# )


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Render density and conditional attribute images using the "
#             "fully GPU-resident scientific renderer. Measure cold-start, "
#             "warm-up, and steady-state interactive performance separately."
#         )
#     )

#     parser.add_argument(
#         "--model",
#         type=Path,
#         default=DEFAULT_MODEL,
#     )

#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=Path(
#             "outputs/gpu_conditional_pipeline"
#         ),
#     )

#     parser.add_argument(
#         "--width",
#         type=int,
#         default=600,
#     )

#     parser.add_argument(
#         "--height",
#         type=int,
#         default=600,
#     )

#     parser.add_argument(
#         "--tile-size",
#         type=int,
#         default=16,
#     )

#     parser.add_argument(
#         "--minimum-eigenvalue",
#         type=float,
#         default=1.0e-6,
#     )

#     parser.add_argument(
#         "--minimum-pixel-variance",
#         type=float,
#         default=0.25,
#     )

#     parser.add_argument(
#         "--sigma-extent",
#         type=float,
#         default=3.0,
#     )

#     parser.add_argument(
#         "--relative-density-threshold",
#         type=float,
#         default=1.0e-6,
#     )

#     parser.add_argument(
#         "--warmup-frames",
#         type=int,
#         default=5,
#         help=(
#             "Number of complete frames rendered before steady-state "
#             "performance measurement."
#         ),
#     )

#     parser.add_argument(
#         "--timed-frames",
#         type=int,
#         default=50,
#         help=(
#             "Number of complete frames used to calculate steady-state "
#             "average timings."
#         ),
#     )

#     parser.add_argument(
#         "--unnormalized-mass",
#         action="store_true",
#         help=(
#             "Use Gaussian mass directly as peak amplitude instead of "
#             "normalizing each projected Gaussian to integrate to its mass."
#         ),
#     )

#     return parser.parse_args()


# def validate_args(
#     args: argparse.Namespace,
# ) -> None:
#     if not args.model.exists():
#         raise FileNotFoundError(
#             f"Model file does not exist: {args.model}"
#         )

#     if args.width <= 0:
#         raise ValueError(
#             f"width must be positive, got {args.width}"
#         )

#     if args.height <= 0:
#         raise ValueError(
#             f"height must be positive, got {args.height}"
#         )

#     if args.tile_size <= 0:
#         raise ValueError(
#             f"tile_size must be positive, got {args.tile_size}"
#         )

#     if args.minimum_eigenvalue <= 0.0:
#         raise ValueError(
#             "minimum_eigenvalue must be positive, "
#             f"got {args.minimum_eigenvalue}"
#         )

#     if args.minimum_pixel_variance <= 0.0:
#         raise ValueError(
#             "minimum_pixel_variance must be positive, "
#             f"got {args.minimum_pixel_variance}"
#         )

#     if args.sigma_extent <= 0.0:
#         raise ValueError(
#             f"sigma_extent must be positive, got {args.sigma_extent}"
#         )

#     if args.relative_density_threshold < 0.0:
#         raise ValueError(
#             "relative_density_threshold must be nonnegative, "
#             f"got {args.relative_density_threshold}"
#         )

#     if args.warmup_frames < 0:
#         raise ValueError(
#             "warmup_frames must be nonnegative, "
#             f"got {args.warmup_frames}"
#         )

#     if args.timed_frames <= 0:
#         raise ValueError(
#             "timed_frames must be positive, "
#             f"got {args.timed_frames}"
#         )


# def build_camera(
#     model: GaussianModel,
#     *,
#     image_width: int,
#     image_height: int,
# ) -> OrthographicCamera:
#     minimum = np.asarray(
#         model.means.min(axis=0),
#         dtype=np.float64,
#     )

#     maximum = np.asarray(
#         model.means.max(axis=0),
#         dtype=np.float64,
#     )

#     center = 0.5 * (
#         minimum + maximum
#     )

#     span = maximum - minimum

#     view_width = float(
#         max(
#             float(span[0]),
#             float(span[1]),
#         )
#         * 1.02
#     )

#     camera_distance = float(
#         max(
#             float(span.max()) * 2.0,
#             1.0,
#         )
#     )

#     return OrthographicCamera(
#         position=np.array(
#             [
#                 center[0],
#                 center[1],
#                 center[2] + camera_distance,
#             ],
#             dtype=np.float64,
#         ),
#         target=center.astype(
#             np.float64,
#             copy=False,
#         ),
#         up=np.array(
#             [0.0, 1.0, 0.0],
#             dtype=np.float64,
#         ),
#         view_width=view_width,
#         image_width=image_width,
#         image_height=image_height,
#         near=0.0,
#         far=2.0 * camera_distance,
#     )


# def create_cuda_event() -> torch.cuda.Event:
#     return torch.cuda.Event(
#         enable_timing=True,
#     )


# def elapsed_seconds(
#     start: torch.cuda.Event,
#     end: torch.cuda.Event,
# ) -> float:
#     return (
#         start.elapsed_time(end)
#         / 1000.0
#     )


# def render_gpu_frame(
#     gpu_model: GpuGaussianModel,
#     camera: OrthographicCamera,
#     *,
#     image_width: int,
#     image_height: int,
#     tile_size: int,
#     minimum_eigenvalue: float,
#     minimum_pixel_variance: float,
#     sigma_extent: float,
#     normalize_gaussian_mass: bool,
#     relative_density_threshold: float,
# ) -> tuple[
#     GpuProjectedGaussians,
#     GpuProjectedConditionalAttributes,
#     GpuTileIntersections,
#     GpuConditionalRenderResult,
# ]:
#     projected = project_gaussians_orthographic_gpu(
#         gpu_model,
#         camera,
#         minimum_eigenvalue=minimum_eigenvalue,
#         minimum_pixel_variance=minimum_pixel_variance,
#         sigma_extent=sigma_extent,
#     )

#     conditional = (
#         project_conditional_attributes_orthographic_gpu(
#             gpu_model,
#             projected,
#             camera,
#         )
#     )

#     intersections = build_gpu_tile_intersections(
#         projected,
#         image_width=image_width,
#         image_height=image_height,
#         tile_size=tile_size,
#     )

#     result = render_conditional_attribute_gpu(
#         projected,
#         conditional,
#         intersections,
#         normalize_gaussian_mass=normalize_gaussian_mass,
#         relative_density_threshold=relative_density_threshold,
#     )

#     return (
#         projected,
#         conditional,
#         intersections,
#         result,
#     )


# def save_density_image(
#     path: Path,
#     density: np.ndarray,
# ) -> None:
#     positive = density[
#         np.isfinite(density)
#         & (density > 0.0)
#     ]

#     if positive.size == 0:
#         image = np.zeros_like(
#             density,
#             dtype=np.float64,
#         )

#     else:
#         upper = float(
#             np.percentile(
#                 positive,
#                 99.9,
#             )
#         )

#         image = np.log1p(
#             np.maximum(
#                 density,
#                 0.0,
#             )
#         )

#         denominator = max(
#             np.log1p(upper),
#             1.0e-12,
#         )

#         image = np.clip(
#             image / denominator,
#             0.0,
#             1.0,
#         )

#     plt.imsave(
#         path,
#         image,
#         origin="lower",
#         cmap="inferno",
#     )


# def save_attribute_image(
#     path: Path,
#     attribute: np.ndarray,
#     valid_mask: np.ndarray,
# ) -> tuple[float, float]:
#     finite_values = attribute[
#         valid_mask
#         & np.isfinite(attribute)
#     ]

#     if finite_values.size == 0:
#         raise RuntimeError(
#             "No finite attribute values were rendered."
#         )

#     lower = float(
#         np.percentile(
#             finite_values,
#             1.0,
#         )
#     )

#     upper = float(
#         np.percentile(
#             finite_values,
#             99.0,
#         )
#     )

#     if upper <= lower:
#         upper = lower + 1.0

#     image = np.asarray(
#         attribute,
#         dtype=np.float64,
#     ).copy()

#     image[~valid_mask] = np.nan

#     plt.imsave(
#         path,
#         image,
#         origin="lower",
#         cmap="viridis",
#         vmin=lower,
#         vmax=upper,
#     )

#     return lower, upper


# def save_combined_figure(
#     path: Path,
#     density: np.ndarray,
#     attribute: np.ndarray,
#     valid_mask: np.ndarray,
#     *,
#     attribute_name: str,
# ) -> None:
#     positive_density = density[
#         np.isfinite(density)
#         & (density > 0.0)
#     ]

#     if positive_density.size > 0:
#         density_upper = float(
#             np.percentile(
#                 positive_density,
#                 99.9,
#             )
#         )
#     else:
#         density_upper = 1.0

#     finite_attribute = attribute[
#         valid_mask
#         & np.isfinite(attribute)
#     ]

#     if finite_attribute.size == 0:
#         raise RuntimeError(
#             "No finite attribute pixels are available."
#         )

#     attribute_lower = float(
#         np.percentile(
#             finite_attribute,
#             1.0,
#         )
#     )

#     attribute_upper = float(
#         np.percentile(
#             finite_attribute,
#             99.0,
#         )
#     )

#     if attribute_upper <= attribute_lower:
#         attribute_upper = (
#             attribute_lower + 1.0
#         )

#     figure, axes = plt.subplots(
#         1,
#         2,
#         figsize=(12, 5),
#         constrained_layout=True,
#     )

#     density_image = axes[0].imshow(
#         np.log1p(
#             np.maximum(
#                 density,
#                 0.0,
#             )
#         ),
#         origin="lower",
#         cmap="inferno",
#         vmin=0.0,
#         vmax=np.log1p(
#             density_upper
#         ),
#     )

#     axes[0].set_title(
#         "Projected density"
#     )

#     axes[0].set_axis_off()

#     figure.colorbar(
#         density_image,
#         ax=axes[0],
#         fraction=0.046,
#         pad=0.04,
#         label="log(1 + density)",
#     )

#     masked_attribute = np.ma.masked_where(
#         ~valid_mask,
#         attribute,
#     )

#     attribute_image = axes[1].imshow(
#         masked_attribute,
#         origin="lower",
#         cmap="viridis",
#         vmin=attribute_lower,
#         vmax=attribute_upper,
#     )

#     axes[1].set_title(
#         attribute_name
#     )

#     axes[1].set_axis_off()

#     figure.colorbar(
#         attribute_image,
#         ax=axes[1],
#         fraction=0.046,
#         pad=0.04,
#         label=attribute_name,
#     )

#     figure.savefig(
#         path,
#         dpi=180,
#     )

#     plt.close(
#         figure
#     )


# def main() -> None:
#     args = parse_args()
#     validate_args(args)

#     if not torch.cuda.is_available():
#         raise RuntimeError(
#             "CUDA is unavailable."
#         )

#     device = torch.device(
#         "cuda:0"
#     )

#     normalize_gaussian_mass = (
#         not args.unnormalized_mass
#     )

#     args.output.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     print("=" * 80)
#     print(
#         "Fully GPU-resident scientific conditional rendering"
#     )
#     print("=" * 80)

#     print(
#         f"Model                   : {args.model}"
#     )

#     print(
#         f"Output                  : "
#         f"{args.output.resolve()}"
#     )

#     print(
#         f"Image                   : "
#         f"{args.width} x {args.height}"
#     )

#     print(
#         f"Tile size               : "
#         f"{args.tile_size}"
#     )

#     print(
#         f"Minimum eigenvalue      : "
#         f"{args.minimum_eigenvalue}"
#     )

#     print(
#         f"Minimum pixel variance  : "
#         f"{args.minimum_pixel_variance}"
#     )

#     print(
#         f"Sigma extent            : "
#         f"{args.sigma_extent}"
#     )

#     print(
#         f"Normalize mass          : "
#         f"{normalize_gaussian_mass}"
#     )

#     print(
#         f"Warm-up frames          : "
#         f"{args.warmup_frames}"
#     )

#     print(
#         f"Timed interactive frames: "
#         f"{args.timed_frames}"
#     )

#     print(
#         f"GPU                     : "
#         f"{torch.cuda.get_device_name(device)}"
#     )

#     print()

#     # --------------------------------------------------------------
#     # Model loading
#     # --------------------------------------------------------------

#     load_start = perf_counter()

#     model = GaussianModel.load(
#         args.model
#     )

#     load_seconds = (
#         perf_counter()
#         - load_start
#     )

#     camera = build_camera(
#         model,
#         image_width=args.width,
#         image_height=args.height,
#     )

#     # --------------------------------------------------------------
#     # One-time GPU upload and covariance stabilization
#     # --------------------------------------------------------------

#     torch.cuda.synchronize(
#         device
#     )

#     upload_start = perf_counter()

#     gpu_model = GpuGaussianModel.from_cpu(
#         model,
#         device=device,
#         minimum_eigenvalue=(
#             args.minimum_eigenvalue
#         ),
#     )

#     gpu_model.synchronize()

#     upload_seconds = (
#         perf_counter()
#         - upload_start
#     )

#     # --------------------------------------------------------------
#     # Cold first frame
#     # --------------------------------------------------------------

#     cold_start = create_cuda_event()
#     cold_end = create_cuda_event()

#     cold_start.record()

#     (
#         projected,
#         conditional,
#         intersections,
#         result,
#     ) = render_gpu_frame(
#         gpu_model,
#         camera,
#         image_width=args.width,
#         image_height=args.height,
#         tile_size=args.tile_size,
#         minimum_eigenvalue=(
#             args.minimum_eigenvalue
#         ),
#         minimum_pixel_variance=(
#             args.minimum_pixel_variance
#         ),
#         sigma_extent=args.sigma_extent,
#         normalize_gaussian_mass=(
#             normalize_gaussian_mass
#         ),
#         relative_density_threshold=(
#             args.relative_density_threshold
#         ),
#     )

#     cold_end.record()
#     cold_end.synchronize()

#     cold_frame_seconds = elapsed_seconds(
#         cold_start,
#         cold_end,
#     )

#     # --------------------------------------------------------------
#     # Warm-up frames
#     # --------------------------------------------------------------

#     warmup_total_seconds = 0.0
#     warmup_average_seconds = 0.0

#     if args.warmup_frames > 0:
#         warmup_start = create_cuda_event()
#         warmup_end = create_cuda_event()

#         warmup_start.record()

#         for _ in range(
#             args.warmup_frames
#         ):
#             (
#                 projected,
#                 conditional,
#                 intersections,
#                 result,
#             ) = render_gpu_frame(
#                 gpu_model,
#                 camera,
#                 image_width=args.width,
#                 image_height=args.height,
#                 tile_size=args.tile_size,
#                 minimum_eigenvalue=(
#                     args.minimum_eigenvalue
#                 ),
#                 minimum_pixel_variance=(
#                     args.minimum_pixel_variance
#                 ),
#                 sigma_extent=(
#                     args.sigma_extent
#                 ),
#                 normalize_gaussian_mass=(
#                     normalize_gaussian_mass
#                 ),
#                 relative_density_threshold=(
#                     args.relative_density_threshold
#                 ),
#             )

#         warmup_end.record()
#         warmup_end.synchronize()

#         warmup_total_seconds = elapsed_seconds(
#             warmup_start,
#             warmup_end,
#         )

#         warmup_average_seconds = (
#             warmup_total_seconds
#             / float(args.warmup_frames)
#         )

#     # --------------------------------------------------------------
#     # Steady-state interactive timing
#     # --------------------------------------------------------------

#     projection_total = 0.0
#     conditional_projection_total = 0.0
#     intersection_total = 0.0
#     rendering_total = 0.0
#     frame_total = 0.0

#     for _ in range(
#         args.timed_frames
#     ):
#         frame_start = create_cuda_event()
#         frame_end = create_cuda_event()

#         projection_start = create_cuda_event()
#         projection_end = create_cuda_event()

#         conditional_start = create_cuda_event()
#         conditional_end = create_cuda_event()

#         intersection_start = create_cuda_event()
#         intersection_end = create_cuda_event()

#         rendering_start = create_cuda_event()
#         rendering_end = create_cuda_event()

#         frame_start.record()

#         projection_start.record()

#         projected = (
#             project_gaussians_orthographic_gpu(
#                 gpu_model,
#                 camera,
#                 minimum_eigenvalue=(
#                     args.minimum_eigenvalue
#                 ),
#                 minimum_pixel_variance=(
#                     args.minimum_pixel_variance
#                 ),
#                 sigma_extent=(
#                     args.sigma_extent
#                 ),
#             )
#         )

#         projection_end.record()

#         conditional_start.record()

#         conditional = (
#             project_conditional_attributes_orthographic_gpu(
#                 gpu_model,
#                 projected,
#                 camera,
#             )
#         )

#         conditional_end.record()

#         intersection_start.record()

#         intersections = (
#             build_gpu_tile_intersections(
#                 projected,
#                 image_width=args.width,
#                 image_height=args.height,
#                 tile_size=args.tile_size,
#             )
#         )

#         intersection_end.record()

#         rendering_start.record()

#         result = (
#             render_conditional_attribute_gpu(
#                 projected,
#                 conditional,
#                 intersections,
#                 normalize_gaussian_mass=(
#                     normalize_gaussian_mass
#                 ),
#                 relative_density_threshold=(
#                     args.relative_density_threshold
#                 ),
#             )
#         )

#         rendering_end.record()

#         frame_end.record()
#         frame_end.synchronize()

#         projection_total += elapsed_seconds(
#             projection_start,
#             projection_end,
#         )

#         conditional_projection_total += (
#             elapsed_seconds(
#                 conditional_start,
#                 conditional_end,
#             )
#         )

#         intersection_total += elapsed_seconds(
#             intersection_start,
#             intersection_end,
#         )

#         rendering_total += elapsed_seconds(
#             rendering_start,
#             rendering_end,
#         )

#         frame_total += elapsed_seconds(
#             frame_start,
#             frame_end,
#         )

#     number_of_timed_frames = float(
#         args.timed_frames
#     )

#     projection_average = (
#         projection_total
#         / number_of_timed_frames
#     )

#     conditional_projection_average = (
#         conditional_projection_total
#         / number_of_timed_frames
#     )

#     intersection_average = (
#         intersection_total
#         / number_of_timed_frames
#     )

#     rendering_average = (
#         rendering_total
#         / number_of_timed_frames
#     )

#     frame_average = (
#         frame_total
#         / number_of_timed_frames
#     )

#     interactive_fps = (
#         1.0 / frame_average
#         if frame_average > 0.0
#         else float("inf")
#     )

#     # --------------------------------------------------------------
#     # Copy only final frame to CPU
#     # --------------------------------------------------------------

#     density = (
#         result
#         .density_numpy()
#         .astype(
#             np.float64,
#             copy=False,
#         )
#     )

#     attribute_numerator = (
#         result
#         .attribute_numerator_numpy()
#         .astype(
#             np.float64,
#             copy=False,
#         )
#     )

#     attribute = (
#         result
#         .attribute_numpy()
#         .astype(
#             np.float64,
#             copy=False,
#         )
#     )

#     valid_mask = (
#         result
#         .valid_mask_numpy()
#         .astype(
#             np.bool_,
#             copy=False,
#         )
#     )

#     # --------------------------------------------------------------
#     # Save arrays
#     # --------------------------------------------------------------

#     np.save(
#         args.output / "density.npy",
#         density.astype(
#             np.float32
#         ),
#     )

#     np.save(
#         args.output
#         / "attribute_numerator.npy",
#         attribute_numerator.astype(
#             np.float32
#         ),
#     )

#     np.save(
#         args.output
#         / "conditional_attribute.npy",
#         attribute.astype(
#             np.float32
#         ),
#     )

#     np.save(
#         args.output / "valid_mask.npy",
#         valid_mask,
#     )

#     # --------------------------------------------------------------
#     # Save images
#     # --------------------------------------------------------------

#     save_density_image(
#         args.output / "density.png",
#         density,
#     )

#     (
#         attribute_lower,
#         attribute_upper,
#     ) = save_attribute_image(
#         args.output
#         / "conditional_attribute.png",
#         attribute,
#         valid_mask,
#     )

#     attribute_name = (
#         model.attribute_name
#         if model.attribute_name is not None
#         else "Conditional attribute"
#     )

#     save_combined_figure(
#         args.output / "comparison.png",
#         density,
#         attribute,
#         valid_mask,
#         attribute_name=attribute_name,
#     )

#     valid_attribute_values = attribute[
#         valid_mask
#         & np.isfinite(attribute)
#     ]

#     if valid_attribute_values.size == 0:
#         raise RuntimeError(
#             "No valid finite attribute values were rendered."
#         )

#     # --------------------------------------------------------------
#     # Rendering result
#     # --------------------------------------------------------------

#     print()
#     print("=" * 80)
#     print("Rendering result")
#     print("=" * 80)

#     print(
#         f"Gaussians               : "
#         f"{gpu_model.n_gaussians:,}"
#     )

#     print(
#         f"Valid Gaussians         : "
#         f"{projected.n_valid:,}"
#     )

#     print(
#         f"Tile intersections      : "
#         f"{intersections.n_intersections:,}"
#     )

#     print(
#         f"Attribute               : "
#         f"{attribute_name}"
#     )

#     print()

#     print(
#         f"Model load              : "
#         f"{load_seconds:.6f} s"
#     )

#     print(
#         f"One-time GPU upload     : "
#         f"{upload_seconds:.6f} s"
#     )

#     print(
#         f"GPU model memory        : "
#         f"{gpu_model.memory_megabytes():.3f} MiB"
#     )

#     # --------------------------------------------------------------
#     # Performance
#     # --------------------------------------------------------------

#     print()
#     print("=" * 80)
#     print("Performance")
#     print("=" * 80)

#     print(
#         f"Cold first frame        : "
#         f"{cold_frame_seconds * 1000.0:.3f} ms"
#     )

#     cold_fps = (
#         1.0 / cold_frame_seconds
#         if cold_frame_seconds > 0.0
#         else float("inf")
#     )

#     print(
#         f"Cold first-frame FPS    : "
#         f"{cold_fps:.2f}"
#     )

#     print(
#         f"Warm-up frames          : "
#         f"{args.warmup_frames}"
#     )

#     if args.warmup_frames > 0:
#         print(
#             f"Warm-up total           : "
#             f"{warmup_total_seconds * 1000.0:.3f} ms"
#         )

#         print(
#             f"Warm-up average         : "
#             f"{warmup_average_seconds * 1000.0:.3f} ms"
#         )

#     else:
#         print(
#             "Warm-up average         : "
#             "not measured"
#         )

#     print(
#         f"Timed interactive frames: "
#         f"{args.timed_frames}"
#     )

#     print()
#     print("Steady-state stage averages")
#     print("-" * 80)

#     print(
#         f"GPU projection          : "
#         f"{projection_average * 1000.0:.3f} ms"
#     )

#     print(
#         f"Conditional projection  : "
#         f"{conditional_projection_average * 1000.0:.3f} ms"
#     )

#     print(
#         f"Tile intersections      : "
#         f"{intersection_average * 1000.0:.3f} ms"
#     )

#     print(
#         f"Conditional rendering   : "
#         f"{rendering_average * 1000.0:.3f} ms"
#     )

#     print(
#         f"Interactive frame       : "
#         f"{frame_average * 1000.0:.3f} ms"
#     )

#     print(
#         f"Interactive FPS         : "
#         f"{interactive_fps:.2f}"
#     )

#     print()
#     print("Final render-call internal timing")
#     print("-" * 80)

#     print(
#         f"Render preparation      : "
#         f"{result.preparation_seconds * 1000.0:.3f} ms"
#     )

#     print(
#         f"CUDA rasterization      : "
#         f"{result.rasterization_seconds * 1000.0:.3f} ms"
#     )

#     print(
#         f"Attribute normalization : "
#         f"{result.normalization_seconds * 1000.0:.3f} ms"
#     )

#     # --------------------------------------------------------------
#     # Final-frame statistics
#     # --------------------------------------------------------------

#     print()
#     print("=" * 80)
#     print("Final-frame statistics")
#     print("=" * 80)

#     print(
#         f"Density minimum         : "
#         f"{density.min():.12g}"
#     )

#     print(
#         f"Density maximum         : "
#         f"{density.max():.12g}"
#     )

#     print(
#         f"Density mean            : "
#         f"{density.mean():.12g}"
#     )

#     print(
#         f"Density sum             : "
#         f"{density.sum():.12g}"
#     )

#     print(
#         f"Density threshold       : "
#         f"{result.density_threshold:.12g}"
#     )

#     print(
#         f"Valid attribute pixels  : "
#         f"{np.count_nonzero(valid_mask):,}"
#     )

#     print(
#         f"Attribute minimum       : "
#         f"{valid_attribute_values.min():.12g}"
#     )

#     print(
#         f"Attribute maximum       : "
#         f"{valid_attribute_values.max():.12g}"
#     )

#     print(
#         f"Attribute mean          : "
#         f"{valid_attribute_values.mean():.12g}"
#     )

#     print(
#         f"Display range           : "
#         f"[{attribute_lower:.6g}, "
#         f"{attribute_upper:.6g}]"
#     )

#     # --------------------------------------------------------------
#     # Saved files
#     # --------------------------------------------------------------

#     print()
#     print("Saved files")
#     print("-" * 80)

#     for name in (
#         "density.npy",
#         "attribute_numerator.npy",
#         "conditional_attribute.npy",
#         "valid_mask.npy",
#         "density.png",
#         "conditional_attribute.png",
#         "comparison.png",
#     ):
#         print(
#             args.output / name
#         )


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from scientific_gsplat_renderer.camera.orthographic import (
    OrthographicCamera,
)
from scientific_gsplat_renderer.data.gaussian_model import (
    GaussianModel,
)
from scientific_gsplat_renderer.gpu import (
    GpuConditionalRenderResult,
    GpuGaussianModel,
    GpuProjectedConditionalAttributes,
    GpuProjectedGaussians,
    GpuTileIntersections,
    build_gpu_tile_intersections,
    project_conditional_attributes_orthographic_gpu,
    project_gaussians_orthographic_gpu,
    render_conditional_attribute_gpu,
)


DEFAULT_MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render density and a conditional scalar attribute using the "
            "fully GPU-resident scientific renderer. Reports cold-start, "
            "warm-up, and steady-state interactive performance separately."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/gpu_conditional_pipeline"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=600,
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
        default=0.0,
        help=(
            "Paper-style covariance smoothing parameter. "
            "The model covariance becomes (1 + beta) * Sigma."
        ),
    )

    parser.add_argument(
        "--blob",
        type=float,
        default=0.0,
        help=(
            "Isotropic screen-space Gaussian blob sigma in pixels. "
            "Use the same blob value used to generate the particle GT."
        ),
    )

    parser.add_argument(
        "--relative-density-threshold",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help=(
            "Number of complete GPU frames executed before "
            "steady-state timing."
        ),
    )

    parser.add_argument(
        "--timed-frames",
        type=int,
        default=50,
        help=(
            "Number of complete GPU frames used for "
            "steady-state timing."
        ),
    )

    parser.add_argument(
        "--unnormalized-mass",
        action="store_true",
        help=(
            "Use Gaussian mass directly as peak amplitude instead of "
            "normalizing every projected Gaussian to integrate to its mass."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {args.model}"
        )

    if args.width <= 0:
        raise ValueError(
            f"width must be positive, got {args.width}"
        )

    if args.height <= 0:
        raise ValueError(
            f"height must be positive, got {args.height}"
        )

    if args.tile_size <= 0:
        raise ValueError(
            "tile-size must be positive, "
            f"got {args.tile_size}"
        )

    if (
        not math.isfinite(args.minimum_eigenvalue)
        or args.minimum_eigenvalue <= 0.0
    ):
        raise ValueError(
            "minimum-eigenvalue must be finite and positive, "
            f"got {args.minimum_eigenvalue}"
        )

    if (
        not math.isfinite(args.minimum_pixel_variance)
        or args.minimum_pixel_variance <= 0.0
    ):
        raise ValueError(
            "minimum-pixel-variance must be finite and positive, "
            f"got {args.minimum_pixel_variance}"
        )

    if (
        not math.isfinite(args.sigma_extent)
        or args.sigma_extent <= 0.0
    ):
        raise ValueError(
            "sigma-extent must be finite and positive, "
            f"got {args.sigma_extent}"
        )

    if (
        not math.isfinite(args.beta)
        or args.beta < 0.0
    ):
        raise ValueError(
            "beta must be finite and nonnegative, "
            f"got {args.beta}"
        )

    if (
        not math.isfinite(args.blob)
        or args.blob < 0.0
    ):
        raise ValueError(
            "blob must be finite and nonnegative, "
            f"got {args.blob}"
        )

    if (
        not math.isfinite(args.relative_density_threshold)
        or args.relative_density_threshold < 0.0
    ):
        raise ValueError(
            "relative-density-threshold must be finite and nonnegative, "
            f"got {args.relative_density_threshold}"
        )

    if args.warmup_frames < 0:
        raise ValueError(
            "warmup-frames must be nonnegative, "
            f"got {args.warmup_frames}"
        )

    if args.timed_frames <= 0:
        raise ValueError(
            "timed-frames must be positive, "
            f"got {args.timed_frames}"
        )


def build_camera(
    model: GaussianModel,
    *,
    image_width: int,
    image_height: int,
) -> OrthographicCamera:
    minimum = np.asarray(
        model.means.min(axis=0),
        dtype=np.float64,
    )

    maximum = np.asarray(
        model.means.max(axis=0),
        dtype=np.float64,
    )

    center = 0.5 * (
        minimum + maximum
    )

    span = maximum - minimum

    view_width = float(
        max(
            float(span[0]),
            float(span[1]),
        )
        * 1.02
    )

    camera_distance = float(
        max(
            float(span.max()) * 2.0,
            1.0,
        )
    )

    return OrthographicCamera(
        position=np.array(
            [
                center[0],
                center[1],
                center[2] + camera_distance,
            ],
            dtype=np.float64,
        ),
        target=center.astype(
            np.float64,
            copy=False,
        ),
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=view_width,
        image_width=image_width,
        image_height=image_height,
        near=0.0,
        far=2.0 * camera_distance,
    )


def render_gpu_frame(
    gpu_model: GpuGaussianModel,
    camera: OrthographicCamera,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    minimum_eigenvalue: float,
    minimum_pixel_variance: float,
    sigma_extent: float,
    beta: float,
    blob_sigma_pixels: float,
    normalize_gaussian_mass: bool,
    relative_density_threshold: float,
) -> tuple[
    GpuProjectedGaussians,
    GpuProjectedConditionalAttributes,
    GpuTileIntersections,
    GpuConditionalRenderResult,
]:
    projected = project_gaussians_orthographic_gpu(
        gpu_model,
        camera,
        minimum_eigenvalue=minimum_eigenvalue,
        minimum_pixel_variance=minimum_pixel_variance,
        sigma_extent=sigma_extent,
        beta=beta,
        blob_sigma_pixels=blob_sigma_pixels,
    )

    conditional = (
        project_conditional_attributes_orthographic_gpu(
            gpu_model,
            projected,
            camera,
        )
    )

    intersections = build_gpu_tile_intersections(
        projected,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    result = render_conditional_attribute_gpu(
        projected,
        conditional,
        intersections,
        normalize_gaussian_mass=normalize_gaussian_mass,
        relative_density_threshold=(
            relative_density_threshold
        ),
    )

    return (
        projected,
        conditional,
        intersections,
        result,
    )


def create_event() -> torch.cuda.Event:
    return torch.cuda.Event(
        enable_timing=True,
    )


def elapsed_seconds(
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> float:
    return (
        start.elapsed_time(end)
        / 1000.0
    )


def save_density_image(
    path: Path,
    density: np.ndarray,
) -> None:
    positive = density[
        np.isfinite(density)
        & (density > 0.0)
    ]

    if positive.size == 0:
        normalized = np.zeros_like(
            density,
            dtype=np.float64,
        )
    else:
        upper = float(
            np.percentile(
                positive,
                99.9,
            )
        )

        transformed = np.log1p(
            np.maximum(
                density,
                0.0,
            )
        )

        denominator = max(
            np.log1p(upper),
            1.0e-12,
        )

        normalized = np.clip(
            transformed / denominator,
            0.0,
            1.0,
        )

    plt.imsave(
        path,
        normalized,
        origin="lower",
        cmap="inferno",
    )


def save_attribute_image(
    path: Path,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    finite_values = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if finite_values.size == 0:
        raise RuntimeError(
            "No finite attribute values were rendered."
        )

    lower = float(
        np.percentile(
            finite_values,
            1.0,
        )
    )

    upper = float(
        np.percentile(
            finite_values,
            99.0,
        )
    )

    if upper <= lower:
        upper = lower + 1.0

    image = np.asarray(
        attribute,
        dtype=np.float64,
    ).copy()

    image[~valid_mask] = np.nan

    plt.imsave(
        path,
        image,
        origin="lower",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )

    return lower, upper


def save_combined_figure(
    path: Path,
    density: np.ndarray,
    attribute: np.ndarray,
    valid_mask: np.ndarray,
    *,
    attribute_name: str,
) -> None:
    positive_density = density[
        np.isfinite(density)
        & (density > 0.0)
    ]

    density_upper = (
        float(
            np.percentile(
                positive_density,
                99.9,
            )
        )
        if positive_density.size > 0
        else 1.0
    )

    finite_attribute = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if finite_attribute.size == 0:
        raise RuntimeError(
            "No finite attribute pixels are available."
        )

    attribute_lower = float(
        np.percentile(
            finite_attribute,
            1.0,
        )
    )

    attribute_upper = float(
        np.percentile(
            finite_attribute,
            99.0,
        )
    )

    if attribute_upper <= attribute_lower:
        attribute_upper = (
            attribute_lower + 1.0
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    density_image = axes[0].imshow(
        np.log1p(
            np.maximum(
                density,
                0.0,
            )
        ),
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=np.log1p(
            density_upper
        ),
    )

    axes[0].set_title(
        "Projected density"
    )

    axes[0].set_axis_off()

    figure.colorbar(
        density_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="log(1 + density)",
    )

    masked_attribute = np.ma.masked_where(
        ~valid_mask,
        attribute,
    )

    attribute_image = axes[1].imshow(
        masked_attribute,
        origin="lower",
        cmap="viridis",
        vmin=attribute_lower,
        vmax=attribute_upper,
    )

    axes[1].set_title(
        attribute_name
    )

    axes[1].set_axis_off()

    figure.colorbar(
        attribute_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label=attribute_name,
    )

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalize_gaussian_mass = (
        not args.unnormalized_mass
    )

    print("=" * 80)
    print(
        "Fully GPU-resident scientific conditional rendering"
    )
    print("=" * 80)

    print(
        f"Model                   : {args.model}"
    )

    print(
        f"Output                  : "
        f"{args.output.resolve()}"
    )

    print(
        f"Image                   : "
        f"{args.width} x {args.height}"
    )

    print(
        f"Tile size               : "
        f"{args.tile_size}"
    )

    print(
        f"Minimum eigenvalue      : "
        f"{args.minimum_eigenvalue}"
    )

    print(
        f"Minimum pixel variance  : "
        f"{args.minimum_pixel_variance}"
    )

    print(
        f"Sigma extent            : "
        f"{args.sigma_extent}"
    )

    print(
        f"Beta smoothing          : "
        f"{args.beta}"
    )

    print(
        f"Blob sigma              : "
        f"{args.blob} pixels"
    )

    print(
        f"Normalize mass          : "
        f"{normalize_gaussian_mass}"
    )

    print(
        f"Warm-up frames          : "
        f"{args.warmup_frames}"
    )

    print(
        f"Timed interactive frames: "
        f"{args.timed_frames}"
    )

    print(
        f"GPU                     : "
        f"{torch.cuda.get_device_name(device)}"
    )

    print()

    # ------------------------------------------------------------------
    # CPU model loading
    # ------------------------------------------------------------------

    load_start = perf_counter()

    model = GaussianModel.load(
        args.model
    )

    load_seconds = (
        perf_counter()
        - load_start
    )

    camera = build_camera(
        model,
        image_width=args.width,
        image_height=args.height,
    )

    # ------------------------------------------------------------------
    # One-time GPU upload and covariance stabilization
    # ------------------------------------------------------------------

    torch.cuda.synchronize(
        device
    )

    upload_start = perf_counter()

    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device=device,
        minimum_eigenvalue=(
            args.minimum_eigenvalue
        ),
    )

    gpu_model.synchronize()

    upload_seconds = (
        perf_counter()
        - upload_start
    )

    # ------------------------------------------------------------------
    # Cold first frame
    # ------------------------------------------------------------------

    cold_start = create_event()
    cold_end = create_event()

    cold_start.record()

    (
        projected,
        conditional,
        intersections,
        result,
    ) = render_gpu_frame(
        gpu_model,
        camera,
        image_width=args.width,
        image_height=args.height,
        tile_size=args.tile_size,
        minimum_eigenvalue=(
            args.minimum_eigenvalue
        ),
        minimum_pixel_variance=(
            args.minimum_pixel_variance
        ),
        sigma_extent=args.sigma_extent,
        beta=args.beta,
        blob_sigma_pixels=args.blob,
        normalize_gaussian_mass=(
            normalize_gaussian_mass
        ),
        relative_density_threshold=(
            args.relative_density_threshold
        ),
    )

    cold_end.record()
    cold_end.synchronize()

    cold_frame_seconds = elapsed_seconds(
        cold_start,
        cold_end,
    )

    # ------------------------------------------------------------------
    # Warm-up frames
    # ------------------------------------------------------------------

    warmup_total_seconds = 0.0

    if args.warmup_frames > 0:
        warmup_start = create_event()
        warmup_end = create_event()

        warmup_start.record()

        for _ in range(
            args.warmup_frames
        ):
            (
                projected,
                conditional,
                intersections,
                result,
            ) = render_gpu_frame(
                gpu_model,
                camera,
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
                blob_sigma_pixels=args.blob,
                normalize_gaussian_mass=(
                    normalize_gaussian_mass
                ),
                relative_density_threshold=(
                    args.relative_density_threshold
                ),
            )

        warmup_end.record()
        warmup_end.synchronize()

        warmup_total_seconds = (
            elapsed_seconds(
                warmup_start,
                warmup_end,
            )
        )

        warmup_average_seconds = (
            warmup_total_seconds
            / float(args.warmup_frames)
        )
    else:
        warmup_average_seconds = 0.0

    # ------------------------------------------------------------------
    # Steady-state interactive frame timing
    # ------------------------------------------------------------------

    projection_total = 0.0
    conditional_projection_total = 0.0
    intersection_total = 0.0
    render_total = 0.0
    frame_total = 0.0

    projection_start = create_event()
    projection_end = create_event()

    conditional_start = create_event()
    conditional_end = create_event()

    intersection_start = create_event()
    intersection_end = create_event()

    render_start = create_event()
    render_end = create_event()

    frame_start = create_event()
    frame_end = create_event()

    for _ in range(
        args.timed_frames
    ):
        frame_start.record()

        projection_start.record()

        projected = (
            project_gaussians_orthographic_gpu(
                gpu_model,
                camera,
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
            )
        )

        projection_end.record()

        conditional_start.record()

        conditional = (
            project_conditional_attributes_orthographic_gpu(
                gpu_model,
                projected,
                camera,
            )
        )

        conditional_end.record()

        intersection_start.record()

        intersections = (
            build_gpu_tile_intersections(
                projected,
                image_width=args.width,
                image_height=args.height,
                tile_size=args.tile_size,
            )
        )

        intersection_end.record()

        render_start.record()

        result = (
            render_conditional_attribute_gpu(
                projected,
                conditional,
                intersections,
                normalize_gaussian_mass=(
                    normalize_gaussian_mass
                ),
                relative_density_threshold=(
                    args.relative_density_threshold
                ),
            )
        )

        render_end.record()

        frame_end.record()
        frame_end.synchronize()

        projection_total += elapsed_seconds(
            projection_start,
            projection_end,
        )

        conditional_projection_total += (
            elapsed_seconds(
                conditional_start,
                conditional_end,
            )
        )

        intersection_total += elapsed_seconds(
            intersection_start,
            intersection_end,
        )

        render_total += elapsed_seconds(
            render_start,
            render_end,
        )

        frame_total += elapsed_seconds(
            frame_start,
            frame_end,
        )

    timed_frames = float(
        args.timed_frames
    )

    projection_average = (
        projection_total
        / timed_frames
    )

    conditional_projection_average = (
        conditional_projection_total
        / timed_frames
    )

    intersection_average = (
        intersection_total
        / timed_frames
    )

    render_average = (
        render_total
        / timed_frames
    )

    frame_average = (
        frame_total
        / timed_frames
    )

    interactive_fps = (
        1.0 / frame_average
        if frame_average > 0.0
        else float("inf")
    )

    # ------------------------------------------------------------------
    # Copy only the final rendered frame to CPU
    # ------------------------------------------------------------------

    density = (
        result
        .density_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    attribute_numerator = (
        result
        .attribute_numerator_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    attribute = (
        result
        .attribute_numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    valid_mask = (
        result
        .valid_mask_numpy()
        .astype(
            np.bool_,
            copy=False,
        )
    )

    # ------------------------------------------------------------------
    # Save numerical arrays
    # ------------------------------------------------------------------

    np.save(
        args.output / "density.npy",
        density.astype(
            np.float32
        ),
    )

    np.save(
        args.output
        / "attribute_numerator.npy",
        attribute_numerator.astype(
            np.float32
        ),
    )

    np.save(
        args.output
        / "conditional_attribute.npy",
        attribute.astype(
            np.float32
        ),
    )

    np.save(
        args.output / "valid_mask.npy",
        valid_mask,
    )

    # ------------------------------------------------------------------
    # Save images
    # ------------------------------------------------------------------

    save_density_image(
        args.output / "density.png",
        density,
    )

    (
        attribute_lower,
        attribute_upper,
    ) = save_attribute_image(
        args.output
        / "conditional_attribute.png",
        attribute,
        valid_mask,
    )

    attribute_name = (
        model.attribute_name
        if model.attribute_name is not None
        else "Conditional attribute"
    )

    save_combined_figure(
        args.output / "comparison.png",
        density,
        attribute,
        valid_mask,
        attribute_name=attribute_name,
    )

    finite_attribute_values = attribute[
        valid_mask
        & np.isfinite(attribute)
    ]

    if finite_attribute_values.size == 0:
        raise RuntimeError(
            "No valid finite attribute values were rendered."
        )

    # ------------------------------------------------------------------
    # Report model and rendering statistics
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Rendering result")
    print("=" * 80)

    print(
        f"Gaussians               : "
        f"{gpu_model.n_gaussians:,}"
    )

    print(
        f"Valid Gaussians         : "
        f"{projected.n_valid:,}"
    )

    print(
        f"Tile intersections      : "
        f"{intersections.n_intersections:,}"
    )

    print(
        f"Attribute               : "
        f"{attribute_name}"
    )

    print()

    print(
        f"Model load              : "
        f"{load_seconds:.6f} s"
    )

    print(
        f"One-time GPU upload     : "
        f"{upload_seconds:.6f} s"
    )

    print(
        f"GPU model memory        : "
        f"{gpu_model.memory_megabytes():.3f} MiB"
    )

    # ------------------------------------------------------------------
    # Performance report
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Performance")
    print("=" * 80)

    print(
        f"Cold first frame        : "
        f"{cold_frame_seconds * 1000.0:.3f} ms"
    )

    cold_fps = (
        1.0 / cold_frame_seconds
        if cold_frame_seconds > 0.0
        else float("inf")
    )

    print(
        f"Cold first-frame FPS    : "
        f"{cold_fps:.2f}"
    )

    print(
        f"Warm-up frames          : "
        f"{args.warmup_frames}"
    )

    if args.warmup_frames > 0:
        print(
            f"Warm-up total           : "
            f"{warmup_total_seconds * 1000.0:.3f} ms"
        )

        print(
            f"Warm-up average         : "
            f"{warmup_average_seconds * 1000.0:.3f} ms"
        )
    else:
        print(
            "Warm-up average         : "
            "not measured"
        )

    print(
        f"Timed interactive frames: "
        f"{args.timed_frames}"
    )

    print()
    print("Steady-state stage averages")
    print("-" * 80)

    print(
        f"GPU projection          : "
        f"{projection_average * 1000.0:.3f} ms"
    )

    print(
        f"Conditional projection  : "
        f"{conditional_projection_average * 1000.0:.3f} ms"
    )

    print(
        f"Tile intersections      : "
        f"{intersection_average * 1000.0:.3f} ms"
    )

    print(
        f"Conditional rendering   : "
        f"{render_average * 1000.0:.3f} ms"
    )

    print(
        f"Interactive frame       : "
        f"{frame_average * 1000.0:.3f} ms"
    )

    print(
        f"Interactive FPS         : "
        f"{interactive_fps:.2f}"
    )

    print()
    print("Final render-call internal timing")
    print("-" * 80)

    print(
        f"Render preparation      : "
        f"{result.preparation_seconds * 1000.0:.3f} ms"
    )

    print(
        f"CUDA rasterization      : "
        f"{result.rasterization_seconds * 1000.0:.3f} ms"
    )

    print(
        f"Attribute normalization : "
        f"{result.normalization_seconds * 1000.0:.3f} ms"
    )

    # ------------------------------------------------------------------
    # Image statistics
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Final-frame statistics")
    print("=" * 80)

    print(
        f"Density minimum         : "
        f"{density.min():.12g}"
    )

    print(
        f"Density maximum         : "
        f"{density.max():.12g}"
    )

    print(
        f"Density mean            : "
        f"{density.mean():.12g}"
    )

    print(
        f"Density sum             : "
        f"{density.sum():.12g}"
    )

    print(
        f"Density threshold       : "
        f"{result.density_threshold:.12g}"
    )

    print(
        f"Valid attribute pixels  : "
        f"{np.count_nonzero(valid_mask):,}"
    )

    print(
        f"Attribute minimum       : "
        f"{finite_attribute_values.min():.12g}"
    )

    print(
        f"Attribute maximum       : "
        f"{finite_attribute_values.max():.12g}"
    )

    print(
        f"Attribute mean          : "
        f"{finite_attribute_values.mean():.12g}"
    )

    print(
        f"Display range           : "
        f"[{attribute_lower:.6g}, "
        f"{attribute_upper:.6g}]"
    )

    # ------------------------------------------------------------------
    # Saved files
    # ------------------------------------------------------------------

    print()
    print("Saved files")
    print("-" * 80)

    for name in (
        "density.npy",
        "attribute_numerator.npy",
        "conditional_attribute.npy",
        "valid_mask.npy",
        "density.png",
        "conditional_attribute.png",
        "comparison.png",
    ):
        print(
            args.output / name
        )


if __name__ == "__main__":
    main()