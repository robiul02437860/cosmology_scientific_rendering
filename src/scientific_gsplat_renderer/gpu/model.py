# from __future__ import annotations

# from dataclasses import dataclass

# import torch
# from torch import Tensor

# from ..data.gaussian_model import GaussianModel


# @dataclass(slots=True)
# class GpuGaussianModel:
#     """
#     GPU-resident representation of a GaussianModel.

#     The model arrays are uploaded to the GPU once when this object is created.
#     Rendering code should reuse these tensors instead of repeatedly converting
#     NumPy arrays and copying them from CPU to GPU.

#     Shapes
#     ------
#     means:
#         (N, 3)

#     covariances:
#         (N, 3, 3)

#     weights:
#         (N,)

#     masses:
#         (N,)

#     attribute_means:
#         (N,)

#     position_attribute_cross_covariances:
#         (N, 3)

#     attribute_variances:
#         (N,)
#     """

#     means: Tensor
#     covariances: Tensor
#     weights: Tensor
#     masses: Tensor

#     attribute_means: Tensor
#     position_attribute_cross_covariances: Tensor
#     attribute_variances: Tensor

#     n_particles: int
#     attribute_name: str | None
#     box_size: float | None

#     @property
#     def device(self) -> torch.device:
#         return self.means.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.means.dtype

#     @property
#     def n_gaussians(self) -> int:
#         return int(self.means.shape[0])

#     @classmethod
#     def from_cpu(
#         cls,
#         model: GaussianModel,
#         *,
#         device: str | torch.device = "cuda",
#         dtype: torch.dtype = torch.float32,
#     ) -> GpuGaussianModel:
#         """
#         Upload a CPU GaussianModel to a device.

#         This method performs all NumPy-to-Torch conversion and CPU-to-GPU
#         transfer once. The returned tensors are contiguous and ready for CUDA
#         projection and rasterization.
#         """
#         resolved_device = torch.device(device)

#         if resolved_device.type == "cuda" and not torch.cuda.is_available():
#             raise RuntimeError(
#                 "CUDA was requested, but torch.cuda.is_available() is False."
#             )

#         means = torch.as_tensor(
#             model.means,
#             dtype=dtype,
#             device=resolved_device,
#         ).contiguous()

#         covariances = torch.as_tensor(
#             model.covariances,
#             dtype=dtype,
#             device=resolved_device,
#         ).contiguous()

#         weights = torch.as_tensor(
#             model.weights,
#             dtype=dtype,
#             device=resolved_device,
#         ).contiguous()

#         attribute_means = torch.as_tensor(
#             model.attribute_means,
#             dtype=dtype,
#             device=resolved_device,
#         ).reshape(-1).contiguous()

#         cross_covariances = torch.as_tensor(
#             model.position_attribute_cross_covariances,
#             dtype=dtype,
#             device=resolved_device,
#         ).contiguous()

#         attribute_variances = torch.as_tensor(
#             model.attribute_variances,
#             dtype=dtype,
#             device=resolved_device,
#         ).reshape(-1).contiguous()

#         n_particles = int(model.n_particles)

#         # global_weights are probabilities. Multiplying by the original
#         # particle count gives the physical particle mass represented by each
#         # Gaussian.
#         masses = (weights * float(n_particles)).contiguous()

#         gpu_model = cls(
#             means=means,
#             covariances=covariances,
#             weights=weights,
#             masses=masses,
#             attribute_means=attribute_means,
#             position_attribute_cross_covariances=cross_covariances,
#             attribute_variances=attribute_variances,
#             n_particles=n_particles,
#             attribute_name=model.attribute_name,
#             box_size=model.box_size,
#         )

#         gpu_model.validate()
#         return gpu_model

#     def validate(self) -> None:
#         """Validate tensor shapes, devices, dtypes, and memory layout."""
#         n = self.n_gaussians

#         expected_shapes = {
#             "means": (n, 3),
#             "covariances": (n, 3, 3),
#             "weights": (n,),
#             "masses": (n,),
#             "attribute_means": (n,),
#             "position_attribute_cross_covariances": (n, 3),
#             "attribute_variances": (n,),
#         }

#         tensors = {
#             "means": self.means,
#             "covariances": self.covariances,
#             "weights": self.weights,
#             "masses": self.masses,
#             "attribute_means": self.attribute_means,
#             "position_attribute_cross_covariances":
#                 self.position_attribute_cross_covariances,
#             "attribute_variances": self.attribute_variances,
#         }

#         for name, tensor in tensors.items():
#             expected = expected_shapes[name]

#             if tuple(tensor.shape) != expected:
#                 raise ValueError(
#                     f"{name} must have shape {expected}, "
#                     f"got {tuple(tensor.shape)}."
#                 )

#             if tensor.device != self.device:
#                 raise ValueError(
#                     f"{name} is on {tensor.device}, but means are on "
#                     f"{self.device}."
#                 )

#             if tensor.dtype != self.dtype:
#                 raise ValueError(
#                     f"{name} has dtype {tensor.dtype}, but means use "
#                     f"{self.dtype}."
#                 )

#             if not tensor.is_contiguous():
#                 raise ValueError(f"{name} must be contiguous.")

#         if self.n_particles <= 0:
#             raise ValueError(
#                 f"n_particles must be positive, got {self.n_particles}."
#             )

#         if not torch.isfinite(self.means).all():
#             raise ValueError("means contains NaN or infinite values.")

#         if not torch.isfinite(self.covariances).all():
#             raise ValueError("covariances contains NaN or infinite values.")

#         if not torch.isfinite(self.weights).all():
#             raise ValueError("weights contains NaN or infinite values.")

#         if not torch.isfinite(self.attribute_means).all():
#             raise ValueError(
#                 "attribute_means contains NaN or infinite values."
#             )

#         if not torch.isfinite(
#             self.position_attribute_cross_covariances
#         ).all():
#             raise ValueError(
#                 "position_attribute_cross_covariances contains NaN or "
#                 "infinite values."
#             )

#         if not torch.isfinite(self.attribute_variances).all():
#             raise ValueError(
#                 "attribute_variances contains NaN or infinite values."
#             )

#     def synchronize(self) -> None:
#         """Wait for queued CUDA work to finish."""
#         if self.device.type == "cuda":
#             torch.cuda.synchronize(self.device)

#     def memory_bytes(self) -> int:
#         """Return total tensor storage used by the GPU model."""
#         tensors = (
#             self.means,
#             self.covariances,
#             self.weights,
#             self.masses,
#             self.attribute_means,
#             self.position_attribute_cross_covariances,
#             self.attribute_variances,
#         )

#         return sum(
#             tensor.numel() * tensor.element_size()
#             for tensor in tensors
#         )

#     def memory_megabytes(self) -> float:
#         return self.memory_bytes() / (1024.0**2)


from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..data.gaussian_model import GaussianModel


@dataclass(slots=True)
class GpuGaussianModel:
    """GPU-resident scientific Gaussian model.

    The CPU model is uploaded once. World-space covariance stabilization is
    also performed once and cached so it is not repeated during every
    interactive frame.

    Shapes
    ------
    means
        ``(N, 3)``

    covariances
        ``(N, 3, 3)`` original world-space covariance matrices.

    stabilized_covariances
        ``(N, 3, 3)`` positive-definite world-space covariance matrices
        whose eigenvalues have been clamped once during initialization.

    weights
        ``(N,)`` normalized global mixture weights.

    masses
        ``(N,)`` represented particle masses.

    attribute_means
        ``(N,)``

    position_attribute_cross_covariances
        ``(N, 3)``

    attribute_variances
        ``(N,)``
    """

    means: Tensor

    covariances: Tensor
    stabilized_covariances: Tensor

    weights: Tensor
    masses: Tensor

    attribute_means: Tensor
    position_attribute_cross_covariances: Tensor
    attribute_variances: Tensor

    n_particles: int
    attribute_name: str | None
    box_size: float | None

    stabilization_minimum_eigenvalue: float

    @property
    def device(self) -> torch.device:
        """Return the tensor device."""

        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype."""

        return self.means.dtype

    @property
    def n_gaussians(self) -> int:
        """Return the number of Gaussian components."""

        return int(self.means.shape[0])

    @classmethod
    def from_cpu(
        cls,
        model: GaussianModel,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        minimum_eigenvalue: float = 1.0e-6,
    ) -> GpuGaussianModel:
        """Upload a CPU model and cache stabilized covariance matrices.

        Parameters
        ----------
        model
            Source CPU Gaussian model.

        device
            Target device.

        dtype
            Floating-point tensor dtype.

        minimum_eigenvalue
            Minimum eigenvalue used to stabilize world-space covariance
            matrices. This operation is performed once during initialization.

        Returns
        -------
        GpuGaussianModel
            Persistent GPU-resident model.
        """

        if not math.isfinite(minimum_eigenvalue):
            raise ValueError(
                "minimum_eigenvalue must be finite."
            )

        if minimum_eigenvalue <= 0.0:
            raise ValueError(
                "minimum_eigenvalue must be positive, "
                f"got {minimum_eigenvalue}."
            )

        resolved_device = torch.device(
            device
        )

        if (
            resolved_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but "
                "torch.cuda.is_available() is False."
            )

        means = torch.as_tensor(
            model.means,
            dtype=dtype,
            device=resolved_device,
        ).contiguous()

        covariances = torch.as_tensor(
            model.covariances,
            dtype=dtype,
            device=resolved_device,
        ).contiguous()

        weights = torch.as_tensor(
            model.weights,
            dtype=dtype,
            device=resolved_device,
        ).reshape(-1).contiguous()

        attribute_means = torch.as_tensor(
            model.attribute_means,
            dtype=dtype,
            device=resolved_device,
        ).reshape(-1).contiguous()

        cross_covariances = torch.as_tensor(
            model.position_attribute_cross_covariances,
            dtype=dtype,
            device=resolved_device,
        ).contiguous()

        attribute_variances = torch.as_tensor(
            model.attribute_variances,
            dtype=dtype,
            device=resolved_device,
        ).reshape(-1).contiguous()

        n_particles = int(
            model.n_particles
        )

        masses = (
            weights
            * float(n_particles)
        ).contiguous()

        stabilized_covariances = (
            cls._stabilize_covariances(
                covariances,
                minimum_eigenvalue=(
                    minimum_eigenvalue
                ),
            )
        )

        gpu_model = cls(
            means=means,
            covariances=covariances,
            stabilized_covariances=(
                stabilized_covariances
            ),
            weights=weights,
            masses=masses,
            attribute_means=attribute_means,
            position_attribute_cross_covariances=(
                cross_covariances
            ),
            attribute_variances=(
                attribute_variances
            ),
            n_particles=n_particles,
            attribute_name=model.attribute_name,
            box_size=model.box_size,
            stabilization_minimum_eigenvalue=float(
                minimum_eigenvalue
            ),
        )

        gpu_model.validate()

        return gpu_model
    
    @staticmethod
    def _stabilize_covariances(
        covariances: Tensor,
        *,
        minimum_eigenvalue: float,
        chunk_size: int = 65_536,
        ) -> Tensor:
        """Clamp covariance eigenvalues in manageable GPU chunks.

        The model can contain millions of covariance matrices. Processing
        them in one batched ``torch.linalg.eigh`` call may exceed cuSOLVER's
        practical batch limits. Chunking keeps the operation robust while
        preserving the same numerical result.
        """

        if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
            raise ValueError(
                "covariances must have shape (N,3,3), "
                f"got {tuple(covariances.shape)}."
            )

        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {chunk_size}."
            )

        if not torch.isfinite(covariances).all():
            raise ValueError(
                "covariances contains NaN or infinite values."
            )

        symmetric = 0.5 * (
            covariances
            + covariances.transpose(-1, -2)
        )

        stabilized = torch.empty_like(
            symmetric
        )

        total = int(
            symmetric.shape[0]
        )

        for start in range(
            0,
            total,
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                total,
            )

            block = symmetric[
                start:stop
            ]

            eigenvalues, eigenvectors = (
                torch.linalg.eigh(
                    block
                )
            )

            clamped_eigenvalues = torch.clamp(
                eigenvalues,
                min=float(
                    minimum_eigenvalue
                ),
            )

            stabilized_block = torch.matmul(
                torch.matmul(
                    eigenvectors,
                    torch.diag_embed(
                        clamped_eigenvalues
                    ),
                ),
                eigenvectors.transpose(
                    -1,
                    -2,
                ),
            )

            stabilized[
                start:stop
            ] = 0.5 * (
                stabilized_block
                + stabilized_block.transpose(
                    -1,
                    -2,
                )
            )

        return stabilized.contiguous()

    # @staticmethod
    # def _stabilize_covariances(
    #     covariances: Tensor,
    #     *,
    #     minimum_eigenvalue: float,
    # ) -> Tensor:
    #     """Clamp covariance eigenvalues and reconstruct the matrices.

    #     This operation is intentionally performed during model initialization
    #     rather than inside the per-frame projection function.
    #     """

    #     symmetric = 0.5 * (
    #         covariances
    #         + covariances.transpose(-1, -2)
    #     )

    #     eigenvalues, eigenvectors = (
    #         torch.linalg.eigh(
    #             symmetric
    #         )
    #     )

    #     clamped_eigenvalues = torch.clamp(
    #         eigenvalues,
    #         min=float(minimum_eigenvalue),
    #     )

    #     stabilized = torch.matmul(
    #         torch.matmul(
    #             eigenvectors,
    #             torch.diag_embed(
    #                 clamped_eigenvalues
    #             ),
    #         ),
    #         eigenvectors.transpose(-1, -2),
    #     )

    #     stabilized = 0.5 * (
    #         stabilized
    #         + stabilized.transpose(-1, -2)
    #     )

    #     return stabilized.contiguous()

    def validate(self) -> None:
        """Validate tensor shapes, devices, dtypes, and values."""

        n = self.n_gaussians

        expected_shapes = {
            "means": (
                n,
                3,
            ),
            "covariances": (
                n,
                3,
                3,
            ),
            "stabilized_covariances": (
                n,
                3,
                3,
            ),
            "weights": (
                n,
            ),
            "masses": (
                n,
            ),
            "attribute_means": (
                n,
            ),
            "position_attribute_cross_covariances": (
                n,
                3,
            ),
            "attribute_variances": (
                n,
            ),
        }

        tensors = {
            "means": self.means,
            "covariances": self.covariances,
            "stabilized_covariances": (
                self.stabilized_covariances
            ),
            "weights": self.weights,
            "masses": self.masses,
            "attribute_means": (
                self.attribute_means
            ),
            "position_attribute_cross_covariances": (
                self.position_attribute_cross_covariances
            ),
            "attribute_variances": (
                self.attribute_variances
            ),
        }

        for name, tensor in tensors.items():
            expected_shape = (
                expected_shapes[name]
            )

            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape "
                    f"{expected_shape}, got "
                    f"{tuple(tensor.shape)}."
                )

            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, "
                    f"but means are on {self.device}."
                )

            if tensor.dtype != self.dtype:
                raise ValueError(
                    f"{name} has dtype {tensor.dtype}, "
                    f"but means use {self.dtype}."
                )

            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous."
                )

            if not torch.isfinite(
                tensor
            ).all():
                raise ValueError(
                    f"{name} contains NaN or "
                    "infinite values."
                )

        if self.n_particles <= 0:
            raise ValueError(
                "n_particles must be positive, "
                f"got {self.n_particles}."
            )

        if not math.isfinite(
            self.stabilization_minimum_eigenvalue
        ):
            raise ValueError(
                "stabilization_minimum_eigenvalue "
                "must be finite."
            )

        if (
            self.stabilization_minimum_eigenvalue
            <= 0.0
        ):
            raise ValueError(
                "stabilization_minimum_eigenvalue "
                "must be positive."
            )

        if torch.any(
            self.weights < 0.0
        ):
            raise ValueError(
                "weights must be nonnegative."
            )

        if torch.any(
            self.masses < 0.0
        ):
            raise ValueError(
                "masses must be nonnegative."
            )

    def require_stabilization(
        self,
        minimum_eigenvalue: float,
        *,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-12,
    ) -> None:
        """Require a projector to use the cached stabilization threshold.

        Interactive projection must not silently request a different world
        covariance threshold, because that would require recomputing the
        stabilized covariance matrices.
        """

        if not math.isfinite(
            minimum_eigenvalue
        ):
            raise ValueError(
                "minimum_eigenvalue must be finite."
            )

        matches = math.isclose(
            float(minimum_eigenvalue),
            self.stabilization_minimum_eigenvalue,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )

        if not matches:
            raise ValueError(
                "The requested minimum_eigenvalue does not "
                "match the covariance stabilization cached in "
                "GpuGaussianModel. "
                f"Requested {minimum_eigenvalue}, cached "
                f"{self.stabilization_minimum_eigenvalue}. "
                "Create a new GpuGaussianModel with the desired "
                "minimum_eigenvalue."
            )

    def synchronize(self) -> None:
        """Wait for queued CUDA work to complete."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

    def memory_bytes(self) -> int:
        """Return total tensor storage used by the persistent model."""

        tensors = (
            self.means,
            self.covariances,
            self.stabilized_covariances,
            self.weights,
            self.masses,
            self.attribute_means,
            self.position_attribute_cross_covariances,
            self.attribute_variances,
        )

        return sum(
            tensor.numel()
            * tensor.element_size()
            for tensor in tensors
        )

    def memory_megabytes(self) -> float:
        """Return total tensor storage in MiB."""

        return (
            self.memory_bytes()
            / (1024.0**2)
        )