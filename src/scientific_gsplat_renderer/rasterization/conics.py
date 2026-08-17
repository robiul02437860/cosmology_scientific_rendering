from __future__ import annotations

import torch
from torch import Tensor


def inverse_covariances_to_conics(
    inverse_covariances: Tensor,
    *,
    symmetry_tolerance: float = 1.0e-5,
) -> Tensor:
    """Convert 2D inverse covariance matrices into gsplat conic form.

    Parameters
    ----------
    inverse_covariances:
        Tensor with shape ``[..., 2, 2]``.

        Each matrix is expected to have the form

            [[a, b],
             [b, c]]

    symmetry_tolerance:
        Maximum permitted absolute difference between the two off-diagonal
        entries.

    Returns
    -------
    Tensor
        Compact conic coefficients with shape ``[..., 3]``:

            [a, b, c]

    Notes
    -----
    The scientific CUDA kernel evaluates

        sigma = 0.5 * (
            a * dx**2
            + 2 * b * dx * dy
            + c * dy**2
        )

    which is equivalent to

        0.5 * delta.T @ inverse_covariance @ delta.
    """
    if not isinstance(inverse_covariances, Tensor):
        raise TypeError(
            "inverse_covariances must be a torch.Tensor"
        )

    if inverse_covariances.ndim < 2:
        raise ValueError(
            "inverse_covariances must have at least two dimensions, "
            f"got shape {tuple(inverse_covariances.shape)}"
        )

    if inverse_covariances.shape[-2:] != (2, 2):
        raise ValueError(
            "inverse_covariances must have shape [..., 2, 2], "
            f"got {tuple(inverse_covariances.shape)}"
        )

    if not inverse_covariances.is_floating_point():
        raise TypeError(
            "inverse_covariances must use a floating-point dtype, "
            f"got {inverse_covariances.dtype}"
        )

    if symmetry_tolerance < 0.0:
        raise ValueError(
            "symmetry_tolerance must be non-negative, "
            f"got {symmetry_tolerance}"
        )

    upper = inverse_covariances[..., 0, 1]
    lower = inverse_covariances[..., 1, 0]

    maximum_asymmetry = torch.max(
        torch.abs(upper - lower)
    )

    if maximum_asymmetry.item() > symmetry_tolerance:
        raise ValueError(
            "inverse_covariances are not symmetric within tolerance: "
            f"maximum asymmetry={maximum_asymmetry.item():.6e}, "
            f"tolerance={symmetry_tolerance:.6e}"
        )

    # Average the two off-diagonal values to remove tiny numerical
    # asymmetries introduced by projection or matrix inversion.
    off_diagonal = 0.5 * (upper + lower)

    return torch.stack(
        (
            inverse_covariances[..., 0, 0],
            off_diagonal,
            inverse_covariances[..., 1, 1],
        ),
        dim=-1,
    ).contiguous()