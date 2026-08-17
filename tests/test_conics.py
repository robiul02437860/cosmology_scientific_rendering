from __future__ import annotations

import pytest
import torch

from scientific_gsplat_renderer.rasterization import (
    inverse_covariances_to_conics,
)


def test_inverse_covariances_to_conics() -> None:
    inverse_covariances = torch.tensor(
        [
            [
                [2.0, 0.5],
                [0.5, 3.0],
            ],
            [
                [4.0, -0.25],
                [-0.25, 5.0],
            ],
        ],
        dtype=torch.float32,
    )

    conics = inverse_covariances_to_conics(
        inverse_covariances
    )

    expected = torch.tensor(
        [
            [2.0, 0.5, 3.0],
            [4.0, -0.25, 5.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(conics, expected)
    assert conics.is_contiguous()


def test_small_asymmetry_is_averaged() -> None:
    inverse_covariances = torch.tensor(
        [
            [
                [2.0, 0.500001],
                [0.499999, 3.0],
            ]
        ],
        dtype=torch.float32,
    )

    conics = inverse_covariances_to_conics(
        inverse_covariances,
        symmetry_tolerance=1.0e-4,
    )

    expected = torch.tensor(
        [[2.0, 0.5, 3.0]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        conics,
        expected,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_large_asymmetry_is_rejected() -> None:
    inverse_covariances = torch.tensor(
        [
            [
                [2.0, 0.8],
                [0.2, 3.0],
            ]
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="not symmetric",
    ):
        inverse_covariances_to_conics(
            inverse_covariances,
            symmetry_tolerance=1.0e-5,
        )


def test_invalid_shape_is_rejected() -> None:
    inverse_covariances = torch.zeros(
        (4, 3, 3),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"\[\.\.\., 2, 2\]",
    ):
        inverse_covariances_to_conics(
            inverse_covariances
        )


def test_quadratic_form_matches_matrix_expression() -> None:
    inverse_covariance = torch.tensor(
        [
            [2.0, 0.5],
            [0.5, 3.0],
        ],
        dtype=torch.float64,
    )

    delta = torch.tensor(
        [1.5, -0.75],
        dtype=torch.float64,
    )

    conic = inverse_covariances_to_conics(
        inverse_covariance
    )

    matrix_sigma = (
        0.5
        * delta
        @ inverse_covariance
        @ delta
    )

    a, b, c = conic

    compact_sigma = 0.5 * (
        a * delta[0] * delta[0]
        + c * delta[1] * delta[1]
    ) + b * delta[0] * delta[1]

    torch.testing.assert_close(
        compact_sigma,
        matrix_sigma,
    )