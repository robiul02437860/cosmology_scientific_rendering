from __future__ import annotations

import numpy as np
import pytest

from scientific_gsplat_renderer.projection.conditional_attribute import (
    ProjectedConditionalAttributes,
)
from scientific_gsplat_renderer.projection.orthographic import (
    ProjectedGaussians,
)
from scientific_gsplat_renderer.rasterization.conditional_cpu import (
    rasterize_conditional_attribute_cpu,
)


def test_single_gaussian_conditional_gradient() -> None:
    projected = ProjectedGaussians(
        means_camera=np.array(
            [[0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        means_pixel=np.array(
            [[5.5, 7.5]],
            dtype=np.float32,
        ),
        covariances_camera=np.array(
            [np.eye(3)],
            dtype=np.float32,
        ),
        covariances_pixel=np.array(
            [np.eye(2)],
            dtype=np.float32,
        ),
        inverse_covariances_pixel=np.array(
            [np.eye(2)],
            dtype=np.float32,
        ),
        radii_pixel=np.array(
            [4.0],
            dtype=np.float32,
        ),
        depths=np.array(
            [1.0],
            dtype=np.float32,
        ),
        masses=np.array(
            [1.0],
            dtype=np.float32,
        ),
        valid=np.array(
            [True],
            dtype=np.bool_,
        ),
    )

    conditional = ProjectedConditionalAttributes(
        means=np.array(
            [10.0],
            dtype=np.float32,
        ),
        cross_covariances_pixel=np.array(
            [[2.0, -3.0]],
            dtype=np.float32,
        ),
        slopes_pixel=np.array(
            [[2.0, -3.0]],
            dtype=np.float32,
        ),
        valid=np.array(
            [True],
            dtype=np.bool_,
        ),
    )

    result = rasterize_conditional_attribute_cpu(
        projected,
        conditional,
        image_width=12,
        image_height=12,
        exponent_cutoff=9.0,
        normalize_gaussian_mass=False,
        relative_density_threshold=0.0,
    )

    # Pixel [7, 5] has center (5.5, 7.5), exactly at the
    # Gaussian mean.
    assert result.attribute[7, 5] == pytest.approx(
        10.0,
        abs=1.0e-6,
    )

    # Pixel [7, 6] has center (6.5, 7.5):
    # 10 + 2*(1) - 3*(0) = 12.
    assert result.attribute[7, 6] == pytest.approx(
        12.0,
        abs=1.0e-6,
    )

    # Pixel [8, 5] has center (5.5, 8.5):
    # 10 + 2*(0) - 3*(1) = 7.
    assert result.attribute[8, 5] == pytest.approx(
        7.0,
        abs=1.0e-6,
    )


def test_zero_slope_reproduces_attribute_mean() -> None:
    projected = ProjectedGaussians(
        means_camera=np.array(
            [[0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        means_pixel=np.array(
            [[2.5, 2.5]],
            dtype=np.float32,
        ),
        covariances_camera=np.array(
            [np.eye(3)],
            dtype=np.float32,
        ),
        covariances_pixel=np.array(
            [np.eye(2)],
            dtype=np.float32,
        ),
        inverse_covariances_pixel=np.array(
            [np.eye(2)],
            dtype=np.float32,
        ),
        radii_pixel=np.array(
            [3.0],
            dtype=np.float32,
        ),
        depths=np.array(
            [1.0],
            dtype=np.float32,
        ),
        masses=np.array(
            [1.0],
            dtype=np.float32,
        ),
        valid=np.array(
            [True],
            dtype=np.bool_,
        ),
    )

    conditional = ProjectedConditionalAttributes(
        means=np.array(
            [12.0],
            dtype=np.float32,
        ),
        cross_covariances_pixel=np.zeros(
            (1, 2),
            dtype=np.float32,
        ),
        slopes_pixel=np.zeros(
            (1, 2),
            dtype=np.float32,
        ),
        valid=np.array(
            [True],
            dtype=np.bool_,
        ),
    )

    result = rasterize_conditional_attribute_cpu(
        projected,
        conditional,
        image_width=6,
        image_height=6,
        normalize_gaussian_mass=False,
        relative_density_threshold=0.0,
    )

    assert np.allclose(
        result.attribute[result.valid_mask],
        12.0,
    )
