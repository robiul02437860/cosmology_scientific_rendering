from __future__ import annotations

import numpy as np
import pytest
import torch

from scientific_gsplat_renderer.projection import ProjectedGaussians
from scientific_gsplat_renderer.rendering import (
    render_projected_gaussians_cuda,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def test_end_to_end_single_gaussian() -> None:
    projected = ProjectedGaussians(
        means_camera=np.array(
            [[0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        means_pixel=np.array(
            [[0.5, 0.5]],
            dtype=np.float32,
        ),
        covariances_camera=np.array(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        covariances_pixel=np.array(
            [[[1.0, 0.0],
              [0.0, 1.0]]],
            dtype=np.float32,
        ),
        inverse_covariances_pixel=np.array(
            [[[1.0, 0.0],
              [0.0, 1.0]]],
            dtype=np.float32,
        ),
        radii_pixel=np.array(
            [1.0],
            dtype=np.float32,
        ),
        depths=np.array(
            [1.0],
            dtype=np.float32,
        ),
        masses=np.array(
            [2.0],
            dtype=np.float32,
        ),
        valid=np.array(
            [True],
            dtype=np.bool_,
        ),
    )

    attribute_means = np.array(
        [12.0],
        dtype=np.float32,
    )

    result = render_projected_gaussians_cuda(
        projected,
        image_width=1,
        image_height=1,
        attribute_means=attribute_means,
        tile_size=1,
        normalize_gaussian_mass=False,
    )

    assert result.density.shape == (1, 1)
    assert result.attribute is not None
    assert result.attribute.shape == (1, 1)

    torch.testing.assert_close(
        result.density,
        torch.tensor(
            [[2.0]],
            dtype=torch.float32,
            device="cuda",
        ),
    )

    torch.testing.assert_close(
        result.attribute,
        torch.tensor(
            [[12.0]],
            dtype=torch.float32,
            device="cuda",
        ),
    )

    assert result.n_input_gaussians == 1
    assert result.n_rendered_gaussians == 1
    assert result.n_intersections == 1