from __future__ import annotations

import pytest
import torch

from scientific_gsplat_renderer.rasterization.gsplat_scientific import (
    mass_to_normalized_amplitude,
    scientific_rasterize_to_pixels,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def test_scientific_rasterizer_is_additive() -> None:
    device = torch.device("cuda")

    means2d = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.float32,
        device=device,
    )

    conics = torch.tensor(
        [[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )

    values = torch.tensor(
        [[[10.0], [20.0]]],
        dtype=torch.float32,
        device=device,
    )

    amplitudes = torch.tensor(
        [[2.0, 3.0]],
        dtype=torch.float32,
        device=device,
    )

    tile_offsets = torch.tensor(
        [[[0]]],
        dtype=torch.int32,
        device=device,
    )

    flatten_ids = torch.tensor(
        [0, 1],
        dtype=torch.int32,
        device=device,
    )

    result = scientific_rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        values=values,
        amplitudes=amplitudes,
        image_width=1,
        image_height=1,
        tile_size=1,
        tile_offsets=tile_offsets,
        flatten_ids=flatten_ids,
    )

    expected_values = torch.tensor(
        [[[[80.0]]]],
        dtype=torch.float32,
        device=device,
    )

    expected_weights = torch.tensor(
        [[[[5.0]]]],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(
        result.accumulated_values,
        expected_values,
    )

    torch.testing.assert_close(
        result.accumulated_weights,
        expected_weights,
    )

    assert result.last_intersection_ids.item() == 1


def test_intensive_attribute_normalization() -> None:
    device = torch.device("cuda")

    means2d = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.float32,
        device=device,
    )

    conics = torch.tensor(
        [[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )

    attribute_values = torch.tensor(
        [[[10.0], [20.0]]],
        dtype=torch.float32,
        device=device,
    )

    masses = torch.tensor(
        [[2.0, 3.0]],
        dtype=torch.float32,
        device=device,
    )

    tile_offsets = torch.tensor(
        [[[0]]],
        dtype=torch.int32,
        device=device,
    )

    flatten_ids = torch.tensor(
        [0, 1],
        dtype=torch.int32,
        device=device,
    )

    result = scientific_rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        values=attribute_values,
        amplitudes=masses,
        image_width=1,
        image_height=1,
        tile_size=1,
        tile_offsets=tile_offsets,
        flatten_ids=flatten_ids,
    )

    attribute = result.normalized_values()

    expected = torch.tensor(
        [[[[16.0]]]],
        dtype=torch.float32,
        device=device,
    )

    # Weighted mean:
    # (10*2 + 20*3) / (2+3) = 16
    torch.testing.assert_close(attribute, expected)


def test_mass_normalization_for_identity_conic() -> None:
    device = torch.device("cuda")

    conics = torch.tensor(
        [[[1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )

    masses = torch.tensor(
        [[2.0]],
        dtype=torch.float32,
        device=device,
    )

    amplitudes = mass_to_normalized_amplitude(
        masses,
        conics,
    )

    expected = torch.tensor(
        [[2.0 / (2.0 * torch.pi)]],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(amplitudes, expected)