import numpy as np
import pytest

from scientific_gsplat_renderer import (
    ProjectedGaussians,
    TileIntersections,
    build_tile_intersections,
)


def make_projected(
    *,
    means_pixel: np.ndarray,
    radii_pixel: np.ndarray,
    valid: np.ndarray | None = None,
) -> ProjectedGaussians:
    means_pixel = np.asarray(
        means_pixel,
        dtype=np.float64,
    )

    radii_pixel = np.asarray(
        radii_pixel,
        dtype=np.float64,
    )

    n_gaussians = means_pixel.shape[0]

    if valid is None:
        valid = np.ones(
            n_gaussians,
            dtype=np.bool_,
        )

    means_camera = np.zeros(
        (n_gaussians, 3),
        dtype=np.float64,
    )

    means_camera[:, 2] = 10.0

    covariances_pixel = np.repeat(
        np.eye(
            2,
            dtype=np.float64,
        )[None, :, :],
        repeats=n_gaussians,
        axis=0,
    )

    covariances_camera = np.repeat(
        np.eye(
            3,
            dtype=np.float64,
        )[None, :, :],
        repeats=n_gaussians,
        axis=0,
    )

    inverse_covariances_pixel = (
        covariances_pixel.copy()
    )

    return ProjectedGaussians(
        means_camera=means_camera,
        means_pixel=means_pixel,
        covariances_camera=covariances_camera,
        covariances_pixel=covariances_pixel,
        inverse_covariances_pixel=(
            inverse_covariances_pixel
        ),
        radii_pixel=radii_pixel,
        depths=np.full(
            n_gaussians,
            10.0,
            dtype=np.float64,
        ),
        masses=np.ones(
            n_gaussians,
            dtype=np.float64,
        ),
        valid=np.asarray(
            valid,
            dtype=np.bool_,
        ),
    )


def test_tile_grid_dimensions() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [[8.0, 8.0]]
        ),
        radii_pixel=np.array([2.0]),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=35,
        image_height=33,
        tile_size=16,
    )

    assert intersections.tiles_x == 3
    assert intersections.tiles_y == 3
    assert intersections.n_tiles == 9


def test_gaussian_inside_single_tile() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [[8.0, 8.0]]
        ),
        radii_pixel=np.array([2.0]),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    np.testing.assert_array_equal(
        intersections.gaussian_indices_for_tile(
            0,
            0,
        ),
        [0],
    )

    assert intersections.n_intersections == 1


def test_gaussian_intersects_four_tiles() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [[16.0, 16.0]]
        ),
        radii_pixel=np.array([2.0]),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    assert intersections.n_intersections == 4

    for tile_y in range(2):
        for tile_x in range(2):
            np.testing.assert_array_equal(
                intersections.gaussian_indices_for_tile(
                    tile_x,
                    tile_y,
                ),
                [0],
            )


def test_multiple_gaussians_in_same_tile() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [
                [4.0, 4.0],
                [8.0, 8.0],
                [12.0, 12.0],
            ]
        ),
        radii_pixel=np.array(
            [1.0, 1.0, 1.0]
        ),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    np.testing.assert_array_equal(
        intersections.gaussian_indices_for_tile(
            0,
            0,
        ),
        [0, 1, 2],
    )


def test_invalid_gaussian_is_excluded() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [
                [4.0, 4.0],
                [8.0, 8.0],
            ]
        ),
        radii_pixel=np.array(
            [1.0, 1.0]
        ),
        valid=np.array(
            [True, False]
        ),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    np.testing.assert_array_equal(
        intersections.gaussian_indices,
        [0],
    )


def test_gaussian_outside_image_produces_no_intersection() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [[100.0, 100.0]]
        ),
        radii_pixel=np.array([2.0]),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=32,
        tile_size=16,
    )

    assert intersections.n_intersections == 0


def test_counts_and_offsets() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [
                [4.0, 4.0],
                [20.0, 4.0],
            ]
        ),
        radii_pixel=np.array(
            [1.0, 1.0]
        ),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=16,
        tile_size=16,
    )

    np.testing.assert_array_equal(
        intersections.counts_per_tile,
        [1, 1],
    )

    np.testing.assert_array_equal(
        intersections.offsets,
        [0, 1, 2],
    )


def test_maximum_gaussians_limits_selection() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [
                [4.0, 4.0],
                [8.0, 8.0],
                [12.0, 12.0],
            ]
        ),
        radii_pixel=np.ones(3),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=16,
        image_height=16,
        tile_size=16,
        maximum_gaussians=2,
    )

    assert intersections.n_intersections == 2
    assert len(
        np.unique(
            intersections.gaussian_indices
        )
    ) == 2


def test_tile_id_and_statistics() -> None:
    projected = make_projected(
        means_pixel=np.array(
            [
                [4.0, 4.0],
                [20.0, 4.0],
            ]
        ),
        radii_pixel=np.array(
            [1.0, 1.0]
        ),
    )

    intersections = build_tile_intersections(
        projected,
        image_width=32,
        image_height=16,
        tile_size=16,
    )

    assert isinstance(
        intersections,
        TileIntersections,
    )

    assert intersections.tile_id(0, 0) == 0
    assert intersections.tile_id(1, 0) == 1

    assert (
        intersections.maximum_gaussians_per_tile
        == 1
    )

    assert (
        intersections.mean_gaussians_per_tile
        == pytest.approx(1.0)
    )


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("image_width", 0, ValueError),
        ("image_width", -1, ValueError),
        ("image_width", 32.0, TypeError),
        ("image_height", 0, ValueError),
        ("image_height", -1, ValueError),
        ("image_height", 32.0, TypeError),
        ("tile_size", 0, ValueError),
        ("tile_size", -1, ValueError),
        ("tile_size", 16.0, TypeError),
        ("maximum_gaussians", 0, ValueError),
        ("maximum_gaussians", -1, ValueError),
        ("maximum_gaussians", 2.5, TypeError),
    ],
)
def test_rejects_invalid_build_parameters(
    name: str,
    value: object,
    exception: type[Exception],
) -> None:
    projected = make_projected(
        means_pixel=np.array(
            [[8.0, 8.0]]
        ),
        radii_pixel=np.array([2.0]),
    )

    parameters: dict[str, object] = {
        "image_width": 32,
        "image_height": 32,
        "tile_size": 16,
        "maximum_gaussians": None,
    }

    parameters[name] = value

    with pytest.raises(exception):
        build_tile_intersections(
            projected,
            **parameters,  # type: ignore[arg-type]
        )


def test_rejects_invalid_tile_coordinate() -> None:
    intersections = TileIntersections(
        tile_size=16,
        tiles_x=2,
        tiles_y=2,
        offsets=np.zeros(
            5,
            dtype=np.int64,
        ),
        gaussian_indices=np.empty(
            0,
            dtype=np.int64,
        ),
    )

    with pytest.raises(ValueError):
        intersections.gaussian_indices_for_tile(
            2,
            0,
        )

    with pytest.raises(ValueError):
        intersections.gaussian_indices_for_tile(
            0,
            2,
        )