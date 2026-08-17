import numpy as np
import pytest

from scientific_gsplat_renderer import (
    DensityRenderResult,
    ProjectedGaussians,
    rasterize_density_cpu,
)


def make_projected_gaussians(
    *,
    means_pixel: np.ndarray,
    covariances_pixel: np.ndarray,
    masses: np.ndarray,
    radii_pixel: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> ProjectedGaussians:
    means_pixel = np.asarray(
        means_pixel,
        dtype=np.float64,
    )

    covariances_pixel = np.asarray(
        covariances_pixel,
        dtype=np.float64,
    )

    masses = np.asarray(
        masses,
        dtype=np.float64,
    )

    n_gaussians = means_pixel.shape[0]

    if radii_pixel is None:
        eigenvalues = np.linalg.eigvalsh(
            covariances_pixel
        )

        radii_pixel = (
            4.0
            * np.sqrt(
                eigenvalues[:, -1]
            )
        )

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

    covariances_camera = np.zeros(
        (n_gaussians, 3, 3),
        dtype=np.float64,
    )

    covariances_camera[:, :2, :2] = (
        covariances_pixel
    )

    covariances_camera[:, 2, 2] = 1.0

    inverse_covariances_pixel = np.linalg.inv(
        covariances_pixel
    )

    depths = np.full(
        n_gaussians,
        10.0,
        dtype=np.float64,
    )

    return ProjectedGaussians(
        means_camera=means_camera,
        means_pixel=means_pixel,
        covariances_camera=covariances_camera,
        covariances_pixel=covariances_pixel,
        inverse_covariances_pixel=(
            inverse_covariances_pixel
        ),
        radii_pixel=np.asarray(
            radii_pixel,
            dtype=np.float64,
        ),
        depths=depths,
        masses=masses,
        valid=np.asarray(
            valid,
            dtype=np.bool_,
        ),
    )


def test_single_gaussian_produces_nonnegative_image() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [[16.0, 16.0]]
        ),
        covariances_pixel=np.array(
            [
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                ]
            ]
        ),
        masses=np.array([100.0]),
    )

    result = rasterize_density_cpu(
        projected,
        image_width=32,
        image_height=32,
    )

    assert isinstance(
        result,
        DensityRenderResult,
    )

    assert result.density.shape == (
        32,
        32,
    )

    assert np.all(
        result.density >= 0.0
    )

    assert np.max(result.density) > 0.0
    assert result.rendered_gaussians == 1
    assert result.skipped_gaussians == 0


def test_single_centered_gaussian_approximately_preserves_mass() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [[64.0, 64.0]]
        ),
        covariances_pixel=np.array(
            [
                [
                    [16.0, 0.0],
                    [0.0, 16.0],
                ]
            ]
        ),
        masses=np.array([1000.0]),
        radii_pixel=np.array([24.0]),
    )

    result = rasterize_density_cpu(
        projected,
        image_width=128,
        image_height=128,
    )

    assert result.input_mass == pytest.approx(
        1000.0
    )

    assert result.image_mass == pytest.approx(
        1000.0,
        rel=1e-6,
    )

    assert result.retained_mass_fraction == pytest.approx(
        1.0,
        rel=1e-6,
    )


def test_center_pixel_has_largest_value() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [[16.5, 16.5]]
        ),
        covariances_pixel=np.array(
            [
                [
                    [2.0, 0.0],
                    [0.0, 2.0],
                ]
            ]
        ),
        masses=np.array([100.0]),
    )

    result = rasterize_density_cpu(
        projected,
        image_width=33,
        image_height=33,
    )

    center_value = result.density[
        16,
        16,
    ]

    assert center_value == pytest.approx(
        np.max(result.density)
    )


def test_two_gaussians_add_linearly() -> None:
    covariance = np.array(
        [
            [
                [4.0, 0.0],
                [0.0, 4.0],
            ]
        ]
    )

    first = make_projected_gaussians(
        means_pixel=np.array(
            [[20.0, 20.0]]
        ),
        covariances_pixel=covariance,
        masses=np.array([100.0]),
    )

    second = make_projected_gaussians(
        means_pixel=np.array(
            [[28.0, 20.0]]
        ),
        covariances_pixel=covariance,
        masses=np.array([200.0]),
    )

    combined = make_projected_gaussians(
        means_pixel=np.array(
            [
                [20.0, 20.0],
                [28.0, 20.0],
            ]
        ),
        covariances_pixel=np.concatenate(
            [
                covariance,
                covariance,
            ],
            axis=0,
        ),
        masses=np.array(
            [100.0, 200.0]
        ),
    )

    first_result = rasterize_density_cpu(
        first,
        image_width=48,
        image_height=48,
    )

    second_result = rasterize_density_cpu(
        second,
        image_width=48,
        image_height=48,
    )

    combined_result = rasterize_density_cpu(
        combined,
        image_width=48,
        image_height=48,
    )

    np.testing.assert_allclose(
        combined_result.density,
        (
            first_result.density
            + second_result.density
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_invalid_gaussian_is_skipped() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [
                [16.0, 16.0],
                [20.0, 20.0],
            ]
        ),
        covariances_pixel=np.array(
            [
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                ],
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                ],
            ]
        ),
        masses=np.array(
            [100.0, 200.0]
        ),
        valid=np.array(
            [True, False]
        ),
    )

    result = rasterize_density_cpu(
        projected,
        image_width=32,
        image_height=32,
    )

    assert result.rendered_gaussians == 1
    assert result.skipped_gaussians == 1
    assert result.input_mass == pytest.approx(
        100.0
    )


def test_maximum_gaussians_limits_processing() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
            ]
        ),
        covariances_pixel=np.repeat(
            np.eye(2)[None, :, :],
            repeats=3,
            axis=0,
        ),
        masses=np.array(
            [10.0, 20.0, 30.0]
        ),
    )

    result = rasterize_density_cpu(
        projected,
        image_width=40,
        image_height=40,
        maximum_gaussians=2,
    )

    assert result.rendered_gaussians == 2
    assert result.skipped_gaussians == 1

    # Uniform selection over indices [0, 1, 2] with two samples
    # selects Gaussian indices [0, 2].
    assert result.input_mass == pytest.approx(
        40.0
    )


def test_zero_mass_gaussian_has_no_effect() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [[16.0, 16.0]]
        ),
        covariances_pixel=np.array(
            [
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                ]
            ]
        ),
        masses=np.array([0.0]),
    )

    result = rasterize_density_cpu(
        projected,
        image_width=32,
        image_height=32,
    )

    np.testing.assert_array_equal(
        result.density,
        np.zeros(
            (32, 32),
            dtype=np.float64,
        ),
    )

    assert result.image_mass == 0.0


def test_exponent_cutoff_reduces_support() -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [[32.0, 32.0]]
        ),
        covariances_pixel=np.array(
            [
                [
                    [16.0, 0.0],
                    [0.0, 16.0],
                ]
            ]
        ),
        masses=np.array([100.0]),
        radii_pixel=np.array([24.0]),
    )

    full_result = rasterize_density_cpu(
        projected,
        image_width=64,
        image_height=64,
    )

    cutoff_result = rasterize_density_cpu(
        projected,
        image_width=64,
        image_height=64,
        exponent_cutoff=4.0,
    )

    assert cutoff_result.image_mass < (
        full_result.image_mass
    )

    assert np.count_nonzero(
        cutoff_result.density
    ) < np.count_nonzero(
        full_result.density
    )


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value", "exception"),
    [
        ("image_width", 0, ValueError),
        ("image_width", -1, ValueError),
        ("image_width", 32.0, TypeError),
        ("image_width", True, TypeError),
        ("image_height", 0, ValueError),
        ("image_height", -1, ValueError),
        ("image_height", 32.0, TypeError),
        ("image_height", False, TypeError),
        ("maximum_gaussians", 0, ValueError),
        ("maximum_gaussians", -1, ValueError),
        ("maximum_gaussians", 1.5, TypeError),
        ("maximum_gaussians", True, TypeError),
        ("exponent_cutoff", 0.0, ValueError),
        ("exponent_cutoff", -1.0, ValueError),
        ("exponent_cutoff", np.inf, ValueError),
        ("exponent_cutoff", np.nan, ValueError),
    ],
)
def test_rejects_invalid_parameters(
    parameter_name: str,
    parameter_value: object,
    exception: type[Exception],
) -> None:
    projected = make_projected_gaussians(
        means_pixel=np.array(
            [[16.0, 16.0]]
        ),
        covariances_pixel=np.array(
            [
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                ]
            ]
        ),
        masses=np.array([100.0]),
    )

    parameters: dict[str, object] = {
        "image_width": 32,
        "image_height": 32,
        "maximum_gaussians": None,
        "exponent_cutoff": None,
    }

    parameters[parameter_name] = parameter_value

    with pytest.raises(exception):
        rasterize_density_cpu(
            projected,
            **parameters,  # type: ignore[arg-type]
        )