import numpy as np
import pytest

from scientific_gsplat_renderer import (
    GaussianModel,
    OrthographicCamera,
    ProjectedGaussians,
    project_gaussians_orthographic,
)


def make_camera() -> OrthographicCamera:
    return OrthographicCamera(
        position=np.array(
            [0.0, 0.0, 10.0],
            dtype=np.float64,
        ),
        target=np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        up=np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
        view_width=20.0,
        image_width=200,
        image_height=100,
        near=0.0,
        far=100.0,
    )


def make_model(
    means: np.ndarray,
    covariances: np.ndarray,
) -> GaussianModel:
    n_gaussians = means.shape[0]

    return GaussianModel(
        means=np.asarray(
            means,
            dtype=np.float64,
        ),
        covariances=np.asarray(
            covariances,
            dtype=np.float64,
        ),
        weights=np.full(
            n_gaussians,
            1.0 / n_gaussians,
            dtype=np.float64,
        ),
        n_particles=1000,
    )


def test_center_projects_to_image_center() -> None:
    model = make_model(
        means=np.array(
            [[0.0, 0.0, 0.0]]
        ),
        covariances=np.eye(3)[None, :, :],
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    np.testing.assert_allclose(
        projected.means_pixel,
        [[100.0, 50.0]],
    )

    np.testing.assert_allclose(
        projected.depths,
        [10.0],
    )

    assert projected.valid[0]


def test_positive_camera_x_moves_right() -> None:
    model = make_model(
        means=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        covariances=np.stack(
            [
                np.eye(3),
                np.eye(3),
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    np.testing.assert_allclose(
        projected.means_pixel,
        [
            [100.0, 50.0],
            [120.0, 50.0],
        ],
    )


def test_positive_camera_y_moves_upward_in_image() -> None:
    model = make_model(
        means=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        covariances=np.stack(
            [
                np.eye(3),
                np.eye(3),
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    np.testing.assert_allclose(
        projected.means_pixel,
        [
            [100.0, 50.0],
            [100.0, 30.0],
        ],
    )


def test_isotropic_covariance_projects_to_pixel_covariance() -> None:
    model = make_model(
        means=np.array(
            [[0.0, 0.0, 0.0]]
        ),
        covariances=np.array(
            [
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 9.0],
                ]
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
        sigma_extent=3.0,
    )

    # The camera has 10 pixels per world-space unit.
    np.testing.assert_allclose(
        projected.covariances_pixel,
        [
            [
                [400.0, 0.0],
                [0.0, 400.0],
            ]
        ],
    )

    np.testing.assert_allclose(
        projected.radii_pixel,
        [60.0],
    )


def test_off_diagonal_covariance_changes_sign_in_image_space() -> None:
    model = make_model(
        means=np.array(
            [[0.0, 0.0, 0.0]]
        ),
        covariances=np.array(
            [
                [
                    [2.0, 0.5, 0.0],
                    [0.5, 3.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    np.testing.assert_allclose(
        projected.covariances_pixel,
        [
            [
                [200.0, -50.0],
                [-50.0, 300.0],
            ]
        ],
    )


def test_inverse_covariance_is_correct() -> None:
    model = make_model(
        means=np.array(
            [[0.0, 0.0, 0.0]]
        ),
        covariances=np.array(
            [
                [
                    [2.0, 0.3, 0.0],
                    [0.3, 3.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    product = (
        projected.covariances_pixel[0]
        @ projected.inverse_covariances_pixel[0]
    )

    np.testing.assert_allclose(
        product,
        np.eye(2),
        rtol=1e-12,
        atol=1e-12,
    )


def test_mass_is_preserved() -> None:
    model = GaussianModel(
        means=np.zeros(
            (2, 3),
            dtype=np.float64,
        ),
        covariances=np.stack(
            [
                np.eye(3),
                np.eye(3),
            ]
        ),
        weights=np.array(
            [0.25, 0.75],
            dtype=np.float64,
        ),
        n_particles=1000,
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    np.testing.assert_allclose(
        projected.masses,
        [250.0, 750.0],
    )


def test_depth_clipping() -> None:
    model = make_model(
        means=np.array(
            [
                [0.0, 0.0, 11.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -95.0],
            ]
        ),
        covariances=np.stack(
            [
                np.eye(3),
                np.eye(3),
                np.eye(3),
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    np.testing.assert_allclose(
        projected.depths,
        [-1.0, 10.0, 105.0],
    )

    np.testing.assert_array_equal(
        projected.valid,
        [False, True, False],
    )


def test_gaussian_outside_image_is_invalid() -> None:
    model = make_model(
        means=np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
        covariances=np.stack(
            [
                np.eye(3),
                np.eye(3),
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
        sigma_extent=3.0,
    )

    np.testing.assert_array_equal(
        projected.valid,
        [True, False],
    )


def test_small_covariance_is_stabilized() -> None:
    model = make_model(
        means=np.array(
            [[0.0, 0.0, 0.0]]
        ),
        covariances=np.array(
            [
                [
                    [1e-12, 0.0, 0.0],
                    [0.0, 1e-12, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
        minimum_eigenvalue=1e-12,
        minimum_pixel_variance=0.25,
    )

    eigenvalues = np.linalg.eigvalsh(
        projected.covariances_pixel
    )

    assert np.all(
        eigenvalues >= 0.25 - 1e-12
    )


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("minimum_eigenvalue", 0.0),
        ("minimum_eigenvalue", -1.0),
        ("minimum_eigenvalue", np.inf),
        ("minimum_pixel_variance", 0.0),
        ("minimum_pixel_variance", -1.0),
        ("minimum_pixel_variance", np.nan),
        ("sigma_extent", 0.0),
        ("sigma_extent", -3.0),
        ("sigma_extent", np.inf),
    ],
)
def test_rejects_invalid_projection_parameters(
    parameter_name: str,
    parameter_value: float,
) -> None:
    model = make_model(
        means=np.array(
            [[0.0, 0.0, 0.0]]
        ),
        covariances=np.eye(3)[None, :, :],
    )

    parameters = {
        "minimum_eigenvalue": 1e-6,
        "minimum_pixel_variance": 1e-4,
        "sigma_extent": 3.0,
    }

    parameters[parameter_name] = parameter_value

    with pytest.raises(ValueError):
        project_gaussians_orthographic(
            model,
            make_camera(),
            **parameters,
        )


def test_projected_gaussians_properties() -> None:
    model = make_model(
        means=np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
        covariances=np.stack(
            [
                np.eye(3),
                np.eye(3),
            ]
        ),
    )

    projected = project_gaussians_orthographic(
        model,
        make_camera(),
    )

    assert isinstance(
        projected,
        ProjectedGaussians,
    )

    assert projected.n_gaussians == 2
    assert projected.n_valid == 1