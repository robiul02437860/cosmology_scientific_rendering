import numpy as np

from scientific_gsplat_renderer.projection.conditional_attribute import (
    compute_pixel_conditional_parameters,
)


def test_identity_camera_without_y_flip() -> None:
    attribute_means = np.array(
        [10.0],
        dtype=np.float64,
    )

    # Cov(world_position, attribute)
    position_attribute_cross_covariances = np.array(
        [[2.0, 3.0, 7.0]],
        dtype=np.float64,
    )

    covariances_pixel = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )

    world_to_camera_rotation = np.eye(
        3,
        dtype=np.float64,
    )

    parameters = compute_pixel_conditional_parameters(
        attribute_means=attribute_means,
        position_attribute_cross_covariances=(
            position_attribute_cross_covariances
        ),
        covariances_pixel=covariances_pixel,
        world_to_camera_rotation=world_to_camera_rotation,
        pixel_scale_x=1.0,
        pixel_scale_y=1.0,
        flip_y=False,
        regularization=0.0,
    )

    np.testing.assert_allclose(
        parameters,
        np.array([[10.0, 2.0, 3.0]], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_image_y_flip_changes_y_slope_sign() -> None:
    attribute_means = np.array(
        [10.0],
        dtype=np.float64,
    )

    position_attribute_cross_covariances = np.array(
        [[2.0, 3.0, 0.0]],
        dtype=np.float64,
    )

    covariances_pixel = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )

    parameters = compute_pixel_conditional_parameters(
        attribute_means=attribute_means,
        position_attribute_cross_covariances=(
            position_attribute_cross_covariances
        ),
        covariances_pixel=covariances_pixel,
        world_to_camera_rotation=np.eye(3, dtype=np.float64),
        pixel_scale_x=1.0,
        pixel_scale_y=1.0,
        flip_y=True,
        regularization=0.0,
    )

    np.testing.assert_allclose(
        parameters,
        np.array([[10.0, 2.0, -3.0]], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_pixel_scaling_produces_attribute_per_pixel_slopes() -> None:
    attribute_means = np.array(
        [5.0],
        dtype=np.float64,
    )

    # World covariance:
    #
    # Sigma_xx = identity
    # Cov(x,a) = [4, 6, 0]
    #
    # Pixel mapping:
    #
    # pixel_x = 2 * x
    # pixel_y = 3 * y
    #
    # Therefore:
    #
    # Sigma_pp = diag(4, 9)
    # Sigma_pa = [8, 18]
    # slope = [8/4, 18/9] = [2, 2]
    position_attribute_cross_covariances = np.array(
        [[4.0, 6.0, 0.0]],
        dtype=np.float64,
    )

    covariances_pixel = np.array(
        [
            [
                [4.0, 0.0],
                [0.0, 9.0],
            ]
        ],
        dtype=np.float64,
    )

    parameters = compute_pixel_conditional_parameters(
        attribute_means=attribute_means,
        position_attribute_cross_covariances=(
            position_attribute_cross_covariances
        ),
        covariances_pixel=covariances_pixel,
        world_to_camera_rotation=np.eye(3, dtype=np.float64),
        pixel_scale_x=2.0,
        pixel_scale_y=3.0,
        flip_y=False,
        regularization=0.0,
    )

    np.testing.assert_allclose(
        parameters,
        np.array([[5.0, 2.0, 2.0]], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_camera_rotation_is_applied_to_cross_covariance() -> None:
    attribute_means = np.array(
        [1.0],
        dtype=np.float64,
    )

    # Attribute varies along world x.
    cross_world = np.array(
        [[4.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    # Camera x axis points along world y.
    # Camera y axis points along negative world x.
    rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    covariances_pixel = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )

    parameters = compute_pixel_conditional_parameters(
        attribute_means=attribute_means,
        position_attribute_cross_covariances=cross_world,
        covariances_pixel=covariances_pixel,
        world_to_camera_rotation=rotation,
        pixel_scale_x=1.0,
        pixel_scale_y=1.0,
        flip_y=False,
        regularization=0.0,
    )

    # World x becomes negative camera y.
    np.testing.assert_allclose(
        parameters,
        np.array([[1.0, 0.0, -4.0]], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )