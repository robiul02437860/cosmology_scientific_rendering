import numpy as np
import pytest

from scientific_gsplat_renderer.camera import OrthographicCamera


def make_default_camera() -> OrthographicCamera:
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
        image_width=800,
        image_height=400,
        near=0.0,
        far=100.0,
    )


def test_camera_basis() -> None:
    camera = make_default_camera()

    np.testing.assert_allclose(
        camera.forward,
        [0.0, 0.0, -1.0],
    )

    np.testing.assert_allclose(
        camera.right,
        [1.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        camera.true_up,
        [0.0, 1.0, 0.0],
    )


def test_camera_basis_is_orthonormal_with_positive_forward_depth() -> None:
    camera = OrthographicCamera(
        position=np.array(
            [2.0, 3.0, 4.0],
            dtype=np.float64,
        ),
        target=np.array(
            [-1.0, 2.0, 0.0],
            dtype=np.float64,
        ),
        up=np.array(
            [0.0, 1.0, 0.2],
            dtype=np.float64,
        ),
        view_width=10.0,
        image_width=500,
        image_height=500,
    )

    rotation = camera.rotation_matrix

    np.testing.assert_allclose(
        rotation @ rotation.T,
        np.eye(3),
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.linalg.det(rotation) == pytest.approx(
        -1.0,
        abs=1e-12,
    )


def test_view_dimensions() -> None:
    camera = make_default_camera()

    assert camera.aspect_ratio == pytest.approx(
        2.0
    )

    assert camera.view_width == pytest.approx(
        20.0
    )

    assert camera.view_height == pytest.approx(
        10.0
    )

    assert camera.world_units_per_pixel_x == pytest.approx(
        0.025
    )

    assert camera.world_units_per_pixel_y == pytest.approx(
        0.025
    )

    assert camera.pixels_per_world_unit_x == pytest.approx(
        40.0
    )

    assert camera.pixels_per_world_unit_y == pytest.approx(
        40.0
    )


def test_camera_position_maps_to_origin() -> None:
    camera = make_default_camera()

    camera_point = camera.world_to_camera(
        camera.position
    )

    np.testing.assert_allclose(
        camera_point,
        [0.0, 0.0, 0.0],
    )


def test_target_maps_to_positive_depth() -> None:
    camera = make_default_camera()

    camera_target = camera.world_to_camera(
        camera.target
    )

    np.testing.assert_allclose(
        camera_target,
        [0.0, 0.0, 10.0],
    )


def test_world_to_camera_multiple_points() -> None:
    camera = make_default_camera()

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 5.0],
            [-4.0, -2.0, 9.0],
        ],
        dtype=np.float64,
    )

    transformed = camera.world_to_camera(
        points
    )

    expected = np.array(
        [
            [0.0, 0.0, 10.0],
            [2.0, 3.0, 5.0],
            [-4.0, -2.0, 1.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        transformed,
        expected,
    )


def test_world_camera_round_trip() -> None:
    camera = OrthographicCamera(
        position=np.array(
            [3.0, -2.0, 8.0],
            dtype=np.float64,
        ),
        target=np.array(
            [1.0, 4.0, -5.0],
            dtype=np.float64,
        ),
        up=np.array(
            [0.2, 1.0, 0.1],
            dtype=np.float64,
        ),
        view_width=50.0,
        image_width=1000,
        image_height=600,
    )

    world_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, -5.0, 2.0],
            [-3.0, 8.0, 12.0],
        ],
        dtype=np.float64,
    )

    camera_points = camera.world_to_camera(
        world_points
    )

    reconstructed = camera.camera_to_world(
        camera_points
    )

    np.testing.assert_allclose(
        reconstructed,
        world_points,
        rtol=1e-12,
        atol=1e-12,
    )


def test_view_matrix_matches_world_to_camera() -> None:
    camera = make_default_camera()

    world_point = np.array(
        [2.0, 3.0, 5.0],
        dtype=np.float64,
    )

    homogeneous_point = np.append(
        world_point,
        1.0,
    )

    transformed_homogeneous = (
        camera.view_matrix
        @ homogeneous_point
    )

    transformed_directly = camera.world_to_camera(
        world_point
    )

    np.testing.assert_allclose(
        transformed_homogeneous[:3],
        transformed_directly,
    )

    assert transformed_homogeneous[3] == pytest.approx(
        1.0
    )


def test_identity_covariance_is_preserved_by_rotation() -> None:
    camera = OrthographicCamera(
        position=np.array(
            [5.0, 4.0, 3.0],
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
        image_width=800,
        image_height=800,
    )

    covariance = np.eye(
        3,
        dtype=np.float64,
    )

    rotated = camera.rotate_covariances_to_camera(
        covariance
    )

    np.testing.assert_allclose(
        rotated,
        covariance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_covariance_rotation_matches_manual_result() -> None:
    camera = make_default_camera()

    covariance = np.array(
        [
            [4.0, 0.5, 0.2],
            [0.5, 3.0, 0.1],
            [0.2, 0.1, 2.0],
        ],
        dtype=np.float64,
    )

    expected = (
        camera.rotation_matrix
        @ covariance
        @ camera.rotation_matrix.T
    )

    actual = camera.rotate_covariances_to_camera(
        covariance
    )

    np.testing.assert_allclose(
        actual,
        expected,
    )


def test_multiple_covariance_rotation() -> None:
    camera = make_default_camera()

    covariances = np.stack(
        [
            np.eye(
                3,
                dtype=np.float64,
            ),
            np.diag(
                [2.0, 3.0, 4.0]
            ),
        ]
    )

    result = camera.rotate_covariances_to_camera(
        covariances
    )

    assert result.shape == (2, 3, 3)

    for index in range(2):
        expected = (
            camera.rotation_matrix
            @ covariances[index]
            @ camera.rotation_matrix.T
        )

        np.testing.assert_allclose(
            result[index],
            expected,
        )


def test_depth_mask() -> None:
    camera = make_default_camera()

    camera_points = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 50.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 101.0],
        ],
        dtype=np.float64,
    )

    mask = camera.depth_mask(
        camera_points
    )

    np.testing.assert_array_equal(
        mask,
        [False, True, True, True, False],
    )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("view_width", 0.0),
        ("view_width", -1.0),
        ("view_width", np.inf),
        ("image_width", 0),
        ("image_width", -1),
        ("image_height", 0),
        ("image_height", -1),
        ("near", -1.0),
    ],
)
def test_rejects_invalid_camera_values(
    field_name: str,
    field_value: float,
) -> None:
    parameters: dict[str, object] = {
        "position": np.array(
            [0.0, 0.0, 10.0]
        ),
        "target": np.array(
            [0.0, 0.0, 0.0]
        ),
        "up": np.array(
            [0.0, 1.0, 0.0]
        ),
        "view_width": 20.0,
        "image_width": 800,
        "image_height": 600,
        "near": 0.0,
        "far": 100.0,
    }

    parameters[field_name] = field_value

    with pytest.raises(ValueError):
        OrthographicCamera(
            **parameters,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("image_width", 800.0),
        ("image_height", 600.0),
        ("image_width", True),
        ("image_height", False),
    ],
)
def test_rejects_invalid_camera_types(
    field_name: str,
    field_value: object,
) -> None:
    parameters: dict[str, object] = {
        "position": np.array(
            [0.0, 0.0, 10.0]
        ),
        "target": np.array(
            [0.0, 0.0, 0.0]
        ),
        "up": np.array(
            [0.0, 1.0, 0.0]
        ),
        "view_width": 20.0,
        "image_width": 800,
        "image_height": 600,
        "near": 0.0,
        "far": 100.0,
    }

    parameters[field_name] = field_value

    with pytest.raises(TypeError):
        OrthographicCamera(
            **parameters,  # type: ignore[arg-type]
        )


def test_rejects_far_less_than_near() -> None:
    with pytest.raises(
        ValueError,
        match="far must be greater than near",
    ):
        OrthographicCamera(
            position=np.array(
                [0.0, 0.0, 10.0]
            ),
            target=np.array(
                [0.0, 0.0, 0.0]
            ),
            up=np.array(
                [0.0, 1.0, 0.0]
            ),
            view_width=20.0,
            image_width=800,
            image_height=600,
            near=10.0,
            far=5.0,
        )


def test_rejects_same_position_and_target() -> None:
    with pytest.raises(
        ValueError,
        match="target - position",
    ):
        OrthographicCamera(
            position=np.array(
                [1.0, 2.0, 3.0]
            ),
            target=np.array(
                [1.0, 2.0, 3.0]
            ),
            up=np.array(
                [0.0, 1.0, 0.0]
            ),
            view_width=10.0,
            image_width=400,
            image_height=400,
        )


def test_rejects_parallel_up_vector() -> None:
    with pytest.raises(
        ValueError,
        match="parallel",
    ):
        OrthographicCamera(
            position=np.array(
                [0.0, 0.0, 10.0]
            ),
            target=np.array(
                [0.0, 0.0, 0.0]
            ),
            up=np.array(
                [0.0, 0.0, 1.0]
            ),
            view_width=10.0,
            image_width=400,
            image_height=400,
        )


def test_rejects_invalid_point_shape() -> None:
    camera = make_default_camera()

    with pytest.raises(
        ValueError,
        match="points must have shape",
    ):
        camera.world_to_camera(
            np.zeros(
                (2, 2, 2),
                dtype=np.float64,
            )
        )


def test_rejects_nonfinite_points() -> None:
    camera = make_default_camera()

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        camera.world_to_camera(
            np.array(
                [0.0, np.nan, 1.0],
                dtype=np.float64,
            )
        )


def test_rejects_invalid_covariance_shape() -> None:
    camera = make_default_camera()

    with pytest.raises(
        ValueError,
        match="covariance must have shape",
    ):
        camera.rotate_covariances_to_camera(
            np.zeros(
                (2, 2),
                dtype=np.float64,
            )
        )


def test_rejects_invalid_covariance_batch_shape() -> None:
    camera = make_default_camera()

    with pytest.raises(
        ValueError,
        match="Multiple covariances must have shape",
    ):
        camera.rotate_covariances_to_camera(
            np.zeros(
                (4, 2, 2),
                dtype=np.float64,
            )
        )


def test_rejects_nonfinite_covariances() -> None:
    camera = make_default_camera()

    covariance = np.eye(
        3,
        dtype=np.float64,
    )

    covariance[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        camera.rotate_covariances_to_camera(
            covariance
        )