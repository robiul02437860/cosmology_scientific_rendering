from pathlib import Path

import numpy as np
import pytest

from scientific_gsplat_renderer.data import GaussianModel


def create_basic_model_file(path: Path) -> None:
    means = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    covariances = np.array(
        [
            np.eye(3),
            np.diag([2.0, 3.0, 4.0]),
        ],
        dtype=np.float32,
    )

    weights = np.array(
        [0.25, 0.75],
        dtype=np.float32,
    )

    np.savez(
        path,
        means=means,
        covariances=covariances,
        global_weights=weights,
        n_particles=np.array(1000),
        box_size=np.array(50.0),
    )


def create_attribute_model_file(path: Path) -> None:
    means = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    covariances = np.array(
        [
            np.eye(3),
            np.diag([2.0, 3.0, 4.0]),
        ],
        dtype=np.float32,
    )

    weights = np.array(
        [0.4, 0.6],
        dtype=np.float32,
    )

    attribute_means = np.array(
        [10.0, 20.0],
        dtype=np.float32,
    )

    cross_covariances = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    attribute_covariances = np.array(
        [
            [[1.5]],
            [[2.5]],
        ],
        dtype=np.float32,
    )

    np.savez(
        path,
        means=means,
        covariances=covariances,
        global_weights=weights,
        n_particles=np.array(5000),
        attribute_means=attribute_means,
        position_attribute_cross_covariances=(
            cross_covariances
        ),
        attribute_covariances=attribute_covariances,
        attribute_name=np.array("temperature"),
        box_size=np.array(100.0),
    )


def test_construct_basic_model() -> None:
    model = GaussianModel(
        means=np.zeros((2, 3), dtype=np.float64),
        covariances=np.stack(
            [
                np.eye(3, dtype=np.float64),
                np.eye(3, dtype=np.float64),
            ]
        ),
        weights=np.array(
            [0.25, 0.75],
            dtype=np.float64,
        ),
        n_particles=1000,
    )

    assert model.n_gaussians == 2
    assert model.n_particles == 1000
    assert model.has_attribute is False
    assert model.weight_sum == pytest.approx(1.0)

    np.testing.assert_allclose(
        model.masses,
        [250.0, 750.0],
    )


def test_load_basic_model(tmp_path: Path) -> None:
    path = tmp_path / "basic_model.npz"
    create_basic_model_file(path)

    model = GaussianModel.load(path)

    assert model.n_gaussians == 2
    assert model.n_particles == 1000
    assert model.box_size == pytest.approx(50.0)
    assert model.has_attribute is False

    np.testing.assert_allclose(
        model.weights,
        [0.25, 0.75],
    )

    np.testing.assert_allclose(
        model.masses,
        [250.0, 750.0],
    )


def test_load_attribute_model(tmp_path: Path) -> None:
    path = tmp_path / "attribute_model.npz"
    create_attribute_model_file(path)

    model = GaussianModel.load(path)

    assert model.n_gaussians == 2
    assert model.n_particles == 5000
    assert model.has_attribute is True
    assert model.attribute_name == "temperature"

    assert model.attribute_means is not None
    assert (
        model.position_attribute_cross_covariances
        is not None
    )
    assert model.attribute_variances is not None

    np.testing.assert_allclose(
        model.attribute_means,
        [10.0, 20.0],
    )

    np.testing.assert_allclose(
        model.position_attribute_cross_covariances,
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    np.testing.assert_allclose(
        model.attribute_variances,
        [1.5, 2.5],
    )


def test_load_supports_alias_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alias_model.npz"

    np.savez(
        path,
        means=np.zeros(
            (1, 3),
            dtype=np.float32,
        ),
        covs=np.eye(
            3,
            dtype=np.float32,
        )[None, :, :],
        weights=np.array(
            [1.0],
            dtype=np.float32,
        ),
    )

    model = GaussianModel.load(path)

    assert model.n_gaussians == 1
    assert model.n_particles == 1
    assert model.weight_sum == pytest.approx(1.0)


def test_loads_column_attribute_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "column_attribute_model.npz"

    np.savez(
        path,
        means=np.zeros(
            (2, 3),
            dtype=np.float32,
        ),
        covariances=np.stack(
            [
                np.eye(3, dtype=np.float32),
                np.eye(3, dtype=np.float32),
            ]
        ),
        global_weights=np.array(
            [[0.5], [0.5]],
            dtype=np.float32,
        ),
        n_particles=np.array(100),
        attribute_means=np.array(
            [
                [10.0],
                [20.0],
            ],
            dtype=np.float32,
        ),
        position_attribute_cross_covariances=np.array(
            [
                [[0.1], [0.2], [0.3]],
                [[0.4], [0.5], [0.6]],
            ],
            dtype=np.float32,
        ),
        attribute_covariances=np.array(
            [
                [[1.5]],
                [[2.5]],
            ],
            dtype=np.float32,
        ),
        attribute_name=np.array("temperature"),
    )

    model = GaussianModel.load(path)

    assert model.weights.shape == (2,)
    assert model.attribute_means is not None
    assert (
        model.position_attribute_cross_covariances
        is not None
    )
    assert model.attribute_variances is not None

    assert model.attribute_means.shape == (2,)
    assert (
        model.position_attribute_cross_covariances.shape
        == (2, 3)
    )
    assert model.attribute_variances.shape == (2,)

    np.testing.assert_allclose(
        model.weights,
        [0.5, 0.5],
    )

    np.testing.assert_allclose(
        model.attribute_means,
        [10.0, 20.0],
    )

    np.testing.assert_allclose(
        model.position_attribute_cross_covariances,
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    np.testing.assert_allclose(
        model.attribute_variances,
        [1.5, 2.5],
    )


def test_loads_transposed_cross_covariances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transposed_cross_model.npz"

    np.savez(
        path,
        means=np.zeros(
            (2, 3),
            dtype=np.float32,
        ),
        covariances=np.stack(
            [
                np.eye(3, dtype=np.float32),
                np.eye(3, dtype=np.float32),
            ]
        ),
        global_weights=np.array(
            [0.5, 0.5],
            dtype=np.float32,
        ),
        attribute_means=np.array(
            [1.0, 2.0],
            dtype=np.float32,
        ),
        position_attribute_cross_covariances=np.array(
            [
                [[0.1, 0.2, 0.3]],
                [[0.4, 0.5, 0.6]],
            ],
            dtype=np.float32,
        ),
    )

    model = GaussianModel.load(path)

    assert (
        model.position_attribute_cross_covariances
        is not None
    )

    assert (
        model.position_attribute_cross_covariances.shape
        == (2, 3)
    )

    np.testing.assert_allclose(
        model.position_attribute_cross_covariances,
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )


def test_rejects_invalid_mean_shape() -> None:
    with pytest.raises(
        ValueError,
        match="means must have shape",
    ):
        GaussianModel(
            means=np.zeros(
                (2, 2),
                dtype=np.float64,
            ),
            covariances=np.zeros(
                (2, 3, 3),
                dtype=np.float64,
            ),
            weights=np.array(
                [0.5, 0.5],
                dtype=np.float64,
            ),
            n_particles=100,
        )


def test_rejects_invalid_covariance_shape() -> None:
    with pytest.raises(
        ValueError,
        match="covariances must have shape",
    ):
        GaussianModel(
            means=np.zeros(
                (2, 3),
                dtype=np.float64,
            ),
            covariances=np.zeros(
                (2, 2, 2),
                dtype=np.float64,
            ),
            weights=np.array(
                [0.5, 0.5],
                dtype=np.float64,
            ),
            n_particles=100,
        )


def test_rejects_negative_weights() -> None:
    with pytest.raises(
        ValueError,
        match="weights must be nonnegative",
    ):
        GaussianModel(
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
                [1.1, -0.1],
                dtype=np.float64,
            ),
            n_particles=100,
        )


def test_rejects_non_symmetric_covariance() -> None:
    covariances = np.eye(
        3,
        dtype=np.float64,
    )[None, :, :]

    covariances[0, 0, 1] = 2.0

    with pytest.raises(
        ValueError,
        match="must be symmetric",
    ):
        GaussianModel(
            means=np.zeros(
                (1, 3),
                dtype=np.float64,
            ),
            covariances=covariances,
            weights=np.array(
                [1.0],
                dtype=np.float64,
            ),
            n_particles=100,
        )


def test_rejects_missing_required_array(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_weights.npz"

    np.savez(
        path,
        means=np.zeros(
            (1, 3),
            dtype=np.float32,
        ),
        covariances=np.eye(
            3,
            dtype=np.float32,
        )[None, :, :],
    )

    with pytest.raises(
        KeyError,
        match="weights",
    ):
        GaussianModel.load(path)


def test_rejects_missing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "does_not_exist.npz"

    with pytest.raises(FileNotFoundError):
        GaussianModel.load(path)


def test_rejects_attribute_name_without_means() -> None:
    with pytest.raises(
        ValueError,
        match="attribute_name was provided",
    ):
        GaussianModel(
            means=np.zeros(
                (1, 3),
                dtype=np.float64,
            ),
            covariances=np.eye(
                3,
                dtype=np.float64,
            )[None, :, :],
            weights=np.array(
                [1.0],
                dtype=np.float64,
            ),
            n_particles=100,
            attribute_name="temperature",
        )


def test_rejects_negative_attribute_variance() -> None:
    with pytest.raises(
        ValueError,
        match="attribute variances must be nonnegative",
    ):
        GaussianModel(
            means=np.zeros(
                (1, 3),
                dtype=np.float64,
            ),
            covariances=np.eye(
                3,
                dtype=np.float64,
            )[None, :, :],
            weights=np.array(
                [1.0],
                dtype=np.float64,
            ),
            n_particles=100,
            attribute_means=np.array(
                [10.0],
                dtype=np.float64,
            ),
            attribute_variances=np.array(
                [-1.0],
                dtype=np.float64,
            ),
        )
        

def test_stabilized_covariances_clamp_negative_eigenvalues() -> None:
    covariances = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, -1e-5],
            ],
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
        ],
        dtype=np.float64,
    )

    model = GaussianModel(
        means=np.zeros((2, 3), dtype=np.float64),
        covariances=covariances,
        weights=np.array([0.5, 0.5], dtype=np.float64),
        n_particles=100,
    )

    stabilized = model.stabilized_covariances(
        minimum_eigenvalue=1e-6
    )

    eigenvalues = np.linalg.eigvalsh(stabilized)

    assert stabilized.shape == (2, 3, 3)
    assert np.all(eigenvalues >= 1e-6 - 1e-12)

    # The original model must remain unchanged.
    assert model.covariances[0, 2, 2] == pytest.approx(
        -1e-5
    )


def test_stabilized_covariances_preserve_valid_covariance() -> None:
    covariances = np.array(
        [
            [
                [2.0, 0.3, 0.1],
                [0.3, 3.0, 0.2],
                [0.1, 0.2, 4.0],
            ]
        ],
        dtype=np.float64,
    )

    model = GaussianModel(
        means=np.zeros((1, 3), dtype=np.float64),
        covariances=covariances,
        weights=np.array([1.0], dtype=np.float64),
        n_particles=100,
    )

    stabilized = model.stabilized_covariances(
        minimum_eigenvalue=1e-6
    )

    np.testing.assert_allclose(
        stabilized,
        covariances,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "minimum_eigenvalue",
    [0.0, -1e-6, np.inf, np.nan],
)
def test_stabilized_covariances_reject_invalid_threshold(
    minimum_eigenvalue: float,
) -> None:
    model = GaussianModel(
        means=np.zeros((1, 3), dtype=np.float64),
        covariances=np.eye(
            3,
            dtype=np.float64,
        )[None, :, :],
        weights=np.array([1.0], dtype=np.float64),
        n_particles=100,
    )

    with pytest.raises(ValueError):
        model.stabilized_covariances(
            minimum_eigenvalue=minimum_eigenvalue
        )