from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def _as_vector3(
    value: FloatArray | tuple[float, float, float] | list[float],
    *,
    name: str,
) -> FloatArray:
    """Convert an input value to a finite float64 vector of shape ``(3,)``."""

    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,), got {vector.shape}"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return vector


def _normalize(
    vector: FloatArray,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Return a normalized copy of a three-dimensional vector."""

    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= tolerance:
        raise ValueError(
            f"{name} must have a nonzero finite length"
        )

    return vector / norm


def _validate_points(
    points: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Validate one point or a batch of three-dimensional points."""

    result = np.asarray(
        points,
        dtype=np.float64,
    )

    if result.ndim == 1:
        if result.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,) or (N, 3), "
                f"got {result.shape}"
            )

    elif result.ndim == 2:
        if result.shape[1] != 3:
            raise ValueError(
                f"{name} must have shape (3,) or (N, 3), "
                f"got {result.shape}"
            )

    else:
        raise ValueError(
            f"{name} must have shape (3,) or (N, 3), "
            f"got {result.shape}"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return result


@dataclass(frozen=True, slots=True)
class OrthographicCamera:
    """Orthographic camera for scientific Gaussian rendering.

    The camera uses the following local coordinate convention:

    - camera ``x`` points right,
    - camera ``y`` points up,
    - camera ``z`` points forward from the camera toward the target.

    Therefore, points in front of the camera have positive camera-space
    depth.

    Parameters
    ----------
    position
        Camera position in world coordinates with shape ``(3,)``.
    target
        World-space point toward which the camera looks.
    up
        Approximate world-space up direction.
    view_width
        Width of the orthographic viewing region in world units.
    image_width
        Output image width in pixels.
    image_height
        Output image height in pixels.
    near
        Minimum visible camera-space depth.
    far
        Maximum visible camera-space depth.
    """

    position: FloatArray
    target: FloatArray
    up: FloatArray
    view_width: float
    image_width: int
    image_height: int
    near: float = 0.0
    far: float = float("inf")

    def __post_init__(self) -> None:
        position = _as_vector3(
            self.position,
            name="position",
        )

        target = _as_vector3(
            self.target,
            name="target",
        )

        up = _as_vector3(
            self.up,
            name="up",
        )

        object.__setattr__(
            self,
            "position",
            position,
        )

        object.__setattr__(
            self,
            "target",
            target,
        )

        object.__setattr__(
            self,
            "up",
            up,
        )

        self._validate()

    def _validate(self) -> None:
        if not np.isfinite(self.view_width):
            raise ValueError(
                "view_width must be finite"
            )

        if self.view_width <= 0.0:
            raise ValueError(
                "view_width must be positive, "
                f"got {self.view_width}"
            )

        if isinstance(self.image_width, bool) or not isinstance(
            self.image_width,
            int,
        ):
            raise TypeError(
                "image_width must be an integer"
            )

        if self.image_width <= 0:
            raise ValueError(
                "image_width must be positive, "
                f"got {self.image_width}"
            )

        if isinstance(self.image_height, bool) or not isinstance(
            self.image_height,
            int,
        ):
            raise TypeError(
                "image_height must be an integer"
            )

        if self.image_height <= 0:
            raise ValueError(
                "image_height must be positive, "
                f"got {self.image_height}"
            )

        if not np.isfinite(self.near):
            raise ValueError(
                "near must be finite"
            )

        if np.isnan(self.far):
            raise ValueError(
                "far must not be NaN"
            )

        if self.near < 0.0:
            raise ValueError(
                f"near must be nonnegative, got {self.near}"
            )

        if self.far <= self.near:
            raise ValueError(
                "far must be greater than near, "
                f"got near={self.near} and far={self.far}"
            )

        forward = _normalize(
            self.target - self.position,
            name="target - position",
        )

        normalized_up = _normalize(
            self.up,
            name="up",
        )

        right_candidate = np.cross(
            forward,
            normalized_up,
        )

        right_length = float(
            np.linalg.norm(right_candidate)
        )

        if right_length <= 1e-12:
            raise ValueError(
                "up must not be parallel to the viewing direction"
            )

    @property
    def aspect_ratio(self) -> float:
        """Return image width divided by image height."""

        return (
            float(self.image_width)
            / float(self.image_height)
        )

    @property
    def image_size(self) -> tuple[int, int]:
        """Return image dimensions as ``(width, height)``."""

        return self.image_width, self.image_height

    @property
    def view_height(self) -> float:
        """Return the orthographic view height in world units."""

        return self.view_width / self.aspect_ratio

    @property
    def forward(self) -> FloatArray:
        """Return the normalized camera-forward direction."""

        return _normalize(
            self.target - self.position,
            name="target - position",
        )

    @property
    def right(self) -> FloatArray:
        """Return the normalized camera-right direction."""

        right = np.cross(
            self.forward,
            _normalize(
                self.up,
                name="up",
            ),
        )

        return _normalize(
            right,
            name="camera right",
        )

    @property
    def true_up(self) -> FloatArray:
        """Return the corrected orthonormal camera-up direction."""

        corrected_up = np.cross(
            self.right,
            self.forward,
        )

        return _normalize(
            corrected_up,
            name="camera true up",
        )

    @property
    def rotation_matrix(self) -> FloatArray:
        """Return the world-to-camera rotation matrix.

        The rows contain the camera right, up, and forward vectors in
        world coordinates.

        Because positive camera ``z`` points forward, this basis has
        determinant ``-1``.
        """

        return np.stack(
            [
                self.right,
                self.true_up,
                self.forward,
            ],
            axis=0,
        )

    @property
    def view_matrix(self) -> FloatArray:
        """Return the homogeneous world-to-camera transformation matrix."""

        rotation = self.rotation_matrix
        translation = -rotation @ self.position

        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation

        return matrix

    @property
    def world_units_per_pixel_x(self) -> float:
        """Return horizontal world units represented by one pixel."""

        return (
            self.view_width
            / float(self.image_width)
        )

    @property
    def world_units_per_pixel_y(self) -> float:
        """Return vertical world units represented by one pixel."""

        return (
            self.view_height
            / float(self.image_height)
        )

    @property
    def pixels_per_world_unit_x(self) -> float:
        """Return horizontal pixels represented by one world unit."""

        return (
            float(self.image_width)
            / self.view_width
        )

    @property
    def pixels_per_world_unit_y(self) -> float:
        """Return vertical pixels represented by one world unit."""

        return (
            float(self.image_height)
            / self.view_height
        )

    def world_to_camera(
        self,
        points: FloatArray,
    ) -> FloatArray:
        """Transform world-space points into camera coordinates."""

        world_points = _validate_points(
            points,
            name="points",
        )

        centered = world_points - self.position

        return centered @ self.rotation_matrix.T

    def camera_to_world(
        self,
        points: FloatArray,
    ) -> FloatArray:
        """Transform camera-space points into world coordinates."""

        camera_points = _validate_points(
            points,
            name="points",
        )

        return (
            camera_points @ self.rotation_matrix
            + self.position
        )

    def rotate_covariances_to_camera(
        self,
        covariances: FloatArray,
    ) -> FloatArray:
        """Rotate world-space covariances into camera coordinates.

        For a world covariance matrix ``Sigma`` and world-to-camera
        rotation matrix ``R``:

        ``Sigma_camera = R @ Sigma @ R.T``
        """

        covariance_array = np.asarray(
            covariances,
            dtype=np.float64,
        )

        if covariance_array.ndim == 2:
            if covariance_array.shape != (3, 3):
                raise ValueError(
                    "A single covariance must have shape "
                    f"(3, 3), got {covariance_array.shape}"
                )

        elif covariance_array.ndim == 3:
            if covariance_array.shape[1:] != (3, 3):
                raise ValueError(
                    "Multiple covariances must have shape "
                    f"(N, 3, 3), got {covariance_array.shape}"
                )

        else:
            raise ValueError(
                "covariances must have shape (3, 3) or "
                f"(N, 3, 3), got {covariance_array.shape}"
            )

        if not np.all(np.isfinite(covariance_array)):
            raise ValueError(
                "covariances must contain only finite values"
            )

        rotation = self.rotation_matrix

        if covariance_array.ndim == 2:
            result = (
                rotation
                @ covariance_array
                @ rotation.T
            )

            return 0.5 * (
                result
                + result.T
            )

        result = np.einsum(
            "ij,njk,lk->nil",
            rotation,
            covariance_array,
            rotation,
            optimize=True,
        )

        return 0.5 * (
            result
            + np.swapaxes(
                result,
                1,
                2,
            )
        )

    def depth_mask(
        self,
        camera_points: FloatArray,
    ) -> NDArray[np.bool_]:
        """Return whether camera-space points lie in the depth interval."""

        points = _validate_points(
            camera_points,
            name="camera_points",
        )

        if points.ndim == 1:
            depth = float(points[2])

            return np.asarray(
                self.near <= depth <= self.far,
                dtype=np.bool_,
            )

        depths = points[:, 2]

        return np.asarray(
            (depths >= self.near)
            & (depths <= self.far),
            dtype=np.bool_,
        )