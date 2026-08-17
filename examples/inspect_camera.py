from __future__ import annotations

import numpy as np

from scientific_gsplat_renderer import OrthographicCamera


def main() -> None:
    camera = OrthographicCamera(
        position=np.array(
            [37_500.0, 37_500.0, 112_500.0]
        ),
        target=np.array(
            [37_500.0, 37_500.0, 37_500.0]
        ),
        up=np.array(
            [0.0, 1.0, 0.0]
        ),
        view_width=75_000.0,
        image_width=600,
        image_height=600,
        near=0.0,
        far=150_000.0,
    )

    points = np.array(
        [
            [37_500.0, 37_500.0, 37_500.0],
            [0.0, 0.0, 0.0],
            [75_000.0, 75_000.0, 75_000.0],
        ]
    )

    camera_points = camera.world_to_camera(points)

    print("=" * 72)
    print("Orthographic Camera")
    print("=" * 72)

    print(f"Position    : {camera.position}")
    print(f"Target      : {camera.target}")
    print(f"Right       : {camera.right}")
    print(f"Up          : {camera.true_up}")
    print(f"Forward     : {camera.forward}")
    print(f"View width  : {camera.view_width}")
    print(f"View height : {camera.view_height}")
    print(f"Image size  : {camera.image_size}")
    print()

    print("View matrix")
    print("-----------")
    print(camera.view_matrix)
    print()

    print("World points")
    print("------------")
    print(points)
    print()

    print("Camera-space points")
    print("-------------------")
    print(camera_points)


if __name__ == "__main__":
    main()