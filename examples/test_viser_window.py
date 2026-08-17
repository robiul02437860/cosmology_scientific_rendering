from __future__ import annotations

import time

import numpy as np
import viser


def make_test_image(
    width: int,
    height: int,
) -> np.ndarray:
    x = np.linspace(
        0.0,
        1.0,
        width,
        dtype=np.float32,
    )

    y = np.linspace(
        0.0,
        1.0,
        height,
        dtype=np.float32,
    )

    grid_x, grid_y = np.meshgrid(
        x,
        y,
    )

    image = np.stack(
        (
            grid_x,
            grid_y,
            0.5 * np.ones_like(grid_x),
        ),
        axis=-1,
    )

    return np.asarray(
        np.clip(
            image * 255.0,
            0.0,
            255.0,
        ),
        dtype=np.uint8,
    )


def main() -> None:
    server = viser.ViserServer(
    host="127.0.0.1",
    port=8080,
)

    server.gui.configure_theme(
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
    )

    server.gui.set_panel_label(
        "Scientific Gaussian Renderer"
    )

    server.scene.set_up_direction(
        "+y"
    )

    server.initial_camera.position = (
        0.0,
        0.0,
        3.0,
    )

    server.initial_camera.look_at = (
        0.0,
        0.0,
        0.0,
    )

    mode = server.gui.add_dropdown(
        "View mode",
        options=(
            "Density",
            "Attribute",
        ),
        initial_value="Density",
    )

    beta = server.gui.add_slider(
        "Beta",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=0.5,
    )

    blob = server.gui.add_slider(
        "Blob sigma",
        min=0.0,
        max=5.0,
        step=0.1,
        initial_value=2.0,
    )

    resolution = server.gui.add_dropdown(
        "Resolution",
        options=(
            "600",
            "1024",
            "2048",
        ),
        initial_value="1024",
    )

    status = server.gui.add_markdown(
        """
### Status

Viewer initialized.
"""
    )

    image = make_test_image(
        width=1024,
        height=1024,
    )

    server.scene.set_background_image(
        image,
        format="jpeg",
        jpeg_quality=90,
    )

    @mode.on_update
    def _(_) -> None:
        status.content = (
            "### Status\n\n"
            f"Mode: **{mode.value}**  \n"
            f"Beta: **{beta.value:.2f}**  \n"
            f"Blob: **{blob.value:.2f} px**  \n"
            f"Resolution: **{resolution.value}**"
        )

    @beta.on_update
    def _(_) -> None:
        status.content = (
            "### Status\n\n"
            f"Mode: **{mode.value}**  \n"
            f"Beta: **{beta.value:.2f}**  \n"
            f"Blob: **{blob.value:.2f} px**  \n"
            f"Resolution: **{resolution.value}**"
        )

    @blob.on_update
    def _(_) -> None:
        status.content = (
            "### Status\n\n"
            f"Mode: **{mode.value}**  \n"
            f"Beta: **{beta.value:.2f}**  \n"
            f"Blob: **{blob.value:.2f} px**  \n"
            f"Resolution: **{resolution.value}**"
        )

    @resolution.on_update
    def _(_) -> None:
        status.content = (
            "### Status\n\n"
            f"Mode: **{mode.value}**  \n"
            f"Beta: **{beta.value:.2f}**  \n"
            f"Blob: **{blob.value:.2f} px**  \n"
            f"Resolution: **{resolution.value}**"
        )

    @server.on_client_connect
    def _(
        client: viser.ClientHandle,
    ) -> None:
        print(
            "Client connected:",
            client.client_id,
        )

        @client.camera.on_update
        def _(
            camera: viser.CameraHandle,
        ) -> None:
            print(
                "Camera:",
                "position=",
                np.asarray(camera.position),
                "look_at=",
                np.asarray(camera.look_at),
                "up=",
                np.asarray(camera.up_direction),
            )

    print()
    print("Viser viewer running.")
    print("Open the URL printed above in your browser.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping viewer.")


if __name__ == "__main__":
    main()