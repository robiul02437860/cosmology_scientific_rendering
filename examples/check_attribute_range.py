from pathlib import Path
import numpy as np

from scientific_gsplat_renderer.data.gaussian_model import GaussianModel
from scientific_gsplat_renderer.camera.orthographic import OrthographicCamera
from scientific_gsplat_renderer.projection.orthographic import (
    project_gaussians_orthographic,
)
from scientific_gsplat_renderer.projection.conditional_attribute import (
    project_conditional_attributes_orthographic,
)

MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)

model = GaussianModel.load(MODEL)

print("=" * 70)
print("Original Gaussian attribute means")
print("=" * 70)

print("min :", model.attribute_means.min())
print("max :", model.attribute_means.max())
print("mean:", model.attribute_means.mean())
print("negative:", np.sum(model.attribute_means < 0))

minimum = model.means.min(axis=0)
maximum = model.means.max(axis=0)
center = 0.5 * (minimum + maximum)
span = maximum - minimum

camera = OrthographicCamera(
    position=np.array(
        [center[0], center[1], center[2] + span.max() * 2],
        dtype=np.float64,
    ),
    target=center,
    up=np.array([0.0, 1.0, 0.0]),
    view_width=float(max(span[0], span[1]) * 1.02),
    image_width=600,
    image_height=600,
    near=0.0,
    far=float(span.max() * 4),
)

projected = project_gaussians_orthographic(model, camera)

conditional = project_conditional_attributes_orthographic(
    model,
    projected,
    camera,
)

print()
print("=" * 70)
print("Projected conditional means")
print("=" * 70)

print("min :", conditional.means.min())
print("max :", conditional.means.max())
print("mean:", conditional.means.mean())

print()
print("=" * 70)
print("Slope statistics")
print("=" * 70)

sx = conditional.slopes_pixel[:, 0]
sy = conditional.slopes_pixel[:, 1]

print("Slope X")
print("  min :", sx.min())
print("  max :", sx.max())
print("  mean:", sx.mean())

print("Slope Y")
print("  min :", sy.min())
print("  max :", sy.max())
print("  mean:", sy.mean())

print()
print("Maximum slope magnitude:")
print(np.sqrt(sx**2 + sy**2).max())