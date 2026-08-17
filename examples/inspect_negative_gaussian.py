from pathlib import Path
import numpy as np

from scientific_gsplat_renderer.data.gaussian_model import GaussianModel

MODEL = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)

model = GaussianModel.load(MODEL)

idx = int(np.argmin(model.attribute_means))

print("=" * 80)
print(f"Negative Gaussian: {idx}")
print("=" * 80)

print(f"Attribute mean           : {model.attribute_means[idx]}")
print(f"Global weight            : {model.weights[idx]}")
print(f"Position                 : {model.means[idx]}")
print()

print("Spatial covariance")
print(model.covariances[idx])
print()

print("Cross covariance")
print(model.position_attribute_cross_covariances[idx])
print()

print("Attribute covariance")
print(model.attribute_variances[idx])

print()

eigvals = np.linalg.eigvalsh(model.covariances[idx])

print("Spatial covariance eigenvalues")
print(eigvals)