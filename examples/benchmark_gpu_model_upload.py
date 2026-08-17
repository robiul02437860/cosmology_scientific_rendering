from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from scientific_gsplat_renderer.data.gaussian_model import GaussianModel
from scientific_gsplat_renderer.gpu import GpuGaussianModel


MODEL_PATH = Path(
    "/home/robiul/Particle_flow/HACC_project/output/"
    "illustris3_missing_tests/full_94m_0_5pct/simple_model.npz"
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 80)
    print("GPU model upload benchmark")
    print("=" * 80)

    t0 = perf_counter()
    model = GaussianModel.load(MODEL_PATH)
    t1 = perf_counter()

    torch.cuda.synchronize()

    t2 = perf_counter()
    gpu_model = GpuGaussianModel.from_cpu(
        model,
        device="cuda",
    )
    gpu_model.synchronize()
    t3 = perf_counter()

    print(f"Model path       : {MODEL_PATH}")
    print(f"Gaussians        : {gpu_model.n_gaussians:,}")
    print(f"Particles        : {gpu_model.n_particles:,}")
    print(f"Attribute        : {gpu_model.attribute_name}")
    print(f"Device           : {gpu_model.device}")
    print(f"Dtype            : {gpu_model.dtype}")
    print(f"Model load time  : {t1 - t0:.6f} s")
    print(f"GPU upload time  : {t3 - t2:.6f} s")
    print(f"GPU memory       : {gpu_model.memory_megabytes():.3f} MiB")

    print()
    print("Tensor shapes")
    print(f"means            : {tuple(gpu_model.means.shape)}")
    print(f"covariances      : {tuple(gpu_model.covariances.shape)}")
    print(f"masses           : {tuple(gpu_model.masses.shape)}")
    print(
        "cross covariance : "
        f"{tuple(gpu_model.position_attribute_cross_covariances.shape)}"
    )


if __name__ == "__main__":
    main()