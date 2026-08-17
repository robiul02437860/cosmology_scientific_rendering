# Cosmology Scientific Rendering

GPU-accelerated scientific Gaussian splatting for large cosmological particle datasets.

This repository contains a scientific Gaussian renderer built on a modified version of **gsplat v1.5.3**. It reconstructs projected particle density using additive Gaussian accumulation and scalar scientific attributes using conditional Gaussian statistics. It also includes an interactive Viser-based viewer with GPU-resident field caching, color transfer functions, and opacity transfer functions.

## Features

- Orthographic scientific Gaussian projection
- Additive projected-density rendering
- Conditional scalar-attribute reconstruction
- Custom CUDA scientific rasterization built on gsplat
- Tile-based GPU rendering
- Interactive Viser viewer
- Linear/logarithmic color transfer functions
- Independent color and opacity transfer functions
- GPU-resident cached display updates
- Resolution/quality benchmarking utilities

## Repository Layout

```text
cosmology_scientific_rendering/
├── examples/
├── external/
│   └── gsplat/                  # Git submodule: modified gsplat
├── scripts/
│   └── install.sh
├── src/
│   └── scientific_gsplat_renderer/
├── tests/
├── .gitmodules
├── pyproject.toml
└── README.md
```

The `external/gsplat` directory is a Git submodule pointing to the `scientific-renderer` branch of the modified gsplat fork.

The parent repository pins an exact gsplat commit, so users obtain the same CUDA implementation used by this renderer.

## Requirements

A CUDA-capable NVIDIA GPU is required for the GPU renderer.

You need:

- Linux
- Python 3.10 or newer
- NVIDIA GPU and driver
- CUDA toolkit with `nvcc`
- CUDA-enabled PyTorch
- Git

The CUDA toolkit used to compile the modified gsplat extension should be compatible with the installed PyTorch CUDA build.

## Installation

### 1. Clone the Repository and Submodule

```bash
git clone --recurse-submodules \
    https://github.com/robiul02437860/cosmology_scientific_rendering.git

cd cosmology_scientific_rendering
```

If the repository was cloned without its submodules:

```bash
git submodule update --init --recursive
```

You can verify the pinned gsplat version with:

```bash
git submodule status
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install CUDA-enabled PyTorch

Install a CUDA-enabled PyTorch build appropriate for your NVIDIA driver and CUDA environment.

Verify the installation:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`CUDA available` should be `True`.

Also verify that the CUDA compiler is available:

```bash
which nvcc
nvcc --version
```

If necessary, select the CUDA toolkit explicitly:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

### 4. Install Modified gsplat and the Renderer

Run:

```bash
bash scripts/install.sh
```

The installation script:

1. Initializes the gsplat Git submodule.
2. Checks CUDA-enabled PyTorch.
3. Checks the CUDA compiler.
4. Installs the required build tools.
5. Compiles and installs the modified gsplat implementation.
6. Installs `scientific-gsplat-renderer`.

CUDA compilation parallelism can be controlled with:

```bash
MAX_JOBS=8 bash scripts/install.sh
```

## Manual Installation

The installation can also be performed manually:

```bash
git submodule update --init --recursive

python -m pip install --upgrade \
    pip setuptools wheel ninja packaging

MAX_JOBS=8 python -m pip install \
    -e external/gsplat \
    --no-build-isolation

python -m pip install -e .
```

> **Important:** Do not replace the submodule with a stock `pip install gsplat`. This project uses custom scientific CUDA rasterization kernels that are not part of upstream gsplat v1.5.3.

## Verify the Installation

Run:

```bash
python - <<'PY'
import torch
import gsplat
import scientific_gsplat_renderer

from scientific_gsplat_renderer.interactive import (
    InteractiveScientificRenderer,
)

print("scientific_gsplat_renderer: OK")
print("gsplat:", getattr(gsplat, "__version__", "unknown"))
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Interactive renderer import: OK")
PY
```

## Running the Interactive Viewer

Provide a saved scientific Gaussian model:

```bash
python examples/run_viser_scientific_viewer.py \
    --model /path/to/simple_model.npz
```

The viewer supports saved Gaussian models containing spatial Gaussian parameters and optional scientific scalar-attribute statistics.

## Remote Visualization

When the renderer runs on a remote machine, the Viser server can be accessed from a local computer using SSH port forwarding.

On the **remote machine**, start the viewer:

```bash
python examples/run_viser_scientific_viewer.py \
    --model /path/to/simple_model.npz
```

On the **local computer**, create the SSH tunnel:

```bash
ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST
```

Then open the following address in the local web browser:

```text
http://localhost:8080
```

## Interactive Controls

The viewer supports:

- Density or scalar-attribute visualization
- Linear/logarithmic color mapping
- Independent linear/logarithmic opacity mapping
- Color range control
- Opacity range control
- Automatic percentile-based ranges
- Colormap selection
- Black/white background
- Interactive camera movement
- Gaussian smoothing controls
- Cached transfer-function updates
- Screenshot export

Camera or Gaussian-kernel changes require a new scientific render.

Color, opacity, and transfer-function changes can reuse the cached GPU scientific fields, avoiding unnecessary rerasterization.

## Scientific Rendering Formulation

### Projected Density

For projected density, the renderer performs additive Gaussian accumulation:

$$
D(\mathbf{u}) =
\sum_{k=1}^{K} M_k G_k(\mathbf{u}),
$$

where:

- $\mathbf{u}$ is the image-space position,
- $M_k$ is the mass represented by Gaussian $k$,
- $G_k(\mathbf{u})$ is the projected Gaussian footprint.

Unlike conventional 3D Gaussian Splatting for novel-view synthesis, this scientific density representation does not use front-to-back alpha compositing. Contributions from overlapping Gaussians are accumulated additively.

### Conditional Scalar Attribute

For a scalar scientific attribute, the renderer reconstructs the local attribute using conditional Gaussian statistics.

The resulting projected attribute field has the form:

$$
A(\mathbf{u}) =
\frac{
\sum_{k=1}^{K}
M_k G_k(\mathbf{u})m_k(\mathbf{u})
}{
\sum_{k=1}^{K}
M_k G_k(\mathbf{u})
},
$$

where $m_k(\mathbf{u})$ is the conditional scalar estimate associated with Gaussian $k$ at image-space position $\mathbf{u}$.

This separates the scientific field reconstruction from the subsequent visualization transfer function.

## Data and Models

Large simulation datasets, ground-truth images, rendered outputs, and saved `.npz` Gaussian models are intentionally not stored in this repository.

The renderer has been developed and evaluated using cosmological particle datasets including:

- Illustris-3 dark matter particles
- HACC `m000` particles

A compatible Gaussian model can be supplied using:

```bash
--model /path/to/simple_model.npz
```

## Benchmarks

Benchmarking utilities are available under `examples/`.

Install the benchmark dependencies:

```bash
python -m pip install -e ".[benchmark]"
```

For example:

```bash
python examples/benchmark_resolution_quality.py
```

Additional scripts are available for GPU projection, density rendering, model upload, and conditional rendering benchmarks.

## Tests

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Modified gsplat

This project builds on a modified fork of gsplat.

**Upstream project:**

https://github.com/nerfstudio-project/gsplat

**Scientific fork:**

https://github.com/robiul02437860/gsplat

The scientific fork adds custom CUDA functionality for scientific Gaussian rasterization, including additive density and conditional scalar-attribute rendering.

The main repository records the exact compatible gsplat commit through the Git submodule:

```bash
git submodule status
```

## Updating the gsplat Submodule

Changes to the custom CUDA implementation should first be committed inside the gsplat repository:

```bash
cd external/gsplat

git status
git add ...
git commit -m "Describe gsplat change"
git push
```

Then return to the parent repository and update its submodule pointer:

```bash
cd ../..

git add external/gsplat
git commit -m "Update scientific gsplat submodule"
git push
```

This ensures that the main repository always points to a specific compatible version of the modified gsplat implementation.

## License

A project license should be added before the public research release.

The gsplat submodule retains its own upstream license and copyright notices.
