Cosmology Scientific Rendering

GPU-accelerated scientific Gaussian splatting for large cosmological particle datasets.

This repository contains a scientific Gaussian renderer built on a modified version of gsplat v1.5.3. It reconstructs projected particle density using additive Gaussian accumulation and scalar scientific attributes using conditional Gaussian statistics. It also includes an interactive Viser-based viewer with GPU-resident field caching, color transfer functions, and opacity transfer functions.

Features

Orthographic scientific Gaussian projection

Additive projected-density rendering

Conditional scalar-attribute reconstruction

Custom CUDA scientific rasterization built on gsplat

Tile-based GPU rendering

Interactive Viser viewer

Linear/logarithmic color transfer functions

Independent color and opacity transfer functions

GPU-resident cached display updates

Resolution/quality benchmarking utilities

Repository Layout

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

The external/gsplat directory is a Git submodule pointing to the scientific-renderer branch of:

https://github.com/robiul02437860/gsplat

The parent repository pins an exact gsplat commit, so users obtain the same CUDA implementation used by this renderer.

Requirements

A CUDA-capable NVIDIA GPU is required for the GPU renderer.

You need:

Linux

Python 3.10 or newer

NVIDIA GPU and driver

CUDA toolkit with nvcc

CUDA-enabled PyTorch

Git

The CUDA toolkit used to compile the modified gsplat extension should be compatible with the installed PyTorch CUDA build.

Installation

1. Clone the repository and submodule

git clone --recurse-submodules \
    https://github.com/robiul02437860/cosmology_scientific_rendering.git

cd cosmology_scientific_rendering

If you already cloned the repository without submodules:

git submodule update --init --recursive

2. Create a virtual environment

python -m venv .venv
source .venv/bin/activate

3. Install CUDA-enabled PyTorch

Install a PyTorch build appropriate for your NVIDIA driver and CUDA environment.

Verify the installation:

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

CUDA available must be True.

Also verify the CUDA compiler:

which nvcc
nvcc --version

If necessary, select the CUDA toolkit explicitly:

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

4. Install modified gsplat and the renderer

bash scripts/install.sh

To control CUDA compilation parallelism:

MAX_JOBS=8 bash scripts/install.sh

Manual Installation

git submodule update --init --recursive

python -m pip install --upgrade \
    pip setuptools wheel ninja packaging

MAX_JOBS=8 python -m pip install \
    -e external/gsplat \
    --no-build-isolation

python -m pip install -e .

Important: Do not replace the submodule with a stock pip install gsplat. This project uses custom scientific CUDA rasterization kernels that are not part of upstream gsplat v1.5.3.

Verify the Installation

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

Run the Interactive Viewer

python examples/run_viser_scientific_viewer.py \
    --model /path/to/simple_model.npz

By default, the Viser server listens on:

http://127.0.0.1:8080

When running on a remote machine, create an SSH tunnel from your local computer:

ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST

Then open:

http://localhost:8080

Interactive Controls

The viewer supports:

Density or scalar-attribute display

Linear/logarithmic color mapping

Independent linear/logarithmic opacity mapping

Color and opacity ranges

Automatic percentile-based ranges

Colormap selection

Black/white background

Camera interaction

Gaussian smoothing controls

Cached transfer-function updates

Screenshot export

Camera or Gaussian-kernel changes trigger a new scientific render. Color, opacity, and transfer-function changes reuse cached GPU scientific fields.

Scientific Rendering Formulation

Projected density is reconstructed additively:

[
D(\mathbf{u}) = \sum_k M_k G_k(\mathbf{u}).
]

A scalar attribute is reconstructed conditionally:

[
A(\mathbf{u}) =
\frac{\sum_k M_k G_k(\mathbf{u}) m_k(\mathbf{u})}
{\sum_k M_k G_k(\mathbf{u})}.
]

Data and Models

Large simulation data, ground-truth images, outputs, and saved .npz models are intentionally not stored in this repository.

The renderer has been developed and evaluated with cosmological particle data including:

Illustris-3 dark matter particles

HACC m000 particles

Provide a compatible saved Gaussian model using --model.

Benchmarks

Install benchmark dependencies:

python -m pip install -e ".[benchmark]"

Then run, for example:

python examples/benchmark_resolution_quality.py

Tests

Install development dependencies:

python -m pip install -e ".[dev]"

Run:

pytest

Modified gsplat

Upstream gsplat:

https://github.com/nerfstudio-project/gsplat

Scientific fork:

https://github.com/robiul02437860/gsplat

The scientific fork adds custom additive-density and conditional-attribute CUDA rasterization. The parent repository records the exact compatible gsplat commit using a Git submodule:

git submodule status

Updating the gsplat Submodule

Make gsplat changes inside external/gsplat, commit and push them there first, then commit the updated submodule pointer in the parent repository:

cd external/gsplat

git add ...
git commit -m "Describe gsplat change"
git push

cd ../..

git add external/gsplat
git commit -m "Update scientific gsplat submodule"
git push

License

Add a project license before the public research release. The gsplat submodule retains its upstream license and copyright notices.