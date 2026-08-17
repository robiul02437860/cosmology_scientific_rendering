#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MAX_JOBS="${MAX_JOBS:-8}"

echo "============================================================"
echo "Scientific Gaussian Splatting Renderer installation"
echo "============================================================"
echo "Repository : ${ROOT_DIR}"
echo "Python     : $(command -v python || true)"
echo "MAX_JOBS   : ${MAX_JOBS}"
echo

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python was not found in PATH."
    exit 1
fi

echo "[1/6] Initializing Git submodules..."
git submodule update --init --recursive

if [ ! -f "external/gsplat/pyproject.toml" ] && \
   [ ! -f "external/gsplat/setup.py" ]; then
    echo "ERROR: external/gsplat is missing."
    echo "Run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo
echo "[2/6] Checking PyTorch..."
if ! python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"PyTorch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("PyTorch version :", torch.__version__)
print("Torch CUDA      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA-enabled PyTorch is required.", file=sys.stderr)
    raise SystemExit(2)
PY
then
    echo
    echo "Install CUDA-enabled PyTorch first, then rerun:"
    echo "  bash scripts/install.sh"
    exit 1
fi

echo
echo "[3/6] Checking CUDA compiler..."
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc : $(command -v nvcc)"
    nvcc --version | tail -n 1 || true
else
    echo "ERROR: nvcc was not found in PATH."
    echo "Set CUDA_HOME and PATH to a CUDA toolkit compatible with PyTorch."
    exit 1
fi

echo
echo "[4/6] Installing build tools..."
python -m pip install --upgrade \
    pip setuptools wheel ninja packaging

echo
echo "[5/6] Installing modified gsplat submodule..."
echo "Pinned gsplat commit:"
git -C external/gsplat rev-parse --short HEAD

MAX_JOBS="${MAX_JOBS}" python -m pip install \
    -e external/gsplat \
    --no-build-isolation

echo
echo "[6/6] Installing scientific renderer..."
python -m pip install -e .

echo
echo "============================================================"
echo "Installation complete"
echo "============================================================"

python - <<'PY'
import torch
import gsplat
import scientific_gsplat_renderer

print("scientific_gsplat_renderer : OK")
print("gsplat                    :", getattr(gsplat, "__version__", "unknown"))
print("PyTorch                   :", torch.__version__)
print("Torch CUDA                :", torch.version.cuda)
print("CUDA available            :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU                       :", torch.cuda.get_device_name(0))
PY

echo
echo "Viewer example:"
echo "  python examples/run_viser_scientific_viewer.py --model /path/to/simple_model.npz"