# Scientific gsplat Renderer

A GPU-accelerated scientific Gaussian splatting renderer for saved Gaussian
mixture models.

## Initial goals

- Load saved Gaussian models from `.npz`
- Support full 3D covariance matrices
- Support orthographic cameras
- Render projected density fields
- Render scalar attributes
- Use gsplat for GPU tile-based rasterization

## Development status

The project is currently under initial development.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"