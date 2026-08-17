from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scientific_gsplat_renderer.data import GaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a saved scientific Gaussian model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default="/home/robiul/Particle_flow/HACC_project/output/illustris3_missing_tests/full_94m_0_5pct/simple_model.npz",
        help="Path to the saved .npz Gaussian model.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def inspect_model(path: Path) -> None:
    model = GaussianModel.load(path)

    print("=" * 72)
    print("Scientific Gaussian Model Inspector")
    print("=" * 72)

    print(f"File: {path.expanduser().resolve()}")
    print(model)

    print_header("Model summary")
    print(f"Gaussians            : {model.n_gaussians:,}")
    print(f"Particles represented: {model.n_particles:,}")
    print(f"Has attribute        : {model.has_attribute}")
    print(f"Box size             : {model.box_size}")

    print_header("Array shapes")
    print(f"means        : {model.means.shape}")
    print(f"covariances  : {model.covariances.shape}")
    print(f"weights      : {model.weights.shape}")

    if model.attribute_means is not None:
        print(f"attribute means: {model.attribute_means.shape}")

    if model.position_attribute_cross_covariances is not None:
        print(
            "cross covariances: "
            f"{model.position_attribute_cross_covariances.shape}"
        )

    if model.attribute_variances is not None:
        print(f"attribute variances: {model.attribute_variances.shape}")

    print_header("Weight statistics")
    print(f"Sum : {model.weight_sum:.12g}")
    print(f"Min : {np.min(model.weights):.12g}")
    print(f"Max : {np.max(model.weights):.12g}")
    print(f"Mean: {np.mean(model.weights):.12g}")

    print_header("Mass statistics")
    masses = model.masses
    print(f"Sum : {np.sum(masses, dtype=np.float64):.12g}")
    print(f"Min : {np.min(masses):.12g}")
    print(f"Max : {np.max(masses):.12g}")
    print(f"Mean: {np.mean(masses):.12g}")

    print_header("Mean-coordinate bounds")
    minimum = np.min(model.means, axis=0)
    maximum = np.max(model.means, axis=0)
    mean = np.mean(model.means, axis=0)
    std = np.std(model.means, axis=0)

    print(f"Minimum: {format_vector(minimum)}")
    print(f"Maximum: {format_vector(maximum)}")
    print(f"Mean   : {format_vector(mean)}")
    print(f"Std    : {format_vector(std)}")

    print_header("Covariance statistics")

    diagonal = np.diagonal(
        model.covariances,
        axis1=1,
        axis2=2,
    )

    print(f"Diagonal min : {np.min(diagonal):.12g}")
    print(f"Diagonal max : {np.max(diagonal):.12g}")
    print(f"Diagonal mean: {np.mean(diagonal):.12g}")

    eigenvalues = np.linalg.eigvalsh(model.covariances)

    print(f"Eigenvalue min : {np.min(eigenvalues):.12g}")
    print(f"Eigenvalue max : {np.max(eigenvalues):.12g}")
    print(f"Eigenvalue mean: {np.mean(eigenvalues):.12g}")

    negative_eigenvalues = int(np.count_nonzero(eigenvalues < 0.0))
    nonpositive_eigenvalues = int(np.count_nonzero(eigenvalues <= 0.0))

    print(f"Negative eigenvalues   : {negative_eigenvalues:,}")
    print(f"Nonpositive eigenvalues: {nonpositive_eigenvalues:,}")

    symmetry_error = np.max(
        np.abs(
            model.covariances
            - np.swapaxes(model.covariances, 1, 2)
        )
    )

    print(f"Maximum symmetry error: {symmetry_error:.12g}")

    print_header("Finite-value checks")
    print(f"Means finite       : {np.all(np.isfinite(model.means))}")
    print(
        "Covariances finite: "
        f"{np.all(np.isfinite(model.covariances))}"
    )
    print(f"Weights finite     : {np.all(np.isfinite(model.weights))}")

    if model.has_attribute:
        print_header("Attribute statistics")
        print(f"Name: {model.attribute_name}")

        assert model.attribute_means is not None

        print(f"Mean minimum: {np.min(model.attribute_means):.12g}")
        print(f"Mean maximum: {np.max(model.attribute_means):.12g}")
        print(f"Mean average: {np.mean(model.attribute_means):.12g}")
        print(f"Mean std    : {np.std(model.attribute_means):.12g}")

        if model.position_attribute_cross_covariances is not None:
            cross = model.position_attribute_cross_covariances
            print(f"Cross-covariance min : {np.min(cross):.12g}")
            print(f"Cross-covariance max : {np.max(cross):.12g}")
            print(f"Cross-covariance mean: {np.mean(cross):.12g}")

        if model.attribute_variances is not None:
            variances = model.attribute_variances
            print(f"Variance min : {np.min(variances):.12g}")
            print(f"Variance max : {np.max(variances):.12g}")
            print(f"Variance mean: {np.mean(variances):.12g}")

    print()
    print("=" * 72)
    print("Inspection complete")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    inspect_model(args.model)


if __name__ == "__main__":
    main()