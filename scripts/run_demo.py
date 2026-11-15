"""Reproduce a Poisson solution plot for the original CFD-coursework forcing."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from schwarz_dd import (
    AdditiveSchwarz, coursework_forcing, coursework_polynomial,
    dirichlet_rhs, full_solution, pcg_solve, poisson_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    n, blocks, overlap = 39, 4, 2
    A = poisson_matrix(n)
    b = dirichlet_rhs(n, coursework_forcing, coursework_polynomial)
    M = AdditiveSchwarz(A, n, blocks, overlap, two_level=True)
    result = pcg_solve(A, b, M.operator, rtol=1e-10)
    if result.info or result.relative_residual > 1.1e-10:
        raise RuntimeError(f"PCG failed: info={result.info}, residual={result.relative_residual}")

    u = full_solution(result.x, n, coursework_polynomial)
    fine = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(fine, fine)
    exact = coursework_polynomial(X, Y)
    error = np.abs(u - exact)
    print(f"Original coursework forcing: n={n}, DOFs={n*n}, blocks={blocks}x{blocks}, overlap={overlap}")
    print(f"PCG iterations: {result.iterations}, relative residual: {result.relative_residual:.3e}")
    print(f"Max nodal error vs original polynomial: {error.max():.6e}")

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)
    cf = ax[0].contourf(X, Y, u, levels=24, cmap="viridis")
    ax[0].set_title("Computed Poisson solution")
    fig.colorbar(cf, ax=ax[0], shrink=.83, label="u(x, y)")
    ef = ax[1].contourf(X, Y, error, levels=24, cmap="magma")
    ax[1].set_title("Absolute error vs analytical solution")
    fig.colorbar(ef, ax=ax[1], shrink=.83, label="|uₕ − u| ")
    for axes in ax:
        axes.set_aspect("equal")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
    fig.suptitle("Poisson problem from original CFD coursework · five-point FD", fontsize=12)
    (ROOT / "figures").mkdir(exist_ok=True)
    path = ROOT / "figures" / "poisson_solution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
