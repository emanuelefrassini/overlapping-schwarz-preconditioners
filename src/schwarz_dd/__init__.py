"""Sparse Poisson benchmarks with one- and two-level additive Schwarz."""

from .poisson import (
    coursework_forcing, coursework_polynomial, dirichlet_rhs, full_solution,
    interior_grid, poisson_matrix, sine_exact, sine_forcing,
    zero_boundary_polynomial,
)
from .schwarz import (
    AdditiveSchwarz, coarse_prolongation, coarse_prolongation_1d, subdomain_indices,
)
from .solvers import SolveResult, pcg_solve

__all__ = [
    "AdditiveSchwarz", "SolveResult", "coarse_prolongation",
    "coarse_prolongation_1d", "coursework_forcing", "coursework_polynomial",
    "dirichlet_rhs", "full_solution", "interior_grid", "pcg_solve",
    "poisson_matrix", "sine_exact", "sine_forcing", "subdomain_indices",
    "zero_boundary_polynomial",
]
