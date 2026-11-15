"""Sparse five-point Dirichlet Laplacian on the unit square.

The unknown vector uses row-major ordering: its first index is y, second x.
All unknowns are INTERIOR nodes; boundary values are eliminated into the RHS.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

Array = NDArray[np.float64]
Field = Callable[[Array, Array], Array]


def interior_grid(n: int) -> tuple[Array, Array]:
    """Return x/y coordinates of n-by-n interior grid points."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.arange(1, n + 1, dtype=float) / (n + 1)
    return np.meshgrid(x, x, indexing="xy")


def poisson_matrix(n: int) -> sparse.csr_matrix:
    """Assemble the SPD 5-point matrix for -Laplace with zero boundary DOFs."""
    if n < 1:
        raise ValueError("n must be >= 1")
    h = 1.0 / (n + 1)
    t = sparse.diags((-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)),
                     offsets=(-1, 0, 1), format="csr")
    eye = sparse.eye(n, format="csr")
    return ((sparse.kron(eye, t, format="csr") +
             sparse.kron(t, eye, format="csr")) / h**2).tocsr()


def _evaluate(field: Field, x: Array, y: Array) -> Array:
    a = np.asarray(field(x, y), dtype=float)
    result = np.broadcast_to(a, np.broadcast_shapes(x.shape, y.shape)).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("field must return finite values")
    return result


def dirichlet_rhs(n: int, source: Field, boundary: Field | None = None) -> Array:
    """Assemble f including eliminated nonhomogeneous Dirichlet boundary values.

    The boundary function g(x,y) is used only on the four sides. The four
    corners never enter the interior 5-point stencil.
    """
    x, y = interior_grid(n)
    rhs = _evaluate(source, x, y)
    if boundary is not None:
        h = 1.0 / (n + 1)
        points = np.arange(1, n + 1, dtype=float) * h
        rhs[:, 0] += _evaluate(boundary, np.zeros(n), points) / h**2
        rhs[:, -1] += _evaluate(boundary, np.ones(n), points) / h**2
        rhs[0, :] += _evaluate(boundary, points, np.zeros(n)) / h**2
        rhs[-1, :] += _evaluate(boundary, points, np.ones(n)) / h**2
    return rhs.ravel()


def full_solution(interior: Array, n: int, boundary: Field | None = None) -> Array:
    """Embed n-by-n interior solution into the (n+2)-square including boundary."""
    interior = np.asarray(interior, dtype=float)
    if interior.shape != (n * n,):
        raise ValueError("expected an interior vector of length n*n")
    x = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(x, x)
    result = np.zeros_like(X)
    if boundary is not None:
        result[0, :] = _evaluate(boundary, X[0, :], Y[0, :])
        result[-1, :] = _evaluate(boundary, X[-1, :], Y[-1, :])
        result[:, 0] = _evaluate(boundary, X[:, 0], Y[:, 0])
        result[:, -1] = _evaluate(boundary, X[:, -1], Y[:, -1])
    result[1:-1, 1:-1] = interior.reshape(n, n)
    return result


def coursework_forcing(x: Array, y: Array) -> Array:
    """f(x,y) from CFD/additive_prec.m and CFD/Assignment2_1_b.m."""
    return 2.0 * (x * (1.0 - x) + y * (1.0 - y))


def coursework_polynomial(x: Array, y: Array) -> Array:
    """Nonhomogeneous-Dirichlet exact u from CFD/Assignment2_1_b.m."""
    return (x**4 - 2*x**3 + x + y**4 - 2*y**3 + y) / 6.0


def zero_boundary_polynomial(x: Array, y: Array) -> Array:
    """Alternative exact u for the SAME source, with homogeneous Dirichlet data."""
    return x * (1.0 - x) * y * (1.0 - y)


def sine_exact(x: Array, y: Array) -> Array:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def sine_forcing(x: Array, y: Array) -> Array:
    return 2.0 * np.pi**2 * sine_exact(x, y)
