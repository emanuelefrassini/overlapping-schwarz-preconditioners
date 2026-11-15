"""Independent structural and manufactured-solution checks for Poisson assembly."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from schwarz_dd import (
    coursework_forcing, coursework_polynomial, dirichlet_rhs, full_solution,
    interior_grid, poisson_matrix, sine_exact, sine_forcing,
    zero_boundary_polynomial,
)


@pytest.mark.parametrize("n", [1, 3, 9])
def test_matrix_sparse_symmetric_positive_definite(n):
    A = poisson_matrix(n)
    assert A.shape == (n*n, n*n)
    assert (A - A.T).nnz == 0
    assert np.linalg.eigvalsh(A.toarray())[0] > 0


@pytest.mark.parametrize("n", [3, 7, 15])
def test_original_forcing_has_exact_zero_boundary_discrete_solution(n):
    """Quadratic-in-each-variable manufactured u has exact 5-point Laplacian."""
    A = poisson_matrix(n)
    b = dirichlet_rhs(n, coursework_forcing)
    X, Y = interior_grid(n)
    u = zero_boundary_polynomial(X, Y).ravel()
    np.testing.assert_allclose(A @ u, b, atol=3e-13, rtol=5e-13)
    np.testing.assert_allclose(spsolve(A, b), u, atol=1e-13)


@pytest.mark.parametrize("n", [7, 15, 31])
def test_original_nonzero_boundary_polynomial_is_second_order(n):
    A = poisson_matrix(n)
    b = dirichlet_rhs(n, coursework_forcing, coursework_polynomial)
    X, Y = interior_grid(n)
    error = np.max(np.abs(spsolve(A, b) - coursework_polynomial(X, Y).ravel()))
    assert error > 0
    assert error < 0.1 / (n+1)**2


def test_original_nonzero_boundary_error_refines_quadratically():
    e = []
    for n in (7, 15, 31):
        A = poisson_matrix(n)
        b = dirichlet_rhs(n, coursework_forcing, coursework_polynomial)
        X, Y = interior_grid(n)
        e.append(np.max(np.abs(spsolve(A, b) - coursework_polynomial(X,Y).ravel())))
    assert 3.4 < e[0]/e[1] < 4.6
    assert 3.4 < e[1]/e[2] < 4.6


def test_sine_manufactured_solution_converges_second_order():
    e = []
    for n in (7, 15, 31):
        X, Y = interior_grid(n)
        b = dirichlet_rhs(n, sine_forcing)
        numerical = spsolve(poisson_matrix(n), b)
        e.append(np.max(np.abs(numerical - sine_exact(X,Y).ravel())))
    assert 3.6 < e[0]/e[1] < 4.2
    assert 3.6 < e[1]/e[2] < 4.2


def test_rhs_accounts_for_boundary_on_every_side():
    n = 3
    A = poisson_matrix(n)
    def g(x, y):
        return x + 2.0*y + 3.0
    b = dirichlet_rhs(n, lambda x,y: 0*x, g)
    X,Y = interior_grid(n)
    exact = g(X,Y).ravel()
    np.testing.assert_allclose(A @ exact, b, atol=2e-13)


def test_full_solution_reconstructs_nonzero_boundary():
    n = 5
    interior = np.arange(n*n, dtype=float)
    u = full_solution(interior, n, coursework_polynomial)
    assert u.shape == (n+2, n+2)
    np.testing.assert_array_equal(u[1:-1,1:-1], interior.reshape(n,n))
    x = np.linspace(0, 1, n+2)
    np.testing.assert_allclose(u[0, :], coursework_polynomial(x, np.zeros_like(x)))
    np.testing.assert_allclose(u[-1, :], coursework_polynomial(x, np.ones_like(x)))


def test_constant_rhs_broadcasting():
    b = dirichlet_rhs(4, lambda x,y: 2.0)
    np.testing.assert_allclose(b, 2.0)


def test_invalid_grid_sizes():
    for n in (0,-1):
        with pytest.raises(ValueError):
            poisson_matrix(n)
        with pytest.raises(ValueError):
            interior_grid(n)
    with pytest.raises(ValueError):
        full_solution(np.ones(3), 2)


def test_nonfinite_field_is_rejected():
    with pytest.raises(ValueError):
        dirichlet_rhs(5, lambda x,y: float('nan'))
