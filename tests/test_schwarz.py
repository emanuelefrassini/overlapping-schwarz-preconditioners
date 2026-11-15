"""Preconditioner structure, SPD, coarse correction, and PCG regression tests."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from schwarz_dd import (
    AdditiveSchwarz, coarse_prolongation, coarse_prolongation_1d,
    coursework_forcing, dirichlet_rhs, pcg_solve, poisson_matrix,
    subdomain_indices,
)


@pytest.mark.parametrize("n,k,overlap", [(9,3,0), (9,3,2), (11,4,1), (5,5,1)])
def test_subdomain_coverage_and_validity(n,k,overlap):
    idx = subdomain_indices(n,k,overlap)
    assert len(idx)==k*k
    counts = np.zeros(n*n, dtype=int)
    for ix in idx:
        assert 0 < len(ix) <= n*n
        assert np.all((ix >= 0) & (ix < n*n))
        assert len(np.unique(ix)) == len(ix)
        counts[ix] += 1
    assert np.all(counts >= 1)


def test_overlap_expands_and_increases_redundancy():
    no_overlap = subdomain_indices(12,3,0)
    two_layers = subdomain_indices(12,3,2)
    assert sum(map(len,two_layers)) > sum(map(len,no_overlap))
    for a,b in zip(no_overlap,two_layers):
        assert set(a).issubset(set(b))


@pytest.mark.parametrize("kwargs", [
    dict(n=0,blocks_per_axis=2,overlap=1),
    dict(n=4,blocks_per_axis=0,overlap=0),
    dict(n=3,blocks_per_axis=4,overlap=0),
    dict(n=4,blocks_per_axis=2,overlap=-1),
])
def test_invalid_subdomains(kwargs):
    with pytest.raises(ValueError):
        subdomain_indices(**kwargs)


def test_coarse_interpolation_at_aligned_nodes():
    # n+1=12 fine cells, k=3 coarse cells; interior coarse nodes x=4/12,8/12.
    P1 = coarse_prolongation_1d(11,3)
    assert P1.shape == (11,2)
    np.testing.assert_allclose(P1.toarray()[[3,7]], np.eye(2))
    np.testing.assert_allclose(P1.toarray()[[0,10]], [[0.25,0], [0,0.25]])


def test_coarse_prolongation_non_nested_grid():
    P = coarse_prolongation(8,3)
    assert P.shape == (64,4)
    assert P.nnz > 0
    assert np.linalg.matrix_rank(P.toarray()) == 4


@pytest.mark.parametrize("two_level", [False, True])
def test_preconditioner_positive_symmetric(two_level):
    n=9
    M=AdditiveSchwarz(poisson_matrix(n),n,3,2,two_level)
    rng=np.random.default_rng(123)
    v,w=rng.standard_normal((2,n*n))
    assert float(v @ M.apply(v)) > 0
    vw = float(v @ M.apply(w))
    wv = float(w @ M.apply(v))
    assert np.isclose(vw,wv,atol=1e-11,rtol=5e-12)


def test_preconditioner_uses_local_sparse_factors():
    n=12
    M=AdditiveSchwarz(poisson_matrix(n),n,3,1)
    assert len(M.local_factors)==9
    assert all(not isinstance(f,sparse.spmatrix) for f in M.local_factors)
    assert M.P is None
    assert M.A0 is None


def test_single_full_domain_preconditioner_is_exact_inverse():
    n=8
    A=poisson_matrix(n)
    M=AdditiveSchwarz(A,n,1,0)
    r=np.arange(1,n*n+1,dtype=float)
    np.testing.assert_allclose(A @ M.apply(r),r,atol=1e-11,rtol=1e-12)


def test_two_level_coarse_operator_is_galerkin_and_spd():
    n,k=11,3
    A=poisson_matrix(n)
    M=AdditiveSchwarz(A,n,k,1,two_level=True)
    assert M.P.shape==(n*n,(k-1)**2)
    np.testing.assert_allclose(M.A0.toarray(),(M.P.T @ A @ M.P).toarray())
    assert np.linalg.eigvalsh(M.A0.toarray())[0] > 0
    v=np.ones(n*n)
    one=AdditiveSchwarz(A,n,k,1)
    extra=M.apply(v)-one.apply(v)
    expected=M.P @ spsolve(M.A0, M.P.T @ v)
    np.testing.assert_allclose(extra,expected,atol=1e-12)


@pytest.mark.parametrize("n,k,overlap", [(11,3,1),(15,4,2)])
def test_all_cg_variants_match_direct(n,k,overlap):
    A=poisson_matrix(n)
    b=dirichlet_rhs(n,coursework_forcing)
    target=spsolve(A,b)
    for preconditioner in [None,
            AdditiveSchwarz(A,n,k,overlap).operator,
            AdditiveSchwarz(A,n,k,overlap,two_level=True).operator]:
        result=pcg_solve(A,b,M=preconditioner,rtol=1e-10)
        assert result.info == 0
        assert result.relative_residual < 1.2e-10
        assert result.iterations >= 1
        np.testing.assert_allclose(result.x,target,rtol=1e-8,atol=1e-9)


def test_zero_rhs_converges_immediately():
    A=poisson_matrix(3)
    result=pcg_solve(A,np.zeros(9))
    assert result.info==0
    assert result.relative_residual==0.0
    np.testing.assert_allclose(result.x,0.0)


def test_wrong_rhs_size_and_bad_matrix_shape():
    A=poisson_matrix(3)
    with pytest.raises(ValueError):
        pcg_solve(A,np.ones(8))
    with pytest.raises(ValueError):
        AdditiveSchwarz(A,n=4,blocks_per_axis=2)
    with pytest.raises(ValueError):
        AdditiveSchwarz(A,n=3,blocks_per_axis=1,two_level=True)
    with pytest.raises(ValueError):
        coarse_prolongation_1d(3,1)
    M=AdditiveSchwarz(A,n=3,blocks_per_axis=2)
    with pytest.raises(ValueError):
        M.apply(np.ones(8))
