"""SPD additive Schwarz preconditioners with a bilinear Galerkin coarse space.

M_1^{-1} = sum_i R_i.T A_i^{-1} R_i, A_i = R_i A R_i.T
M_2^{-1} = M_1^{-1} + P (P.T A P)^{-1} P.T

Restriction R_i is represented by an array of global DOF indices. All local
solves are factored ONCE and applied matrix-free; a dense inverse is never built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu

Array = NDArray[np.float64]


def subdomain_indices(n: int, blocks_per_axis: int, overlap: int) -> list[NDArray[np.intp]]:
    """Partition interior grid into k-by-k blocks; expand each by overlap nodes.

    Blocks partition all interior degrees of freedom before adding overlap.
    Overlap is measured in interior-node layers and clipped at the domain edge.
    """
    if n < 1 or not 1 <= blocks_per_axis <= n or overlap < 0:
        raise ValueError("require n>=1, 1<=blocks_per_axis<=n, overlap>=0")
    blocks = np.array_split(np.arange(n), blocks_per_axis)
    result = []
    for by in blocks:
        ys = np.arange(max(0, int(by[0]) - overlap),
                       min(n - 1, int(by[-1]) + overlap) + 1)
        for bx in blocks:
            xs = np.arange(max(0, int(bx[0]) - overlap),
                           min(n - 1, int(bx[-1]) + overlap) + 1)
            indices = (ys[:, None] * n + xs[None, :]).ravel().astype(np.intp)
            result.append(indices)
    return result


def coarse_prolongation_1d(n: int, coarse_cells: int) -> sparse.csr_matrix:
    """Piecewise-linear interpolation, with zero Dirichlet values at endpoints.

    Fine and coarse grids need not be nested. Coarse interior nodes are at
    j/coarse_cells, j=1,...,coarse_cells-1.
    """
    if n < 1 or coarse_cells < 2:
        raise ValueError("require n>=1 and coarse_cells>=2")
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for row in range(n):
        t = (row + 1) * coarse_cells / (n + 1)
        lower = int(np.floor(t))
        alpha = t - lower
        for node, weight in ((lower, 1.0 - alpha), (lower + 1, alpha)):
            if 1 <= node <= coarse_cells - 1 and weight > 0:
                rows.append(row)
                cols.append(node - 1)
                weights.append(weight)
    return sparse.coo_matrix((weights, (rows, cols)),
                             shape=(n, coarse_cells - 1)).tocsr()


def coarse_prolongation(n: int, blocks_per_axis: int) -> sparse.csr_matrix:
    """Tensor-product Q1 coarse interpolation to fine interior grid."""
    p1 = coarse_prolongation_1d(n, blocks_per_axis)
    return sparse.kron(p1, p1, format="csr")


@dataclass
class AdditiveSchwarz:
    """Matrix-free additive Schwarz with optional Galerkin coarse correction.

    This is a serial reference implementation, NOT parallel MPI Schwarz.
    """

    A: sparse.spmatrix
    n: int
    blocks_per_axis: int
    overlap: int = 1
    two_level: bool = False

    def __post_init__(self) -> None:
        if self.A.shape != (self.n**2, self.n**2):
            raise ValueError("A must be n*n by n*n")
        if self.two_level and self.blocks_per_axis < 2:
            raise ValueError("two_level requires at least 2 blocks per axis")
        if not sparse.issparse(self.A):
            raise TypeError("A must be a SciPy sparse matrix")
        self.A = self.A.tocsr()
        self.indices = subdomain_indices(self.n, self.blocks_per_axis, self.overlap)
        self.local_factors = [splu(self.A[ix, :][:, ix].tocsc()) for ix in self.indices]
        self.P = None
        self.A0 = None
        self.coarse_factor = None
        if self.two_level:
            self.P = coarse_prolongation(self.n, self.blocks_per_axis)
            self.A0 = (self.P.T @ self.A @ self.P).tocsc()
            self.coarse_factor = splu(self.A0)
        self.operator = LinearOperator(self.A.shape, matvec=self.apply, dtype=float)

    def apply(self, residual: Array) -> Array:
        r = np.asarray(residual, dtype=float)
        if r.shape != (self.n**2,):
            raise ValueError("residual has incorrect shape")
        z = np.zeros_like(r)
        for indices, lu in zip(self.indices, self.local_factors):
            z[indices] += lu.solve(r[indices])
        if self.two_level:
            assert self.P is not None and self.coarse_factor is not None
            z += np.asarray(self.P @ self.coarse_factor.solve(self.P.T @ r)).ravel()
        return z
