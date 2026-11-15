"""Conjugate-gradient solves with measured residual history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

Array = NDArray[np.float64]


@dataclass
class SolveResult:
    x: Array
    residuals: list[float]  # relative ||b-Ax||_2 / ||b||_2; includes initial
    info: int               # SciPy CG exit code: 0 = converged

    @property
    def iterations(self) -> int:
        return len(self.residuals) - 1

    @property
    def relative_residual(self) -> float:
        return self.residuals[-1]


def pcg_solve(A: sparse.spmatrix, b: Array,
              M: LinearOperator | None = None, rtol: float = 1e-9,
              maxiter: int | None = None) -> SolveResult:
    """Solve Ax=b by SciPy CG, optionally preconditioned; record true residuals."""
    b = np.asarray(b, dtype=float)
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b has incorrect shape")
    scale = float(np.linalg.norm(b))
    if scale == 0:
        return SolveResult(np.zeros_like(b), [0.0], 0)
    history = [float(np.linalg.norm(b) / scale)]

    def callback(xk: Array) -> None:
        history.append(float(np.linalg.norm(b - A @ xk) / scale))

    x, info = cg(A, b, M=M, rtol=rtol, atol=0.0,
                 maxiter=maxiter, callback=callback)
    actual = float(np.linalg.norm(b - A @ x) / scale)
    if len(history) == 1 or not np.isclose(history[-1], actual, rtol=1e-12, atol=1e-15):
        # No callback for immediate convergence; otherwise use the fresh true residual.
        if len(history) == 1:
            history.append(actual)
        else:
            history[-1] = actual
    return SolveResult(np.asarray(x), history, int(info))
