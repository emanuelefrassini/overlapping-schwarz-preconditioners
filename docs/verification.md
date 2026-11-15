# Numerical verification and limitations

## Tests

`python -m pytest -q` checks:

- Sparse five-point matrix symmetry, dimensions, and positive definiteness at multiple sizes.
- Recovery of the exact homogeneous-Dirichlet polynomial solution associated with the original coursework source.
- Approximately second-order convergence for the original *nonhomogeneous*-Dirichlet polynomial.
- Approximately second-order convergence for the independent sine manufactured solution.
- Correct elimination of boundary data on all four sides and reconstruction of full-grid fields.
- Subdomain coverage, unique local indices, overlap enlargement, and invalid-input handling.
- Tensor-product coarse interpolation and the SPD Galerkin coarse matrix.
- Positive definiteness / symmetry of the *action* of the additive preconditioners.
- Agreement of unpreconditioned and both preconditioned CG solutions with a sparse direct solve on small systems.
- Exact-inverse behavior in the single-subdomain limit.

The test suite does **not** independently validate the historical MATLAB Schwarz drafts. The modern Python version is a transparent numerical reconstruction of the mathematical topic.

## Measured benchmark protocol

Run:

```bash
python scripts/run_demo.py
python scripts/run_benchmarks.py
```

- Python package: NumPy + SciPy + Matplotlib, no MATLAB licence needed.
- Five-point SPD Poisson matrix with `n = 8k - 1` interior grid nodes per side, `k = 2..6`.
- `k*k` subdomains, each extended by **two interior-node layers**.
- Two-level coarse grid: `k` coarse cells per side, `(k-1)^2` coarse unknowns.
- Fixed RNG seed `1000+k`, different deterministic random RHS per grid.
- Each CG solve starts from zero and stops at `rtol=1e-9`, `atol=0`.
- All reported residuals are explicitly recomputed using the assembled sparse matrix.
- See [`results/iteration_counts.csv`](../results/iteration_counts.csv) for the exact logged outputs.

The iteration counts are a useful controlled illustration of the coarse correction and **not** timing benchmarks: local LU factorisation time, cache/memory use, and distributed communication costs are not included. Increasing `k` also increases `n` in this experiment. No claim of asymptotic iteration-independent behavior is made.

## Scope

- The PDE is scalar **Poisson**, not Navier–Stokes or a velocity-pressure saddle-point system.
- Fixed regular Cartesian grids and unit diffusivity; no spatially varying coefficients, irregular domains, anisotropy, or unstructured finite elements.
- Sparse local LU, serial reference implementation; no MPI/GPU support and no measured parallel speedup.
- Simple unweighted additive Schwarz and a bilinear Galerkin coarse space; alternative Schwarz variants not included.
- The benchmark right-hand sides are synthetic and not physical CFD data.
- Original MATLAB scripts are retained *unchanged* and are not executed in CI; they should not be considered a verified package.
