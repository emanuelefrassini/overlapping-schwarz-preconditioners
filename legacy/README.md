# Original MATLAB CFD coursework (unaltered)

The `.m` files in `coursework/` are copied verbatim from the user's original `CFD.zip`. They are historical coursework references, **not** the working implementation and not required to execute the Python package.

- `additive_prec.m`: attempts an additive overlapping subdomain preconditioner, an additional coarse correction, and PCG; source forcing `2*(x*(1-x)+y*(1-y))`.
- `ass2_2.m`: related experiment, begins with an incomplete `function []=` line.
- `Assignment2_1.m`, `Assignment2_1_a.m`, `Assignment2_1_b.m`: experimental overlapping/alternating Schwarz indexing and sweeps. `Assignment2_1_a.m` contains a comment noting that repeated iterations produce the same solution.
- `Assignment2_1_b.m` defines a polynomial reference solution with **nonzero boundary traces**; the historical script also overwrites reference boundary values with zero, so it is not a verified homogeneous-Dirichlet comparison.

**Caution:** `additive_prec.m` constructs a grid containing boundary nodes without imposing the desired Dirichlet conditions in the assembled matrix; it also allocates a dense preconditioner and has inconsistent restriction/coarse-index handling. These make the historical results unsafe to treat as validated. The new Python implementation uses interior DOFs only, explicit Dirichlet elimination, sparse local factorizations and a Galerkin coarse operator.

The numerical experiment is a scalar Poisson problem from a CFD course, not a Navier–Stokes flow solver. No attribution or endorsement of the new Python rewrite by course instructors is implied.
