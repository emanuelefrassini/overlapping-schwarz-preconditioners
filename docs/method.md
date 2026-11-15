# Numerical formulation

## Model problem and the connection to the original coursework

We solve the *scalar*, steady elliptic problem on the unit square:

$$
-\Delta u=f\quad\text{in }\Omega=(0,1)^2,
\qquad u=g\quad\text{on }\partial\Omega.
$$

The original MATLAB files [`additive_prec.m`](../legacy/coursework/additive_prec.m) and [`Assignment2_1_b.m`](../legacy/coursework/Assignment2_1_b.m) use

$$
f(x,y)=2\big[x(1-x)+y(1-y)\big].
$$

`Assignment2_1_b.m` also defines a polynomial

$$
u_{\rm coursework}(x,y)=\tfrac16(x^4-2x^3+x+y^4-2y^3+y).
$$

This polynomial **does not vanish on all four boundaries**. To verify against it, the Python implementation uses its *actual nonzero* boundary trace as the Dirichlet data `g`. Direct differentiation gives $-\Delta u_{\rm coursework}=f$.

A separate exact function with **homogeneous** boundary conditions and the *same source* is

$$
u_{\rm zero}(x,y)=x(1-x)y(1-y).
$$

The central-difference stencil is exact for this particular polynomial on every interior node, making it a strong indexing and boundary-elimination test. An additional trigonometric manufactured solution tests genuine second-order truncation error instead of relying solely on a polynomial that the scheme reproduces exactly.

## Sparse finite-difference operator

For `n` interior nodes per coordinate, $h=1/(n+1)$, and the unknown vector contains $n^2$ **interior** grid values in row-major order. The five-point discretisation is

$$
(Au)_{i,j}=\frac{4u_{i,j}-u_{i-1,j}-u_{i+1,j}-u_{i,j-1}-u_{i,j+1}}{h^2}.
$$

Boundary-node values are not unknowns; they are added to the appropriate entries of the right-hand side with coefficient $1/h^2$. The matrix is symmetric positive definite (SPD).

Writing $T=\operatorname{tridiag}(-1,2,-1)$, the sparse Kronecker construction is

$$
A=h^{-2}\big(I_n\otimes T+T\otimes I_n\big).
$$

This is a five-point finite-difference **reformulation** of the Poisson coursework, not an exact implementation of the nine-point coefficient matrix attempted in `additive_prec.m`. The latter includes boundary degrees of freedom without consistently eliminating Dirichlet conditions and is not used for reference accuracy claims.

## One-level overlapping additive Schwarz

Split the interior grid into $k\times k$ disjoint rectangular cores, distributing any remainder evenly. Enlarge each core by `overlap` fine-grid node layers in x and y, clipping at the physical boundary. For each enlarged subdomain $\Omega_i$, let $R_i$ select its global degrees of freedom. Form

$$
A_i=R_i A R_i^{\mathsf T},\qquad
M_1^{-1}=\sum_i R_i^{\mathsf T} A_i^{-1}R_i.
$$

Each local sparse matrix is LU-factorized **once**; each preconditioner application performs the local solves and adds overlapping contributions. The dense matrix $M_1^{-1}$ is never explicitly assembled. We deliberately **do not** divide by overlap multiplicity: this is the conventional unweighted additive Schwarz formulation written above.

With SPD $A$, local principal submatrices are SPD, and the restrictions cover every unknown, so the additive preconditioner is also SPD. This makes it suitable for conjugate gradients. The implementation is serial; the additive structure could be parallelized, but MPI/multicore performance is not demonstrated here.

## Two-level coarse correction

Use a separate coarse grid with `k` equal-sized cells per axis and $(k-1)^2$ interior coarse nodes. $P$ is the **tensor-product piecewise-bilinear prolongation** to all fine-grid interior nodes; zero values are assumed on the coarse boundary. The Galerkin coarse matrix is

$$
A_0=P^{\mathsf T} A P.
$$

For the tested grids $n=km-1$, the coarse grid is nested within the fine mesh; the implementation also supports nonnested interpolation. The two-level additive preconditioner is

$$
M_2^{-1}=M_1^{-1}+PA_0^{-1}P^{\mathsf T}.
$$

The coarse matrix is factored once. The purpose of this correction is to address error components that local subdomain solves remove less efficiently. The method is an additive coarse correction, **not** a multiplicative, restricted additive Schwarz, or algebraic multigrid method.

## Solver and measurements

Use SciPy `scipy.sparse.linalg.cg` with $M^{-1}$ supplied via a `LinearOperator`. The callback recomputes the **true** relative Euclidean residual $\|b-Au_j\|_2/\|b\|_2$ at every iteration; termination uses SciPy CG's relative tolerance (`rtol=1e-9` for benchmarks, zero absolute tolerance).

The benchmark uses fixed-seed random vectors as right-hand sides. This is a *spectral solver stress test*, not the physically motivated coursework forcing; the separate Poisson demo uses the original forcing and its polynomial exact solution. Both experiments are reproducible. We report **iteration counts and residuals**, not runtime speedups, parallel scalability, or a proof of mesh-independent convergence. Local sparse factorization and coarse-grid setup are extra costs that would matter in a runtime comparison.
