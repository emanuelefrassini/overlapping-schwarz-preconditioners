"""Serial CG iteration benchmarks; separate setup cost from iteration counts.

Use deterministic random RHS so that the full spectrum of error components is
excited. Numbers are not parallel timings and are not CFD flow benchmarks.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from schwarz_dd import (
    AdditiveSchwarz, pcg_solve, poisson_matrix, subdomain_indices,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("CG", "One-level Schwarz", "Two-level Schwarz")


def benchmark(k: int, subintervals: int = 8, overlap: int = 2):
    n = k * subintervals - 1  # fine grid n+1=km; coarse grid k cells
    A = poisson_matrix(n)
    b = np.random.default_rng(1000 + k).standard_normal(n*n)
    one = AdditiveSchwarz(A, n, k, overlap)
    two = AdditiveSchwarz(A, n, k, overlap, two_level=True)
    return n, {
        "CG": pcg_solve(A, b, None, rtol=1e-9),
        "One-level Schwarz": pcg_solve(A, b, one.operator, rtol=1e-9),
        "Two-level Schwarz": pcg_solve(A, b, two.operator, rtol=1e-9),
    }


def plot_overlap(n: int = 23, k: int = 3, overlap: int = 2) -> None:
    indices = subdomain_indices(n,k,overlap)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    coords = np.arange(1,n+1)/(n+1)
    for a, idx in enumerate(indices):
        ys, xs = np.divmod(idx, n)
        x0, x1 = coords[xs.min()], coords[xs.max()]
        y0, y1 = coords[ys.min()], coords[ys.max()]
        color = f"C{a%10}"
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=color,alpha=.11,
                               edgecolor=color,lw=2.0))
        ax.text((x0+x1)/2,(y0+y1)/2,f"Ω{a+1}",ha="center",va="center",fontsize=10)
    ax.scatter(np.repeat(coords,n),np.tile(coords,n),s=3,alpha=.22,color="black")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(f"{k}×{k} overlapping subdomains · {overlap} fine-grid layers")
    fig.savefig(ROOT / "figures" / "subdomain_overlap.png",dpi=180)
    plt.close(fig)


def main() -> None:
    (ROOT/"figures").mkdir(exist_ok=True)
    (ROOT/"results").mkdir(exist_ok=True)
    overlap = 2
    records = []
    selected = None
    for k in (2,3,4,5,6):
        n, results = benchmark(k, overlap=overlap)
        print(f"k={k}, n={n}, unknowns={n*n}")
        for name in METHODS:
            r=results[name]
            if r.info != 0 or r.relative_residual > 1.2e-9:
                raise RuntimeError(f"{name} failed at k={k}: info={r.info}, r={r.relative_residual}")
            records.append({"subdomains_per_axis":k,"total_subdomains":k*k,
                            "interior_nodes_per_axis":n,"unknowns":n*n,
                            "overlap_layers":overlap,"method":name,
                            "iterations":r.iterations,
                            "relative_residual":f"{r.relative_residual:.12e}"})
            print(f"  {name:20s}: {r.iterations:4d} iterations, residual {r.relative_residual:.2e}")
        if k==5:
            selected=results

    with (ROOT / "results" / "iteration_counts.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    fig, ax=plt.subplots(figsize=(8.2,4.7),constrained_layout=True)
    for method in METHODS:
        rows=[r for r in records if r["method"]==method]
        ax.plot([r["total_subdomains"] for r in rows],
                [r["iterations"] for r in rows],marker="o",lw=2,label=method)
    ax.set_xlabel("Number of subdomains (k²)")
    ax.set_ylabel("CG iterations to relative residual ≤ 10⁻⁹")
    ax.set_title("Iteration counts as the mesh and subdomain count grow")
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(ROOT / "figures" / "scaling_iterations.png",dpi=180)
    plt.close(fig)

    assert selected is not None
    fig,ax=plt.subplots(figsize=(8.2,4.7),constrained_layout=True)
    for method in METHODS:
        vals=selected[method].residuals
        ax.semilogy(range(len(vals)),np.maximum(vals,1e-16),lw=2,label=f"{method} ({len(vals)-1} iters)")
    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("True relative residual ‖b−Ax‖₂ / ‖b‖₂")
    ax.set_title("Convergence at k=5, n=39 (1,521 unknowns)")
    ax.grid(alpha=.3,which="both")
    ax.legend()
    fig.savefig(ROOT / "figures" / "pcg_residuals.png",dpi=180)
    plt.close(fig)

    plot_overlap()
    print("Wrote results/iteration_counts.csv and three figures.")


if __name__ == "__main__":
    main()
