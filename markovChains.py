import numpy as np  # Maths module used for most things
import matplotlib.pyplot as plt  # Plotting library to plot maths things
import argparse, pathlib
from scipy.linalg import (
    eig,
)  # Used because I don't know how to get left eigenvectors using numpy


def load_matrix(matrix_file: pathlib.Path | None, size: int, seed: int) -> np.ndarray:
    if matrix_file:
        P = np.loadtxt(matrix_file, delimiter=",", dtype=float)
    else:
        rng = np.random.default_rng(seed)
        P = rng.random((size, size))
        P /= P.sum(axis=1, keepdims=True)
    return np.asarray(P, dtype=float)


def plateau_step_entry(P, i, j, N=1000, tol=1e-6, streak=3):
    """Function to find the nth-step at which entries at i,j
    tend to the same value (plateau) after n-steps"""
    A = np.eye(P.shape[0])  # Defines the identity matrix of the same size as P
    prev = A[i, j]
    run = 0
    for k in range(1, N + 1):
        A = A @ P
        cur = A[i, j]
        if abs(cur - prev) < tol:
            run += 1
            if run >= streak:
                return k - streak + 1  # first step of the flat streak
        else:
            run = 0
        prev = cur
    return None  # no plateau found within N


def all_entries_vs_n(P, n):
    """Function to compute the entry i,j after n-steps"""
    P = np.array(P, dtype=float)
    m = P.shape[0]
    seqs = np.empty((n + 1, m, m), dtype=float)
    A = np.eye(m)
    for k in range(n + 1):
        seqs[k] = A
        if k < n:
            A = A @ P
    return seqs


def entry_map(P, outdir):
    """Function which draws the map of all entries of the matrix P after n-steps"""
    m = P.shape[0]
    for i in range(m):
        for j in range(m):
            output = plateau_step_entry(P, i, j)

    seqs = all_entries_vs_n(P, output)
    ns = np.arange((output) + 1)
    plt.figure()
    for i in range(m):
        for j in range(m):
            plt.plot(ns, seqs[:, i, j], marker="o", label=f"({i},{j})")
    plt.xlabel("n (steps)")
    plt.ylabel("(P^n)")
    plt.title("n-step transition probability for one entry")
    plt.legend(ncol=m, fontsize="small")
    plt.tight_layout()
    plt.grid()
    plt.savefig(outdir / "entry_map.png", dpi=200)
    plt.close()


def transition_heatmap(P, outdir):
    plt.figure()
    plt.imshow(P, origin="lower")
    plt.colorbar()
    plt.title("Transition matrix P")
    plt.xlabel("to j")
    plt.savefig(outdir / "transition_heatmap.png", dpi=200)
    plt.close()


def stationary_distribution(P, outdir, tol=1e-12):
    vals, left = eig(P, left=True, right=False)
    # index of eigenvalue closest to 1
    idx = np.argmin(np.abs(vals - 1))
    pi = np.real(
        left[:, idx]
    )  # take the corresponding left eigenvector (column), real part
    # fix possible sign/negatives from numerical noise, then normalise
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s < tol:
        raise ValueError(
            "Could not extract a valid stationary distribution (check reducibility/periodicity)."
        )
    pi = pi / s
    np.savetxt(outdir / "stationary_distribution.csv", pi, delimiter=",")
    plt.figure()
    plt.bar(np.arange(pi.size), pi)
    plt.title("Stationary distribution")
    plt.tight_layout()
    plt.savefig(outdir / "stationary_distribution.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Reproducible Markov chain utilities")
    ap.add_argument(
        "--mode", choices=["entry-map", "heatmap", "stationary"], default="stationary"
    )
    ap.add_argument("--matrix-file", type=pathlib.Path)
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    P = load_matrix(args.matrix_file, args.size, args.seed)
    np.savetxt(args.outdir / "P.csv", P, delimiter=",")
    if args.mode == "entry-map":
        entry_map(P, args.outdir)
    elif args.mode == "heatmap":
        transition_heatmap(P, args.outdir)
    elif args.mode == "stationary-distribution":
        stationary_distribution(P, args.outdir)
    elif args.mode == "balance-residuals":
        print("Work in progress...")
    elif args.mode == "row-distribution-convergence":
        print("Work in progress...")


if __name__ == "__main__":
    main()
