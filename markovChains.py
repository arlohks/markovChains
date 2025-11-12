import numpy as np  # Maths module used for most things
import matplotlib.pyplot as plt  # Plotting library to plot maths things
import argparse, pathlib
from scipy.linalg import (
    eig,
)  # Used because I don't know how to get left eigenvectors using numpy

import networkx as nx  # checking irreducability


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
    #    A = np.array(P)
    A = np.eye(m)
    for k in range(n + 1):
        seqs[k] = A
        if k < n:
            A = A @ P
    np.savetxt(
        "out/nth-step-matrix.csv", seqs.reshape(-1, seqs.shape[-1]), delimiter=","
    )
    return seqs


def matrix_of_convergence(P):
    """Make a matrix of convergence e.g. collect the n-step for which an entry ij convergences
    Use plateau_step_entry to find if it returns None or an integer for a specific entry ij
    And then add it to a matrix (this matrix is the same size as the matrix we are analysing)"""
    m = P.shape[0]  # Record the shape of the matrix we analyse to use later
    M = np.empty(
        [m, m]
    )  # Create an empty variable of matrix type with the same size as the matrix we are analysing
    for i in range(m):  # Check each individual i entry
        for j in range(m):  # Check each individual j entry
            M[i, j] = plateau_step_entry(
                P, i, j
            )  # Record the n^th-step at which this entry converges
    np.savetxt(
        "out/convergence_matrix.csv", M, delimiter=","
    )  # Save the convergence matrix to text
    plt.figure()  # Initiate plot
    plt.imshow(M, origin="lower")
    plt.colorbar()  # Convert each value to a heat map
    plt.title("Convergence matrix P")  # Name the table plotted
    plt.xlabel("to j")  # Label the x-axis
    plt.savefig("out/convergence_heatmap.png", dpi=200)  # Saves the image to directory
    plt.close()
    return M  # Return the convergence matrix


def entry_map(P, outdir, plotcond):
    """Function which draws the map of all entries of the matrix P after n-steps"""
    m = P.shape[0]  # Record the shape of the matrix we analyse to use later
    M = matrix_of_convergence(
        P
    )  # Find the matrix of convergence and record it to memory using matrix_of_convergence function
    plateau = np.nanmax(
        M
    ).astype(
        int
    )  # matrix_of_convergence records entries in np.nan (Not a Number) [https://numpy.org/doc/stable/reference/constants.html#numpy.nan]
    # and np.max(M) returns np.nan, so find the next biggest number, and record it as an integer
    # This number is the n^th-step for which all entries converge
    if np.isnan(
        plateau
    ):  # Checks if the whole convergence matrix has all entries np.nan (nothing converges)
        seqs = all_entries_vs_n(P, 20)
        ns = np.arange((20) + 1)
    else:  # Unless all entries are np.nan, evaluate each entry over n-steps as n gets large
        seqs = all_entries_vs_n(P, plateau)
        ns = np.arange((plateau) + 1)

    plt.figure()  # Initiate plot
    for i in range(m):  # Check each individual i entry
        for j in range(m):  # Check each individual j entry
            if plotcond == True:
                if not (
                    np.isnan(M[i, j])
                ):  # Only plot entries which converge (one's which won't be np.nan)
                    plt.plot(ns, seqs[:, i, j], marker="o", label=f"({i},{j})")
            else:
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
    print(vals)
    print(left)
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


def is_reducible(P):
    """A chain is reducible if some states cannot be reached from others.
    This means the state space splits into separate, non-communicating classes,
    and the chain cannot explore the whole state space from every starting point."""
    reducible = np.empty([1, 1])
    G = nx.DiGraph(P > 0)
    reducible[0] = nx.is_strongly_connected(G)
    np.savetxt("out/reducible.txt", np.array(reducible))
    return reducible


def is_periodic(P):
    """A chain is periodic if it can return to a state only at multiples
    of some integer greater than 1. For example, if a state can only be revisited
    every 2 steps, the chain is period 2. Periodicity can prevent convergence to
    a single stationary distribution in the usual sense, even if one exists."""
    periodic = True
    return periodic


def main():
    ap = argparse.ArgumentParser(description="Reproducible Markov chain utilities")
    ap.add_argument(
        "--mode",
        choices=["entry-map", "heatmap", "stationary", "reducible", "periodic"],
        default="stationary",
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
        entry_map(P, args.outdir, plotcond=False)
    elif args.mode == "heatmap":
        transition_heatmap(P, args.outdir)
    elif args.mode == "stationary-distribution":
        stationary_distribution(P, args.outdir)
    elif args.mode == "balance-residuals":
        print("Work in progress...")
    elif args.mode == "row-distribution-convergence":
        print("Work in progress...")
    elif args.mode == "reducible":
        is_reducible(P)
    elif args.mode == "periodic":
        is_periodic(P)


if __name__ == "__main__":
    main()
