import numpy as np

from problem import sample_test_problem
from quantum_ising import build_total_hamiltonian


def compute_gap_for_s(problem, s: float) -> float:
    """
    H(s) のスペクトルを計算し、基底状態と第1励起状態のギャップ Δ(s) を返す
    """
    Hs = build_total_hamiltonian(problem, s)
    evals, _ = np.linalg.eigh(Hs)

    evals = np.sort(evals.real)
    E0 = evals[0]
    E1 = evals[1]
    gap = E1 - E0
    return E0, E1, gap


def main():
    problem = sample_test_problem()
    print("=== Spectrum scan for H(s) ===")
    print(f"N = {problem.N}")

    num_points = 21  # s = 0.00, 0.05, ..., 1.00
    s_values = [i / (num_points - 1) for i in range(num_points)]

    min_gap = float("inf")
    min_gap_s = None

    print("s\tE0\t\tE1\t\tgap")
    for s in s_values:
        E0, E1, gap = compute_gap_for_s(problem, s)
        print(f"{s:.2f}\t{E0:.6f}\t{E1:.6f}\t{gap:.6f}")

        if gap < min_gap:
            min_gap = gap
            min_gap_s = s

    print("\n=== Summary of gap ===")
    print(f"Minimum gap Δ_min ≈ {min_gap:.6f} at s ≈ {min_gap_s:.3f}")


if __name__ == "__main__":
    main()
