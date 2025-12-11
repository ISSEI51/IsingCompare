from problem import sample_test_problem
from brute_force import brute_force_ground_state
from sa_solver import simulated_annealing

# 量子イジング関連
from quantum_ising import (
    build_problem_hamiltonian,
    ground_state,
    dominant_spin_config,
)


def main():
    problem = sample_test_problem()

    print("=== Ising Problem: Classical & Quantum Baseline ===")
    print(f"N = {problem.N}")
    print("J couplings:", problem.J)
    print("h fields   :", problem.h)

    # --- 1. Classical: Brute Force ---
    print("\n[Brute Force] searching exact ground state...")
    exact_E, exact_spins = brute_force_ground_state(problem)
    print("Exact ground energy:", exact_E)
    print("Exact ground spins :", exact_spins)

    # --- 2. Classical: Simulated Annealing ---
    trials = 5
    print(f"\n[Simulated Annealing] {trials} trials")
    best_E_sa = float("inf")
    best_spins_sa = None

    for k in range(trials):
        E_sa, spins_sa = simulated_annealing(problem)
        print(f" Trial {k+1}: E = {E_sa:.4f}, spins = {spins_sa}")
        if E_sa < best_E_sa:
            best_E_sa = E_sa
            best_spins_sa = spins_sa

    # --- 3. Quantum: Problem Hamiltonian H_p^Q の基底状態 ---
    print("\n[Quantum Ising] Diagonalizing H_p^Q ...")
    H_p = build_problem_hamiltonian(problem)
    E0_q, psi0 = ground_state(H_p)
    dom_spins, prob = dominant_spin_config(psi0, problem.N)

    print("Quantum ground energy (H_p^Q):", E0_q)
    print("Dominant classical config    :", dom_spins)
    print("Probability of that config   :", prob)

    # --- 4. Summary ---
    print("\n=== Summary ===")
    print("Exact ground energy (classical H_p):", exact_E)
    print("Best SA energy                :", best_E_sa)
    print("Quantum ground energy (H_p^Q) :", E0_q)
    print("Difference (exact vs quantum) :", E0_q - exact_E)


if __name__ == "__main__":
    main()
