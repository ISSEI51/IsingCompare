import numpy as np
from typing import Tuple

from problem import IsingProblem
from quantum_ops import sigma_z_i, sigma_x_i


def build_problem_hamiltonian(problem: IsingProblem) -> np.ndarray:
    """
    H_p^Q = -Σ J_ij σ_i^z σ_j^z - Σ h_i σ_i^z を行列として構成
    """
    N = problem.N
    dim = 2**N

    H = np.zeros((dim, dim), dtype=complex)

    # 相互作用項
    for (i, j), Jij in problem.J.items():
        Zi = sigma_z_i(N, i)
        Zj = sigma_z_i(N, j)
        H += -Jij * (Zi @ Zj)

    # ローカル場項
    for i, hi in problem.h.items():
        Zi = sigma_z_i(N, i)
        H += -hi * Zi

    return H


def build_driver_hamiltonian(problem: IsingProblem) -> np.ndarray:
    """
    H_d = -Σ σ_i^x
    """
    N = problem.N
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N):
        Xi = sigma_x_i(N, i)
        H += -Xi

    return H


def build_total_hamiltonian(
    problem: IsingProblem,
    s: float,
    A0: float = 1.0,
    B0: float = 1.0,
) -> np.ndarray:
    """
    H(s) = A(s) H_d + B(s) H_p^Q
    ここでは簡単のため
      A(s) = A0 * (1 - s)
      B(s) = B0 * s
    とする。
    """
    H_p = build_problem_hamiltonian(problem)
    H_d = build_driver_hamiltonian(problem)

    A_s = A0 * (1.0 - s)
    B_s = B0 * s

    return A_s * H_d + B_s * H_p


def ground_state(H: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    ハミルトニアン H の基底エネルギーと固有ベクトルを返す。
    """
    # eigh: Hermitian行列用
    evals, evecs = np.linalg.eigh(H)
    idx = np.argmin(evals.real)
    E0 = evals[idx].real
    psi0 = evecs[:, idx]
    return E0, psi0


def index_to_spins_z_basis(idx: int, N: int):
    """
    量子側の z 基底 |b_{N-1} ... b_1 b_0> に対して、
    - b = 0 → σ^z = +1
    - b = 1 → σ^z = -1
    とし、かつ「スピン i = ビット i」の対応が
    古典側（int_to_spins）と揃うように、
    上位ビットから読み出している。
    """
    spins = []
    for i in range(N):
        bit_pos = N - 1 - i  # 上位ビットから順に読む
        bit = (idx >> bit_pos) & 1
        spins.append(1 if bit == 0 else -1)
    return spins


def dominant_spin_config(psi: np.ndarray, N: int):
    """
    基底状態ベクトル psi の中で、振幅の絶対値が最大の基底状態
    （|σ^z> 基底）を見つけ、そのスピン配置を ±1 のリストとして返す。
    """
    dim = len(psi)
    assert dim == 2**N

    probs = np.abs(psi) ** 2
    idx = int(np.argmax(probs))

    spins = index_to_spins_z_basis(idx, N)
    return spins, probs[idx]
