import random
import math
from typing import List, Tuple
from problem import IsingProblem


def random_spins(N: int) -> List[int]:
    return [random.choice([-1, 1]) for _ in range(N)]


def delta_energy_flip(problem: IsingProblem, spins: List[int], flip_idx: int) -> float:
    """
    スピン flip_idx を σ_i -> -σ_i に反転したときの ΔE を計算
    ΔE = E(new) - E(old)
    """
    i = flip_idx
    sigma_i = spins[i]

    dE = 0.0
    # 相互作用項: -J_ij σ_i σ_j
    for j in range(problem.N):
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in problem.J:
            Jij = problem.J[key]
            sigma_j = spins[j]
            # 元の寄与: -J_ij σ_i σ_j
            # 反転後:   -J_ij (-σ_i) σ_j = +J_ij σ_i σ_j
            dE += (+Jij * sigma_i * sigma_j) - (-Jij * sigma_i * sigma_j)

    # ローカル場項: -h_i σ_i
    if i in problem.h:
        hi = problem.h[i]
        # 元: -h_i σ_i, 新: -h_i (-σ_i) = +h_i σ_i
        dE += (+hi * sigma_i) - (-hi * sigma_i)

    return dE


def simulated_annealing(
    problem: IsingProblem,
    T_start: float = 5.0,
    T_end: float = 0.1,
    steps: int = 10000,
) -> Tuple[float, List[int]]:
    """
    シミュレーテッドアニーリングでイジング問題を解く。
    """
    N = problem.N
    spins = random_spins(N)

    def energy_current() -> float:
        return problem.energy(spins)

    E = energy_current()

    for t in range(steps):
        # 線形スケジュール
        T = T_start + (T_end - T_start) * (t / steps)
        beta = 1.0 / T

        i = random.randrange(N)
        dE = delta_energy_flip(problem, spins, i)

        if dE <= 0:
            # 受理
            spins[i] *= -1
            E += dE
        else:
            # メトロポリス
            if random.random() < math.exp(-beta * dE):
                spins[i] *= -1
                E += dE

    return E, spins
