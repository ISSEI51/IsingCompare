from typing import Tuple, List
from problem import IsingProblem


def int_to_spins(x: int, N: int) -> List[int]:
    """
    整数 x (0..2^N-1) を ±1 のスピン列に変換
    bitが1 -> +1, bitが0 -> -1 のように定義
    """
    spins = []
    for i in range(N):
        bit = (x >> i) & 1
        spins.append(1 if bit == 1 else -1)
    return spins


def brute_force_ground_state(problem: IsingProblem) -> Tuple[float, List[int]]:
    """全探索で基底エネルギーと対応するスピン配置を求める"""
    N = problem.N
    best_E = float("inf")
    best_spins = None

    for x in range(2**N):
        spins = int_to_spins(x, N)
        E = problem.energy(spins)
        if E < best_E:
            best_E = E
            best_spins = spins

    return best_E, best_spins
