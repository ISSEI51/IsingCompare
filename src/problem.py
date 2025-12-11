from dataclasses import dataclass
from typing import Dict, Tuple, List

SpinConfig = List[int]  # 各成分が +1 or -1


@dataclass
class IsingProblem:
    """イジング最適化問題 H_p(σ) = -Σ J_ij σ_i σ_j - Σ h_i σ_i"""

    N: int
    J: Dict[Tuple[int, int], float]  # (i, j) with i < j
    h: Dict[int, float]  # i -> h_i

    def energy(self, spins: SpinConfig) -> float:
        assert len(spins) == self.N
        E = 0.0
        # 相互作用項
        for (i, j), Jij in self.J.items():
            E += -Jij * spins[i] * spins[j]
        # ローカル場
        for i, hi in self.h.items():
            E += -hi * spins[i]
        return E


def sample_test_problem() -> IsingProblem:
    """
    テスト用の小さな問題を1つ定義する。
    N=6 の完全グラフっぽいものなど、適当に用意。
    """
    N = 11
    # 簡単な例: ランダムでもいいが、ここでは手で決める
    J = {
        (0, 1): 1.0,
        (0, 2): -0.5,
        (1, 2): 0.8,
        (2, 3): 1.2,
        (3, 4): -1.0,
        (4, 5): 0.7,
        (0, 5): -0.9,
    }
    h = {
        0: 0.2,
        3: -0.3,
    }
    return IsingProblem(N=N, J=J, h=h)
