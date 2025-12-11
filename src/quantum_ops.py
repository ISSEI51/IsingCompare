import numpy as np

# 2x2 Pauli and identity
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)

PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

IDENTITY_2 = np.eye(2, dtype=complex)


def kron_n(matrices):
    """
    行列のリスト [A0, A1, ..., A_{N-1}] に対して
    A0 ⊗ A1 ⊗ ... ⊗ A_{N-1} を計算する。
    """
    result = matrices[0]
    for M in matrices[1:]:
        result = np.kron(result, M)
    return result


def sigma_z_i(N: int, i: int) -> np.ndarray:
    """
    N量子ビット系における、i 番目の量子ビット上で作用する σ^z の行列表現。
    （i は 0 〜 N-1）
    """
    mats = []
    for k in range(N):
        if k == i:
            mats.append(PAULI_Z)
        else:
            mats.append(IDENTITY_2)
    return kron_n(mats)


def sigma_x_i(N: int, i: int) -> np.ndarray:
    """
    N量子ビット系における、i 番目の量子ビット上で作用する σ^x の行列表現。
    """
    mats = []
    for k in range(N):
        if k == i:
            mats.append(PAULI_X)
        else:
            mats.append(IDENTITY_2)
    return kron_n(mats)
