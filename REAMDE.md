# Ising Problem: Classical SA Baseline

このプロジェクトは、任意グラフ上のイジング最適化問題

\[
H_p(\sigma) = -\sum_{i<j} J_{ij}\sigma_i\sigma_j - \sum_i h_i \sigma_i
\]

に対して

- 全探索（小さい N 用）
- シミュレーテッドアニーリング（SA）

を実行し、古典解法のベースラインを作ることを目的とする。

将来的には、同じ \(H_p\) を量子イジング模型の「問題ハミルトニアン」として用い、
量子アニーリング（QA）との比較を行う。

## 実行方法

```bash
cd src
python main.py
```

## 今後の拡張予定
- 量子イジング模型
- QA(横磁場アニーリング)の数値対角化
- SA vs QA の性能比較(成功確率・エネルギーギャップなど)



