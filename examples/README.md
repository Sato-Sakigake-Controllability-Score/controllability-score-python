# 使用例

<table>
    <thead>
        <tr>
            <th style="text-align:center"><a href="README_eng.md">English</a></th>
            <th style="text-align:center">日本語</th>
        </tr>
    </thead>
</table>

この README は，controllability score の使い方を Python のコードで順に確認するためのガイドである．

`ex01_*.py` から `ex05_*.py` までのファイルは，このパッケージの基本的な使い方を
まとまった形で実行できる参考スクリプトとして置いている．

## リポジトリの取得

まず，ターミナルで任意の作業ディレクトリに移動し，GitHub からリポジトリを clone する．

```sh
git clone https://github.com/Sato-Sakigake-Controllability-Score/controllability-score-python.git
cd controllability-score-python
```

以下では，この `controllability-score-python` ディレクトリをリポジトリのルートディレクトリと呼ぶ．

## 前提

- Python 3.10 以降
- NumPy
- SciPy
- テストや開発を行う場合は pytest

ソースから利用する場合は，リポジトリのルートディレクトリで次を実行する．

```sh
python -m pip install -e .
```

開発用依存パッケージも入れる場合は，次を実行する．

```sh
python -m pip install -e ".[dev]"
```

可視化を含む例も開発用依存パッケージに含まれる．

サンプルスクリプトは，リポジトリのルートディレクトリから次のように実行する．

```sh
python examples/ex01_minimal_cs.py
```

## 1. システム行列を用意する

基本 API は，システム行列 `A` を受け取ってノードごとの重みを返す．

```python
import numpy as np

A = np.diag([-1.0, -2.0, 0.5, 1.2])
```

一般には，`A` の行と列がノードに対応する正方行列であればよく，次のようなバリエーションが考えられる．

- ネットワークの隣接行列 `A` をそのままシステム行列にする
- 安定化のために，対角成分を足して `A - cI` の形にする
- ラプラシアン `L = D - A` を計算して，`-L` をシステム行列とする

```python
adjacency = np.array(
    [
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.4],
        [0.2, 0.0, 0.0],
    ],
    dtype=float,
)

self_loop_weight = 2.0
A1 = adjacency - self_loop_weight * np.eye(adjacency.shape[0])

degree = adjacency.sum(axis=1)
L = np.diag(degree) - adjacency
A2 = -L
```

## 2. 隣接行列を有向グラフとして扱う

Python 版の最小依存は NumPy と SciPy であるため，examples では graph 描画ライブラリを必須にしていない．
隣接行列の内容を確認したい場合は，行列を表示してから `cs` に渡す．

```python
adjacency = np.array(
    [
        [0.0, 0.8, 0.3, 0.0],
        [0.0, 0.3, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.6],
        [0.3, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)

print(adjacency)
```

この最小例は [`ex05_minimal_graph_cs.py`][ex05] にまとめている．

## 3. VCS と AECS を同時に計算する

VCS と AECS の両方を使う場合は，`cs` を使うのが基本である．

```python
from controllability_scoring import cs

p_v, p_a, info_v, info_a, wlist, transform = cs(A)
```

`p_v` と `p_a` は，それぞれノードごとの重みベクトルである．

```python
print("VCS weights:")
print(p_v)

print("AECS weights:")
print(p_a)
```

VCS だけ，または AECS だけが必要な場合は，個別の関数も使える．

```python
from controllability_scoring import aecs, vcs

p_v, info_v, wlist_v, transform = vcs(A)
p_a, info_a, wlist_a, transform = aecs(A)
```

`cs(A)` は，アルゴリズムの最初に `W_1, ..., W_n` を一度に計算し，それをもとに VCS と AECS を計算するため，両方を用いる場合は `vcs(A)` と `aecs(A)` を順に呼び出すよりも効率的である．

## 4. solver の情報を確認する

計算結果の重みだけでなく，最適化の状態も確認したい場合は，`info_v` と `info_a` を確認する．

```python
p_v, p_a, info_v, info_a, wlist, transform = cs(A)
```

例えば，`info_v` は `CSResult` オブジェクトで，次のような属性を持つ．

```python
print(info_v.objective_value)
print(info_v.iterations)
print(info_v.converged)
print(info_v.exit_flag)
print(info_v.exit_message)
```

個別関数でも，同じように `info` を受け取れる．

```python
p_v, info_v, wlist_v, transform = vcs(A)
p_a, info_a, wlist_a, transform = aecs(A)
```

## 5. 有限時間ホライズンを指定する

既定では `T = inf` として扱われるが，有限時間のスコアを計算したい場合は，第2引数または `T=...` で終端時刻を指定する．

```python
T = 2.0
p_v, p_a, info_v, info_a, wlist, transform = cs(A, T)
```

キーワード引数でも同じ指定ができる．

```python
p_v, p_a, info_v, info_a, wlist, transform = cs(A, T=2.0)
```

## 6. そのほかのオプションを指定する

Python 版では，Gramian 側のオプションは `WOptions`，solver 側のオプションは `PGSolverOptions` で指定する．

- Gramian 側の主な指定: `method`, `steps`, `use_scaling`
- solver 側の主な指定: `max_iter`, `tol`, `verbose`, `store_trace`

例えば，オプションをまとめて指定した例を示す．

```python
from controllability_scoring import cs
from controllability_scoring.options import PGSolverOptions, WOptions

T = 2.0
w_options = WOptions.from_system(
    A,
    T,
    method="integral",
    steps=80,
    use_scaling=False,
)
solver_options = PGSolverOptions(
    max_iter=2000,
    tol=1e-9,
    verbose=True,
    store_trace=True,
)

p_v, p_a, info_v, info_a, wlist, transform = cs(
    A,
    T,
    w_options=w_options,
    solver_options=solver_options,
)
```

オプションの詳細は，リポジトリルートの [README.md][repo-readme] を参照されたい．

## 7. サンプルスクリプトを実行する

まとまった例として実行したい場合は，次のスクリプトを使う．README の各節は，これらのスクリプトを読むときの見取り図としても使える．

| File | 内容 |
| --- | --- |
| [`ex01_minimal_cs.py`][ex01] | `vcs`，`aecs`，`cs` の基本と solver 情報の確認 |
| [`ex02_finite_horizon.py`][ex02] | 複数の有限時間ホライズン `T` によるスコア比較 |
| [`ex03_edge_weight_sweep.py`][ex03] | 1 本のエッジ重みを変えたときのスコア変化 |
| [`ex04_visualize_scores.py`][ex04] | 隣接行列から VCS と AECS を計算し，グラフとスコアを可視化 |
| [`ex05_minimal_graph_cs.py`][ex05] | 隣接行列を表示し，同じ行列から VCS と AECS を計算 |

`ex04_visualize_scores.py` は，画像として保存して確認することもできる．

```sh
python examples/ex04_visualize_scores.py --save examples/ex04_scores.png
```


[repo-readme]: ../README.md
[ex01]: ./ex01_minimal_cs.py
[ex02]: ./ex02_finite_horizon.py
[ex03]: ./ex03_edge_weight_sweep.py
[ex04]: ./ex04_visualize_scores.py
[ex05]: ./ex05_minimal_graph_cs.py
