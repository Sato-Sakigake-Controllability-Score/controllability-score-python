# 可制御性スコア計算プログラム Python 版

<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_eng.md">English</a></th>
      <th style="text-align:center">日本語</th>
    </tr>
  </thead>
</table>

## 1. はじめに

本プロジェクトは，可制御性スコアの計算を目的とした Python パッケージである．
主にシステム行列 `A` と終端時刻 `T` を受け取り，可制御性スコア
(VCS, AECS) を出力する．


### 可制御性スコア・可観測性スコアの解釈

可制御性スコアは，状態ノードと入力ノードが 1:1 に対応する仮想的な枠組みを導入し，
「どのノードにどの程度の重みで介入すれば系全体を効果的に制御できるか」を
定量的に測る指標である．

可観測性スコアは，可制御性スコアの双対概念として，システム

```math
\frac{dx}{dt}=Ax
```

においてどの状態ノードを観測すれば状態全体を把握しやすいかを表す．
Python 版では，システム行列 `A` の代わりに `A.T` を渡すことで計算できる．

```python
p_obs_v, info_v, wlist_v, transform = vcs(A.T)
p_obs_a, info_a, wlist_a, transform = aecs(A.T)
p_obs_v, p_obs_a, info_v, info_a, wlist, transform = cs(A.T)
```

### 動作環境

- Python: 3.10 以上
- 主要依存パッケージ: NumPy, SciPy
- テスト用依存パッケージ: pytest

---

## 2. インストール

ソースから利用する場合:

```bash
python -m pip install -e .
```

開発用依存パッケージも入れる場合:

```bash
python -m pip install -e ".[dev]"
```

このワークスペースでは，親ディレクトリの仮想環境を使う．

```bash
cd ..
source .venv/bin/activate
cd controllability-score-python
python -m pytest
```

Python パッケージ名は `controllability-scoring`，import 名は
`controllability_scoring` である．

---

## 3. 全体構成の概要

```text
project/
├─ src/controllability_scoring/
│  ├─ api.py                    # トップレベル API
│  ├─ problem.py                # 問題設定
│  ├─ results.py                # 結果
│  ├─ options.py                # オプション
│  ├─ gramian/                  # Gramian 計算
│  ├─ pg_solvers/               # Projected gradient solver
│  ├─ projections/              # simplex 射影
│  ├─ solvers/                  # VCS/AECS solver
│  └─ utils/                    # 共通処理
├─ tests/                       # テスト
├─ examples/                    # 使用例
├─ docs_archive/                # 既存ドキュメント
├─ README.md                    # 本ドキュメント
├─ README_eng.md                # 英語版ドキュメント
├─ pyproject.toml               # パッケージ設定
└─ LICENSE                      # ライセンス
```

主な構成要素の役割は以下のとおりである．

- `vcs`: VCS を計算するトップレベル関数
- `aecs`: AECS を計算するトップレベル関数
- `cs`: VCS と AECS を同じ `CSProblem` から計算するトップレベル関数
- `CSProblem`: 問題設定を表すクラス
- `CSResult`: 最適化結果を保持するクラス
- `WOptions`: Gramian 計算用オプション
- `PGSolverOptions`: 最適化用オプション

---

## 4. 使用例

```python
import numpy as np
from controllability_scoring import vcs, aecs, cs

A = np.diag([-1.0, -2.0, 0.5, 1.2])

p_v, info_v, wlist_v, transform = vcs(A)
p_a, info_a, wlist_a, transform = aecs(A)
p_v, p_a, info_v, info_a, wlist, transform = cs(A)
```

より詳しい例は `examples/` にある．

```bash
python examples/ex01_minimal_cs.py
python examples/ex02_finite_horizon.py
python examples/ex03_edge_weight_sweep.py
python examples/ex04_visualize_scores.py
python examples/ex05_minimal_graph_cs.py
```

---

## 5. 出力

`vcs`, `aecs`, `cs` は simplex 上の重みベクトルを返す．

```math
p_i \ge 0,\quad \sum_i p_i = 1
```

重みが大きいノードほど，対応する可制御性指標に対する寄与が大きいと解釈できる．

### `vcs(A, ...)`

戻り値:

- `p_v`: VCS の最適重み
- `info_v`: solver の診断情報
- `wlist_v`: node-wise Gramian 行列
- `transform`: 変換行列

### `aecs(A, ...)`

戻り値:

- `p_a`: AECS の最適重み
- `info_a`: solver の診断情報
- `wlist_a`: node-wise Gramian 行列
- `transform`: 変換行列

### `cs(A, ...)`

戻り値:

- `p_v`: VCS の最適重み
- `p_a`: AECS の最適重み
- `info_v`: VCS solver の診断情報
- `info_a`: AECS solver の診断情報
- `wlist`: node-wise Gramian 行列
- `transform`: 変換行列

---

## 6. 現在の対応範囲

- full-state VCS/AECS に対応
- finite horizon / infinite horizon に対応
- Lyapunov 法と integral 法に対応
- `w_output="orig"` または `w_output="trans"` により，出力 Gramian の座標系を選択可能
- target controllability score は未実装
- Python API は固定長 tuple を返す
- VCS と AECS を同時に計算する関数は `cs`

---

## Citation

このソースコードを論文内で利用した場合は，以下の文献を引用してください．

```bibtex
@article{sato2024controllability,
  title={Controllability scores for selecting control nodes of large-scale network systems},
  author={Sato, Kazuhiro and Terasaki, Shun},
  journal={IEEE Transactions on Automatic Control},
  volume={69},
  number={7},
  pages={4673--4680},
  year={2024},
  publisher={IEEE}
}

@article{sato2025uniqueness,
  title={Uniqueness analysis of controllability scores and their application to brain networks},
  author={Sato, Kazuhiro and Kawamura, Ryohei},
  journal={IEEE Transactions on Control of Network Systems},
  volume={12},
  number={4},
  pages={2568--2580},
  year={2025},
  publisher={IEEE}
}

@article{sato2025target,
  title={Target Controllability Scores for Actuation-Constrained Network Intervention},
  author={Kazuhiro Sato},
  journal={arXiv preprint arXiv:2510.13354},
  year={2025}
}

@article{umezu2026infinite,
  title={Infinite-horizon controllability scores for linear time-invariant systems},
  author={Umezu, Kota and Sato, Kazuhiro},
  journal={arXiv preprint arXiv:2601.10260},
  year={2026}
}
```

## License

This software is released under the MIT License. See `LICENSE` for details.

## Contributors

This software has been developed by members of the Sato Group.

Project lead:

- Kazuhiro Sato

Main contributors:

- Riku Yaegashi
- Michiya Iwata

## Contact

For questions or bug reports, please open an issue on GitHub or contact:

Kazuhiro Sato

kazuhiro[at]mist.i.u-tokyo.ac.jp
