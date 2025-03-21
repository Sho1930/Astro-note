# はじめに


*Astro-note*は、宇宙工学(**Astrodynamics**)を学ぶためのノート（**Note**）を、デジタルによる新しいカタチの**学習帳**として実現することを目指しています。<br>
[機械学習帳](https://chokkan.github.io/mlnote/index.html) に多大なinspirationを受け、機械学習帳の宇宙工学verを作成することを目的としています。
::::{grid} 2
:gutter: 2

:::{grid-item-card} 👩‍💻 理論と実装を一体化した「動く」学習帳 👨‍💻
宇宙工学の理論と実装を一緒に説明することで、理論の実装や応用に触れるだけでなく、プログラムの実行例を通して理論への理解を深めることができます。

<div style="text-align: center;">
<video autoplay loop muted playsinline width="100%" src="_static/starlink.mp4"></video>
<video autoplay loop muted playsinline width="100%" src="_static/3bp.mp4"></video>
</div>
:::

:::{grid-item-card} プログラミング言語としてPythonを採用 ✨
可読性が高く、かつライセンスの必要がない[Python](https://www.python.org/)をプログラミング言語として採用しています。[NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [scipy](https://scipy.org/), [cvxpy](https://www.cvxpy.org/)などのエコシステムとあわせて、宇宙工学の実装を習得できます。

```python
import numpy as np
def quaternion_to_DCM(q):
    # Ensure q is a float array to maintain precision
    q = np.array(q, dtype=np.float64)
    
    # Check that the holonomic constraint of quaternion is satisfied, else normalize it
    q_norm = np.linalg.norm(q)
    if not np.isclose(q_norm, 1.0, atol=1e-8):
        q /= q_norm
    
    # Extract components
    q0, q1, q2, q3 = q
    
    # Compute the elements of the DCM
    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3),         2 * (q1*q3 - q0*q2)],
        [2 * (q1*q2 - q0*q3),           q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)],
        [2 * (q1*q3 + q0*q2),           2 * (q2*q3 - q0*q1),           q0**2 - q1**2 - q2**2 + q3**2]
    ])
    
    return C
```
:::

:::{grid-item-card} 宇宙工学の基礎事項を学べる 📖
*Astro-note*は、キネマティクス、キネティクス、スピン衛星やデュアルスピン衛星の運動、二体問題、Patched conics法、Porkchop Plot、制限三体問題、軌道決定、フィードバック制御、最適制御、凸最適化、カルマンフィルター、画像航法など、宇宙工学における重要事項を広くカバーしています。初学者向けに、その原理や数学的な取り扱いを丁寧に説明します。
```{figure} _images/porkchop.jpg
---
height: 300px
---
```
:::

:::{grid-item-card} オープンソース・プロジェクト 🎁
*Astro-note*は、[クリエイティブ・コモンズ 表示 - 非営利 - 改変禁止 4.0 国際 (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.ja)（プログラム部分以外）および[MITライセンス](https://opensource.org/licenses/MIT)（プログラム部分）で公開されているオープンソース・プロジェクトです。
:::
::::

*Astro-note*は、[Jupyter Lab](https://jupyter.org/#jupyterlab)で書かれたコンテンツを[Jupyter Book](https://jupyterbook.org/)で変換することで生成されています。


