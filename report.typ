#import "@preview/enja-bib:0.1.0": *
#import bib-setting-plain: *
#show: bib-init

#set text(
  font: "Noto Sans JP",
  size: 11pt,
  lang: "ja",
)
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  numbering: "1",
)
#set par(
  justify: true,
  first-line-indent: 1em,
  leading: 0.6em,
  spacing: 0.9em,
)
#show heading: it => {
  it
  par(text(size: 0em, ""))
}

#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[課題11(群れのシミュレーション)]

  #v(1.5em)

  #text(size: 14pt, weight: "regular")[03230421 坂口健人]

  #text(size: 12pt)[2026年01月12日]
]


= 概要

本課題では、Couzinアルゴリズムを用いた群れ行動のシミュレーションを実装し、攪乱効果(Confusion Effect)仮説の検証を行った。

Confusion Effectとは、被捕食者が群れを形成することで捕食者の視覚的な混乱を引き起こし、捕食成功率を低下させるという仮説である @Olson_2013。
群れの中では多数の個体が動き回るため、捕食者が特定の個体を追跡し続けることが困難になる。

本シミュレーションでは、被捕食者（小魚）と捕食者（大魚）の相互作用をモデル化した。
捕食者は視野内に入った被捕食者を追跡し、一定距離以内に近づくと捕食に成功する。
被捕食者はCouzinアルゴリズムに基づいて群れを形成しながら、捕食者から逃避する行動をとる。

Confusion Effectを定量的に評価するため、被捕食者の個体数やCouzinアルゴリズムのパラメータを変化させて「被捕食者が初期個体数の80%に減少するまでの時間」を計測した。

コードはPythonで書いたが、試行回数が多くなるシミュレーションについてはRustに移植して高速化を図った。
その際に、生成AI(Gemini)を用いて移植した。
なお、コードの内容が元のコードと一致することは自身で確認した。

= 実装の詳細

== 実行方法

シミュレーションは `main.py` を実行することで開始できる。
実行すると以下のメニューが表示され、選択したモードに応じて動作する。
1は、可視化モードでシミュレーションをリアルタイムに表示する。
2~6は計測モードで、各種ベンチマークやグラフの生成を行う。
本レポートで使用したモード5と6については、試行回数を多く設定すると時間がかかるため、同等の内容のRustコードを用いた。

```
Select Mode:
1: Run Visualization
2: Run Benchmark (Population Reduction to 80%)
3: Run Benchmark (Time to First Catch)
4: Plot First Catch Time vs Prey Count
5: Plot Time to 80% Reduction vs Prey Count
6: Plot Time to 80% Reduction vs ZoO Parameter
Enter 1, 2, 3, 4, 5, or 6:
```

== Couzinアルゴリズムの実装

本シミュレーションでは、研究室ホームページのコード@AIAL_Support を参考にしてCouzinアルゴリズムを実装した。

各個体は周囲の個体との距離に応じて3つの領域を持つ。

- *Zor*(Zone of repulsion): 斥力領域。この範囲内の個体から離れようとする
- *Zoo*(Zone of orientation): 整列領域。この範囲内の個体と向きを揃えようとする
- *Zoa*(Zone of attraction): 誘引領域。この範囲内の個体に近づこうとする

個体の移動方向は以下の優先順位で決定される。

+ Zor内に個体がいる場合、それらから離れる方向に移動(最優先)
+ それ以外の場合、Zoo内の個体と向きを揃えつつ、Zoa内の個体に近づく

また、生物として自然な動きを再現するため、以下の制約を設けた:

- 視野角の制約があり、後方の個体を認識することはできない
- 最大回転速度を制限し、瞬間的に任意の方向へ向きを変えることはできない

== 被捕食者(Prey)の実装

被捕食者は基本的なCouzinアルゴリズムに加え、捕食者に対する逃避行動を実装した。

捕食者が視野に入ると、同種個体との相互作用よりも優先して捕食者から逃げる方向に移動する。
複数の捕食者が視野内にいる場合、それらから遠ざかる方向の合成ベクトルを計算する。

#figure(
  table(
    columns: (1fr, 1fr),
    // 列の幅を等しく設定
    inset: 8pt,
    align: (x, y) => if x == 0 { left } else { center },
    // 左列は左寄せ、右列は中央
    stroke: none,
    // 罫線を消して論文風にする（必要なら削除してください）
    table.header([*パラメータ*], [*設定値*]),
    table.hline(),
    // ヘッダー下の線
    [Zor], [$1.0$ units],
    [Zoo], [$10.0$ units],
    [Zoa], [$20.0$ units],
    [速さ], [$4.0$ units/s],
    [視野角], [$270 degree$],
    [最大回転速度], [$30 degree$/s],
    table.hline(),
    // 下の線
  ),
  caption: [獲物(Prey)のパラメータ設定],
)

== 捕食者(Predator)の実装

捕食者は獲物よりも高速に移動し、視野内の獲物を追跡する。
捕食者と獲物の距離が捕獲半径(1.0 units)以下になると捕食成功とみなす。
ただし、捕食されるのはターゲットとして追跡している個体のみである。
捕獲半径内にターゲット以外の個体が偶然入っても捕食は行わない。

#figure(
  table(
    columns: (1fr, 1fr),
    // 列の幅を等しく設定
    inset: 8pt,
    align: (x, y) => if x == 0 { left } else { center },
    // 左列は左寄せ、右列は中央
    stroke: none,
    // 罫線を消して論文風にする（必要なら削除してください）
    table.header([*パラメータ*], [*設定値*]),
    table.hline(),
    // ヘッダー下の線
    [Zor], [$2.0$ units],
    [Zoo], [$10.0$ units],
    [Zoa], [$20.0$ units],
    [速さ], [$7.0$ units/s],
    [視野角], [$180 degree$],
    [最大回転速度], [$60 degree$/s],
    [捕獲半径], [$1.0$ units],
    table.hline(),
    // 下の線
  ),
  caption: [捕食者(Predator)のパラメータ設定],
)

=== 捕食処理の実装

捕食処理は生物としての自然さを重視して実装した。
1タイムステップにつき1回のみ捕食判定を行い、明示的にターゲットとして追跡している個体のみを捕食対象とする。
これにより、以下の利点がある。

- 偶然近くを通過した個体を捕食してしまう不自然な挙動を防ぐ
- 捕食者が意図的に選択した個体を追跡・捕獲するという現実的な行動を再現
- ターゲット選択の方針が捕食成功率に与える影響を明確に評価できる


=== 捕食方針の実装

本課題の主題は攪乱効果仮説の検証である。
これを検証するため、2種類の捕食方針を実装し比較した。

*CONFUSION方針*(本課題のメイン)

攪乱効果仮説を検証するための方針。
ステップごとに視野内の獲物を確認し、以下の優先順位でターゲットを選択する。

+ 新しい獲物が視野に入った場合、その中で最も近い個体を新規ターゲットとして選択
+ 現在のターゲットが視野内に残っている場合、そのターゲットを追い続ける
+ ターゲットが視野から外れた場合、視野内で最も近い獲物を新たなターゲットとする

```python
# CONFUSION方針の疑似コード
if new_prey_entered_fov:
    target = select_new_target()  # 新規進入個体から選択
elif current_target_still_visible:
    target = current_target  # 現在のターゲットを維持
else:
    target = closest_prey  # 最も近い個体を選択
```

捕食者がこの方針を取っている場合、獲物が群れを形成しているとターゲットの切り替えが頻繁に発生し、捕食成功率が低下することが期待される。

*CLOSEST方針*(比較対象):

常に視野内で最も近い獲物を追跡する単純な方針。
群れによる視覚的混乱の影響を受けにくいため、CONFUSION方針との比較により、Confusion Effectの有無を定量的に評価できる。


== 周期的境界条件

シミュレーション空間には周期的境界条件(トーラス構造)を採用した。
これは、空間が壁に囲まれていると、壁に群れがぶつかるか否かでシミュレーションごとに挙動が大きく変わってしまい、統計的な評価が困難になるためである。

=== 実装方法

個体が境界を超えると反対側から出現するように実装した。さらに、距離計算や視野判定においても周期的境界を考慮し、常に最短経路を使用する:

```python
def get_wrapped_diff(self, other, boundary):
    diff = other.pos - self.pos
    half = boundary / 2
    # X軸方向の補正
    if diff[x] > half: diff[x] -= boundary
    if diff[x] < -half: diff[x] += boundary
    # Y軸方向も同様に補正
    return diff
```

この実装により、境界を跨いだ個体間の相互作用も正しく計算される。


== 計測・評価機能

Confusion Effectの検証のため、*80%減少時間*(捕食者が最初にターゲットをロックオンしてから、被捕食者が初期個体数の80%に減少するまでの時間)を評価指標として採用した。

パラメータを変化させて複数回の試行を行い、各条件での80%減少時間の平均値を計測した。

= 実験結果

== 被捕食者個体数を変化させた場合

被捕食者の初期個体数を1匹から50匹まで1匹ずつ変化させ、80%減少時間を計測した。
@count_closet はCLOSEST方針、@count_confusion はCONFUSION方針の結果を示している。

どちらも被捕食者個体数が増加するにつれて80%減少時間が長くなる傾向が見られた。
しかし、増加率には差があり、CLOSEST方針では10秒程度から20秒程度まで増加したのに対し、CONFUSION方針では10秒程度から250秒程度まで急激に増加した。

#figure(
  image("images/benchmark_prey_count_closet.png", width: 95%),
  caption: [被捕食者個体数と80%減少時間の関係（CLOSEST方針）],
)<count_closet>
#figure(
  image("images/benchmark_prey_count_confusion.png", width: 95%),
  caption: [被捕食者個体数と80%減少時間の関係（CONFUSION方針）],
)<count_confusion>

== Zoo(Zone of orientation)を変化させた場合
Couzinアルゴリズムの変化させるパラメータとして、Zoo(Zone of orientation)を選んだ。
被捕食者のZooを1.0から20.0まで1.0刻みで変化させ、80%減少時間を計測した(Zorは1.0、Zoaは20.0)。
被捕食者個体数は50匹に固定した。


@zoo_closet はCLOSEST方針、@zoo_confusion はCONFUSION方針の結果を示している。
CLOSEST方針では、Zooが変化しても80%減少時間はほぼ横這いだった。

一方、CONFUSION方針では、Zooを大きくしていくと、9.0付近までは緩やかに80%減少時間が短くなり、その後はしばらく緩やかに増加した。
そして、20.0(Zoaと同じ値)になると急激に80%減少時間が長くなった。


#figure(
  image("images/benchmark_zoo_closet.png", width: 95%),
  caption: [Zooと80%減少時間の関係（CLOSEST方針）],
)<zoo_closet>
#figure(
  image("images/benchmark_zoo_confusion.png", width: 95%),
  caption: [Zooと80%減少時間の関係（CONFUSION方針）],
)<zoo_confusion>

= 考察

== 被捕食者個体数の影響
実験結果(@count_closet, @count_confusion)より、被捕食者の個体数が増加するにつれて、捕食者が全個体の80%を捕食するまでに要する時間が増加することが確認された。
特に、CLOSEST方針とCONFUSION方針の比較において、その増加率に顕著な差が見られた点は重要である。

CLOSEST方針における時間の増加は、単純に捕食対象の数が増えたことによる処理時間の増加（「10匹捕まえる時間」対「50匹捕まえる時間」の差）であると考えられる。
一方、CONFUSION方針では、個体数の増加に伴い比較的急激な時間の増加が確認された。 これは、群れの密度が高まることで、捕食者の視野内に頻繁に新しい個体が進入し、ターゲットの切り替え（スイッチング）が多発したためであると推察される。

ターゲットが頻繁に切り替わると、捕食者は特定の個体を追い詰めることができず、方向転換を繰り返すだけの無駄な動きが増加する。
この結果は、個体数が多い（群れが大きい）ほど捕食者の注意が分散し、捕食成功率が低下するという「攪乱効果(Confusion Effect)」の仮説を強く支持するものである。

== Zoo(Zone of orientation)の影響
Zooパラメータの変化に対する80%減少時間の挙動( @zoo_confusion )は、複雑な特性を示した。
これは、Zooの値によって群れの形態が変化し、それに応じて攪乱効果の質が変わるためであると考えられる。

まず、Zooが小さい領域(1.0付近)では、@fig_zoo_shapes(a),(b) に示すように、個体間の整列作用が弱く、群れは無秩序に動き回る「スウォーム（Swarm）」や、中心を回る「トーラス（Torus）」に近い形態をとる。
この状態では、個体同士が頻繁に交差して移動するため、捕食者の視野内における視覚的な変化が激しく、「新規個体へのターゲット切り替え」が頻繁に誘発されたと考えられる。
その結果、個体ごとの逃避方向はバラバラであるものの、強い攪乱効果により比較的高い生存時間を維持できた。

次に、Zooが中程度の領域(10.0付近)で生存時間が短縮した点についてである。
この領域では、@fig_zoo_shapes (c) に示すように、群れは密集しつつ、整列した「パラレル（Parallel）」な形態をとる。
この状態では、個体同士が比較的近接して移動するため、捕食者の視野内に新規で入る個体の数が減少し、ターゲットの切り替え頻度が低下したと考えられる。
さらに、整列した群れは捕食者からの逃避方向が類似するため、捕食者が一度ターゲットをロックオンすると、その個体を追跡しやすくなったと推察される。

最後に、Zooが20.0(Zoaと同値)で急激に生存時間が増加した点についてである。
ZooとZoaが等しい場合、誘引される範囲の個体すべてと向きを揃えることになる。
そのため、群れは非常に高い整列性を持ち、@fig_zoo_shapes(d) に示すような「高度に整列したパラレル（Highly Parallel）」な形態をとる。
この状態では、Zooが中程度のときよりは並進の作用が強くなって密度が低下しており、捕食者の視野内に新規個体が進入する頻度が再び増加したと考えられる。
また、Zooが低いときは逃避して一度離散すると再び集まるのが難しいが、ZooがZoaと等しい場合は個体同士が引き合うため、捕食者から逃げた後も再び群れを形成しやすい。
これらの要因により、長期的な攪乱効果が発揮され、生存時間が大幅に延長されたと考えられる。

以上の考察から、攪乱効果を最大化するには、「ほどよい密度で集まって捕食者の視界に頻繁に新規個体が入ること」と「逃避後も再び群れを形成しやすいこと」が重要であることが推察される。

#figure(
  grid(
    columns: (auto, auto),
    rows: (auto, auto),
    gutter: 1em,
    [#image("images/fig_swarm.png", width: 100%)], [#image("images/fig_torus.png", width: 100%)],
    align(center)[(a) Swarm\ (Zoo=1.0)], align(center)[(b) Torus\ (Zoo=1.1)],
    [#image("images/fig_parallel.png", width: 100%)], [#image("images/fig_highly_parallel.png", width: 100%)],
    // キャプション用の行（必要なら）
    align(center)[(c) Parallel\ (Zoo=10.0)], align(center)[(d) Highly Parallel\ (Zoo=20.0)],
  ),
  caption: [Zooパラメータによる群れ形状の変化],
) <fig_zoo_shapes>

= 授業に関するコメント

今回の課題7や課題9にあるような、「LLMの訓練データに類似のものがない質問」を考えるのはかなり難しく感じたので、もう少し参考の難易度を上げても良いと感じました。



= 参考文献

#bibliography-list(
  ..bib-file(read("ref.bib")),
  title: none,
)
