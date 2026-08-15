[English](README.md) | [Português](README.pt-BR.md) | [Español](README.es.md) | [简体中文](README.zh-CN.md) | [Русский](README.ru.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

---

# 🧠 AdamV: 幾何学的適応型オプティマイザー

![AdamV vs AdamW CIFAR-10](assets/cifar_10_train.png)

AdamV（Adam-Vedic）は、古代ヴェーダ数学と数値トポロジーを融合させ、100%自律的で幾何学的に適応するオプティマイザーを作成する、PyTorch向けの最先端の最適化アルゴリズムです。

CIFAR-10（ResNet-9）での厳格なテストにおいて、**AdamV 2.0はAdamWを上回り**（89.68% vs 89.44%）、`CosineAnnealingLR`のような厳格な外部スケジューラーに依存することなく、完全に自律的に動作しました。

## ⚙️ AdamV 2.0の4つの柱

AdamVは高度に最適化されたC++/CUDAコア上に構築されており、損失の地形をナビゲートするための4つの数学的革新に依存しています。

### 1. 曲率駆動型ダイナミックイナーシャ (SNR変調 $\beta_1$)
固定の momentum 減衰（$\beta_1 = 0.9$）を使用する代わりに、AdamVは局所的なS/N比（$m_t^2 / v_t$）を計算します。平坦なプラトーでは、慣性を低下させて直ちに加速します。混沌とした谷では、ノイズを無視して降下を安定させるために慣性を増加させます。

### 2. 準ニュートン近似 (バクシャーリー ヘシアンフリーゲート)
AdamVは、2次重力ブレーキとして古代のバクシャーリー四次ブレーキを使用します。分母を疑似ヘシアン（$\sqrt{v_t}$）でスケーリングすることにより、完全なヘシアン行列の$O(N^2)$のメモリコストなしに、構造的曲率の知識を用いて爆発的な勾配をインテリジェントにクリップします。

### 3. 自律冷却 (ラマヌジャン 対数周期エンベロープ)
AdamVは、ラマヌジャンの連分数展開を使用してトポロジカル空間を横断する際に、その learning rate を有機的に冷却します。これにより、従来のスケジューラーの「盲目的な押しつぶし」を防ぎ、ネットワークが堅牢な大域的最小値に落ち着く前に広く探索することを可能にします。

### 4. 量子エスケープ (Type-Punning を介した OMNI-ModBH)
AdamVが不毛なプラトーを検出すると、絶対的なベイスンホップをトリガーします。GPUのベアメタル速度で実行され、IEEE 754 float32の仮数部に直接ビット単位のマスクを適用し (Type-Punning)、warp divergence を引き起こしたりスケール指数を破壊したりすることなく、重みを隣接する盆地にテレポートさせます。

## 📦 インストール

AdamVには、最新のC++17コンパイラとCUDAツールキット（GPUアクセラレーションを希望する場合）が必要です。

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 使用方法

AdamVの使用は、PyTorchのトレーニングループに組み込むのと同じくらい簡単です。外部のスケジューラーは必要ありません。

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# AdamVを初期化します (外部スケジューラーは不要です！)
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # 基本の learning rate
    betas=(0.9, 0.999),# 基本のbetas (beta1は動的に振動します)
    enable_omni=True   # トポロジカルエスケープの有効化 (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # ステップを実行します！ (ネイティブ混合精度 / AMPをサポート)
        optimizer.step(loss=loss)
```

## 🧪 ベンチマーク
付属のCIFAR-10アリーナを実行して、マシン上でAdamVとAdamWを直接ベンチマークします：
```bash
python benchmarks/cifar10_arena.py
```
