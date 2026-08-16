[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: 幾何学的適応型オプティマイザー

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Vedic) は、古代ヴェーダ数学と数値位相幾何学を融合させて幾何学的に適応するオプティマイザーを構築する、PyTorch向けの最先端の最適化アルゴリズムです。ベアメタルに近い速度で動作するために、高度に最適化された **C++/CUDA 実装** に依存しています。

厳格なマルチシードのストレステストにおいて、**AdamV 2.0.2 alpha は NanoGPT において AdamW を圧倒し**、グローバルストレススイートで検証されたように、複数の独立したシードにわたって検証損失を大幅に低下させました。

## ⚙️ AdamV 2.0.2 alpha の柱

AdamVは、画期的な数学的革新を用いて非凸損失地形をナビゲートします。

### 1. バクシャーリー平方根とBRCM幾何学的モメンタムブレーキ
AdamV は、古代のバクシャーリー近似法とバクシャーリー残差結合モメンタム (BRCM) を組み合わせて活用します。残差衝突力 ($\sqrt{v_t}$) に基づいてモメンタム減衰 ($\beta_1$) を指数関数的にスケーリングすることで、オプティマイザーは動的なショックアブソーバーとして機能します。狭い位相幾何学的な峡谷では幾何学的なモメンタムブレーキをかけて爆発的な勾配を抑制し、不毛な台地では直線的に加速します。

### 2. 曲率駆動型動的慣性
固定されたモメンタム減衰を使用する代わりに、AdamV は局所的なSN比 (Signal-to-Noise Ratio) を計算します。平坦な台地では慣性を落として直ちに加速します。混沌とした峡谷では慣性を増し、ノイズを無視して降下を安定させます。

### 3. 対数周期冷却 (Log-Periodic Cooling)
AdamV には、ラマヌジャン連分数展開を用いた自律的な**対数周期冷却**が組み込まれています。これにより、学習率が対数周期的な包絡線の中で動的にスケールダウンし、従来のステップスケジューラーで見られる「ブラインドクラッシュ」効果を防ぎ、より深い最小値の盆地 (basin) への有機的な収束を可能にします。

### 4. 量子エスケープ (型パンニングによる OMNI-ModBH)
AdamV は不毛な台地を検出すると、絶対的な盆地ホップ (Basin Hop) をトリガーします。GPU 上のベアメタル速度で実行され、IEEE 754 float32 の仮数部に直接ビットワイズマスクを適用し (型パンニング)、ワープの分岐を引き起こしたりスケール指数を破壊したりすることなく、重みを隣接する盆地にテレポートさせます。

## 📦 インストール

AdamV には、最新の C++17 コンパイラと（GPU アクセラレーションが必要な場合は）CUDA ツールキットが必要です。

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 使い方

`AdamVCpp` の使用は、PyTorchのトレーニングループに組み込むだけで非常に簡単です。

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# AdamV 2.0.2 alpha の初期化
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # 基本学習率
    betas=(0.9, 0.999),# 基本ベータ (beta1 は動的に振動します)
    enable_omni=True   # 位相幾何学的エスケープ (型パンニング) を有効化
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # ステップを実行！ (ネイティブ混合精度 / AMP をサポート)
        optimizer.step(loss=loss)
```

## 🧪 ベンチマーク
AdamV をご自身のマシンで直接 AdamW とベンチマークするために、同梱されている100%中立的なマルチシード統計的検証スイートを実行してください。これは、p値の検証とともに、ResNet-18、VAE、NanoGPT にわたって5つのシードを実行します。

```bash
python benchmarks/run_global_stress_suite.py
```

![Global Stress Test Results](assets/global_stress_results.png)

ベンチマーク、p値、および方法論の詳細な分析については、[ベンチマークレポート](benchmarks/README.md)を参照してください。
