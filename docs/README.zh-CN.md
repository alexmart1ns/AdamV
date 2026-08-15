[English](README.md) | [Português](README.pt-BR.md) | [Español](README.es.md) | [简体中文](README.zh-CN.md) | [Русский](README.ru.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

---

# 🧠 AdamV: 几何自适应优化器

![AdamV vs AdamW CIFAR-10](assets/cifar_10_train.png)

AdamV (Adam-Vedic) 是一种用于 PyTorch 的最先进的优化算法，它将古老的吠陀数学与数值拓扑学相融合，创建了一个 100% 自主、几何自适应的优化器。

在 CIFAR-10 (ResNet-9) 的严格测试中，**AdamV 2.0 的表现优于 AdamW** (89.68% 对 89.44%)，它完全自主运行，不依赖于像 `CosineAnnealingLR` 这样死板的外部调度器。

## ⚙️ AdamV 2.0 的 4 大支柱

AdamV 基于高度优化的 C++/CUDA 核心构建，依靠四项数学创新来在损失地形中导航：

### 1. 曲率驱动的动态惯性 (信噪比调制 $\beta_1$)
AdamV 没有使用固定的 momentum 衰减 ($\beta_1 = 0.9$)，而是计算局部信噪比 ($m_t^2 / v_t$)。在平坦的高原上，它会降低惯性以立即加速。在混乱的峡谷中，它会增加惯性以忽略噪声并稳定下降。

### 2. 拟牛顿近似 (Bakhshali 无海森门)
AdamV 使用古老的 Bakhshali 四次制动作为二阶引力制动。通过用伪海森矩阵 ($\sqrt{v_t}$) 缩放分母，它利用结构曲率知识智能地裁剪爆炸性梯度，而没有完整海森矩阵的 $O(N^2)$ 内存开销。

### 3. 自主冷却 (Ramanujan 对数周期包络)
当 AdamV 使用 Ramanujan 连分数展开在拓扑空间中遍历时，它会有机地冷却其 learning rate。这防止了传统调度器的“盲目压碎”，允许网络在稳定于一个稳健的全局极小值之前进行广泛的探索。

### 4. 量子逃逸 (通过 Type-Punning 的 OMNI-ModBH)
当 AdamV 检测到荒芜的高原时，它会触发绝对的盆地跳跃 (Basin Hop)。在 GPU 上以裸机速度运行时，它将按位掩码直接应用于 IEEE 754 float32 尾数 (Type-Punning)，将权重传送到相邻的盆地，而不会导致 warp divergence 或破坏比例指数。

## 📦 安装

AdamV 需要现代 C++17 编译器和 CUDA 工具包 (如果需要 GPU 加速)。

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 用法

使用 AdamV 非常简单，只需将其放入您的 PyTorch 训练循环中即可。您不需要外部调度器。

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# 初始化 AdamV (不需要外部调度器！)
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # 基础 learning rate
    betas=(0.9, 0.999),# 基础 betas (beta1 将动态振荡)
    enable_omni=True   # 启用拓扑逃逸 (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # 迈出一步！(支持原生混合精度 / AMP)
        optimizer.step(loss=loss)
```

## 🧪 基准测试
运行包含的 CIFAR-10 Arena，直接在您的机器上对 AdamV 和 AdamW 进行基准测试：
```bash
python benchmarks/cifar10_arena.py
```
