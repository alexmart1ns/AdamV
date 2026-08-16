[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: 几何自适应优化器

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Vedic) 是一种用于 PyTorch 的最先进的优化算法，它将古老的吠陀数学与数值拓扑学相融合，创建了一个几何自适应优化器。它依赖于高度优化的 **C++/CUDA 实现** 以裸机速度运行。

在严格的、多随机种子的压力测试中，**AdamV 2.0.2 alpha 在 NanoGPT 上碾压了 AdamW**，在多个独立的种子上实现了显著更低的验证损失，这已得到全球压力套件的验证。

## ⚙️ AdamV 2.0.2 alpha 的核心支柱

AdamV 使用突破性的数学创新在非凸损失景观中导航：

### 1. 巴克沙利平方根和 BRCM 几何动量制动器
AdamV 利用古老的巴克沙利近似法结合巴克沙利残差耦合动量 (BRCM)。通过基于残差碰撞力 ($\sqrt{v_t}$) 对动量衰减 ($\beta_1$) 进行指数级缩放，优化器充当动态减震器。在狭窄的拓扑峡谷中，它会应用几何动量制动器来抑制爆炸性梯度，而在贫瘠的平原上则线性加速。

### 2. 曲率驱动的动态惯性
AdamV 没有使用刚性的动量衰减，而是计算局部的信噪比。在平坦的平原上，它会降低惯性以立即加速。在混乱的峡谷中，它会增加惯性以忽略噪声并稳定下降。

### 3. 对数周期冷却
AdamV 结合了使用拉马努金连分数展开的自主**对数周期冷却**。这会在对数周期包络中动态缩减学习率，防止在传统步进调度器中看到的“盲目粉碎”效应，并允许有机地收敛到更深的最小值盆地。

### 4. 量子逃逸 (基于类型惩罚的 OMNI-ModBH)
当 AdamV 检测到贫瘠的平原时，它会触发绝对的盆地跳跃 (Basin Hop)。在 GPU 上以裸机速度运行，它直接将按位掩码应用于 IEEE 754 float32 尾数（类型惩罚），将权重传送到相邻的盆地，而不会引起 warp 分歧或破坏比例指数。

## 📦 安装

AdamV 需要现代的 C++17 编译器和 CUDA 工具包（如果需要 GPU 加速）。

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🎛️ 校准指南（特定场景）

AdamV 具有高度的自我调节能力，但由于不同的神经架构具有根本不同的数学拓扑结构，因此您必须根据您的模型正确初始化 AdamV。

### 1. 视觉与 NLP（确定性与深度注意力）
对于标准模型（如 **ResNet**）和自回归模型（如 **NanoGPT**），请使用 **黄金校准 (Golden Calibration)**。这利用了 3 世纪的巴克沙利平方根和几何动量制动器来自动阻止梯度爆炸。

```python
# The Golden Calibration (Default)
optimizer = AdamVCpp(
    model.parameters(),
    lr=1e-3,            # Flat Learning Rate (No Schedulers Needed!)
    betas=(0.9, 0.999),
    weight_decay=0.1
    # Hidden defaults: use_bakhshali=True, bakhshali_threshold=50.0, enable_brcm=True
)
```

### 2. 生成模型（VAE、GAN、扩散模型）
生成模型会原生地将**高斯噪声**注入梯度中。这种噪声会导致与 AdamV 的动量制动器发生假阳性碰撞。对于这些模型，请使用 **随机校准 (Stochastic Calibration)**。

```python
# The Stochastic Calibration
optimizer = AdamVCpp(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    enable_brcm=False,             # Turn OFF Geometric Brakes (let noise flow)
    bakhshali_threshold=1000.0     # Expand the shock tolerance for noise
)
```

## 🚀 用法

使用 `AdamVCpp` 就像将它放入您的 PyTorch 训练循环一样简单。

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# 初始化 AdamV 2.0.2 alpha
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # 基础学习率
    betas=(0.9, 0.999),# 基础 beta (beta1 将动态振荡)
    enable_omni=True   # 启用拓扑逃逸 (类型惩罚)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # 走一步！(支持原生混合精度 / AMP)
        optimizer.step(loss=loss)
```

## 🧪 基准测试与验证指南

我们坚信**开放科学和可重复的结果**。您不必只听我们的片面之词——您可以运行完整的、100% 中立的、多随机种子统计验证套件，在您自己的机器上对 AdamV 和 AdamW 进行基准测试。

该套件在三种不同的架构上跨 5 个独立的随机种子测试这两种优化器：
- **ResNet-18** (确定性图像分类)
- **VAE** (随机生成噪声)
- **NanoGPT** (深度自回归注意力)

### 如何重现结果：
1. **硬件要求**：强烈建议使用具有至少 15GB VRAM 的支持 CUDA 的 GPU (例如，NVIDIA T4、RTX 3090，或标准的免费 Kaggle GPU 实例)。
2. **运行全球压力套件**：
```bash
python benchmarks/run_global_stress_suite.py
```
3. **预期情况**：该脚本将自动下载数据集 (FashionMNIST、TinyShakespeare)，编译 AdamV C++ 内核，并运行所有 30 种组合 (5 个种子 × 3 个场景 × 2 种优化器)。在标准的 NVIDIA T4 GPU 上，此过程大约需要 1.5 小时。
4. **输出**：完成后，脚本将自动生成包含原始指标的 `global_stress_results.csv` 文件以及展示最小-最大方差阴影和韦尔奇 t 检验 p 值的 `global_stress_plot.png`。

![Global Stress Test Results](assets/global_stress_results.png)

有关基准测试、p 值和方法论的详细分析，请参阅我们的 [基准测试报告](benchmarks/README.md)。
