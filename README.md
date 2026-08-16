[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: Geometrically Adaptive Optimizer

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Vedic) is a state-of-the-art optimization algorithm for PyTorch that fuses ancient Vedic mathematics with numerical topology to create a geometrically adaptive optimizer. It relies on a highly optimized **C++/CUDA implementation** to operate at bare-metal speeds.

In rigorous, multi-seed stress testing, **AdamV 2.0.2 alpha massacred AdamW on NanoGPT**, achieving significantly lower validation loss across multiple independent seeds, as validated by the global stress suite.

## ⚙️ The Pillars of AdamV 2.0.2 alpha

AdamV navigates non-convex loss landscapes using breakthrough mathematical innovations:

### 1. The Bakhshali Root & BRCM Geometric Momentum Brakes
AdamV utilizes the ancient Bakhshali approximation method combined with Bakhshali Residual-Coupled Momentum (BRCM). By scaling the momentum decay ($\beta_1$) exponentially based on residual collision force ($\sqrt{v_t}$), the optimizer acts as a dynamic shock absorber. It applies geometric momentum brakes in tight topological ravines to restrain explosive gradients, while accelerating linearly on barren plateaus.

### 2. Curvature-Driven Dynamic Inertia
Instead of using a rigid momentum decay, AdamV calculates the local Signal-to-Noise Ratio. On flat plateaus, it drops inertia to accelerate immediately. In chaotic ravines, it increases inertia to ignore noise and stabilize descent.

### 3. Log-Periodic Cooling
AdamV incorporates autonomous **Log-Periodic Cooling** using Ramanujan Continued Fraction expansions. This dynamically scales down the learning rate in a log-periodic envelope, preventing the "blind crushing" effect seen with traditional step schedulers and allowing organic convergence into deeper minima basins.

### 4. Quantum Escape (OMNI-ModBH via Type-Punning)
When AdamV detects a barren plateau, it triggers an absolute Basin Hop. Running at bare-metal speeds on the GPU, it applies bitwise masks directly to the IEEE 754 float32 mantissa (Type-Punning), teleporting weights to adjacent basins without causing warp divergence or destroying scale exponents.

## 📦 Installation

AdamV requires a modern C++17 compiler and CUDA toolkit (if GPU acceleration is desired).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Usage

Using `AdamVCpp` is as simple as dropping it into your PyTorch training loop.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Initialize AdamV 2.0.2 alpha
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # Base learning rate
    betas=(0.9, 0.999),# Base betas (beta1 will oscillate dynamically)
    enable_omni=True   # Enable topological escapes (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # Take a step! (Supports Native Mixed-Precision / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks
Run the included 100% neutral, multi-seed statistical validation suite to benchmark AdamV against AdamW directly on your machine. It runs 5 seeds across ResNet-18, VAE, and NanoGPT with p-value validation.

```bash
python benchmarks/run_global_stress_suite.py
```

![Global Stress Test Results](assets/global_stress_results.png)

For a detailed analysis of the benchmarks, p-values, and methodology, please see our [Benchmark Report](benchmarks/README.md).
