# 🧠 AdamV: Geometrically Adaptive Optimizer

![AdamV vs AdamW CIFAR-10](grand_arena.png)

AdamV (Adam-Vedic) is a state-of-the-art optimization algorithm for PyTorch that fuses ancient Vedic mathematics with numerical topology to create a 100% autonomous, geometrically adaptive optimizer.

In rigorous testing on CIFAR-10 (ResNet-9), **AdamV 2.0 outperformed AdamW** (89.68% vs 89.44%) running completely autonomously, without relying on rigid external schedulers like `CosineAnnealingLR`.

## ⚙️ The 4 Pillars of AdamV 2.0

AdamV is built on a highly optimized C++/CUDA core, relying on four mathematical innovations to navigate loss landscapes:

### 1. Curvature-Driven Dynamic Inertia (SNR Modulated $\beta_1$)
Instead of using a rigid momentum decay ($\beta_1 = 0.9$), AdamV calculates the local Signal-to-Noise Ratio ($m_t^2 / v_t$). On flat plateaus, it drops inertia to accelerate immediately. In chaotic ravines, it increases inertia to ignore noise and stabilize descent.

### 2. Quasi-Newtonian Approximation (Bakhshali Hessian-Free Gate)
AdamV uses the ancient Bakhshali Quartic Brake as a second-order gravitational brake. By scaling the denominator with the pseudo-Hessian ($\sqrt{v_t}$), it intelligently clips explosive gradients using structural curvature knowledge, without the $O(N^2)$ memory cost of full Hessian matrices.

### 3. Autonomous Cooling (Ramanujan Log-Periodic Envelope)
AdamV organically cools down its learning rate as it traverses the topological space using a Ramanujan Continued Fraction expansion. This prevents the "blind crushing" of traditional schedulers, allowing the network to explore widely before settling into a robust global minimum.

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

Using AdamV is as simple as dropping it into your PyTorch training loop. You do NOT need external schedulers.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Initialize AdamV (No external scheduler needed!)
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
Run the included CIFAR-10 Arena to benchmark AdamV against AdamW directly on your machine:
```bash
python benchmarks/cifar10_arena.py
```
