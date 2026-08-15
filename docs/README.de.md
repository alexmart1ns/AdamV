[English](README.md) | [Português](README.pt-BR.md) | [Español](README.es.md) | [简体中文](README.zh-CN.md) | [Русский](README.ru.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

---

# 🧠 AdamV: Geometrisch Adaptiver Optimizer

![AdamV vs AdamW CIFAR-10](assets/cifar_10_train.png)

AdamV (Adam-Vedic) ist ein hochmoderner Optimierungsalgorithmus für PyTorch, der antike vedische Mathematik mit numerischer Topologie verschmilzt, um einen 100% autonomen, geometrisch adaptiven Optimizer zu erschaffen.

In rigorosen Tests auf CIFAR-10 (ResNet-9) **übertrumpfte AdamV 2.0 AdamW** (89,68% vs 89,44%), während er komplett autonom lief, ohne sich auf starre externe Scheduler wie `CosineAnnealingLR` zu verlassen.

## ⚙️ Die 4 Säulen von AdamV 2.0

AdamV basiert auf einem hochoptimierten C++/CUDA-Kern und stützt sich auf vier mathematische Innovationen, um durch Loss-Landschaften zu navigieren:

### 1. Krümmungsgesteuerte dynamische Trägheit (SNR-moduliertes $\beta_1$)
Anstatt einen starren momentum decay ($\beta_1 = 0.9$) zu verwenden, berechnet AdamV das lokale Signal-Rausch-Verhältnis ($m_t^2 / v_t$). Auf flachen Plateaus verringert er die Trägheit, um sofort zu beschleunigen. In chaotischen Schluchten erhöht er die Trägheit, um Rauschen zu ignorieren und den Abstieg zu stabilisieren.

### 2. Quasi-Newtonsche Approximation (Bakhshali Hessian-Free Gate)
AdamV verwendet die antike Bakhshali Quartic Brake als gravitative Bremse zweiter Ordnung. Durch die Skalierung des Nenners mit der Pseudo-Hesse-Matrix ($\sqrt{v_t}$) beschneidet er auf intelligente Weise explosive Gradienten unter Verwendung von strukturellem Krümmungswissen, ohne die $O(N^2)$ Speicherkosten vollständiger Hesse-Matrizen.

### 3. Autonome Abkühlung (Ramanujan Log-Periodic Envelope)
AdamV kühlt seine learning rate organisch ab, während er den topologischen Raum durchquert, indem er eine Ramanujan-Kettenbruchentwicklung verwendet. Dies verhindert das "blinde Zerschlagen" traditioneller Scheduler und ermöglicht es dem Netzwerk, weitreichend zu explorieren, bevor es sich in einem robusten globalen Minimum niederlässt.

### 4. Quantum Escape (OMNI-ModBH via Type-Punning)
Wenn AdamV ein karges Plateau erkennt, löst er einen absoluten Basin Hop aus. Er läuft mit Bare-Metal-Geschwindigkeiten auf der GPU und wendet bitweise Masken direkt auf die IEEE 754 float32 Mantisse an (Type-Punning), um Gewichte in benachbarte Becken zu teleportieren, ohne eine warp divergence zu verursachen oder Skalenexponenten zu zerstören.

## 📦 Installation

AdamV erfordert einen modernen C++17 Compiler und das CUDA Toolkit (falls GPU-Beschleunigung erwünscht ist).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Verwendung

Die Verwendung von AdamV ist so einfach wie das Einfügen in Ihre PyTorch Trainingsschleife. Sie benötigen KEINE externen Scheduler.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Initialisiere AdamV (Kein externer Scheduler nötig!)
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # Basis learning rate
    betas=(0.9, 0.999),# Basis betas (beta1 wird dynamisch oszillieren)
    enable_omni=True   # Aktiviere topologische Escapes (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # Führe einen Schritt aus! (Unterstützt Native Mixed-Precision / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks
Führen Sie die beigefügte CIFAR-10 Arena aus, um AdamV direkt auf Ihrer Maschine gegen AdamW zu benchmarken:
```bash
python benchmarks/cifar10_arena.py
```
