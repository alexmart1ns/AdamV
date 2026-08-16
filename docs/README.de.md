[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: Geometrisch Adaptiver Optimierer

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Vedic) ist ein hochmoderner Optimierungsalgorithmus für PyTorch, der alte vedische Mathematik mit numerischer Topologie verbindet, um einen geometrisch adaptiven Optimierer zu schaffen. Er verlässt sich auf eine hochoptimierte **C++/CUDA-Implementierung**, um mit Bare-Metal-Geschwindigkeiten zu arbeiten.

In strengen, Multi-Seed-Stresstests hat **AdamV 2.0.2 alpha AdamW bei NanoGPT massakriert** und über mehrere unabhängige Seeds hinweg, wie von der globalen Stress-Suite validiert, einen deutlich geringeren Validierungsverlust erzielt.

## ⚙️ Die Säulen von AdamV 2.0.2 alpha

AdamV navigiert durch nicht-konvexe Verlustlandschaften mittels bahnbrechender mathematischer Innovationen:

### 1. Bakhshali-Wurzel & BRCM Geometrische Impulsbremsen
AdamV verwendet die alte Bakhshali-Näherungsmethode in Kombination mit Bakhshali Residual-Coupled Momentum (BRCM). Durch exponentielle Skalierung des Impulsabfalls ($\beta_1$) basierend auf der residualen Kollisionskraft ($\sqrt{v_t}$) wirkt der Optimierer als dynamischer Stoßdämpfer. Er wendet in engen topologischen Schluchten geometrische Impulsbremsen an, um explosive Gradienten einzudämmen, während er auf kargen Plateaus linear beschleunigt.

### 2. Krümmungsgesteuerte Dynamische Trägheit
Anstatt einen starren Impulsabfall zu verwenden, berechnet AdamV das lokale Signal-Rausch-Verhältnis. Auf flachen Plateaus verringert es die Trägheit, um sofort zu beschleunigen. In chaotischen Schluchten erhöht es die Trägheit, um Rauschen zu ignorieren und den Abstieg zu stabilisieren.

### 3. Log-Periodische Kühlung
AdamV integriert eine autonome **Log-Periodische Kühlung** unter Verwendung von Ramanujan-Kettenbruchentwicklungen. Dies skaliert die Lernrate dynamisch in einer log-periodischen Hüllkurve herunter, verhindert den Effekt des "blinden Zermalmens", der bei traditionellen Step-Schedulern auftritt, und ermöglicht eine organische Konvergenz in tiefere Minima-Becken.

### 4. Quantenflucht (OMNI-ModBH via Type-Punning)
Wenn AdamV ein karges Plateau erkennt, löst es einen absoluten Basin Hop aus. Bei Bare-Metal-Geschwindigkeiten auf der GPU wendet es bitweise Masken direkt auf die IEEE 754 float32-Mantisse (Type-Punning) an, teleportiert Gewichte in benachbarte Becken, ohne Warp-Divergenz zu verursachen oder Skalenexponenten zu zerstören.

## 📦 Installation

AdamV erfordert einen modernen C++17-Compiler und das CUDA-Toolkit (falls GPU-Beschleunigung gewünscht wird).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🎛️ Kalibrierungsleitfaden (Szenariospezifisch)

AdamV ist stark selbstregulierend, aber da verschiedene neuronale Architekturen grundlegend unterschiedliche mathematische Topologien aufweisen, müssen Sie AdamV je nach Modell richtig initialisieren.

### 1. Vision & NLP (Deterministisch & Deep Attention)
Für Standardmodelle (wie **ResNet**) und autoregressive Modelle (wie **NanoGPT**) verwenden Sie die **Goldene Kalibrierung**. Diese nutzt die Bakhshali-Wurzel aus dem 3. Jahrhundert und geometrische Impulsbremsen, um Gradientenexplosionen automatisch zu stoppen.

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

### 2. Generative Modelle (VAE, GANs, Diffusion)
Generative Modelle injizieren von Natur aus **Gaußsches Rauschen** in die Gradienten. Dieses Rauschen verursacht falsch-positive Kollisionen mit den Impulsbremsen von AdamV. Verwenden Sie für diese Modelle die **Stochastische Kalibrierung**.

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

## 🚀 Verwendung

Die Verwendung von `AdamVCpp` ist so einfach wie das Einfügen in Ihre PyTorch-Trainingsschleife.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Initialisiere AdamV 2.0.2 alpha
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # Basis-Lernrate
    betas=(0.9, 0.999),# Basis-Betas (beta1 wird dynamisch oszillieren)
    enable_omni=True   # Topologische Fluchten aktivieren (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # Mache einen Schritt! (Unterstützt native gemischte Präzision / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks & Validierungsleitfaden

Wir glauben an **offene Wissenschaft und reproduzierbare Ergebnisse**. Sie müssen uns nicht einfach vertrauen – Sie können die gesamte, 100% neutrale, Multi-Seed-Statistikvalidierungssuite ausführen, um AdamV direkt auf Ihrer eigenen Maschine mit AdamW zu vergleichen.

Die Suite testet beide Optimierer über 5 unabhängige zufällige Seeds hinweg auf drei verschiedenen Architekturen:
- **ResNet-18** (Deterministische Bildklassifizierung)
- **VAE** (Stochastisches generatives Rauschen)
- **NanoGPT** (Deep Autoregressive Attention)

### Wie man die Ergebnisse reproduziert:
1. **Hardwareanforderungen**: Eine CUDA-fähige GPU mit mindestens 15 GB VRAM wird dringend empfohlen (z. B. NVIDIA T4, RTX 3090 oder eine standardmäßige kostenlose Kaggle-GPU-Instanz).
2. **Führen Sie die Global Stress Suite aus**:
```bash
python benchmarks/run_global_stress_suite.py
```
3. **Was Sie erwartet**: Das Skript lädt automatisch die Datensätze (FashionMNIST, TinyShakespeare) herunter, kompiliert die AdamV C++-Kernel und führt alle 30 Kombinationen (5 Seeds × 3 Szenarien × 2 Optimierer) aus. Auf einer Standard-NVIDIA-T4-GPU dauert dieser Vorgang etwa ~1,5 Stunden.
4. **Ausgaben**: Nach Abschluss generiert das Skript automatisch eine Datei `global_stress_results.csv` mit den Rohmetriken und eine Datei `global_stress_plot.png`, die die Min-Max-Varianzschattierung und die p-Werte des Welch-t-Tests zeigt.

![Global Stress Test Results](assets/global_stress_results.png)

Für eine detaillierte Analyse der Benchmarks, p-Werte und Methodik lesen Sie bitte unseren [Benchmark-Bericht](benchmarks/README.md).
