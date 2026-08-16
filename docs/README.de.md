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

## 🧪 Benchmarks
Führen Sie die enthaltene 100% neutrale, Multi-Seed-Statistikvalidierungssuite aus, um AdamV direkt auf Ihrer Maschine mit AdamW zu vergleichen. Sie führt 5 Seeds über ResNet-18, VAE und NanoGPT mit p-Wert-Validierung aus.

```bash
python benchmarks/run_global_stress_suite.py
```

![Global Stress Test Results](assets/global_stress_results.png)

Für eine detaillierte Analyse der Benchmarks, p-Werte und Methodik lesen Sie bitte unseren [Benchmark-Bericht](benchmarks/README.md).
