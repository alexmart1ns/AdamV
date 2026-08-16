[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: Optimizador Geométricamente Adaptativo

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Vedic) es un algoritmo de optimización de última generación para PyTorch que fusiona la antigua matemática védica con la topología numérica para crear un optimizador geométricamente adaptativo. Se basa en una **implementación en C++/CUDA** altamente optimizada para operar a velocidades bare-metal.

En pruebas de estrés rigurosas con múltiples semillas, **AdamV 2.0.2 alpha masacró a AdamW en NanoGPT**, logrando una pérdida de validación significativamente menor en múltiples semillas independientes, validado por la suite de estrés global.

## ⚙️ Los Pilares de AdamV 2.0.2 alpha

AdamV navega por paisajes de pérdida no convexos utilizando innovaciones matemáticas revolucionarias:

### 1. La Raíz de Bakhshali y los Frenos de Momento Geométrico BRCM
AdamV utiliza el antiguo método de aproximación de Bakhshali combinado con el Momento Acoplado a Residuos de Bakhshali (BRCM). Al escalar el decaimiento del momento ($\beta_1$) exponencialmente basándose en la fuerza de colisión residual ($\sqrt{v_t}$), el optimizador actúa como un amortiguador dinámico. Aplica frenos de momento geométrico en estrechos barrancos topológicos para restringir gradientes explosivos, mientras acelera linealmente en mesetas áridas.

### 2. Inercia Dinámica Impulsada por Curvatura
En lugar de utilizar un decaimiento de momento rígido, AdamV calcula la Relación Señal-Ruido local. En mesetas planas, reduce la inercia para acelerar de inmediato. En barrancos caóticos, aumenta la inercia para ignorar el ruido y estabilizar el descenso.

### 3. Enfriamiento Log-Periódico
AdamV incorpora un **Enfriamiento Log-Periódico** autónomo mediante expansiones de Fracciones Continuas de Ramanujan. Esto reduce dinámicamente la tasa de aprendizaje en una envolvente log-periódica, previniendo el efecto de "aplastamiento ciego" visto en los programadores de pasos tradicionales y permitiendo una convergencia orgánica en cuencas de mínimos más profundas.

### 4. Escape Cuántico (OMNI-ModBH vía Type-Punning)
Cuando AdamV detecta una meseta árida, desencadena un Salto de Cuenca (Basin Hop) absoluto. Ejecutándose a velocidades bare-metal en la GPU, aplica máscaras bit a bit directamente a la mantisa del float32 IEEE 754 (Type-Punning), teletransportando pesos a cuencas adyacentes sin causar divergencia de warp o destruir los exponentes de escala.

## 📦 Instalación

AdamV requiere un compilador C++17 moderno y el toolkit de CUDA (si se desea aceleración por GPU).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Uso

Usar `AdamVCpp` es tan simple como integrarlo en su bucle de entrenamiento de PyTorch.

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
Ejecute la suite de validación estadística 100% neutral y de múltiples semillas incluida para evaluar AdamV frente a AdamW directamente en su máquina. Ejecuta 5 semillas en ResNet-18, VAE y NanoGPT con validación de valor p.

```bash
python benchmarks/run_global_stress_suite.py
```

![Global Stress Test Results](assets/global_stress_results.png)

Para un análisis detallado de los benchmarks, valores p y la metodología, consulte nuestro [Informe de Benchmark](benchmarks/README.md).
