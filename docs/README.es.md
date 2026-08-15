[English](README.md) | [Português](README.pt-BR.md) | [Español](README.es.md) | [简体中文](README.zh-CN.md) | [Русский](README.ru.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

---

# 🧠 AdamV: Optimizador Geométricamente Adaptativo

![AdamV vs AdamW CIFAR-10](assets/cifar_10_train.png)

AdamV (Adam-Védico) es un algoritmo de optimización de vanguardia para PyTorch que fusiona la antigua matemática védica con la topología numérica para crear un optimizador geométricamente adaptativo 100% autónomo.

En pruebas rigurosas en CIFAR-10 (ResNet-9), **AdamV 2.0 superó a AdamW** (89.68% vs 89.44%) ejecutándose de manera completamente autónoma, sin depender de schedulers externos rígidos como `CosineAnnealingLR`.

## ⚙️ Los 4 Pilares de AdamV 2.0

AdamV está construido sobre un núcleo de C++/CUDA altamente optimizado, basándose en cuatro innovaciones matemáticas para navegar los paisajes de pérdida:

### 1. Inercia Dinámica Impulsada por la Curvatura (SNR Modulado $\beta_1$)
En lugar de usar un decaimiento de momentum rígido ($\beta_1 = 0.9$), AdamV calcula la Relación Señal-Ruido local ($m_t^2 / v_t$). En mesetas planas, reduce la inercia para acelerar inmediatamente. En barrancos caóticos, incrementa la inercia para ignorar el ruido y estabilizar el descenso.

### 2. Aproximación Cuasi-Newtoniana (Bakhshali Hessian-Free Gate)
AdamV utiliza el antiguo Freno Cuártico de Bakhshali como un freno gravitacional de segundo orden. Al escalar el denominador con el pseudo-Hessiano ($\sqrt{v_t}$), recorta inteligentemente los gradientes explosivos usando conocimiento estructural de la curvatura, sin el costo de memoria $O(N^2)$ de las matrices Hessianas completas.

### 3. Enfriamiento Autónomo (Envolvente Log-Periódica de Ramanujan)
AdamV enfría orgánicamente su learning rate a medida que atraviesa el espacio topológico utilizando una expansión de Fracción Continua de Ramanujan. Esto previene el "aplastamiento ciego" de los schedulers tradicionales, permitiendo que la red explore ampliamente antes de establecerse en un mínimo global robusto.

### 4. Escape Cuántico (OMNI-ModBH vía Type-Punning)
Cuando AdamV detecta una meseta estéril, desencadena un Salto de Cuenca absoluto. Ejecutándose a velocidades nativas en la GPU, aplica máscaras de bits directamente a la mantisa del float32 IEEE 754 (Type-Punning), teletransportando los pesos a cuencas adyacentes sin causar warp divergence ni destruir los exponentes de escala.

## 📦 Instalación

AdamV requiere un compilador C++17 moderno y el toolkit de CUDA (si se desea aceleración por GPU).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Uso

Usar AdamV es tan simple como integrarlo en tu bucle de entrenamiento de PyTorch. NO necesitas schedulers externos.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Inicializar AdamV (¡No se necesita scheduler externo!)
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # learning rate base
    betas=(0.9, 0.999),# betas base (beta1 oscilará dinámicamente)
    enable_omni=True   # Habilitar escapes topológicos (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # ¡Dar un paso! (Soporta Precisión Mixta Nativa / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks
Ejecuta el Arena de CIFAR-10 incluido para hacer benchmark de AdamV frente a AdamW directamente en tu máquina:
```bash
python benchmarks/cifar10_arena.py
```
