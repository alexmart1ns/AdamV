[English](README.md) | [Português](README.pt-BR.md) | [Español](README.es.md) | [简体中文](README.zh-CN.md) | [Русский](README.ru.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

---

# 🧠 AdamV: Optimiseur géométriquement adaptatif

![AdamV vs AdamW CIFAR-10](assets/cifar_10_train.png)

AdamV (Adam-Védique) est un algorithme d'optimisation de pointe pour PyTorch qui fusionne les anciennes mathématiques védiques avec la topologie numérique pour créer un optimiseur 100 % autonome et géométriquement adaptatif.

Lors de tests rigoureux sur CIFAR-10 (ResNet-9), **AdamV 2.0 a surpassé AdamW** (89.68% vs 89.44%) en fonctionnant de manière totalement autonome, sans s'appuyer sur des planificateurs externes rigides comme `CosineAnnealingLR`.

## ⚙️ Les 4 Piliers d'AdamV 2.0

AdamV est construit sur un cœur C++/CUDA hautement optimisé, s'appuyant sur quatre innovations mathématiques pour naviguer dans les paysages de perte :

### 1. Inertie dynamique pilotée par la courbure ($\beta_1$ modulé par le SNR)
Au lieu d'utiliser une décroissance de momentum rigide ($\beta_1 = 0.9$), AdamV calcule le rapport signal sur bruit local ($m_t^2 / v_t$). Sur les plateaux plats, il réduit l'inertie pour accélérer immédiatement. Dans les ravins chaotiques, il augmente l'inertie pour ignorer le bruit et stabiliser la descente.

### 2. Approximation quasi-newtonienne (Porte sans Hessien de Bakhshali)
AdamV utilise l'ancien frein quartique de Bakhshali comme frein gravitationnel de second ordre. En mettant à l'échelle le dénominateur avec la pseudo-hessienne ($\sqrt{v_t}$), il limite intelligemment les gradients explosifs en utilisant la connaissance structurelle de la courbure, sans le coût mémoire de $O(N^2)$ des matrices hessiennes complètes.

### 3. Refroidissement autonome (Enveloppe log-périodique de Ramanujan)
AdamV refroidit organiquement son learning rate à mesure qu'il traverse l'espace topologique en utilisant un développement en fraction continue de Ramanujan. Cela empêche l'"écrasement aveugle" des planificateurs traditionnels, permettant au réseau d'explorer largement avant de se stabiliser dans un minimum global robuste.

### 4. Évasion quantique (OMNI-ModBH via Type-Punning)
Lorsqu'AdamV détecte un plateau stérile, il déclenche un Basin Hop absolu. S'exécutant à des vitesses "bare-metal" sur le GPU, il applique des masques au niveau du bit directement à la mantisse du float32 de la norme IEEE 754 (Type-Punning), téléportant les poids vers des bassins adjacents sans provoquer de warp divergence ni détruire les exposants d'échelle.

## 📦 Installation

AdamV nécessite un compilateur C++17 moderne et le toolkit CUDA (si l'accélération GPU est souhaitée).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Utilisation

L'utilisation d'AdamV est aussi simple que de l'intégrer dans votre boucle d'entraînement PyTorch. Vous n'avez PAS besoin de planificateurs externes.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Initialiser AdamV (Aucun planificateur externe n'est nécessaire !)
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # learning rate de base
    betas=(0.9, 0.999),# Betas de base (beta1 oscillera dynamiquement)
    enable_omni=True   # Active les évasions topologiques (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # Fait un pas d'optimisation ! (Prend en charge la précision mixte native / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks
Exécutez l'arène CIFAR-10 incluse pour évaluer AdamV par rapport à AdamW directement sur votre machine :
```bash
python benchmarks/cifar10_arena.py
```
