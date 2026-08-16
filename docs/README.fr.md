[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: Optimiseur Géométriquement Adaptatif

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Védique) est un algorithme d'optimisation de pointe pour PyTorch qui fusionne les mathématiques védiques anciennes avec la topologie numérique pour créer un optimiseur géométriquement adaptatif. Il s'appuie sur une **implémentation C++/CUDA** hautement optimisée pour fonctionner à des vitesses bare-metal.

Lors de tests de résistance rigoureux et multi-graines, **AdamV 2.0.2 alpha a massacré AdamW sur NanoGPT**, atteignant une perte de validation significativement inférieure sur plusieurs graines indépendantes, tel que validé par la suite de tests de résistance globale.

## ⚙️ Les Piliers d'AdamV 2.0.2 alpha

AdamV navigue dans des paysages de perte non convexes en utilisant des innovations mathématiques révolutionnaires :

### 1. Racine de Bakhshali & Freins d'Impulsion Géométriques BRCM
AdamV utilise l'ancienne méthode d'approximation de Bakhshali combinée au Momentum à Couplage Résiduel de Bakhshali (BRCM). En mettant à l'échelle la décroissance de l'impulsion ($\beta_1$) de manière exponentielle en fonction de la force de collision résiduelle ($\sqrt{v_t}$), l'optimiseur agit comme un amortisseur dynamique. Il applique des freins d'impulsion géométriques dans les ravins topologiques étroits pour restreindre les gradients explosifs, tout en accélérant linéairement sur les plateaux arides.

### 2. Inertie Dynamique Guidée par la Courbure
Au lieu d'utiliser une décroissance de l'impulsion rigide, AdamV calcule le ratio signal sur bruit local. Sur les plateaux plats, il réduit l'inertie pour accélérer immédiatement. Dans les ravins chaotiques, il augmente l'inertie pour ignorer le bruit et stabiliser la descente.

### 3. Refroidissement Log-Périodique
AdamV intègre un **Refroidissement Log-Périodique** autonome utilisant les développements en fractions continues de Ramanujan. Cela réduit dynamiquement le taux d'apprentissage dans une enveloppe log-périodique, empêchant l'effet d'"écrasement aveugle" vu avec les ordonnanceurs par étapes traditionnels et permettant une convergence organique vers des bassins de minima plus profonds.

### 4. Échappement Quantique (OMNI-ModBH via Type-Punning)
Lorsqu'AdamV détecte un plateau aride, il déclenche un Basin Hop absolu. Fonctionnant à des vitesses bare-metal sur le GPU, il applique des masques au niveau du bit directement à la mantisse IEEE 754 float32 (Type-Punning), téléportant les poids vers des bassins adjacents sans causer de divergence de warp ni détruire les exposants d'échelle.

## 📦 Installation

AdamV nécessite un compilateur C++17 moderne et la boîte à outils CUDA (si l'accélération GPU est souhaitée).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Utilisation

L'utilisation d'`AdamVCpp` est aussi simple que de l'intégrer dans votre boucle d'entraînement PyTorch.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Initialiser AdamV 2.0.2 alpha
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # Taux d'apprentissage de base
    betas=(0.9, 0.999),# Betas de base (beta1 oscillera dynamiquement)
    enable_omni=True   # Activer les échappements topologiques (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # Faire un pas ! (Supporte la Précision Mixte Native / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks
Exécutez la suite de validation statistique multi-graines incluse, 100% neutre, pour comparer directement AdamV à AdamW sur votre machine. Elle exécute 5 graines sur ResNet-18, VAE et NanoGPT avec validation de la valeur p.

```bash
python benchmarks/run_global_stress_suite.py
```

![Global Stress Test Results](assets/global_stress_results.png)

Pour une analyse détaillée des benchmarks, des valeurs p et de la méthodologie, veuillez consulter notre [Rapport de Benchmark](benchmarks/README.md).
