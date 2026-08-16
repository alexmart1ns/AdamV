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

## 🎛️ Guide de Calibration (Spécifique au Scénario)

AdamV est hautement auto-régulé, mais comme les différentes architectures neuronales ont des topologies mathématiques fondamentalement différentes, vous devez initialiser AdamV correctement en fonction de votre modèle.

### 1. Vision & NLP (Déterministe & Attention Profonde)
Pour les modèles standards (comme **ResNet**) et les modèles autorégressifs (comme **NanoGPT**), utilisez la **Calibration Dorée**. Cela exploite la Racine de Bakhshali du 3ème siècle et les freins d'impulsion géométriques pour arrêter automatiquement les explosions de gradient.

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

### 2. Modèles Génératifs (VAE, GANs, Diffusion)
Les modèles génératifs injectent nativement du **bruit Gaussien** dans les gradients. Ce bruit provoque de fausses collisions positives avec les freins d'impulsion d'AdamV. Pour ces modèles, utilisez la **Calibration Stochastique**.

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

## 🧪 Benchmarks & Guide de Validation

Nous croyons en **la science ouverte et aux résultats reproductibles**. Vous n'avez pas à nous croire sur parole—vous pouvez exécuter l'intégralité de la suite de validation statistique multi-graines 100% neutre pour comparer AdamV à AdamW sur votre propre machine.

La suite teste les deux optimiseurs sur 5 graines aléatoires indépendantes sur trois architectures distinctes :
- **ResNet-18** (Classification d'Images Déterministe)
- **VAE** (Bruit Génératif Stochastique)
- **NanoGPT** (Attention Autorégressive Profonde)

### Comment reproduire les résultats :
1. **Exigences Matérielles** : Un GPU compatible CUDA avec au moins 15 Go de VRAM est fortement recommandé (par ex., NVIDIA T4, RTX 3090, ou une instance GPU gratuite standard sur Kaggle).
2. **Exécuter la Global Stress Suite** :
```bash
python benchmarks/run_global_stress_suite.py
```
3. **À quoi s'attendre** : Le script téléchargera automatiquement les ensembles de données (FashionMNIST, TinyShakespeare), compilera les noyaux C++ d'AdamV, et exécutera toutes les 30 combinaisons (5 Graines × 3 Scénarios × 2 Optimiseurs). Sur un GPU NVIDIA T4 standard, ce processus prend environ ~1,5 heures.
4. **Sorties** : Une fois terminé, le script générera automatiquement un fichier `global_stress_results.csv` avec les métriques brutes et un fichier `global_stress_plot.png` montrant l'ombrage de variance Min-Max et les valeurs p du test t de Welch.

![Global Stress Test Results](assets/global_stress_results.png)

Pour une analyse détaillée des benchmarks, des valeurs p et de la méthodologie, veuillez consulter notre [Rapport de Benchmark](benchmarks/README.md).
