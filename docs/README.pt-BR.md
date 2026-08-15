[English](README.md) | [Português](README.pt-BR.md) | [Español](README.es.md) | [简体中文](README.zh-CN.md) | [Русский](README.ru.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

---

# 🧠 AdamV: Otimizador Geometricamente Adaptativo

![AdamV vs AdamW CIFAR-10](assets/cifar_10_train.png)

O AdamV (Adam-Vedic) é um algoritmo de otimização de última geração para o PyTorch que funde a antiga matemática védica com a topologia numérica para criar um otimizador 100% autônomo e geometricamente adaptativo.

Em testes rigorosos no CIFAR-10 (ResNet-9), **o AdamV 2.0 superou o AdamW** (89,68% vs 89,44%) rodando de forma completamente autônoma, sem depender de agendadores externos rígidos como o `CosineAnnealingLR`.

## ⚙️ Os 4 Pilares do AdamV 2.0

O AdamV é construído sobre um núcleo C++/CUDA altamente otimizado, baseando-se em quatro inovações matemáticas para navegar nas paisagens de perda (loss landscapes):

### 1. Inércia Dinâmica Movida por Curvatura (SNR Modulado por $\beta_1$)
Em vez de usar um decaimento rígido de momentum ($\beta_1 = 0.9$), o AdamV calcula a Relação Sinal-Ruído (Signal-to-Noise Ratio) local ($m_t^2 / v_t$). Em platôs planos, ele diminui a inércia para acelerar imediatamente. Em ravinas caóticas, ele aumenta a inércia para ignorar o ruído e estabilizar a descida.

### 2. Aproximação Quase-Newtoniana (Porta Livre de Hessiana de Bakhshali)
O AdamV usa o antigo Freio Quártico de Bakhshali como um freio gravitacional de segunda ordem. Ao escalar o denominador com a pseudo-Hessiana ($\sqrt{v_t}$), ele recorta (clips) de forma inteligente gradientes explosivos usando conhecimento de curvatura estrutural, sem o custo de memória de $O(N^2)$ das matrizes Hessianas completas.

### 3. Resfriamento Autônomo (Envelope Log-Periódico de Ramanujan)
O AdamV resfria organicamente seu learning rate à medida que atravessa o espaço topológico usando uma expansão de Fração Contínua de Ramanujan. Isso evita o "esmagamento cego" dos agendadores tradicionais, permitindo que a rede explore amplamente antes de se estabelecer em um mínimo global robusto.

### 4. Fuga Quântica (OMNI-ModBH via Type-Punning)
Quando o AdamV detecta um platô estéril, ele aciona um Basin Hop absoluto. Rodando em velocidades bare-metal na GPU, ele aplica máscaras bit a bit (bitwise) diretamente à mantissa float32 do IEEE 754 (Type-Punning), teletransportando os pesos para bacias adjacentes sem causar warp divergence ou destruir os expoentes de escala.

## 📦 Instalação

O AdamV requer um compilador C++17 moderno e o kit de ferramentas CUDA (se a aceleração por GPU for desejada).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🚀 Uso

Usar o AdamV é tão simples quanto colocá-lo no seu loop de treinamento do PyTorch. Você NÃO precisa de agendadores externos.

```python
import torch
from adamv import AdamVCpp

model = YourNeuralNetwork()

# Inicializa o AdamV (Nenhum agendador externo necessário!)
optimizer = AdamVCpp(
    model.parameters(),
    lr=0.01,           # learning rate base
    betas=(0.9, 0.999),# betas base (beta1 oscilará dinamicamente)
    enable_omni=True   # Habilita fugas topológicas (Type-Punning)
)

for epoch in range(15):
    for batch, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        
        # Dá um passo! (Suporta Native Mixed-Precision / AMP)
        optimizer.step(loss=loss)
```

## 🧪 Benchmarks
Rode a Arena CIFAR-10 incluída para realizar o benchmark do AdamV contra o AdamW diretamente na sua máquina:
```bash
python benchmarks/cifar10_arena.py
```
