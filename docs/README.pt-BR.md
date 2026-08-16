[English](README.md) | [Português](docs/README.pt-BR.md) | [Español](docs/README.es.md) | [简体中文](docs/README.zh-CN.md) | [Русский](docs/README.ru.md) | [日本語](docs/README.ja.md) | [Deutsch](docs/README.de.md) | [Français](docs/README.fr.md)

---

# 🧠 AdamV 2.0.2 alpha: Otimizador Geometricamente Adaptativo

![AdamV vs AdamW Duel](assets/adamv_duel_banner.jpg)

AdamV 2.0.2 alpha (Adam-Vedic) é um algoritmo de otimização de última geração para PyTorch que funde a antiga matemática védica com topologia numérica para criar um otimizador geometricamente adaptativo. Ele depende de uma **implementação em C++/CUDA** altamente otimizada para operar em velocidades bare-metal.

Em rigorosos testes de estresse com múltiplas sementes, **AdamV 2.0.2 alpha massacrou o AdamW no NanoGPT**, alcançando uma perda de validação significativamente menor em várias sementes independentes, conforme validado pela suíte de estresse global.

## ⚙️ Os Pilares do AdamV 2.0.2 alpha

AdamV navega em paisagens de perda não convexas usando inovações matemáticas revolucionárias:

### 1. A Raiz de Bakhshali & Freios de Momento Geométrico BRCM
AdamV utiliza o antigo método de aproximação de Bakhshali combinado com o Momento Acoplado a Resíduos de Bakhshali (BRCM). Ao dimensionar o decaimento do momento ($\beta_1$) exponencialmente com base na força de colisão residual ($\sqrt{v_t}$), o otimizador atua como um amortecedor dinâmico. Ele aplica freios de momento geométrico em ravinas topológicas estreitas para conter gradientes explosivos, enquanto acelera linearmente em platôs áridos.

### 2. Inércia Dinâmica Movida por Curvatura
Em vez de usar um decaimento de momento rígido, AdamV calcula a Relação Sinal-Ruído local. Em platôs planos, ele diminui a inércia para acelerar imediatamente. Em ravinas caóticas, ele aumenta a inércia para ignorar o ruído e estabilizar a descida.

### 3. Resfriamento Log-Periódico
AdamV incorpora um **Resfriamento Log-Periódico** autônomo usando expansões de Frações Contínuas de Ramanujan. Isso reduz dinamicamente a taxa de aprendizado em um envelope log-periódico, prevenindo o efeito de "esmagamento cego" visto com agendadores de passos tradicionais e permitindo uma convergência orgânica em bacias de mínimos mais profundos.

### 4. Fuga Quântica (OMNI-ModBH via Type-Punning)
Quando o AdamV detecta um platô árido, ele aciona um Salto de Bacia (Basin Hop) absoluto. Operando em velocidades bare-metal na GPU, ele aplica máscaras bit a bit diretamente à mantissa do float32 IEEE 754 (Type-Punning), teletransportando pesos para bacias adjacentes sem causar divergência de warp ou destruir os expoentes de escala.

## 📦 Instalação

AdamV requer um compilador C++17 moderno e o toolkit CUDA (se a aceleração por GPU for desejada).

```bash
git clone https://github.com/your-username/AdamV.git
cd AdamV
pip install -e .
```

## 🎛️ Guia de Calibração (Específico por Cenário)

AdamV é altamente autorregulado, mas como diferentes arquiteturas neurais têm topologias matemáticas fundamentalmente diferentes, você deve inicializar o AdamV corretamente dependendo do seu modelo.

### 1. Visão & PNL (Determinístico & Atenção Profunda)
Para modelos padrão (como **ResNet**) e modelos autorregressivos (como **NanoGPT**), use a **Calibração Dourada** (Golden Calibration). Isso aproveita a Raiz de Bakhshali do século III e os freios de momento geométrico para parar explosões de gradiente automaticamente.

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

### 2. Modelos Generativos (VAE, GANs, Difusão)
Modelos generativos injetam nativamente **ruído Gaussiano** nos gradientes. Esse ruído causa colisões falsas positivas com os freios de momento do AdamV. Para esses modelos, use a **Calibração Estocástica** (Stochastic Calibration).

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

## 🚀 Uso

Usar `AdamVCpp` é tão simples quanto inseri-lo no seu loop de treinamento do PyTorch.

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

## 🧪 Guia de Benchmarks & Validação

Acreditamos na **ciência aberta e em resultados reprodutíveis**. Você não precisa acreditar apenas na nossa palavra — você pode executar toda a suíte de validação estatística 100% neutra e com múltiplas sementes para comparar o AdamV com o AdamW na sua própria máquina.

A suíte testa ambos os otimizadores em 5 sementes aleatórias independentes em três arquiteturas distintas:
- **ResNet-18** (Classificação de Imagens Determinística)
- **VAE** (Ruído Generativo Estocástico)
- **NanoGPT** (Atenção Autorregressiva Profunda)

### Como reproduzir os resultados:
1. **Requisitos de Hardware**: Uma GPU compatível com CUDA com pelo menos 15GB de VRAM é altamente recomendada (por exemplo, NVIDIA T4, RTX 3090 ou uma instância de GPU padrão gratuita do Kaggle).
2. **Execute a Suíte de Estresse Global**:
```bash
python benchmarks/run_global_stress_suite.py
```
3. **O que esperar**: O script fará automaticamente o download dos conjuntos de dados (FashionMNIST, TinyShakespeare), compilará os kernels C++ do AdamV e executará todas as 30 combinações (5 Sementes × 3 Cenários × 2 Otimizadores). Em uma GPU NVIDIA T4 padrão, este processo leva aproximadamente ~1,5 horas.
4. **Saídas**: Após a conclusão, o script gerará automaticamente um arquivo `global_stress_results.csv` com as métricas brutas e um `global_stress_plot.png` mostrando o sombreamento de variância Min-Max e os valores p do teste t de Welch.

![Global Stress Test Results](assets/global_stress_results.png)

Para uma análise detalhada dos benchmarks, valores p e metodologia, consulte nosso [Relatório de Benchmark](benchmarks/README.md).
