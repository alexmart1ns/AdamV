# AdamV (Adam Vedic) Optimizer

<p align="center">
  <em>Um Otimizador PyTorch com Fused Kernels C++ unindo a estabilidade estrutural do Adam com a matemática não-linear de Ramanujan e Bakhshali.</em>
</p>

---

## ⚡ Visão Geral

O **AdamV** é o estágio final de evolução de um projeto de pesquisa matemática. 

Nosso objetivo foi descobrir se podíamos melhorar algoritmos de otimização de Deep Learning injetando teorias matemáticas antigas (Védicas). Após testes severos e engenharia de ponta a ponta, desenvolvemos um otimizador nativo para PyTorch que não apenas superou o `AdamW` em acurácia em datasets densos, mas também é **15 a 23% mais rápido** quando compilado no backend via extensão C++.

### A Matemática: Como Funciona
A arquitetura do AdamV usa 4 pilares:
1. **O Motor Base (Adam):** Utilizamos a média móvel exponencial confiável dos gradientes ($m_t$) e da variância não-centrada ($v_t$).
2. **O Envelope de Ramanujan:** Em vez de usar Schedulers tradicionais (como `CosineAnnealingLR`), o otimizador lê a **norma do tensor direcional** em cada passo e ajusta o passo de aprendizado dinamicamente usando uma equação de fração contínua inspirada na teoria termodinâmica/números de Ramanujan.
3. **O Freio ABS de Bakhshali:** Um mecanismo heurístico de *Gating* (Máscara). Quando o gradiente de uma dimensão específica "explode", o AdamV detecta a distorção e aplica uma Equação Quártica (o método de extração de raízes de Bakhshali) atuando como um *Gradient Clipping* ultra-localizado e perfeitamente estável.
4. **O Salto OMNI:** Em caso de estagnação prolongada num platô ou mínimo local, o otimizador injeta ruído direcional isotrópico para reiniciar o momento (inspirado no *Basin Hopping* e sequências de Recamán).

---

## 📂 Estrutura do Repositório

Organizamos o código para garantir máxima clareza:

* `torch_adamv.py`: **[A BIBLIOTECA PRINCIPAL]** Contém as classes `AdamV` (Python) e `AdamVCpp` (C++ Nativo). 
* `csrc/`: Contém os núcleos otimizados em `C++` (`adamv_kernel.cpp`) desenhados via ponteiros e laços de memória contígua (`/O2`).
* `setup_adamv.py`: Script de instalação e compilação do C++.
* `benchmarks_pytorch/`: Laboratórios reais de benchmarking da versão final testando Deep MLPs e CNNs (como a temida *Arena do Abismo* com 15 camadas densas e Ruído de Labels).
* `archive_numpy/`: Scripts de laboratório antigos baseados apenas em NumPy (historicamente guardados para prova de conceito).

---

## 🚀 Instalação e Compilação

Você pode usar o otimizador puramente em Python (que é estável e amigável), mas para destrancar a verdadeira velocidade competitiva e vencer os padrões da Meta/OpenAI, **recomendamos compilar a extensão nativa C++**.

### Pré-requisitos (Windows)
* Microsoft Visual C++ Build Tools (MSVC) instalado no seu sistema.

### Compilando a "Ferrari" (C++)
Na raiz do repositório, execute:
```bash
python setup_adamv.py build_ext --inplace
```
Isso vai gerar um arquivo `.pyd` (ex: `adamv_cpp.cp311-win_amd64.pyd`). Pronto! O PyTorch agora pode chamar o núcleo acelerado.

---

## 💻 Como Usar

O AdamV herda nativamente de `torch.optim.Optimizer`. Apenas troque o seu otimizador antigo por ele:

```python
import torch
from torch_adamv import AdamV, AdamVCpp

# Defina a sua rede neural (ResNet, Transformer, MLP, etc)
model = MinhaRedeNeural()

# Opção 1: Motor em puro Python (Universal)
optimizer = AdamV(model.parameters(), lr=1e-3, weight_decay=0.01, total_steps=10000)

# Opção 2: Fused Kernel em C++ (Requer compilação, MUITO mais rápido)
optimizer = AdamVCpp(model.parameters(), lr=1e-3, weight_decay=0.01, total_steps=10000)

# Loop de Treinamento
for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    
    # IMPORTANTE: O AdamV precisa saber o loss atual em cada passo 
    # para poder calcular estagnação e realizar o Salto OMNI
    optimizer.step(current_loss=loss.item())
```

---

## 📊 Benchmarks Finais

No arquivo `benchmarks_pytorch/abyss_arena.py` testamos uma Rede Densa (MLP) profunda de 15 camadas sem `Skip-Connections` e com 20% de ruído falsificado:

| Otimizador | Loss Final (CrossEntropy) | Acurácia (200 features, 20% ruído) | Tempo Total CPU |
| :--- | :---: | :---: | :---: |
| **AdamW** (Padrão SOTA) | 0.2620 | 96.7% | 9.37s |
| **AdamV** (Python) | 0.0874 | **97.2%** | 12.71s |
| **AdamVCpp** (C++) | **0.0904** | **97.2%** | **7.97s** 🏆 |

*O `AdamVCpp` demonstrou-se mais robusto à perda em paisagens não-lineares severas (graças à matemática védica) e bateu o tempo de execução do AdamW nativo do PyTorch devido à ausência de sobrecarga de instanciamento de kernel para tensores menores.*

