import json
import os

def create_notebook():
    # Carregar o código do AdamV e do Kernel CUDA
    with open("torch_adamv.py", "r", encoding="utf-8") as f:
        adamv_code = f.read()
    with open("csrc/adamv_kernel.cu", "r", encoding="utf-8") as f:
        cuda_code = f.read()
    
    # Remover importações e inibir as dependências estáticas
    adamv_clean_code = adamv_code.replace("from csrc import adamv_cpp", "")
    
    # Suppress ImportError for adamv_cpp in Kaggle
    adamv_clean_code = adamv_clean_code.replace(
        "        try:\n            import adamv_cpp\n            self.adamv_cpp = adamv_cpp\n        except ImportError:\n            raise ImportError(\"AdamVCpp requer que a extensão C++ seja compilada rodando 'python setup_adamv.py build_ext --inplace'\")",
        "        self.adamv_cpp = None"
    )
    
    # Substituir a injeção do JIT
    adamv_clean_code = adamv_clean_code.replace(
        "        try:\n            import adamv_cuda\n            self.adamv_cuda = adamv_cuda\n        except ImportError:\n            self.adamv_cuda = None",
        "        self.adamv_cuda = adamv_cuda_module"
    )
    
    jit_setup_code = f"""import torch
import torch.utils.cpp_extension
import time

print("🔥 Compilando o Fused CUDA Kernel do AdamV (JIT)... Isso levará cerca de 40 segundos na primeira vez.")
start_time = time.time()

cuda_source = \"\"\"
{cuda_code}
\"\"\"

cpp_source = \"\"\"
#include <torch/extension.h>
void adamv_step_cuda(at::Tensor p, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor direcao, float lr, float beta1, float beta2, float eps, float weight_decay, float progresso, float bakh_thresh_eff, int step, int D, bool omni_triggered, int64_t punning_mask);
\"\"\"

adamv_cuda_module = torch.utils.cpp_extension.load_inline(
    name='adamv_cuda_jit',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['adamv_step_cuda'],
    with_cuda=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-use_fast_math']
)
print(f"✅ CUDA Kernel compilado com sucesso em {{time.time()-start_time:.2f}} segundos!")
"""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 The Honest Ring Benchmark: AdamV vs AdamW\n",
                "Welcome to the official Kaggle benchmark for **AdamV** (Adam Vedic Optimizer).\n",
                "AdamV is a stochastic optimizer designed to cure Barren Plateaus in Deep Neural Networks using Bakhshali Quartic clipping and Pingala Basin Hopping."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. JIT CUDA Compilation (The Fused Kernel)\n",
                "Instead of running on pure Python and suffering from CPU-GPU sync overhead, we use PyTorch's `load_inline` to compile our Bakhshali Quartic Brake kernel directly on Kaggle's Nvidia GPUs on-the-fly!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in jit_setup_code.split("\n")]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. The AdamV Optimizer Source Code\n",
                "We inject the Python wrapper of AdamV directly into this notebook. It will automatically detect the `adamv_cuda_module` compiled above and route all GPU tensors to it."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import math\n",
                "from torch.optim.optimizer import Optimizer\n\n"
            ] + [line + "\n" for line in adamv_clean_code.split("\n") if not line.startswith("import") and not line.startswith("from")]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. The Benchmark Setup (Abyss Arena)\n",
                "We will test AdamV against AdamW on a Deep MLP (40 layers, no residual connections) designed to suffer from vanishing gradients."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n\n",
                "class AbyssMLP(nn.Module):\n",
                "    def __init__(self, input_dim=512, hidden_dim=2048, num_layers=15, num_classes=10):\n",
                "        super().__init__()\n",
                "        layers = []\n",
                "        for i in range(num_layers):\n",
                "            in_f = input_dim if i == 0 else hidden_dim\n",
                "            layers.append(nn.Linear(in_f, hidden_dim))\n",
                "            layers.append(nn.GELU())\n",
                "        layers.append(nn.Linear(hidden_dim, num_classes))\n",
                "        self.net = nn.Sequential(*layers)\n\n",
                "    def forward(self, x):\n",
                "        return self.net(x)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def run_abyss_arena():\n",
                "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "    print(f'Running on {device}')\n",
                "    \n",
                "    N = 20000\n",
                "    X = torch.randn(N, 512).to(device)\n",
                "    # Create a real mathematical signal for the 10 classes\n",
                "    signal = torch.sin(X[:, 0] * 5.0) + torch.cos(X[:, 1] * 5.0) + X[:, 2:10].sum(dim=1)\n",
                "    # Use modulo to create perfectly balanced classes (0 to 9) with high non-linearity\n",
                "    Y = (torch.abs(signal * 10).long()) % 10\n",
                "    \n",
                "    # Add 50% destructive noise\n",
                "    noise_idx = torch.randperm(N)[:int(0.5 * N)]\n",
                "    Y[noise_idx] = torch.randint(0, 10, (len(noise_idx),)).to(device)\n",
                "    \n",
                "    optimizers = ['AdamW', 'AdamV']\n",
                "    results = {}\n",
                "    \n",
                "    import gc\n",
                "    for opt_name in optimizers:\n",
                "        torch.cuda.empty_cache()\n",
                "        gc.collect()\n",
                "        torch.manual_seed(42)\n",
                "        model = AbyssMLP().to(device)\n",
                "        \n",
                "        if opt_name == 'AdamW':\n",
                "            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)\n",
                "        else:\n",
                "            opt = AdamVCpp(model.parameters(), lr=1e-3, weight_decay=0.01)\n",
                "            \n",
                "        criterion = nn.CrossEntropyLoss()\n",
                "        \n",
                "        history = []\n",
                "        start_time = time.time()\n",
                "        \n",
                "        for step in range(2000):\n",
                "            # Mini-batch massivo para saturar a GPU (Reduzindo overhead de CPU)\n",
                "            idx = torch.randint(0, 20000, (4096,), device=device)\n",
                "            X_batch = X[idx]\n",
                "            Y_batch = Y[idx]\n",
                "            \n",
                "            opt.zero_grad()\n",
                "            out = model(X_batch)\n",
                "            loss = criterion(out, Y_batch)\n",
                "            loss.backward()\n",
                "            \n",
                "            if opt_name == 'AdamW':\n",
                "                opt.step()\n",
                "            else:\n",
                "                opt.step(current_loss=loss.item())\n",
                "                \n",
                "            history.append(loss.item())\n",
                "            \n",
                "            if step % 200 == 0 or step == 1999:\n",
                "                print(f'[{opt_name}] Step {step}/2000 | Loss: {loss.item():.4f}')\n",
                "            \n",
                "        results[opt_name] = history\n",
                "        print(f'{opt_name} finished in {time.time()-start_time:.2f}s | Final Loss: {history[-1]:.4f}')\n",
                "        \n",
                "    plt.figure(figsize=(10,6))\n",
                "    for name, hist in results.items():\n",
                "        plt.plot(hist, label=name)\n",
                "    plt.title('Abyss Arena (15-Layer Deep MLP w/ 50% Noise)')\n",
                "    plt.xlabel('Steps')\n",
                "    plt.ylabel('CrossEntropy Loss')\n",
                "    plt.legend()\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    plt.show()\n\n",
                "run_abyss_arena()\n"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("benchmarks_pytorch/kaggle_adamv_arena.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    create_notebook()
