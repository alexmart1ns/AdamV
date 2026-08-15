import json
import base64
import os

def create_cifar_kaggle_notebook():
    # Read the C++ kernel source
    with open('csrc/adamv_kernel.cu', 'r') as f:
        cuda_code = f.read()
        
    with open('csrc/adamv_kernel.cpp', 'r') as f:
        cpp_code = f.read()

    # Read the Python optimizer source
    with open('torch_adamv.py', 'r') as f:
        python_opt = f.read()

    # Read the CIFAR-10 arena source
    with open('benchmarks_pytorch/cifar10_arena.py', 'r') as f:
        cifar_arena = f.read()

    # Define the JIT compilation block
    jit_compilation = """
import os
import torch
from torch.utils.cpp_extension import load_inline

# Configure CUDA for T4 GPUs on Kaggle
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5"
os.environ['NVIDIA_VISIBLE_DEVICES'] = "all"
os.environ['OMP_NUM_THREADS'] = "1"

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Include CUDAContext to fix stream issues
cpp_source = \"\"\"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void adamv_step_cuda(at::Tensor p, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor direcao, float lr, float beta1, float beta2, float eps, float weight_decay, float progresso, float bakh_thresh_eff, int step, int D, bool omni_triggered, int64_t punning_mask);
\"\"\"

cuda_source = \"\"\"
""" + cuda_code + """
\"\"\"

print("Compiling AdamV CUDA Kernel (JIT) ...")
adamv_cuda = load_inline(
    name='adamv_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['adamv_step_cuda'],
    with_cuda=True,
    extra_cflags=['-O3', '-fopenmp'],
    extra_cuda_cflags=['-O3', '-use_fast_math', '-lineinfo'],
    verbose=True
)
print("CUDA Kernel loaded successfully!")

class DummyCPU:
    pass
adamv_cpp = DummyCPU()
"""


    # Replace the initialization in torch_adamv to use our JIT compiled versions
    python_opt_clean = python_opt.replace("import adamv_cpp", "pass # import adamv_cpp")
    python_opt_clean = python_opt_clean.replace("import adamv_cuda", "pass # import adamv_cuda")
    python_opt_clean = python_opt_clean.replace("raise ImportError(\"AdamVCpp requer que a extensão C++ seja compilada rodando 'python setup_adamv.py build_ext --inplace'\")", "pass")


    # Clean the arena code (remove local imports and main block)
    cifar_arena_clean = cifar_arena.replace("from adamv import AdamVCpp", "pass")
    cifar_arena_clean = cifar_arena_clean.replace("sys.path.append", "# sys.path.append")
    cifar_arena_clean = cifar_arena_clean.replace("if __name__ == '__main__':", "if False:")
    
    # Notebook structure
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 AdamV: CIFAR-10 Grand Finale Benchmark\n",
                    "This notebook tests the real-world computer vision performance of **AdamV** vs **AdamW**.\n",
                    "We train a **ResNet-9** architecture on the official CIFAR-10 dataset.\n",
                    "- **AdamW** uses CosineAnnealingLR.\n",
                    "- **AdamV** uses its native Ramanujan Envelope and OMNI-ModBH."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install ninja matplotlib torchvision\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    jit_compilation
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    python_opt_clean
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    cifar_arena_clean
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "run_cifar10_arena()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
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

    output_path = 'benchmarks_pytorch/kaggle_cifar10_arena.ipynb'
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
        
    print(f"[SUCCESS] Generated Kaggle Notebook: {output_path}")

if __name__ == '__main__':
    create_cifar_kaggle_notebook()
