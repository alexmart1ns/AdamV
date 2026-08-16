import json
import os

def build_notebook(py_script_path, output_nb_path):
    # Read AdamV core
    with open("adamv/torch_adamv.py", "r", encoding="utf-8") as f:
        adamv_code = f.read()
    with open("adamv/csrc/adamv_kernel.cu", "r", encoding="utf-8") as f:
        cuda_code = f.read()
        
    adamv_clean_code = adamv_code.replace("from csrc import adamv_cpp", "")
    adamv_clean_code = adamv_clean_code.replace(
        "        try:\n            import adamv_cpp\n            self.adamv_cpp = adamv_cpp\n        except ImportError:\n            raise ImportError(\"AdamVCpp requer que a extensao C++ seja compilada\")\n            \n        try:\n            import adamv_cuda\n            self.adamv_cuda = adamv_cuda\n        except ImportError:\n            self.adamv_cuda = None",
        "        self.adamv_cpp = globals().get('adamv_cuda', None)\n        self.adamv_cuda = globals().get('adamv_cuda', None)\n        if self.adamv_cuda is None:\n            pass # Fallback to python"
    )

    with open(py_script_path, "r", encoding="utf-8") as f:
        arena_code = f.read()
        
    arena_code = arena_code.replace("from adamv import AdamVCpp", "")
    arena_code = arena_code.replace("from adamv.torch_adamv import AdamVCpp", "")

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install ninja\n",
                    "import os\n",
                    "import torch\n",
                    "from torch.utils.cpp_extension import load_inline\n\n",
                    "cuda_source = \"\"\"\n" + cuda_code + "\n\"\"\"\n\n",
                    "cpp_source = \"\"\"\n",
                    "void adamv_step_cuda(\n",
                    "    at::Tensor p,\n",
                    "    at::Tensor grad,\n",
                    "    at::Tensor exp_avg,\n",
                    "    at::Tensor exp_avg_sq,\n",
                    "    at::Tensor direcao,\n",
                    "    float lr,\n",
                    "    float beta1,\n",
                    "    float beta2,\n",
                    "    float eps,\n",
                    "    float weight_decay,\n",
                    "    float progresso,\n",
                    "    float bakh_thresh_eff,\n",
                    "    int step,\n",
                    "    int D,\n",
                    "    bool enable_cooling,\n",
                    "    bool enable_brake,\n",
                    "    bool omni_triggered,\n",
                    "    int64_t punning_mask\n",
                    ");\n",
                    "\"\"\"\n\n",
                    "print('Compiling JIT C++ Kernel for Kaggle T4...')\n",
                    "adamv_cuda = load_inline(\n",
                    "    name='adamv_cuda_v8',\n",
                    "    cpp_sources=cpp_source,\n",
                    "    cuda_sources=cuda_source,\n",
                    "    functions=['adamv_step_cuda'],\n",
                    "    with_cuda=True,\n",
                    "    extra_cuda_cflags=['-O3', '-use_fast_math', '-arch=sm_75']\n",
                    ")\n",
                    "print('JIT Compilation complete! Kernel injected globally.')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in adamv_clean_code.split('\n')]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in arena_code.split('\n')]
            }
        ],
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

    with open(output_nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated {output_nb_path}")

build_notebook("benchmarks/transformer_arena.py", "benchmarks/kaggle_transformer_arena.ipynb")
build_notebook("benchmarks/vision_arena.py", "benchmarks/kaggle_vision_arena.ipynb")

# For adamv_only, we just create a version of transformer_arena that removes AdamW from the loop
with open("benchmarks/transformer_arena.py", "r", encoding="utf-8") as f:
    arena_code = f.read()
arena_code = arena_code.replace('["AdamW", "AdamVCpp"]', '["AdamVCpp"]')
with open("benchmarks/transformer_adamv_only_tmp.py", "w", encoding="utf-8") as f:
    f.write(arena_code)
build_notebook("benchmarks/transformer_adamv_only_tmp.py", "benchmarks/kaggle_transformer_adamv_only.ipynb")
os.remove("benchmarks/transformer_adamv_only_tmp.py")

print("All notebooks rebuilt cleanly!")
