# AdamV: Kaggle Benchmark Setup

This directory contains the ready-to-deploy Jupyter Notebook (`kaggle_adamv_arena.ipynb`) for the Kaggle platform.

## 🚀 How to deploy on Kaggle

1. Go to [Kaggle.com](https://www.kaggle.com/) and sign in.
2. Click on **Create > Notebook**.
3. In the notebook editor, go to **File > Import Notebook** and upload the `kaggle_adamv_arena.ipynb` file from this folder.
4. On the right-side panel, ensure **Accelerator** is set to **GPU T4 x2** or **GPU P100** (You have 30 hours of free GPU per week on Kaggle).
5. Click **Run All**!

## 🧪 What is inside the Notebook?

Because Kaggle relies heavily on isolated notebook environments, we embedded the core `AdamV` logic directly into the first cell of the notebook. 
This bypasses the need for the users to compile our C++ kernels via `gcc`, leveraging the automatic **GPU Python Fallback** we programmed. The math is 100% identical and accelerated by the PyTorch CUDA backend directly on Kaggle's GPUs.

The notebook executes the **Abyss Arena Benchmark**:
- A 15-Layer Deep MLP without residual connections.
- It proves that standard `AdamW` suffers from *Barren Plateaus* (vanishing gradients causing complete stagnation).
- It proves that `AdamV` injects the *Bakhshali-Pingala Basin Hopping* correctly to break symmetry and rescue the network, reaching convergence in seconds.

## 🏷️ Suggested Tags for Kaggle Publication
When publishing this notebook to the community, we suggest using these tags:
`optimization`, `pytorch`, `research`, `barren-plateaus`, `math`, `gradient-descent`
