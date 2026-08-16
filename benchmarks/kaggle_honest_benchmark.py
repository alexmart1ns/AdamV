"""
# AdamV 3.0 — Honest Benchmark Suite
# ============================================================
# Kaggle Notebook | GPU T4/P100 Required
# 
# Methodology:
#   - 3 Scenarios: Vision (CIFAR-10), Generative (VAE), NLP (NanoGPT)
#   - 3 Optimizers: AdamW (baseline), AdamV 3.0, SGD+Momentum
#   - 5 Random Seeds for statistical significance
#   - Identical initial weights across optimizers (per seed)
#   - Flat learning rate (no scheduler) for fair comparison
#   - Welch's t-test on final metrics
#   - 95% CI shading on convergence plots
#
# What we DON'T do:
#   - No cherry-picked hyperparameters per optimizer
#   - No inline code redefinitions
#   - No monkey-patching
#   - No separate LR schedules that favor one optimizer
# ============================================================
"""

# %% [markdown]
# # AdamV 3.0 — Honest Benchmark Suite
# 
# **3 Scenarios** × **3 Optimizers** × **5 Seeds** = 45 training runs
# 
# This notebook uses the **real AdamV 3.0 library** — no inline code overrides.
# 
# | Scenario | Architecture | Dataset | Metric | Epochs/Steps |
# |:---|:---|:---|:---|:---|
# | Vision | ResNet-18 | CIFAR-10 | Val Accuracy % | 10 epochs |
# | Generative | VAE | FashionMNIST | Val ELBO Loss | 15 epochs |
# | NLP | NanoGPT (4L-4H) | TinyShakespeare | Val CrossEntropy | 2000 steps |

# %% 
# ============================================================
# 0. INSTALL ADAMV FROM GITHUB (Pure Python — no C++ build needed)
# ============================================================
import subprocess
import sys
import os

# Clone the repo (avoids C++ extension build issues on Kaggle)
if not os.path.exists('AdamV'):
    subprocess.check_call(['git', 'clone', '--depth', '1', 
                           'https://github.com/alexmart1ns/AdamV.git'])

# Add to Python path so we can import directly
sys.path.insert(0, os.path.join(os.getcwd(), 'AdamV'))

# Verify import
import adamv
from adamv import AdamV
print(f"✓ AdamV {adamv.__version__} loaded (Pure Python — GPU via PyTorch)")

# %% 
# ============================================================
# 1. IMPORTS
# ============================================================
import os
import gc
import math
import time
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind
from adamv import AdamV

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% 
# ============================================================
# 2. CONFIGURATION — ALL IN ONE PLACE
# ============================================================
# Hyperparameters are SHARED across optimizers (fair comparison)
CONFIG = {
    "seeds": [42, 1337, 2024, 3141, 8888],
    "optimizers": ["AdamW", "AdamV", "SGD"],
    
    "Vision": {
        "lr": 1e-3,
        "wd": 0.01,
        "epochs": 10,
        "batch_size": 128,
    },
    "Generative": {
        "lr": 1e-3,
        "wd": 0.0,
        "epochs": 15,
        "batch_size": 128,
    },
    "NLP": {
        "lr": 1e-3,
        "wd": 0.1,
        "max_iters": 2000,
        "eval_interval": 200,
        "eval_iters": 50,
        "batch_size": 64,
        "block_size": 128,
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.1,
    }
}

print("\n— Configuration —")
for k, v in CONFIG.items():
    if isinstance(v, dict):
        print(f"  {k}: {v}")

# %% 
# ============================================================
# 3. UTILITIES
# ============================================================
def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(model, opt_name, scenario, total_steps):
    """Create optimizer with SHARED hyperparameters. No per-optimizer tuning."""
    cfg = CONFIG[scenario]
    lr = cfg["lr"]
    wd = cfg["wd"]
    
    # Separate decay/no-decay params (standard practice)
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    if opt_name == "AdamW":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "AdamV":
        return AdamV(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                     total_steps=total_steps,
                     bakhshali_threshold=50.0,
                     enable_brake=True,
                     enable_omni=False)
    elif opt_name == "SGD":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)

# %% 
# ============================================================
# 4. ARCHITECTURES
# ============================================================

# --- 4a. ResNet-18 for CIFAR-10 ---
class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 images, 3 channels)."""
    def __init__(self, num_classes=10):
        super().__init__()
        import torchvision.models as models
        self.model = models.resnet18(weights=None)
        # CIFAR-10 uses 32x32, so use smaller conv1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)

# --- 4b. VAE for FashionMNIST ---
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- 4c. NanoGPT for TinyShakespeare ---
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

# %% 
# ============================================================
# 5. DATA LOADERS
# ============================================================
def get_vision_data(seed, batch_size):
    """CIFAR-10 with standard augmentation."""
    seed_everything(seed)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    g = torch.Generator()
    g.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True, generator=g)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True)
    return trainloader, testloader

def get_generative_data(seed, batch_size):
    """FashionMNIST for VAE."""
    seed_everything(seed)
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    g = torch.Generator()
    g.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True, generator=g)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True)
    return trainloader, testloader

def get_nlp_data():
    """TinyShakespeare character-level corpus."""
    if not os.path.exists('input.txt'):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, 'input.txt')
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size

def get_batch_nlp(split, train_data, val_data, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss_nlp(model, train_data, val_data, cfg):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(cfg['eval_iters'])
        for k in range(cfg['eval_iters']):
            X, Y = get_batch_nlp(split, train_data, val_data, cfg['batch_size'], cfg['block_size'])
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# %% 
# ============================================================
# 6. TRAINING LOOPS
# ============================================================
def train_vision(model, optimizer, trainloader, testloader, epochs):
    """Train ResNet-18 on CIFAR-10. Returns list of (epoch, val_acc)."""
    history = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                _, predicted = model(inputs).max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * correct / total
        history.append((epoch + 1, val_acc))
        print(f"      Epoch {epoch+1:2d} | Val Acc: {val_acc:.2f}%")
    
    return history

def train_generative(model, optimizer, trainloader, testloader, epochs):
    """Train VAE on FashionMNIST. Returns list of (epoch, val_elbo)."""
    history = []
    for epoch in range(epochs):
        model.train()
        for inputs, _ in trainloader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(inputs)
            loss = vae_loss(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, _ in testloader:
                inputs = inputs.to(DEVICE)
                recon, mu, logvar = model(inputs)
                test_loss += vae_loss(recon, inputs, mu, logvar).item()
        
        val_elbo = test_loss / len(testloader.dataset)
        history.append((epoch + 1, val_elbo))
        print(f"      Epoch {epoch+1:2d} | Val ELBO: {val_elbo:.4f}")
    
    return history

def train_nlp(model, optimizer, train_data, val_data, cfg):
    """Train NanoGPT on TinyShakespeare. Returns list of (step, val_loss)."""
    history = []
    max_iters = cfg['max_iters']
    
    for step in range(max_iters + 1):
        if step % cfg['eval_interval'] == 0 or step == max_iters:
            losses = estimate_loss_nlp(model, train_data, val_data, cfg)
            history.append((step, losses['val']))
            print(f"      Step {step:5d} | Val Loss: {losses['val']:.4f}")
        
        if step < max_iters:
            xb, yb = get_batch_nlp('train', train_data, val_data, cfg['batch_size'], cfg['block_size'])
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    return history

# %% 
# ============================================================
# 7. MAIN EXECUTION ENGINE
# ============================================================
def run_benchmark():
    """Run the full benchmark suite."""
    
    seeds = CONFIG["seeds"]
    optimizers = CONFIG["optimizers"]
    scenarios = ["Vision", "Generative", "NLP"]
    
    all_results = []
    final_metrics = {s: {o: [] for o in optimizers} for s in scenarios}
    timing = {s: {o: [] for o in optimizers} for s in scenarios}
    
    total_runs = len(seeds) * len(scenarios) * len(optimizers)
    run_count = 0
    
    print(f"\n{'='*60}")
    print(f"  AdamV 3.0 — Honest Benchmark Suite")
    print(f"  {total_runs} training runs ({len(seeds)} seeds x {len(scenarios)} scenarios x {len(optimizers)} optimizers)")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"  GLOBAL SEED: {seed}")
        print(f"{'='*50}")
        
        for scenario in scenarios:
            print(f"\n  --- Scenario: {scenario} ---")
            cfg = CONFIG[scenario]
            
            # Load data ONCE per scenario per seed
            if scenario == "Vision":
                trainloader, testloader = get_vision_data(seed, cfg["batch_size"])
                total_steps = len(trainloader) * cfg["epochs"]
            elif scenario == "Generative":
                trainloader, testloader = get_generative_data(seed, cfg["batch_size"])
                total_steps = len(trainloader) * cfg["epochs"]
            elif scenario == "NLP":
                train_data, val_data, vocab_size = get_nlp_data()
                total_steps = cfg["max_iters"]
            
            # Create base weights ONCE (shared by all optimizers in this seed)
            seed_everything(seed)
            if scenario == "Vision":
                base_model = ResNet18CIFAR(num_classes=10)
            elif scenario == "Generative":
                base_model = VAE()
            elif scenario == "NLP":
                base_model = NanoGPT(
                    vocab_size, cfg["n_embd"], cfg["n_head"], 
                    cfg["n_layer"], cfg["block_size"], cfg["dropout"]
                )
            
            base_state = base_model.state_dict()
            param_count = count_params(base_model)
            print(f"  Model: {base_model.__class__.__name__} ({param_count:,} params)")
            del base_model
            gc.collect()
            
            for opt_name in optimizers:
                run_count += 1
                print(f"\n  [{run_count}/{total_runs}] Optimizer: {opt_name}")
                seed_everything(seed)
                
                # Create model with IDENTICAL weights
                if scenario == "Vision":
                    model = ResNet18CIFAR(num_classes=10).to(DEVICE)
                elif scenario == "Generative":
                    model = VAE().to(DEVICE)
                elif scenario == "NLP":
                    model = NanoGPT(
                        vocab_size, cfg["n_embd"], cfg["n_head"],
                        cfg["n_layer"], cfg["block_size"], cfg["dropout"]
                    ).to(DEVICE)
                
                model.load_state_dict(base_state)
                
                optimizer = get_optimizer(model, opt_name, scenario, total_steps)
                
                # Train
                t0 = time.time()
                
                if scenario == "Vision":
                    history = train_vision(model, optimizer, trainloader, testloader, cfg["epochs"])
                elif scenario == "Generative":
                    history = train_generative(model, optimizer, trainloader, testloader, cfg["epochs"])
                elif scenario == "NLP":
                    history = train_nlp(model, optimizer, train_data, val_data, cfg)
                
                elapsed = time.time() - t0
                timing[scenario][opt_name].append(elapsed)
                print(f"  Time: {elapsed:.1f}s")
                
                # Record results
                for epoch_or_step, metric in history:
                    all_results.append({
                        "Seed": seed, "Scenario": scenario, 
                        "Optimizer": opt_name, "Step": epoch_or_step,
                        "Metric": metric
                    })
                
                # Record final metric
                final_metric = history[-1][1]
                final_metrics[scenario][opt_name].append(final_metric)
                
                # Cleanup
                del model, optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Cleanup data
            if scenario != "NLP":
                del trainloader, testloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_results, final_metrics, timing

# %% 
# ============================================================
# 8. RUN THE BENCHMARK
# ============================================================
all_results, final_metrics, timing = run_benchmark()

# %% 
# ============================================================
# 9. STATISTICAL ANALYSIS
# ============================================================
print(f"\n{'='*70}")
print(f"  STATISTICAL ANALYSIS (Welch's t-test, two-sided)")
print(f"{'='*70}\n")

scenarios = ["Vision", "Generative", "NLP"]
metric_names = {"Vision": "Val Accuracy %", "Generative": "Val ELBO Loss", "NLP": "Val CrossEntropy"}
higher_is_better = {"Vision": True, "Generative": False, "NLP": False}

stats_rows = []

for scenario in scenarios:
    print(f"-- {scenario} ({metric_names[scenario]}) --")
    
    optimizers = list(final_metrics[scenario].keys())
    
    for opt_name in optimizers:
        vals = final_metrics[scenario][opt_name]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0
        t_mean = np.mean(timing[scenario][opt_name])
        print(f"  {opt_name:8s}: {mean:.4f} +/- {std:.4f}  (mean time: {t_mean:.1f}s)")
        stats_rows.append({
            "Scenario": scenario, "Optimizer": opt_name,
            "Mean": mean, "Std": std, "Time_s": t_mean
        })
    
    # Pairwise t-tests vs AdamW (baseline)
    baseline = "AdamW"
    baseline_vals = final_metrics[scenario][baseline]
    
    for opt_name in optimizers:
        if opt_name == baseline:
            continue
        test_vals = final_metrics[scenario][opt_name]
        t_stat, p_val = ttest_ind(baseline_vals, test_vals, equal_var=False)
        
        baseline_mean = np.mean(baseline_vals)
        test_mean = np.mean(test_vals)
        
        if higher_is_better[scenario]:
            better = "BETTER" if test_mean > baseline_mean else "WORSE"
        else:
            better = "BETTER" if test_mean < baseline_mean else "WORSE"
        
        sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        print(f"  {opt_name} vs {baseline}: p={p_val:.4f} ({sig}) [{better}]")
    
    print()

# %% 
# ============================================================
# 10. CONVERGENCE PLOTS
# ============================================================
df = pd.DataFrame(all_results)
df.to_csv("benchmark_results_v3.csv", index=False)
print("Saved: benchmark_results_v3.csv")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = {"AdamW": "#2563EB", "AdamV": "#DC2626", "SGD": "#059669"}
titles = {
    "Vision": "CIFAR-10 / ResNet-18\nValidation Accuracy (%)",
    "Generative": "FashionMNIST / VAE\nValidation ELBO Loss",
    "NLP": "TinyShakespeare / NanoGPT\nValidation CrossEntropy"
}

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    scenario_df = df[df["Scenario"] == scenario]
    
    for opt_name in CONFIG["optimizers"]:
        opt_df = scenario_df[scenario_df["Optimizer"] == opt_name]
        if opt_df.empty:
            continue
        
        agg = opt_df.groupby("Step")["Metric"].agg(["mean", "std", "min", "max"]).reset_index()
        
        # Final metric for legend
        final_mean = agg["mean"].iloc[-1]
        final_std = agg["std"].iloc[-1]
        label = f"{opt_name} ({final_mean:.2f}+/-{final_std:.2f})"
        
        ax.plot(agg["Step"], agg["mean"], label=label, color=colors[opt_name], linewidth=2)
        
        # Min-max shading (more informative than CI for n=5)
        ax.fill_between(agg["Step"], agg["min"], agg["max"], 
                        color=colors[opt_name], alpha=0.12)
    
    ax.set_title(titles[scenario], fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Epoch" if scenario != "NLP" else "Step", fontsize=11)
    ax.set_ylabel(metric_names[scenario], fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Invert y-axis for loss metrics (lower is better)
    if not higher_is_better[scenario]:
        ax.invert_yaxis()

plt.suptitle("AdamV 3.0 Benchmark (5 seeds, flat LR, identical init weights)", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("benchmark_convergence_v3.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: benchmark_convergence_v3.png")

# %% 
# ============================================================
# 11. SUMMARY TABLE
# ============================================================
print(f"\n{'='*70}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'='*70}\n")

summary_df = pd.DataFrame(stats_rows)
print(summary_df.to_string(index=False))
print()

# Timing comparison
print(f"\n{'='*70}")
print(f"  TIMING (seconds per run)")
print(f"{'='*70}\n")

for scenario in scenarios:
    print(f"  {scenario}:")
    for opt_name in CONFIG["optimizers"]:
        times = timing[scenario][opt_name]
        print(f"    {opt_name:8s}: {np.mean(times):.1f}s +/- {np.std(times):.1f}s")

print(f"\n{'='*70}")
print(f"  Benchmark complete. All results saved to benchmark_results_v3.csv")
print(f"{'='*70}")
