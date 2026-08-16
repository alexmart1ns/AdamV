import os
import gc
import math
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind
from adamv import AdamV, AdamVCpp

# =========================================================
# 2. SEEDING & UTILS
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, opt_name, scenario, lr=1e-3, wd=0.0):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    if opt_name == "AdamW":
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        
    elif opt_name == "AdamV":
        if scenario == "Vision" or scenario == "NLP":
            return AdamVCpp(optim_groups, lr=lr, betas=(0.9, 0.999), 
                            bakhshali_threshold=50.0, enable_brake=True, 
                            enable_omni=False)
        elif scenario == "Generative":
            return AdamVCpp(optim_groups, lr=lr, betas=(0.9, 0.999), 
                            bakhshali_threshold=1000.0, enable_brake=False, 
                            enable_omni=False)

# =========================================================
# 3. ARCHITECTURES
# =========================================================
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        import torchvision.models as models
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# NanoGPT
batch_size_nlp = 64
block_size = 128
max_iters_nlp = 1000
eval_interval_nlp = 200
eval_iters_nlp = 50
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

class Head(nn.Module):
    def __init__(self, head_size):
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
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

def get_batch_nlp(split, train_data, val_data, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size_nlp,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss_nlp(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters_nlp)
        for k in range(eval_iters_nlp):
            X, Y = get_batch_nlp(split, train_data, val_data, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =========================================================
# 4. DATA DOWNLOADERS
# =========================================================
def get_dataloaders(scenario, seed):
    seed_everything(seed)
    if scenario == "Vision":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        return trainloader, testloader, None
    elif scenario == "Generative":
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        return trainloader, testloader, None
    elif scenario == "NLP":
        if not os.path.exists('input.txt'):
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, 'input.txt')
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s]
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data, vocab_size

# =========================================================
# 5. EXECUTION ENGINE
# =========================================================
def run_global_stress_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 5 Seeds recommended by the Benchmark Specialist
    seeds = [42, 1337, 2024, 3141, 8888]
    scenarios = ["Vision", "Generative", "NLP"]
    optimizers = ["AdamW", "AdamV"]
    
    all_results = []
    final_metrics = {"Vision": {}, "Generative": {}, "NLP": {}}
    
    for seed in seeds:
        print(f"\n{'='*50}\nGLOBAL SEED: {seed}\n{'='*50}")
        
        for scenario in scenarios:
            print(f"\n--- Running Scenario: {scenario} ---")
            
            # Setup Data
            dl1, dl2, vocab_size = get_dataloaders(scenario, seed)
            
            # Create base weights for fair comparison within this seed
            seed_everything(seed)
            if scenario == "Vision":
                base_model = ResNet18(num_classes=10)
            elif scenario == "Generative":
                base_model = VAE()
            elif scenario == "NLP":
                base_model = NanoGPT(vocab_size)
            
            torch.save(base_model.state_dict(), f"base_weights_{scenario}.pt")
            del base_model
            gc.collect()
            
            for opt_name in optimizers:
                print(f"  > Optimizer: {opt_name}")
                seed_everything(seed)
                
                # Load identical weights
                if scenario == "Vision":
                    model = ResNet18(num_classes=10).to(device)
                    total_steps = len(dl1) * 4 # 4 epochs
                elif scenario == "Generative":
                    model = VAE().to(device)
                    total_steps = len(dl1) * 8 # 8 epochs
                elif scenario == "NLP":
                    model = NanoGPT(vocab_size).to(device)
                    total_steps = max_iters_nlp
                    
                model.load_state_dict(torch.load(f"base_weights_{scenario}.pt"))
                
                # Configure LR and WD
                lr = 1e-3
                wd = 0.1 if scenario == "NLP" else (0.1 if scenario == "Vision" else 0.0)
                optimizer = get_optimizer(model, opt_name, scenario, lr=lr, wd=wd)
                
                if opt_name == "AdamV":
                    for group in optimizer.param_groups:
                        group['total_steps'] = total_steps
                scheduler = None

                # Training Loops
                if scenario == "Vision":
                    step = 0
                    for epoch in range(4):
                        model.train()
                        for inputs, targets in dl1:
                            inputs, targets = inputs.to(device), targets.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            outputs = model(inputs)
                            loss = F.cross_entropy(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            if scheduler: scheduler.step()
                            step += 1
                        
                        # Eval
                        model.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for inputs, targets in dl2:
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()
                        val_acc = 100. * correct / total
                        all_results.append({"Seed": seed, "Scenario": scenario, "Optimizer": opt_name, "Epoch": epoch+1, "Metric": val_acc})
                        print(f"      Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")
                        if epoch == 3:
                            if opt_name not in final_metrics[scenario]: final_metrics[scenario][opt_name] = []
                            final_metrics[scenario][opt_name].append(val_acc)
                        
                elif scenario == "Generative":
                    step = 0
                    for epoch in range(8):
                        model.train()
                        for inputs, _ in dl1:
                            inputs = inputs.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            recon_batch, mu, logvar = model(inputs)
                            loss = vae_loss_function(recon_batch, inputs, mu, logvar)
                            loss.backward()
                            optimizer.step()
                            if scheduler: scheduler.step()
                            step += 1
                        
                        # Eval
                        model.eval()
                        test_loss = 0
                        with torch.no_grad():
                            for inputs, _ in dl2:
                                inputs = inputs.to(device)
                                recon_batch, mu, logvar = model(inputs)
                                test_loss += vae_loss_function(recon_batch, inputs, mu, logvar).item()
                        val_loss = test_loss / len(dl2.dataset)
                        all_results.append({"Seed": seed, "Scenario": scenario, "Optimizer": opt_name, "Epoch": epoch+1, "Metric": val_loss})
                        print(f"      Epoch {epoch+1} | Val Loss (ELBO): {val_loss:.4f}")
                        if epoch == 7:
                            if opt_name not in final_metrics[scenario]: final_metrics[scenario][opt_name] = []
                            final_metrics[scenario][opt_name].append(val_loss)
                        
                elif scenario == "NLP":
                    for step in range(max_iters_nlp + 1):
                        if step % 200 == 0 or step == max_iters_nlp:
                            losses = estimate_loss_nlp(model, dl1, dl2, device)
                            all_results.append({"Seed": seed, "Scenario": scenario, "Optimizer": opt_name, "Epoch": step, "Metric": losses['val']})
                            print(f"      Step {step:04d} | Val Loss: {losses['val']:.4f}")
                            if step == max_iters_nlp:
                                if opt_name not in final_metrics[scenario]: final_metrics[scenario][opt_name] = []
                                final_metrics[scenario][opt_name].append(losses['val'])

                        if step < max_iters_nlp:
                            xb, yb = get_batch_nlp('train', dl1, dl2, device)
                            logits, loss = model(xb, yb)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            if scheduler: scheduler.step()

                # CRITICAL MEMORY CLEANUP
                del model, optimizer, scheduler
                gc.collect()
                torch.cuda.empty_cache()
            
            # Clean up dataloaders
            del dl1, dl2
            gc.collect()
            torch.cuda.empty_cache()

    # =========================================================
    # 6. AGGREGATION, STATISTICS & PLOTTING (MIN-MAX SHADING)
    # =========================================================
    print("\n=== STATISTICAL SIGNIFICANCE (p-values) ===")
    for scenario in scenarios:
        adamw_vals = final_metrics[scenario]["AdamW"]
        adamv_vals = final_metrics[scenario]["AdamV"]
        t_stat, p_val = ttest_ind(adamw_vals, adamv_vals, equal_var=False)
        print(f"[{scenario}] AdamW Mean: {np.mean(adamw_vals):.4f} | AdamV Mean: {np.mean(adamv_vals):.4f}")
        print(f"[{scenario}] Welch's t-test p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("  -> Result is STATISTICALLY SIGNIFICANT!")
        else:
            print("  -> Result is NOT statistically significant (overlap in distributions).")

    df = pd.DataFrame(all_results)
    df.to_csv("global_stress_results.csv", index=False)
    print("\nSaved global_stress_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Vision (ResNet-18) [Val Acc %]", "Generative (VAE) [Val ELBO Loss]", "NLP (NanoGPT) [Val CrossEntropy]"]
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        scenario_df = df[df["Scenario"] == scenario]
        
        for opt, color in [("AdamW", "#1f77b4"), ("AdamV", "#ff7f0e")]:
            # Adjust legend to reflect Fair Fight Protocol
            label = f"{opt} (Flat LR)"
            opt_df = scenario_df[scenario_df["Optimizer"] == opt]
            if opt_df.empty: continue
            
            # Group by Epoch for Min-Max Shading
            agg_df = opt_df.groupby("Epoch")["Metric"].agg(["mean", "min", "max"]).reset_index()
            
            ax.plot(agg_df["Epoch"], agg_df["mean"], label=label, color=color, marker='o')
            ax.fill_between(agg_df["Epoch"], agg_df["min"], agg_df["max"], color=color, alpha=0.2)
            
        ax.set_title(titles[i])
        ax.set_xlabel("Epoch / Step")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Invert y-axis for Loss metrics so "Better" is always visually upwards or clearly marked
        if scenario != "Vision":
            ax.invert_yaxis()
            
    plt.tight_layout()
    plt.savefig("global_stress_plot.png", dpi=300)
    print("Saved global_stress_plot.png")

if __name__ == "__main__":
    run_global_stress_test()
