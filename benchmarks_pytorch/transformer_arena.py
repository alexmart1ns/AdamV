import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adamv import AdamVCpp

# ==========================================
# 1. ARQUITETURA MINI-GPT (Decoder-Only)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, block_size=256):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, block_size=256):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(dropout),
        ))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        m = self.mlp.c_fc(self.ln_2(x))
        m = self.mlp.act(m)
        m = self.mlp.c_proj(m)
        x = x + self.mlp.dropout(m)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=65, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# ==========================================
# 2. TREINAMENTO
# ==========================================
def generate_synthetic_batch(batch_size, block_size, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    return x, y

def configure_optimizers(model, weight_decay, learning_rate, betas, optimizer_type, total_steps):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    decay = {pn for pn in decay if pn in param_dict}
    no_decay = {pn for pn in no_decay if pn in param_dict}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    elif optimizer_type == "AdamVCpp":
        return AdamVCpp(optim_groups, lr=learning_rate, betas=betas, total_steps=total_steps, bakhshali_threshold=2.0)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def train_transformer(optimizer_name, seed=42, iterations=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(vocab_size=65, block_size=256).to(device)
    
    lr = 5e-4
    betas = (0.9, 0.95)
    weight_decay = 0.1
    
    optimizer = configure_optimizers(model, weight_decay, lr, betas, optimizer_name, iterations)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    loss_history = []
    
    model.train()
    for it in range(iterations):
        x, y = generate_synthetic_batch(16, 256, 65, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if optimizer_name == "AdamVCpp":
            optimizer.step(current_loss=loss.item())
        else:
            optimizer.step()
        
        scheduler.step()
        
        if it % 10 == 0:
            loss_history.append(loss.item())
            print(f"[{optimizer_name} - Seed {seed}] Iter {it}/{iterations} Loss: {loss.item():.4f}", flush=True)
            
    return loss_history

def run_transformer_arena():
    print("=========================================================")
    print(" TRANSFORMER ARENA: MINI-GPT")
    print("=========================================================")
    seeds = [42, 1024, 2024]
    optimizers = ["AdamW", "AdamVCpp"]
    colors = {'AdamW': '#f38ba8', 'AdamVCpp': '#89b4fa'}
    results = {}
    
    for opt in optimizers:
        print(f"Running {opt}...")
        all_hists = []
        for s in seeds:
            hist = train_transformer(opt, seed=s, iterations=100)
            all_hists.append(hist)
        results[opt] = {
            "mean": np.mean(all_hists, axis=0),
            "std": np.std(all_hists, axis=0)
        }
        
    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    x_axis = np.arange(0, 100, 10)
    for opt in optimizers:
        mean = results[opt]["mean"]
        std = results[opt]["std"]
        ax.plot(x_axis, mean, label=opt, color=colors[opt], lw=2)
        ax.fill_between(x_axis, mean - std, mean + std, color=colors[opt], alpha=0.2)
        
    plt.title('Mini-GPT Optimization Stability', color='#cdd6f4', fontsize=14)
    plt.xlabel('Iterations', color='#cdd6f4')
    plt.ylabel('CrossEntropy Loss', color='#cdd6f4')
    plt.tick_params(colors='#cdd6f4')
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    plt.grid(color='#313244', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('transformer_arena.png', dpi=200, facecolor='#1e1e2e')
    print("Done! Saved as transformer_arena.png")

if __name__ == '__main__':
    run_transformer_arena()
