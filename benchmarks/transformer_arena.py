import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import urllib.request
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
from adamv import AdamVCpp

# =============================================================================
# 1. Strict Determinism (Data Expert + Skeptical Critic)
# =============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed locked to {seed}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_deterministic_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# =============================================================================
# 2. Dataset Pipeline (Shakespeare-char)
# =============================================================================
class ShakespeareCharDataset(Dataset):
    def __init__(self, seq_len: int = 256, split='train'):
        self.seq_len = seq_len
        file_path = "tinyshakespeare.txt"
        if not os.path.exists(file_path):
            print("Downloading Tiny Shakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, file_path)
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        
        # 90% train, 10% val
        n = int(0.9 * len(data))
        self.data = data[:n] if split == 'train' else data[n:]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

def get_nlp_dataloaders(batch_size=64, seq_len=256, seed=42):
    train_ds = ShakespeareCharDataset(seq_len=seq_len, split='train')
    val_ds = ShakespeareCharDataset(seq_len=seq_len, split='val')
    gen = get_deterministic_generator(seed)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          num_workers=0, worker_init_fn=seed_worker, generator=gen, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dl, val_dl, train_ds.vocab_size

# =============================================================================
# 3. Model Architecture (nanoGPT - PhD Design)
# =============================================================================
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
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
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_layer=6, n_head=6, block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

# =============================================================================
# 4. Decoupled Weight Decay & Training Loop
# =============================================================================
@torch.no_grad()
def estimate_loss(model, eval_iters, val_dl, device):
    model.eval()
    losses = []
    val_iter = iter(val_dl)
    for _ in range(eval_iters):
        try:
            X, Y = next(val_iter)
        except StopIteration:
            break
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses) if len(losses) > 0 else 0.0

def configure_optimizers(model, weight_decay, learning_rate, device_type, is_adamv=False):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    if is_adamv:
        return AdamVCpp(optim_groups, lr=learning_rate, betas=(0.9, 0.999), enable_omni=False)
    else:
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

def run_transformer_arena():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    block_size = 128
    max_iters = 2000
    eval_interval = 200
    eval_iters = 50
    seeds = [42, 1337, 2026] # Multi-seed rigor
    
    results = {"AdamW": [], "AdamVCpp": []}
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        seed_everything(seed)
        train_dl, val_dl, vocab_size = get_nlp_dataloaders(batch_size, block_size, seed)
        
        for opt_name in ["AdamW", "AdamVCpp"]:
            print(f"Training with {opt_name}...")
            # Re-initialize model to guarantee same starting weights per optimizer
            seed_everything(seed)
            model = NanoGPT(vocab_size, block_size=block_size).to(device)
            
            # Independent Hyperparameters
            # We strictly match 1e-3 for a fair 1-to-1 comparison
            lr = 1e-3 if opt_name == "AdamW" else 1e-3
            wd = 0.1
            optimizer = configure_optimizers(model, wd, lr, device, is_adamv=(opt_name=="AdamVCpp"))
            
            if opt_name == "AdamVCpp":
                for group in optimizer.param_groups:
                    group['total_steps'] = max_iters
            
            # Scheduler ONLY for AdamW
            if opt_name == "AdamW":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_iters, pct_start=0.1)
            
            train_iter = iter(train_dl)
            history = []
            
            for iter_num in range(max_iters):
                if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                    val_loss = estimate_loss(model, eval_iters, val_dl, device)
                    ppl = math.exp(val_loss)
                    print(f"Step {iter_num}: Val Loss {val_loss:.4f}, PPL {ppl:.4f}")
                    history.append((iter_num, val_loss))
                
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dl)
                    xb, yb = next(train_iter)
                
                xb, yb = xb.to(device), yb.to(device)
                
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if opt_name == "AdamW":
                    scheduler.step()
                    
            results[opt_name].append(history)
            
    # Plotting Logic
    plt.figure(figsize=(10, 6))
    for opt_name, histories in results.items():
        steps = [h[0] for h in histories[0]]
        losses = np.array([[h[1] for h in history] for history in histories])
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)
        plt.plot(steps, mean_loss, label=opt_name, marker='o')
        plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        
    plt.title("Transformer Arena (NanoGPT on Shakespeare-char)")
    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("assets/transformer_arena.png")
    print("Transformer Arena Benchmark complete! Saved to assets/transformer_arena.png")

if __name__ == "__main__":
    run_transformer_arena()
