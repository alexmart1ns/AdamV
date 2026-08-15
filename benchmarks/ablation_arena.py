import torch
import torch.nn as nn
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from adamv.torch_adamv import AdamV

# Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Small Transformer for fast testing
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        return self.fc_out(output)

def train_ablation(variant_name, enable_ignition, enable_cooling, enable_brake, steps=500):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer().to(device)
    
    # We enforce C++ = False so it uses the pure python code we patched
    optimizer = AdamV(model.parameters(), lr=1e-3, total_steps=steps, 
                      enable_ignition=enable_ignition, 
                      enable_cooling=enable_cooling, 
                      enable_brake=enable_brake)
    optimizer.adamv_cpp = None
    optimizer.adamv_cuda = None
    
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    # Dummy data
    vocab_size = 65
    batch_size = 32
    seq_len = 128
    
    print(f"Training variant: {variant_name}")
    for step in range(steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
    return losses

if __name__ == "__main__":
    variants = {
        "All Enabled (Baseline 3.0)": (True, True, True),
        "No Ignition (No Warmup)": (False, True, True),
        "No Cooling (No Cosine/Envelope)": (True, False, True),
        "No Brake (No Bakhshali Quartic)": (True, True, False),
    }
    
    results = {}
    for name, (ig, cool, brake) in variants.items():
        losses = train_ablation(name, ig, cool, brake)
        results[name] = losses
        
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        # Smooth the loss for better visualization
        smoothed = []
        alpha = 0.1
        curr = losses[0]
        for l in losses:
            curr = curr * (1 - alpha) + l * alpha
            smoothed.append(curr)
        plt.plot(smoothed, label=name)
        
    plt.title("AdamV 3.1 Ablation Study (Shakespeare-char dummy)")
    plt.xlabel("Steps")
    plt.ylabel("Training Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ablation_arena.png", dpi=300)
    print("Saved plot to ablation_arena.png")
