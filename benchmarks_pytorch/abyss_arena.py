import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification
from torch_omega_optimizer import OmegaOptimizer
from torch_omega_cpp import OmegaCppOptimizer
import numpy as np

# A Arena do Abismo: 15 camadas profundas de pura dor de cabeça para o gradiente
class AbyssMLP(nn.Module):
    def __init__(self, input_dim, num_classes, depth=15):
        super(AbyssMLP, self).__init__()
        layers = []
        
        # Primeira camada
        layers.append(nn.Linear(input_dim, 128))
        layers.append(nn.GELU())
        
        # Camadas Ocultas (Profundidade extrema para causar Vanishing/Exploding Gradients)
        for _ in range(depth - 2):
            layers.append(nn.Linear(128, 128))
            # Sem BatchNorm, Sem Skip Connections (Residuals). Pura dificuldade bruta.
            layers.append(nn.GELU())
            
        # Saída
        layers.append(nn.Linear(128, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_abyss(optimizer_name, model, dataloader, epochs=30, lr=1e-3):
    device = torch.device("cpu")
    model.to(device)
    total_steps = len(dataloader) * epochs
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif optimizer_name == "TrueOmegaCpp":
        # Usamos o Kernel C++ que criamos!
        # Ajustamos o limite do Bakhshali (bakhshali_threshold) para frear as explosões violentas da rede profunda
        optimizer = OmegaCppOptimizer(model.parameters(), lr=lr*1.5, weight_decay=0.01, 
                                      total_steps=total_steps, bakhshali_threshold=2.0)
        scheduler = None
        
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            if optimizer_name == "TrueOmegaCpp":
                optimizer.step(current_loss=loss.item())
            else:
                optimizer.step()
                if scheduler:
                    scheduler.step()
                    
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[{optimizer_name}] Época {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")
        
    return loss_history

def run_abyss_arena():
    print("==================================================================")
    print(" A ARENA DO ABISMO: REDE ULTRAPROFUNDA + DADOS CORROMPIDOS")
    print("==================================================================")
    print("Testando a sobrevivência em 15 camadas sem Skip-Connections")
    print("Dataset: 20% de Ruído de Label (Fake News) + Alta Esparsidade")
    
    # Dataset corrompido
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=20, 
                               n_redundant=80, n_classes=2, flip_y=0.2, random_state=42)
                               
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    epochs = 30
    lr_base = 1e-3
    
    print("\n--- DESCENDO O ABISMO: ADAMW (SOTA) ---")
    model_adam = AbyssMLP(input_dim=100, num_classes=2, depth=15)
    t0 = time.time()
    loss_adam = train_abyss("AdamW", model_adam, dataloader, epochs=epochs, lr=lr_base)
    print(f"-> Tempo total: {time.time()-t0:.2f}s")
    
    print("\n--- DESCENDO O ABISMO: TRUE OMEGA (C++ FUSED) ---")
    model_omega = AbyssMLP(input_dim=100, num_classes=2, depth=15)
    t0 = time.time()
    loss_omega = train_abyss("TrueOmegaCpp", model_omega, dataloader, epochs=epochs, lr=lr_base)
    print(f"-> Tempo total: {time.time()-t0:.2f}s")
    
    # Visualização
    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    plt.plot(range(1, epochs+1), loss_adam, label='AdamW (SOTA)', color='#f38ba8', lw=2, linestyle='--')
    plt.plot(range(1, epochs+1), loss_omega, label='True Omega (C++ Fused)', color='#89b4fa', lw=3)
    
    plt.title('A Arena do Abismo: Sobrevivência em Topologias Extremas', color='#cdd6f4', fontsize=14)
    plt.xlabel('Épocas', color='#cdd6f4')
    plt.ylabel('Erro de Treinamento (BCE)', color='#cdd6f4')
    plt.tick_params(colors='#cdd6f4')
    plt.grid(color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    plt.tight_layout()
    plt.savefig('abyss_arena.png', dpi=200, facecolor='#1e1e2e')
    print("\nGráfico do abismo gerado: 'abyss_arena.png'")

if __name__ == "__main__":
    run_abyss_arena()
