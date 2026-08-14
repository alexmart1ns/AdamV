import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from torch_omega_optimizer import OmegaOptimizer

# Uma CNN rápida projetada para o dataset CIFAR (3x32x32)
class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(optimizer_name, model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    
    total_steps = len(dataloader) * epochs
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # Schedulers tradicionais
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif optimizer_name == "TrueOmega":
        # Nosso otimizador nativo!
        optimizer = OmegaOptimizer(model.parameters(), lr=lr*1.5, weight_decay=0.01, total_steps=total_steps)
        scheduler = None
    
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # O TrueOmega precisa saber do loss para ativar o OMNI Kick
            if optimizer_name == "TrueOmega":
                optimizer.step(current_loss=loss.item())
            else:
                optimizer.step()
                if scheduler:
                    scheduler.step()
                    
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    return loss_history

def run_synthetic_cifar_arena():
    print("=========================================================")
    print(" ARENA PYTORCH: MEMORIZAÇÃO DE RUÍDO (Teste de Expressividade)")
    print("=========================================================")
    print("Em Deep Learning, um teste fundamental de poder de otimização")
    print("é forçar a rede a memorizar imagens aleatórias com labels aleatórias.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Gerando dataset sintético "CIFAR" (ruído branco)
    N = 2048 # 2048 imagens sintéticas
    X = torch.randn(N, 3, 32, 32)
    y = torch.randint(0, 10, (N,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    epochs = 20
    
    print("\nIniciando AdamW SOTA...")
    model_adam = FastCNN()
    t0 = time.time()
    loss_adam = train_model("AdamW", model_adam, dataloader, epochs=epochs, lr=3e-3, device=device)
    print(f"-> Tempo AdamW: {time.time()-t0:.2f}s")
    
    print("\nIniciando True Omega...")
    model_omega = FastCNN()
    t0 = time.time()
    loss_omega = train_model("TrueOmega", model_omega, dataloader, epochs=epochs, lr=3e-3, device=device)
    print(f"-> Tempo True Omega: {time.time()-t0:.2f}s")
    
    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    plt.plot(range(1, epochs+1), loss_adam, label='AdamW (Cosine)', color='#f38ba8', lw=2, linestyle='--')
    plt.plot(range(1, epochs+1), loss_omega, label='True Omega (Nat PyTorch)', color='#a6e3a1', lw=3)
    
    plt.title('Capacidade de Memorização de Caos (CNN ResNet-Style)', color='#cdd6f4', fontsize=14)
    plt.xlabel('Épocas', color='#cdd6f4')
    plt.ylabel('CrossEntropy Loss', color='#cdd6f4')
    plt.tick_params(colors='#cdd6f4')
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pytorch_cnn_arena.png', dpi=200, facecolor='#1e1e2e')
    print("\nGráfico gerado: 'pytorch_cnn_arena.png'")

if __name__ == "__main__":
    run_synthetic_cifar_arena()
