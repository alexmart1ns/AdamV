import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification
from torch_omega_optimizer import OmegaOptimizer
from torch_omega_cpp import OmegaCppOptimizer

# Uma Rede Neural Profunda (MLP) para dados tabulares complexos
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepMLP, self).__init__()
        # Arquitetura profunda para forçar o Otimizador a trabalhar os gradientes através das camadas
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_duel(optimizer_name, model, dataloader, epochs=15, lr=1e-3):
    device = torch.device("cpu")
    model.to(device)
    total_steps = len(dataloader) * epochs
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif optimizer_name == "TrueOmega":
        optimizer = OmegaOptimizer(model.parameters(), lr=lr*1.2, weight_decay=0.01, total_steps=total_steps)
        scheduler = None
    elif optimizer_name == "TrueOmegaCpp":
        optimizer = OmegaCppOptimizer(model.parameters(), lr=lr*1.2, weight_decay=0.01, total_steps=total_steps)
        scheduler = None
        
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            if optimizer_name == "TrueOmega":
                optimizer.step(current_loss=loss.item())
            else:
                optimizer.step()
                if scheduler:
                    scheduler.step()
                    
            epoch_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = epoch_loss / len(dataloader)
        acc = 100 * correct / total
        loss_history.append(avg_loss)
        print(f"[{optimizer_name}] Época {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acurácia: {acc:.1f}%")
        
    return loss_history

def run_real_duel():
    print("==================================================================")
    print(" O DUELO REAL: ADAMW vs TRUE OMEGA EM DADOS ESTRUTURADOS COMPLEXOS")
    print("==================================================================")
    print("Gerando um dataset tabular não-linear de alta dificuldade...")
    
    # Dataset real, com sinal (padrões), mas muito difícil de otimizar
    X, y = make_classification(n_samples=15000, n_features=200, n_informative=40, 
                               n_redundant=10, n_classes=10, random_state=42)
                               
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    epochs = 20
    lr_base = 3e-3
    
    print("\n--- TREINANDO ADAMW (SOTA) ---")
    model_adam = DeepMLP(input_dim=200, num_classes=10)
    t0 = time.time()
    loss_adam = train_duel("AdamW", model_adam, dataloader, epochs=epochs, lr=lr_base)
    print(f"-> Tempo total: {time.time()-t0:.2f}s")
    
    print("\n--- TREINANDO TRUE OMEGA (PYTHON) ---")
    model_omega = DeepMLP(input_dim=200, num_classes=10)
    t0 = time.time()
    loss_omega = train_duel("TrueOmega", model_omega, dataloader, epochs=epochs, lr=lr_base)
    print(f"-> Tempo total: {time.time()-t0:.2f}s")
    
    print("\n--- TREINANDO TRUE OMEGA (C++ FUSED KERNEL) ---")
    model_omega_cpp = DeepMLP(input_dim=200, num_classes=10)
    t0 = time.time()
    loss_omega_cpp = train_duel("TrueOmegaCpp", model_omega_cpp, dataloader, epochs=epochs, lr=lr_base)
    print(f"-> Tempo total: {time.time()-t0:.2f}s")
    
    # Visualização
    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    plt.plot(range(1, epochs+1), loss_adam, label='AdamW (C++)', color='#f38ba8', lw=2, linestyle='--')
    plt.plot(range(1, epochs+1), loss_omega, label='True Omega (Python)', color='#a6e3a1', lw=2)
    plt.plot(range(1, epochs+1), loss_omega_cpp, label='True Omega (C++ Fused)', color='#89b4fa', lw=3)
    
    plt.title('O Duelo Real: Treinamento de MLP Profunda (Dataset Complexo)', color='#cdd6f4', fontsize=14)
    plt.xlabel('Épocas de Treinamento', color='#cdd6f4')
    plt.ylabel('Erro (CrossEntropy Loss)', color='#cdd6f4')
    plt.tick_params(colors='#cdd6f4')
    plt.grid(color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    plt.tight_layout()
    plt.savefig('real_duel_arena.png', dpi=200, facecolor='#1e1e2e')
    print("\nGráfico do duelo gerado: 'real_duel_arena.png'")

if __name__ == "__main__":
    run_real_duel()
