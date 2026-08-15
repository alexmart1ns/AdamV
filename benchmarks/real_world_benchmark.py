import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import sys
import os

# Adiciona o diretório pai para podermos importar o torch_adamv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adamv import AdamVCpp

# ==========================================
# ARQUITETURA FAST CNN (Para rodar rápido em CPU)
# ==========================================
class FastCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FastCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # FashionMNIST é 28x28 -> pool -> 14x14 -> pool -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size=256):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    return trainloader, testloader

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    return acc

def train_sota_model(optimizer_name, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n--- {optimizer_name} | Hardware: {device} ---", flush=True)
    
    trainloader, testloader = get_data_loaders(batch_size=256)
    model = FastCNN(num_classes=10).to(device)
    
    total_steps = len(trainloader) * epochs
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        scheduler = None
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif optimizer_name == "AdamVCpp":
        optimizer = AdamVCpp(model.parameters(), lr=5e-3, weight_decay=1e-2, total_steps=total_steps, bakhshali_threshold=3.0)
        scheduler = None
        
    history_acc = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if optimizer_name == "AdamVCpp":
                optimizer.step(current_loss=loss.item())
            else:
                optimizer.step()
                
            train_loss += loss.item()
            
        if scheduler:
            scheduler.step()
            
        test_acc = evaluate(model, testloader, device)
        history_acc.append(test_acc)
        
        print(f"[{optimizer_name}] Epoca {epoch+1:02d}/{epochs} | Acc: {test_acc:.2f}%", flush=True)
        
    total_time = time.time() - start_time
    print(f"-> {optimizer_name} Finalizado! Tempo: {total_time:.2f}s | Acuracia Final: {history_acc[-1]:.2f}%", flush=True)
    return history_acc, total_time

def run_sota_arena():
    epochs = 10
    print("Iniciando Arena de Otimizadores SOTA (Fast CNN)...", flush=True)
    
    acc_adam, t_adam = train_sota_model("Adam", epochs=epochs)
    acc_adamw, t_adamw = train_sota_model("AdamW", epochs=epochs)
    acc_adamv, t_adamv = train_sota_model("AdamVCpp", epochs=epochs)
    
    # Plot SOTA
    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    plt.plot(range(1, epochs+1), acc_adam, label=f'Adam ({t_adam:.0f}s)', color='#a6e3a1', lw=2)
    plt.plot(range(1, epochs+1), acc_adamw, label=f'AdamW + Cosine ({t_adamw:.0f}s)', color='#f38ba8', lw=2, linestyle='--')
    plt.plot(range(1, epochs+1), acc_adamv, label=f'AdamV C++ Fused (Puro com Endogenous WD) ({t_adamv:.0f}s)', color='#89b4fa', lw=3)
    
    plt.title('Real-World Arena: CNN em FashionMNIST', color='#cdd6f4', fontsize=14)
    plt.xlabel('Épocas', color='#cdd6f4')
    plt.ylabel('Acurácia no Teste (%)', color='#cdd6f4')
    plt.tick_params(colors='#cdd6f4')
    plt.grid(color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    plt.tight_layout()
    plt.savefig('sota_benchmark_ultimate.png', dpi=200, facecolor='#1e1e2e')
    print("\nGrafico final SOTA gerado: 'sota_benchmark_ultimate.png'", flush=True)

if __name__ == "__main__":
    run_sota_arena()
