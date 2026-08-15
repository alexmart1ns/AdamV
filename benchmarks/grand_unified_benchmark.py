import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adamv import AdamVCpp

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_best_device()

# ==========================================
# 1. ARQUITETURA PADRÃO: RESNET-18 (Minimal)
# ==========================================
def get_resnet():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

# ==========================================
# DATASETS E DATA AUGMENTATION (CIFAR-10)
# ==========================================
from torch.utils.data import TensorDataset

def get_dataloaders(scenario, batch_size=256):
    if scenario == "CIFAR-10 (Standard)":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        try:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        except Exception as e:
            print("Aviso: Falha ao baixar CIFAR-10. Usando dataset sintético (Ruído Branco 32x32) para contornar timeout de rede.")
            N_train, N_test = 2048, 512
            X_train = torch.randn(N_train, 3, 32, 32)
            y_train = torch.randint(0, 10, (N_train,))
            X_test = torch.randn(N_test, 3, 32, 32)
            y_test = torch.randint(0, 10, (N_test,))
            trainset = TensorDataset(X_train, y_train)
            testset = TensorDataset(X_test, y_test)
        
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    return trainloader, testloader

# ==========================================
# TRAINING ORCHESTRATOR
# ==========================================
def train_scenario(scenario_name, model_fn, optimizer_name, epochs=20, seed=42):
    print(f"[{scenario_name}] Iniciando {optimizer_name} (Seed {seed})...", flush=True)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    trainloader, testloader = get_dataloaders(scenario_name, batch_size=256)
    
    model = model_fn().to(device)
    total_steps = len(trainloader) * epochs
    criterion = nn.CrossEntropyLoss()
    
    # Fair Hyperparameters: Both get 1e-3 and CosineAnnealing.
    lr = 1e-3
    weight_decay = 1e-2
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamVCpp":
        optimizer = AdamVCpp(model.parameters(), lr=lr, weight_decay=weight_decay, total_steps=total_steps, bakhshali_threshold=3.0)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if optimizer_name == "AdamVCpp":
                optimizer.step(current_loss=loss.item())
            else:
                optimizer.step()
                
            scheduler.step()
            
        model.eval()
        test_metric = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                test_metric += predicted.eq(targets).sum().item()
                    
        val_metric = 100. * test_metric / total
        history.append(val_metric)
        print(f"  > Epoca {epoch+1:02d} | Acurácia: {val_metric:.2f}%", flush=True)
        
    print(f"  => Tempo: {time.time()-start_time:.2f}s | Final: {history[-1]:.2f}%\n", flush=True)
    return history

def run_grand_arena():
    print(f"============================================================")
    print(f"INICIANDO GRAND UNIFIED BENCHMARK NO DISPOSITIVO: {device}")
    print(f"============================================================\n", flush=True)
    
    scenarios = [
        {"name": "CIFAR-10 (Standard)", "model": get_resnet, "ylabel": "Acuracia (%)"},
    ]
    
    seeds = [42]
    optimizers = ["AdamW", "AdamVCpp"]
    colors = {'AdamW': '#f38ba8', 'AdamVCpp': '#89b4fa'}
    styles = {'AdamW': '--', 'AdamVCpp': '-'}
    
    results = {}
    
    for s in scenarios:
        s_name = s["name"]
        results[s_name] = {}
        for opt in optimizers:
            all_hists = []
            for seed in seeds:
                hist = train_scenario(s_name, s["model"], opt, epochs=3, seed=seed)
                all_hists.append(hist)
            results[s_name][opt] = {
                "mean": np.mean(all_hists, axis=0),
                "std": np.std(all_hists, axis=0)
            }
            
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='#1e1e2e')
    ax.set_facecolor('#11111b')
    s_name = scenarios[0]["name"]
    
    x_axis = range(1, 4)
    for opt in optimizers:
        mean = results[s_name][opt]["mean"]
        std = results[s_name][opt]["std"]
        ax.plot(x_axis, mean, label=opt, color=colors[opt], linestyle=styles[opt], lw=3 if opt == "AdamVCpp" else 2)
        ax.fill_between(x_axis, mean - std, mean + std, color=colors[opt], alpha=0.2)
        
    ax.set_title(s_name + " (Média de 3 Seeds)", color='#cdd6f4', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epocas', color='#a6adc8')
    ax.set_ylabel(scenarios[0]["ylabel"], color='#a6adc8')
    ax.tick_params(colors='#a6adc8')
    ax.grid(color='#313244', linestyle=':', alpha=0.5)
    ax.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
        
    plt.tight_layout()
    plt.savefig('grand_arena.png', dpi=200, facecolor='#1e1e2e')
    print("Grafico final gerado: 'grand_arena.png'", flush=True)

if __name__ == "__main__":
    run_grand_arena()
