import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import sys

# Ensure AdamV can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from adamv import AdamVCpp
except ImportError:
    print("Warning: Could not import AdamVCpp. Ensure you are running from the correct directory.")

class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x):
        return x * self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
    )

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            nn.Linear(128, num_classes, bias=False),
            Mul(0.2)
        )
        
    def forward(self, x):
        return self.model(x)

def run_cifar10_arena():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running CIFAR-10 Arena on {device}")
    
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)
    
    optimizers_to_run = ['AdamW', 'AdamV']
    results = {}
    epochs = 15
    
    import gc
    
    for opt_name in optimizers_to_run:
        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(42)
        model = ResNet9().to(device)
        
        total_steps = epochs * len(trainloader)
        
        if opt_name == 'AdamW':
            opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        else:
            opt = AdamVCpp(model.parameters(), lr=0.01, weight_decay=1e-4, total_steps=total_steps, bakhshali_threshold=10.0, enable_omni=False)
            scheduler = None
            
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        test_accs = []
        
        print(f"\n[{opt_name}] Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                opt.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                if opt_name == 'AdamW':
                    opt.step()
                    scheduler.step()
                else:
                    opt.step(current_loss=loss.item())
                    
                epoch_loss += loss.item()
                
            avg_train_loss = epoch_loss / len(trainloader)
            train_losses.append(avg_train_loss)
            
            # Validation
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
            test_accs.append(acc)
            print(f"[{opt_name}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Acc: {acc:.2f}%")
            
        total_time = time.time() - start_time
        print(f"[{opt_name}] Finished in {total_time:.2f}s | Final Acc: {test_accs[-1]:.2f}%")
        results[opt_name] = {'loss': train_losses, 'acc': test_accs}
        
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, data in results.items():
        ax1.plot(data['loss'], label=name, marker='o')
        ax2.plot(data['acc'], label=name, marker='o')
        
    ax1.set_title('CIFAR-10 Train Loss (ResNet-9)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('CIFAR-10 Test Accuracy (ResNet-9)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cifar10_arena_results.png', dpi=200)
    print("\nResults saved to cifar10_arena_results.png")

if __name__ == '__main__':
    run_cifar10_arena()
