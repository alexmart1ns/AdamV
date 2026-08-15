import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from adamv import AdamVCpp

# =============================================================================
# 1. Strict Determinism
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_deterministic_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# =============================================================================
# 2. Vision Pipeline (CIFAR-100)
# =============================================================================
def get_vision_dataloaders(batch_size=128, seed=42):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], 
                             std=[0.2673, 0.2564, 0.2762])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], 
                             std=[0.2673, 0.2564, 0.2762])
    ])
    
    # Using CIFAR-100 to push complexity further than CIFAR-10
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    gen = get_deterministic_generator(seed)
    
    train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, worker_init_fn=seed_worker, generator=gen, drop_last=True)
    val_dl = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl

# =============================================================================
# 3. Model Architecture (Modified ResNet-18 for 32x32)
# =============================================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Modified Stem for 32x32 inputs (CIFAR)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# =============================================================================
# 4. Decoupled Weight Decay & Training Loop
# =============================================================================
def configure_optimizers(model, weight_decay, learning_rate, is_adamv=False):
    # Decouple weight decay: disable for BatchNorm and biases
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    if is_adamv:
        return AdamVCpp(optim_groups, lr=learning_rate, betas=(0.9, 0.999), enable_omni=True)
    else:
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()
    correct_1 = 0
    total = 0
    losses = []
    
    for inputs, targets in val_dl:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        losses.append(loss.item())
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct_1 += predicted.eq(targets).sum().item()
        
    model.train()
    acc = 100.0 * correct_1 / total
    return sum(losses)/len(losses), acc

def run_vision_arena():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    seeds = [42, 1337] # Multi-seed rigor
    
    results = {"AdamW": [], "AdamVCpp": []}
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        seed_everything(seed)
        train_dl, val_dl = get_vision_dataloaders(batch_size=128, seed=seed)
        
        for opt_name in ["AdamW", "AdamVCpp"]:
            print(f"Training with {opt_name}...")
            seed_everything(seed)
            model = ResNet18().to(device)
            
            lr = 1e-3 if opt_name == "AdamW" else 3e-3
            wd = 0.05
            optimizer = configure_optimizers(model, wd, lr, is_adamv=(opt_name=="AdamVCpp"))
            
            # Scheduler for both optimizers
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs, pct_start=0.1)
                
            history_acc = []
            
            for epoch in range(epochs):
                model.train()
                for batch_idx, (inputs, targets) in enumerate(train_dl):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets, label_smoothing=0.1)
                    loss.backward()
                    
                    optimizer.step()
                    scheduler.step()
                        
                val_loss, val_acc = evaluate(model, val_dl, device)
                print(f"Epoch {epoch}: Val Acc {val_acc:.2f}%")
                history_acc.append((epoch, val_acc))
                
            results[opt_name].append(history_acc)
            
    # Plotting Logic
    plt.figure(figsize=(10, 6))
    for opt_name, histories in results.items():
        epochs_arr = [h[0] for h in histories[0]]
        accs = np.array([[h[1] for h in history] for history in histories])
        mean_acc = accs.mean(axis=0)
        std_acc = accs.std(axis=0)
        plt.plot(epochs_arr, mean_acc, label=opt_name, marker='o')
        plt.fill_between(epochs_arr, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
        
    plt.title("Vision Arena (ResNet-18 on CIFAR-100)")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("assets/vision_arena.png")
    print("Vision Arena Benchmark complete! Saved to assets/vision_arena.png")

if __name__ == "__main__":
    run_vision_arena()
