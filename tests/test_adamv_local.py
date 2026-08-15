import torch
import torch.nn as nn
from adamv import AdamVCpp

def test_adamv():
    print("Testing AdamVCpp...")
    model = nn.Linear(10, 10).cuda()
    opt = AdamVCpp(model.parameters(), lr=0.1)
    
    # Store initial weights
    w_initial = model.weight.clone()
    
    # Forward & backward
    x = torch.randn(5, 10).cuda()
    y = torch.randn(5, 10).cuda()
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    
    # Step
    opt.step(current_loss=loss.item())
    
    # Check if updated
    w_updated = model.weight.clone()
    diff = torch.norm(w_initial - w_updated).item()
    print(f"Weight difference after step: {diff}")
    
    if diff == 0:
        print("FAIL: Weights were not updated!")
    else:
        print("PASS: Weights were updated.")

if __name__ == "__main__":
    test_adamv()
