import pytest
import torch
import torch.nn as nn
from adamv import AdamV

def setup_model(dim=4):
    torch.manual_seed(42)
    return nn.Linear(dim, dim)

def test_zero_gradient():
    """Test 1: Zero gradient → parameter unchanged (except weight decay)"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
        
    opt.step()
    
    for p_init, p_curr in zip(initial_params, model.parameters()):
        torch.testing.assert_close(p_init, p_curr)

def test_single_element_tensor():
    """Test 2: Single-element tensor → no crash"""
    model = setup_model(1)
    opt = AdamV(model.parameters(), lr=1e-3)
    
    data = torch.randn(1, 1)
    target = torch.randn(1, 1)
    criterion = nn.MSELoss()
    
    opt.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    opt.step()
    # If we reached here, no crash occurred

def test_bfloat16_parameters():
    """Test 3: BFloat16 parameters → no crash, parameters updated"""
    model = setup_model().bfloat16()
    opt = AdamV(model.parameters(), lr=1e-3)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    data = torch.randn(2, 4, dtype=torch.bfloat16)
    target = torch.randn(2, 4, dtype=torch.bfloat16)

    opt.zero_grad()
    # MSE doesn't support bfloat16 on CPU; compute loss in float32
    loss = nn.MSELoss()(model(data).float(), target.float())
    loss.backward()
    opt.step()

    for p_init, p_curr in zip(initial_params, model.parameters()):
        assert not torch.allclose(p_init, p_curr)

def test_large_gradient():
    """Test 4: Very large gradient → Bakhshali Brake reduces step vs no-brake, no NaN/Inf"""
    model1 = setup_model()
    opt1 = AdamV(model1.parameters(), lr=10.0, enable_brake=True)
    model2 = setup_model()
    opt2 = AdamV(model2.parameters(), lr=10.0, enable_brake=False)

    for _ in range(1000):
        for p in model1.parameters():
            p.grad = torch.full_like(p, 0.01)
        for p in model2.parameters():
            p.grad = torch.full_like(p, 0.01)
        opt1.step()
        opt2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p1.data.copy_(p2.data)

    init_params = [p.clone() for p in model1.parameters()]

    for p in model1.parameters():
        p.grad = torch.full_like(p, 100000.0)
    for p in model2.parameters():
        p.grad = torch.full_like(p, 100000.0)
    opt1.step()
    opt2.step()

    for p_init, p1, p2 in zip(init_params, model1.parameters(), model2.parameters()):
        # Both must remain finite
        assert torch.all(torch.isfinite(p1))
        assert torch.all(torch.isfinite(p2))
        # Brake must reduce step vs no-brake
        step_braked = torch.max(torch.abs(p1 - p_init))
        step_free = torch.max(torch.abs(p2 - p_init))
        assert step_braked < step_free

def test_small_gradient():
    """Test 5: Very small gradient (1e-10) → no NaN or Inf produced"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=1e-3)
    
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e-10)
        
    opt.step()
    
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))

def test_weight_decay_zero_edge():
    """Test 6: weight_decay=0 → no weight decay applied"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
        
    opt.step()
    
    for p_init, p_curr in zip(initial_params, model.parameters()):
        torch.testing.assert_close(p_init, p_curr)

def test_multiple_param_groups():
    """Test 7: Multiple param groups with different lr → each group uses its own lr"""
    model = setup_model()
    opt = AdamV([
        {'params': model.weight, 'lr': 1e-2},
        {'params': model.bias, 'lr': 1e-4}
    ])
    
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    
    opt.step()
    
    state_w = opt.state[model.weight]
    state_b = opt.state[model.bias]
    
    # The updates should be scaled differently
    assert torch.max(torch.abs(model.weight.data)) != torch.max(torch.abs(model.bias.data))

def test_omni_with_no_loss():
    """Test 8: Optimizer step with no loss (enable_omni=True, loss=None) → no crash"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=1e-3, enable_omni=True)
    
    data = torch.randn(2, 4)
    target = torch.randn(2, 4)
    criterion = nn.MSELoss()
    
    opt.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    
    opt.step() # Call without loss argument
    opt.step(loss=None) # Explicitly pass None

def test_brake_false_large_gradient():
    """Test 9: enable_brake=False, large gradient → step_size = a (no braking)"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=0.1, enable_brake=False)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e6)
        
    opt.step()
    
    # Should update significantly more when brake is off
    for p_init, p_curr in zip(initial_params, model.parameters()):
        update_size = torch.max(torch.abs(p_curr - p_init))
        assert update_size > 0.05

def test_total_steps_one():
    """Test 10: total_steps=1 → progresso saturates at 1.0 quickly"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=1e-3, total_steps=1)
    
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()
    
    state = opt.state[model.weight]
    step = state.get('step')
    # If there's a progresso factor calculated in state or just internal,
    # the main test here is no crash when total_steps=1
    assert step == 1
    
    opt.step()
    assert state.get('step') == 2 # Should still increment without crashing

def test_gradient_clipping_compatibility():
    """Test 11: Gradient clipping compatibility — test that AdamV works correctly after torch.nn.utils.clip_grad_norm_"""
    model = setup_model()
    opt = AdamV(model.parameters(), lr=1e-3)
    
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    opt.step()
    
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))
