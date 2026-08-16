import pytest
import torch
import torch.nn as nn
import copy
from adamv import AdamV, AdamVCpp

def setup_model():
    torch.manual_seed(42)
    model = nn.Linear(4, 4)
    return model

def test_pure_python_matches_cpp_fallback():
    """Test 1: Pure Python AdamV matches AdamVCpp Python fallback (no C++ extension)."""
    model1 = setup_model()
    model2 = copy.deepcopy(model1)

    opt1 = AdamV(model1.parameters(), lr=1e-3, weight_decay=0.01)
    # Assuming AdamVCpp falls back to Python if no extension or has a force_python flag,
    # or just by comparing their results since they should implement the same algorithm.
    opt2 = AdamVCpp(model2.parameters(), lr=1e-3, weight_decay=0.01)

    data = torch.randn(2, 4)
    target = torch.randn(2, 4)
    criterion = nn.MSELoss()

    for _ in range(3):
        opt1.zero_grad()
        opt2.zero_grad()

        loss1 = criterion(model1(data), target)
        loss2 = criterion(model2(data), target)

        loss1.backward()
        loss2.backward()

        opt1.step()
        opt2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2, atol=1e-4, rtol=1e-3)

def test_adamw_baseline_difference():
    """Test 2: AdamW as baseline — verify AdamV behavior is different from AdamW in expected ways."""
    model_adamv = setup_model()
    model_adamw = copy.deepcopy(model_adamv)

    opt_adamv = AdamV(model_adamv.parameters(), lr=1e-3)
    opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3)

    data = torch.randn(2, 4)
    target = torch.randn(2, 4)
    criterion = nn.MSELoss()

    opt_adamv.zero_grad()
    opt_adamw.zero_grad()

    loss_v = criterion(model_adamv(data), target)
    loss_w = criterion(model_adamw(data), target)

    loss_v.backward()
    loss_w.backward()

    opt_adamv.step()
    opt_adamw.step()

    # AdamV and AdamW should produce different parameter updates due to CAMD and Bakhshali Brake
    for p_v, p_w in zip(model_adamv.parameters(), model_adamw.parameters()):
        assert not torch.allclose(p_v, p_w)

def test_determinism():
    """Test 3: Determinism — same seed, same optimizer config produces same results."""
    def run_deterministic():
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        opt = AdamV(model.parameters(), lr=1e-3)
        data = torch.randn(2, 4)
        target = torch.randn(2, 4)
        criterion = nn.MSELoss()
        
        opt.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        opt.step()
        return [p.clone() for p in model.parameters()]

    params_run1 = run_deterministic()
    params_run2 = run_deterministic()

    for p1, p2 in zip(params_run1, params_run2):
        torch.testing.assert_close(p1, p2)

def test_enable_brake_false_closer_to_adam():
    """Test 4: enable_brake=False makes AdamV behave closer to standard Adam."""
    model_brake = setup_model()
    model_no_brake = copy.deepcopy(model_brake)

    opt_brake = AdamV(model_brake.parameters(), lr=0.1, enable_brake=True)
    opt_no_brake = AdamV(model_no_brake.parameters(), lr=0.1, enable_brake=False)

    # Large gradient to trigger brake
    data = torch.randn(2, 4) * 100
    target = torch.randn(2, 4) * 100
    criterion = nn.MSELoss()

    opt_brake.zero_grad()
    opt_no_brake.zero_grad()

    loss_brake = criterion(model_brake(data), target)
    loss_no_brake = criterion(model_no_brake(data), target)

    loss_brake.backward()
    loss_no_brake.backward()

    opt_brake.step()
    opt_no_brake.step()

    # With large gradients, the braked version should have smaller parameter updates
    for p_brake, p_no_brake, p_orig in zip(model_brake.parameters(), model_no_brake.parameters(), setup_model().parameters()):
        update_brake = torch.norm(p_brake - p_orig)
        update_no_brake = torch.norm(p_no_brake - p_orig)
        assert update_brake < update_no_brake

def test_weight_decay_zero():
    """Test 5: weight_decay=0 disables weight decay correctly."""
    model_wd = setup_model()
    model_no_wd = copy.deepcopy(model_wd)

    opt_wd = AdamV(model_wd.parameters(), lr=1e-3, weight_decay=0.1)
    opt_no_wd = AdamV(model_no_wd.parameters(), lr=1e-3, weight_decay=0.0)

    # Zero input to ensure gradients are zero; only weight decay will update params
    data = torch.zeros(2, 4)
    target = torch.zeros(2, 4)
    criterion = nn.MSELoss()

    opt_wd.zero_grad()
    opt_no_wd.zero_grad()

    loss_wd = criterion(model_wd(data), target)
    loss_no_wd = criterion(model_no_wd(data), target)

    loss_wd.backward()
    loss_no_wd.backward()

    # Manually zero out gradients to isolate weight decay
    for p in model_wd.parameters(): p.grad.zero_()
    for p in model_no_wd.parameters(): p.grad.zero_()

    opt_wd.step()
    opt_no_wd.step()

    for p_wd, p_no_wd, p_orig in zip(model_wd.parameters(), model_no_wd.parameters(), setup_model().parameters()):
        # With WD, params should shrink
        assert torch.norm(p_wd) < torch.norm(p_orig)
        # Without WD and zero grads, params should stay exactly the same
        torch.testing.assert_close(p_no_wd, p_orig)
