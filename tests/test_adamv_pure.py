import torch
import torch.nn as nn
import math

class AdamVPure(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=3.0, enable_omni=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps, bakhshali_threshold=bakhshali_threshold,
                        enable_omni=enable_omni)
        super(AdamVPure, self).__init__(params, defaults)
        
    def step(self, closure=None, current_loss=None):
        loss = None
        if closure is not None: loss = closure()
        loss_val = current_loss if current_loss is not None else (loss.item() if loss is not None else 0.0)
        
        if 'global_state' not in self.state:
            self.state['global_state'] = {'step': 0, 'patience': 0, 'loss_ema': loss_val, 'clock_reset_step': 0}
        
        g_state = self.state['global_state']
        current_step = g_state['step']
        
        if loss_val > g_state['loss_ema'] * 0.99:
            g_state['patience'] += 1
        else:
            g_state['patience'] = 0
            
        omni_triggered = False
        if g_state['patience'] >= 100 and self.defaults['enable_omni']:
            omni_triggered = True
            g_state['patience'] = 0
            g_state['clock_reset_step'] = current_step
            g_state['loss_ema'] = loss_val
            
        g_state['loss_ema'] = 0.9 * g_state['loss_ema'] + 0.1 * loss_val
        g_state['step'] += 1
        
        for group in self.param_groups:
            lr_max, beta1, beta2, eps, weight_decay, total_steps, bakh_thresh = (
                group['lr'], group['betas'][0], group['betas'][1], group['eps'], 
                group['weight_decay'], group['total_steps'], group['bakhshali_threshold'])
            
            internal_step = current_step - g_state['clock_reset_step']
            progresso = min(1.0, internal_step / max(1, total_steps))
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                if omni_triggered:
                    std_val = torch.std(p) if p.numel() > 1 else torch.tensor(0.05)
                    noise = torch.randn_like(p) * (0.05 * std_val)
                    p.data.add_(noise)
                    exp_avg.copy_(-noise * 2.0)
                    exp_avg_sq.copy_(noise ** 2)
                
                if weight_decay != 0:
                    wd_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                    p.data.mul_(1.0 - lr_max * weight_decay * wd_factor)
                    
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                direcao = m_hat / (v_hat.sqrt() + eps)
                
                norm_dir = torch.linalg.norm(direcao).item() / math.sqrt(p.numel())
                envelope = 1.0 / (progresso + norm_dir + eps)
                cooling = 0.5 * (1.0 + math.cos(math.pi * progresso))
                lr_efetivo = lr_max * min(envelope * cooling, 1.5)
                
                a = lr_efetivo * direcao
                sqrt_v = v_hat.sqrt()
                
                explosao_mask = torch.abs(grad) > (bakh_thresh * sqrt_v)
                
                denom = torch.abs(p.data) + torch.abs(a) + eps
                correction = (a * a) / (2.0 * denom)
                bakhshali_brake = a - torch.sign(a) * correction
                
                step_size = torch.where(explosao_mask, bakhshali_brake, a)
                p.data.sub_(step_size)
                
        return loss

torch.manual_seed(42)
model = nn.Sequential(*[nn.Linear(200 if i==0 else 200, 200) if i%2==0 else nn.GELU() for i in range(30)], nn.Linear(200, 10))
opt = AdamVPure(model.parameters(), lr=1e-3, weight_decay=0.01)

X = torch.randn(1000, 200)
Y = (torch.sin(X[:, 0]) * 10).long() % 10
criterion = nn.CrossEntropyLoss()

for i in range(1000):
    opt.zero_grad()
    out = model(X)
    loss = criterion(out, Y)
    loss.backward()
    opt.step(current_loss=loss.item())
    if i % 10 == 0:
        print(f"Step {i}: {loss.item():.4f}", flush=True)
