import torch
import math

class AdamV(torch.optim.Optimizer):
    """
    AdamV (Adam Vedic) Optimizer - Pure Python Version.
    Combines Adam Momentum + Ramanujan Scale Envelope + Bakhshali Quartic Gate + OMNI Basin Hopping.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=3.0, enable_omni=True):
                 
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps, bakhshali_threshold=bakhshali_threshold,
                        enable_omni=enable_omni)
        super(AdamV, self).__init__(params, defaults)
        
        self.state['omni_global'] = {
            'loss_ema': float('inf'),
            'patience': 0,
            'clock_reset_step': 0,
            'global_step': 0,
        }

    @torch.no_grad()
    def step(self, current_loss=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        if current_loss is not None:
            loss = current_loss

        g_state = self.state['omni_global']
        g_state['global_step'] += 1
        current_step = g_state['global_step']
        
        omni_triggered = False
        if loss is not None and len(self.param_groups) > 0 and self.param_groups[0]['enable_omni']:
            if g_state['loss_ema'] == float('inf'):
                g_state['loss_ema'] = loss
            else:
                g_state['loss_ema'] = 0.9 * g_state['loss_ema'] + 0.1 * loss
                
            if loss > g_state['loss_ema'] * 0.99:
                g_state['patience'] += 1
            else:
                g_state['patience'] = 0
                
            patience_limit = max(100, int(self.param_groups[0]['total_steps'] * 0.01))
            if g_state['patience'] >= patience_limit:
                omni_triggered = True
                g_state['patience'] = 0
                g_state['clock_reset_step'] = current_step
                
        for group in self.param_groups:
            lr_max = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            total_steps = group['total_steps']
            bakh_thresh = group['bakhshali_threshold']
            
            internal_step = current_step - g_state['clock_reset_step']
            progresso = min(1.0, internal_step / max(1, total_steps))
            cosine_fator = 0.5 * (1 + math.cos(math.pi * progresso))
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                direcao = m_hat / (torch.sqrt(v_hat) + eps)
                
                if omni_triggered:
                    p.add_(torch.randn_like(p) * 0.05 * torch.std(p))
                
                D = p.numel()
                norm_dir_padrao = torch.linalg.norm(direcao) / math.sqrt(D)
                
                envelope = 1.0 / (progresso + norm_dir_padrao + eps)
                lr_efetivo = lr_max * min(float(envelope * cosine_fator), 1.5)
                
                a = lr_efetivo * direcao
                
                sqrt_v = torch.sqrt(v_hat)
                explosao_mask = torch.abs(grad) > (bakh_thresh * sqrt_v)
                
                denom = torch.abs(p) + torch.abs(a) + eps
                correction = (a ** 2) / (2.0 * denom)
                
                bakhshali_brake = a - torch.sign(a) * correction
                step_size = torch.where(explosao_mask, bakhshali_brake, a)
                
                if weight_decay > 0:
                    p.mul_(1 - lr_efetivo * weight_decay)
                    
                p.sub_(step_size)
                
        return loss

class AdamVCpp(torch.optim.Optimizer):
    """
    AdamV (Adam Vedic) Optimizer - C++ Fused Kernel Version.
    Extremely fast execution bypassing Python tensor loops.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=3.0, enable_omni=True):
                 
        try:
            import adamv_cpp
            self.adamv_cpp = adamv_cpp
        except ImportError:
            raise ImportError("AdamVCpp requer que a extensão C++ seja compilada rodando 'python setup_adamv.py build_ext --inplace'")
            
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps, bakhshali_threshold=bakhshali_threshold,
                        enable_omni=enable_omni)
        super(AdamVCpp, self).__init__(params, defaults)
        
        self.state['omni_global'] = {
            'loss_ema': float('inf'),
            'patience': 0,
            'clock_reset_step': 0,
            'global_step': 0,
        }

    @torch.no_grad()
    def step(self, current_loss=None, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if current_loss is not None:
            loss = current_loss

        g_state = self.state['omni_global']
        g_state['global_step'] += 1
        current_step = g_state['global_step']
        
        omni_triggered = False
        if loss is not None and len(self.param_groups) > 0 and self.param_groups[0]['enable_omni']:
            if g_state['loss_ema'] == float('inf'):
                g_state['loss_ema'] = loss
            else:
                g_state['loss_ema'] = 0.9 * g_state['loss_ema'] + 0.1 * loss
                
            if loss > g_state['loss_ema'] * 0.99:
                g_state['patience'] += 1
            else:
                g_state['patience'] = 0
                
            patience_limit = max(100, int(self.param_groups[0]['total_steps'] * 0.01))
            if g_state['patience'] >= patience_limit:
                omni_triggered = True
                g_state['patience'] = 0
                g_state['clock_reset_step'] = current_step
                
        for group in self.param_groups:
            lr_max = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            total_steps = group['total_steps']
            bakh_thresh = group['bakhshali_threshold']
            
            internal_step = current_step - g_state['clock_reset_step']
            progresso = min(1.0, internal_step / max(1, total_steps))
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                if omni_triggered:
                    p.add_(torch.randn_like(p) * 0.05 * torch.std(p))
                
                # FUSED KERNEL C++ CALL
                self.adamv_cpp.adamv_step_cpu(
                    p, grad, exp_avg, exp_avg_sq,
                    lr_max, beta1, beta2, eps, weight_decay,
                    float(progresso), bakh_thresh, state['step'], p.numel()
                )
                
        return loss
