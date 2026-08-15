import torch
import math

class AdamV(torch.optim.Optimizer):
    """
    AdamV (Adam Vedic) Optimizer - Pure Python Version.
    Combines Adam Momentum + Ramanujan Scale Envelope + Bakhshali Quartic Gate + OMNI Basin Hopping.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=10.0, enable_omni=True,
                 lp_kappa=0.1, lp_omega=10.0, punning_mask=0xFFFFE000):
                 
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps, bakhshali_threshold=bakhshali_threshold,
                        enable_omni=enable_omni, lp_kappa=lp_kappa, lp_omega=lp_omega, 
                        punning_mask=punning_mask)
        super(AdamV, self).__init__(params, defaults)
        
        self.state['omni_global'] = {
            'loss_ema': float('inf'),
            'patience': 0,
            'clock_reset_step': 0,
            'global_step': 0,
        }

    @torch.no_grad()
    def step(self, closure=None, current_loss=None):
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
            loss_val = float(loss) if isinstance(loss, torch.Tensor) else loss
            if g_state['loss_ema'] == float('inf'):
                g_state['loss_ema'] = loss_val
                g_state['patience'] = 0
            else:
                g_state['loss_ema'] = 0.9 * g_state['loss_ema'] + 0.1 * loss_val
                
            is_worse = loss_val > g_state['loss_ema'] * 0.99
            g_state['patience'] = g_state['patience'] + 1 if is_worse else 0
            
            patience_limit = max(500, int(self.param_groups[0]['total_steps'] * 0.05))
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
            lp_kappa = group['lp_kappa']
            lp_omega = group['lp_omega']
            punning_mask = group['punning_mask']
            
            internal_step = current_step - g_state['clock_reset_step']
            progresso = min(1.0, internal_step / max(1, total_steps))
            
            # Cálculo no HOST para evitar SFU Starvation na GPU
            LP_Fator = 1.0 + lp_kappa * math.cos(lp_omega * math.log(1.0 + progresso * 10.0))
            bakh_thresh_eff = bakh_thresh * LP_Fator
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                snr = (exp_avg * exp_avg) / (exp_avg_sq + eps)
                beta1_eff = torch.clamp(beta1 + 0.05 * (1.0 - 2.0 * snr), 0.0, 1.0)
                exp_avg = exp_avg * beta1_eff + grad.float() * (1.0 - beta1_eff)
                state['exp_avg'] = exp_avg
                
                exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                direcao = m_hat / (torch.sqrt(v_hat) + eps)
                
                if omni_triggered:
                    std_val = torch.std(p) if p.numel() > 1 else torch.tensor(0.05, dtype=p.dtype, device=p.device)
                    noise = torch.randn_like(p) * (0.05 * std_val)
                    p.add_(noise)
                    # OMNI-ModBH puro no Python
                    if p.dtype == torch.float32:
                        p_int = p.view(torch.int32)
                        mask_val = int(punning_mask)
                        if mask_val > 0x7FFFFFFF:
                            mask_val -= 0x100000000
                        p_int.bitwise_and_(mask_val)
                    pingala_scale = 2.0
                    state['exp_avg'].copy_(-noise * pingala_scale)
                    state['exp_avg_sq'].copy_(noise.float() ** 2)
                
                D = p.numel()
                norm_dir_padrao = torch.linalg.norm(direcao) / math.sqrt(D)
                
                envelope = (1.0 + progresso) / (progresso + norm_dir_padrao + eps)
                cooling_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                lr_efetivo = lr_max * torch.clamp(envelope * cooling_factor, max=1.5)
                
                a = lr_efetivo * direcao
                
                sqrt_v = torch.sqrt(v_hat)
                explosao_mask = torch.abs(grad) > (bakh_thresh_eff * sqrt_v)
                
                denom = sqrt_v + torch.abs(a) + eps
                correction = (a ** 2) / (2.0 * denom)
                
                bakhshali_brake = a - torch.sign(a) * correction
                step_size = torch.where(explosao_mask, bakhshali_brake, a)
                
                if weight_decay != 0:
                    # Dynamic Endogenous Weight Decay (Cosine Annealing interno)
                    wd_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                    p.mul_(1 - lr_max * weight_decay * wd_factor)
                    
                p.sub_(step_size)
                
        return loss

class AdamVCpp(torch.optim.Optimizer):
    """
    AdamV (Adam Vedic) Optimizer - C++ Fused Kernel Version.
    Extremely fast execution bypassing Python tensor loops.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=10.0, enable_omni=True,
                 lp_kappa=0.1, lp_omega=10.0, punning_mask=0xFFFFE000):
                 
        try:
            import adamv_cpp
            self.adamv_cpp = adamv_cpp
        except ImportError:
            raise ImportError("AdamVCpp requer que a extensão C++ seja compilada rodando 'python setup_adamv.py build_ext --inplace'")
            
        try:
            import adamv_cuda
            self.adamv_cuda = adamv_cuda
        except ImportError:
            self.adamv_cuda = None
            
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps, bakhshali_threshold=bakhshali_threshold,
                        enable_omni=enable_omni, lp_kappa=lp_kappa, lp_omega=lp_omega, 
                        punning_mask=punning_mask)
        super(AdamVCpp, self).__init__(params, defaults)
        
        self.state['omni_global'] = {
            'loss_ema': float('inf'),
            'patience': 0,
            'clock_reset_step': 0,
            'global_step': 0,
        }

    @torch.no_grad()
    def step(self, closure=None, current_loss=None):
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
            loss_val = float(loss) if isinstance(loss, torch.Tensor) else loss
            if g_state['loss_ema'] == float('inf'):
                g_state['loss_ema'] = loss_val
                g_state['patience'] = 0
            else:
                g_state['loss_ema'] = 0.9 * g_state['loss_ema'] + 0.1 * loss_val
                
            is_worse = loss_val > g_state['loss_ema'] * 0.99
            g_state['patience'] = g_state['patience'] + 1 if is_worse else 0
            
            patience_limit = max(500, int(self.param_groups[0]['total_steps'] * 0.05))
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
            lp_kappa = group['lp_kappa']
            lp_omega = group['lp_omega']
            punning_mask = group['punning_mask']
            
            internal_step = current_step - g_state['clock_reset_step']
            progresso = min(1.0, internal_step / max(1, total_steps))
            
            LP_Fator = 1.0 + lp_kappa * math.cos(lp_omega * math.log(1.0 + progresso * 10.0))
            bakh_thresh_eff = bakh_thresh * LP_Fator
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                    state['direcao_buffer'] = torch.empty_like(p, memory_format=torch.preserve_format, dtype=torch.float32)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                if omni_triggered:
                    if p.numel() > 1:
                        std_val = torch.std(p)
                    else:
                        std_val = torch.tensor(0.05, dtype=p.dtype, device=p.device)
                        
                    noise = torch.randn_like(p) * (0.05 * std_val)
                    p.add_(noise)
                    # Note: OMNI-ModBH bitmask happens in CUDA kernel for GPU. 
                    pingala_scale = 2.0
                    state['exp_avg'].copy_(-noise * pingala_scale)
                    state['exp_avg_sq'].copy_(noise.float() ** 2)
                
                if p.is_cpu:
                    self.adamv_cpp.adamv_step_cpu(
                        p, grad, exp_avg, exp_avg_sq, state['direcao_buffer'],
                        lr_max, beta1, beta2, eps, weight_decay,
                        float(progresso), float(bakh_thresh_eff), state['step'], p.numel()
                    )
                elif p.is_cuda and self.adamv_cuda is not None and hasattr(self.adamv_cuda, 'adamv_step_cuda'):
                    self.adamv_cuda.adamv_step_cuda(
                        p, grad, exp_avg, exp_avg_sq, state['direcao_buffer'],
                        lr_max, beta1, beta2, eps, weight_decay,
                        float(progresso), float(bakh_thresh_eff), state['step'], p.numel(),
                        bool(omni_triggered), int(punning_mask)
                    )
                else:
                    # Python fallback para GPU
                    
                    # Update moving averages FIRST
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    snr = (exp_avg * exp_avg) / (exp_avg_sq + eps)
                    beta1_eff = torch.clamp(beta1 + 0.05 * (1.0 - 2.0 * snr), 0.0, 1.0)
                    exp_avg = exp_avg * beta1_eff + grad.float() * (1.0 - beta1_eff)
                    state['exp_avg'] = exp_avg
                    
                    exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)
                    
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                    direcao = m_hat / (v_hat.sqrt() + eps)
                    
                    norm_dir = torch.linalg.norm(direcao) / math.sqrt(p.numel())
                    envelope = (1.0 + progresso) / (progresso + norm_dir + eps)
                    cooling = 0.5 * (1.0 + math.cos(math.pi * progresso))
                    lr_efetivo = lr_max * torch.clamp(envelope * cooling, max=1.5)
                    
                    a = lr_efetivo * direcao
                    sqrt_v = v_hat.sqrt()
                    
                    explosao_mask = torch.abs(grad) > (bakh_thresh_eff * sqrt_v)
                    
                    denom = sqrt_v + torch.abs(a) + eps
                    correction = (a * a) / (2.0 * denom)
                    bakhshali_brake = a - torch.sign(a) * correction
                    
                    step_size = torch.where(explosao_mask, bakhshali_brake, a)
                    
                    if weight_decay != 0:
                        wd_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                        p.mul_(1.0 - lr_max * weight_decay * wd_factor)
                        
                    p.sub_(step_size)
                
        return loss
