import torch
import math

class AdamV(torch.optim.Optimizer):
    """
    AdamV (Adam Vedic) Optimizer - Pure Python Version.
    AdamV 3.1: Harmonic Refactor (In-Place VRAM Opt, OMNI State Fix, Modular Flags)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=10.0, enable_omni=True,
                 lp_kappa=0.1, lp_omega=10.0, punning_mask=0xFFFFE000,
                 enable_ignition=True, enable_cooling=False, enable_brake=True):
                 
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        total_steps=total_steps, bakhshali_threshold=bakhshali_threshold,
                        enable_omni=enable_omni, lp_kappa=lp_kappa, lp_omega=lp_omega, 
                        punning_mask=punning_mask,
                        enable_ignition=enable_ignition, enable_cooling=enable_cooling, enable_brake=enable_brake)
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
            total_steps = group['total_steps']
            
            # Autonomous Ignition
            if group.get('enable_ignition', True):
                ignition = min(1.0, current_step / max(1.0, total_steps * 0.10))
                lr_max = lr_max * ignition
                
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
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # In-Place Momentum Update
                snr = (exp_avg * exp_avg) / (exp_avg_sq + eps)
                beta1_eff = torch.clamp(beta1 + 0.05 * (1.0 - 2.0 * snr), 0.0, 1.0)
                exp_avg.mul_(beta1_eff).add_(grad.float(), alpha=1.0 - beta1_eff)
                
                exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1.0 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                sqrt_v = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
                direcao = (exp_avg / bias_correction1) / (sqrt_v + eps)
                
                norm_dir_padrao = torch.linalg.norm(direcao) / math.sqrt(p.numel())
                
                if group.get('enable_cooling', False):
                    envelope = (1.0 + progresso) / (progresso + norm_dir_padrao + eps)
                    cooling_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                    lr_efetivo = lr_max * torch.clamp(envelope * cooling_factor, max=1.5)
                else:
                    lr_efetivo = lr_max
                
                a = direcao.mul_(lr_efetivo)
                
                explosao_mask = torch.abs(grad) > (bakh_thresh_eff * sqrt_v)
                denom = sqrt_v.add_(torch.abs(a)).add_(eps)
                correction = (a * a).div_(denom.mul_(2.0))
                
                bakhshali_brake = a.clone().sub_(torch.sign(a) * correction)
                
                if group.get('enable_brake', True):
                    step_size = torch.where(explosao_mask, bakhshali_brake, a)
                else:
                    step_size = a
                
                if weight_decay != 0:
                    wd_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                    p.mul_(1.0 - lr_max * weight_decay * wd_factor)
                    
                p.sub_(step_size)
                
                if omni_triggered:
                    # Deterministic Mantissa Teleportation (Zero-RAM escape)
                    p_int = p.view(torch.int32)
                    sign_exp = p_int & 0xFF800000
                    mant = p_int & 0x007FFFFF
                    
                    mask_val = int(punning_mask)
                    if mask_val > 0x7FFFFFFF:
                        mask_val -= 0x100000000
                    
                    conditional_mask = mask_val if p.dtype == torch.float32 else -1
                    scrambled_mant = (((mant + 1) * 31337) & 0x007FFFFF) & conditional_mask
                    
                    p_new = (sign_exp | scrambled_mant).view(torch.float32)
                    p.copy_(p_new)
                    
                    # Harmonic State Reset: Do not poison momentum. Flush it gracefully.
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].mul_(0.1)
                
        return loss

class AdamVCpp(torch.optim.Optimizer):
    """
    AdamV (Adam Vedic) Optimizer - C++ Fused Kernel Version.
    AdamV 3.1: Harmonic Refactor
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, total_steps=10000, 
                 bakhshali_threshold=10.0, enable_omni=True,
                 lp_kappa=0.1, lp_omega=10.0, punning_mask=0xFFFFE000,
                 enable_ignition=True, enable_cooling=False, enable_brake=True):
                 
        try:
            import adamv_cpp
            self.adamv_cpp = adamv_cpp
        except ImportError:
            raise ImportError("AdamVCpp requer que a extensao C++ seja compilada")
            
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
                        punning_mask=punning_mask,
                        enable_ignition=enable_ignition, enable_cooling=enable_cooling, enable_brake=enable_brake)
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
            total_steps = group['total_steps']
            
            if group.get('enable_ignition', True):
                ignition = min(1.0, current_step / max(1.0, total_steps * 0.10))
                lr_max = lr_max * ignition
                
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
                
                # C++ Fused Kernel Call
                # OMNI logic is pushed to the END inside the C++ Kernel now
                mask_val = int(punning_mask)
                if mask_val > 0x7FFFFFFF:
                    mask_val -= 0x100000000
                    
                if p.is_cpu:
                    self.adamv_cpp.adamv_step_cpu(
                        p, grad, exp_avg, exp_avg_sq, state['direcao_buffer'],
                        lr_max, beta1, beta2, eps, weight_decay,
                        float(progresso), float(bakh_thresh_eff), state['step'], p.numel(),
                        bool(group.get('enable_cooling', True)),
                        bool(group.get('enable_brake', True)),
                        bool(omni_triggered), mask_val
                    )
                elif p.is_cuda and self.adamv_cuda is not None and hasattr(self.adamv_cuda, 'adamv_step_cuda'):
                    self.adamv_cuda.adamv_step_cuda(
                        p, grad, exp_avg, exp_avg_sq, state['direcao_buffer'],
                        lr_max, beta1, beta2, eps, weight_decay,
                        float(progresso), float(bakh_thresh_eff), state['step'], p.numel(),
                        bool(group.get('enable_cooling', True)),
                        bool(group.get('enable_brake', True)),
                        bool(omni_triggered), mask_val
                    )
                else:
                    # Python fallback para GPU - 100% In-Place BRCM
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    sqrt_v = exp_avg_sq.sqrt()
                    denom_brcm = sqrt_v.clone().add_(torch.abs(grad.float())).add_(eps)
                    bakh_residual = (grad.float() * grad.float()).div_(denom_brcm.mul_(2.0))
                    curvature_shift = bakh_residual.div_(sqrt_v.add_(eps))
                    
                    beta1_eff = torch.exp(-curvature_shift).mul_(beta1)
                    
                    # Update momentum In-Place
                    exp_avg.mul_(beta1_eff).add_(grad.float() * (1.0 - beta1_eff))
                    exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)
                    
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                    direcao = m_hat / (v_hat.sqrt() + eps)
                    
                    norm_dir = torch.linalg.norm(direcao) / math.sqrt(p.numel())
                    if group.get('enable_cooling', False):
                        envelope = (1.0 + progresso) / (progresso + norm_dir + eps)
                        cooling = 0.5 * (1.0 + math.cos(math.pi * progresso))
                        lr_efetivo = lr_max * torch.clamp(envelope * cooling, max=1.5)
                    else:
                        lr_efetivo = lr_max
                    
                    a = direcao.mul_(lr_efetivo)
                    sqrt_v_hat = v_hat.sqrt()
                    
                    explosao_mask = torch.abs(grad) > (bakh_thresh_eff * sqrt_v_hat)
                    
                    denom = sqrt_v_hat.add_(torch.abs(a)).add_(eps)
                    correction = (a * a).div_(denom.mul_(2.0))
                    bakhshali_brake = a.clone().sub_(torch.sign(a) * correction)
                    
                    if group.get('enable_brake', True):
                        step_size = torch.where(explosao_mask, bakhshali_brake, a)
                    else:
                        step_size = a
                        
                    if weight_decay != 0:
                        wd_factor = 0.5 * (1.0 + math.cos(math.pi * progresso))
                        p.mul_(1.0 - lr_max * weight_decay * wd_factor)
                        
                    p.sub_(step_size)
                    
                    if omni_triggered:
                        p_int = p.view(torch.int32)
                        sign_exp = p_int & 0xFF800000
                        mant = p_int & 0x007FFFFF
                        
                        conditional_mask = mask_val if p.dtype == torch.float32 else -1
                        scrambled_mant = (((mant + 1) * 31337) & 0x007FFFFF) & conditional_mask
                        
                        p_new = (sign_exp | scrambled_mant).view(torch.float32)
                        p.copy_(p_new)
                        
                        state['exp_avg'].zero_()
                        state['exp_avg_sq'].mul_(0.1)
                        
        return loss
