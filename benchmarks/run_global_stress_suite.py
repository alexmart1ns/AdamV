import os
import gc
import math
import random
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind

# =========================================================
# 1. ADAMV KERNEL COMPILATION
# =========================================================
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void adamv_prepare_kernel(
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ direcao_buffer,
    const scalar_t* __restrict__ grad,
    float beta1, float beta2, float bias_correction2, float eps, int step, int numel) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float g = static_cast<float>(grad[idx]);
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        v = beta2 * v + (1.0f - beta2) * g * g;
        exp_avg_sq[idx] = v;

        float v_hat = v / bias_correction2;
        float sqrt_v_hat = sqrt(v_hat);

        // BRCM: Excess Shock Isolator (delta)
        // Tighter tuning: 1.0x margin to prevent excess momentum in sharp minima
        float delta = fmaxf(0.0f, std::abs(g) - 1.5f * sqrt_v_hat);
        float denom_brcm = sqrt_v_hat + delta + eps;
        float bakh_residual = (delta * delta) / (2.0f * denom_brcm);
        
        // Full curvature shift to avoid carrying too much momentum
        float curvature_shift = bakh_residual / (sqrt_v_hat + eps);
        float beta1_dynamic = beta1 * std::exp(-0.03f * curvature_shift);
        
        m = beta1_dynamic * m + (1.0f - beta1_dynamic) * g;
        exp_avg[idx] = m;
        
        float bias_correction1_dynamic = 1.0f - std::pow(beta1_dynamic, static_cast<float>(step));

        float m_hat = m / bias_correction1_dynamic;
        
        direcao_buffer[idx] = m_hat / (sqrt_v_hat + eps);
    }
}

template <typename scalar_t>
__global__ void adamv_update_kernel(
    scalar_t* __restrict__ params,
    const scalar_t* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float* __restrict__ direcao_buffer,
    const float* __restrict__ norm_tensor_ptr,
    float progresso, float cooling_factor, float bakh_thresh_eff, float bias_correction2, float eps, 
    float wd_factor, float lr_max, float weight_decay, int numel, int D, int step,
    bool enable_cooling, bool enable_brake,
    bool omni_triggered, uint32_t punning_mask) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t p = params[idx];
        
        float g = static_cast<float>(grad[idx]);
        float dir = direcao_buffer[idx];
        float v = exp_avg_sq[idx];
        
        float lr_efetivo = lr_max;
        if (enable_cooling) {
            float norm_dir = (*norm_tensor_ptr) / sqrt(static_cast<float>(D));
            float envelope = (1.0f + progresso) / (progresso + norm_dir + eps);
            lr_efetivo = lr_max * min(envelope * cooling_factor, 1.5f);
        }
        
        // Inject Golden Calibration Onda (Topological Wave)
        lr_efetivo = lr_efetivo * (1.0f + 0.01f * cosf(M_PI * 4.0f * progresso));
        
        float a = lr_efetivo * dir;
        float v_hat = v / bias_correction2;
        float sqrt_v = sqrt(v_hat);
        
        float step_size = a;
        if (enable_brake) {
            bool explosao_mask = std::abs(g) > (bakh_thresh_eff * sqrt_v);
            if (explosao_mask) {
                float denom = std::abs(static_cast<float>(p)) + (std::abs(a) * 2.0f) + eps;
                float correction = (a * a) / (2.0f * denom);
                float bakhshali_brake = a - copysignf(1.0f, a) * correction;
                step_size = bakhshali_brake;
            }
        }
        
        if (weight_decay != 0.0f) {
            p = static_cast<scalar_t>(static_cast<float>(p) * (1.0f - lr_max * weight_decay));
        }
        
        params[idx] = static_cast<scalar_t>(static_cast<float>(p) - step_size);
        
        if (omni_triggered) {
            if (sizeof(scalar_t) == 4) {
                float p_new = static_cast<float>(params[idx]);
                uint32_t p_int = __float_as_uint(p_new);
                
                uint32_t sign = p_int & 0x80000000;
                uint32_t exp  = p_int & 0x7F800000;
                uint32_t mant = p_int & 0x007FFFFF;
                
                uint32_t mant_mod = (((mant + 1) * 31337) & 0x007FFFFF) & punning_mask;
                
                p_new = __uint_as_float(sign | exp | mant_mod);
                params[idx] = static_cast<scalar_t>(p_new);
            }
            exp_avg[idx] = 0.0f;
            exp_avg_sq[idx] *= 0.1f;
        }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void adamv_step_cuda(
    at::Tensor p,
    at::Tensor grad,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::Tensor direcao,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float progresso,
    float bakh_thresh_eff,
    int step,
    int D,
    bool enable_cooling,
    bool enable_brake,
    bool omni_triggered,
    int64_t punning_mask) 
{
    CHECK_INPUT(p);
    CHECK_INPUT(grad);
    CHECK_INPUT(exp_avg);
    CHECK_INPUT(exp_avg_sq);
    CHECK_INPUT(direcao);

    int numel = p.numel();
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float bias_correction2 = 1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(step));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "adamv_prepare", [&] {
        adamv_prepare_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            direcao.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            beta1, beta2, bias_correction2, eps, step, numel
        );
    });

    // Compute norm asynchronously on GPU
    at::Tensor norm_tensor = at::linalg_norm(direcao);
    
    float cooling_factor;
    if (progresso < 0.1f) {
        cooling_factor = 0.01f + (progresso / 0.1f) * 0.99f;
    } else {
        float cos_progresso = (progresso - 0.1f) / 0.9f;
        cooling_factor = 1.0f;
    }
    float wd_factor = 0.5f * (1.0f + std::cos(M_PI * progresso));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "adamv_update", [&] {
        adamv_update_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            p.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            direcao.data_ptr<float>(),
            norm_tensor.data_ptr<float>(),
            progresso, cooling_factor, bakh_thresh_eff, bias_correction2, eps, wd_factor, lr, weight_decay, numel, D, step,
            enable_cooling, enable_brake,
            omni_triggered, static_cast<uint32_t>(punning_mask)
        );
    });
}

"""
cpp_source = """

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void adamv_step_cuda(
    at::Tensor p,
    at::Tensor grad,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::Tensor direcao,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float progresso,
    float bakh_thresh_eff,
    int step,
    int D,
    bool enable_cooling,
    bool enable_brake,
    bool omni_triggered,
    int64_t punning_mask
);

"""

print('Compiling JIT C++ Kernel for Kaggle T4...')
adamv_cuda = load_inline(
    name='adamv_cuda_v19',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['adamv_step_cuda'],
    extra_cuda_cflags=['-O3', '-use_fast_math', '-arch=sm_75', '-std=c++17'],
    with_cuda=True,
    verbose=False
)
print('JIT Compilation complete!')

# Python Wrapper
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
            pass
            
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


_old_init = AdamVCpp.__init__
def _new_init(self, *args, **kwargs):
    _old_init(self, *args, **kwargs)
    self.adamv_cuda = adamv_cuda
AdamVCpp.__init__ = _new_init


# =========================================================
# 2. SEEDING & UTILS
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, opt_name, scenario, lr=1e-3, wd=0.0):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    if opt_name == "AdamW":
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        
    elif opt_name == "AdamV":
        if scenario == "Vision" or scenario == "NLP":
            # Golden Calibration
            return AdamVCpp(optim_groups, lr=lr, betas=(0.9, 0.999), 
                            bakhshali_threshold=50.0, enable_brake=True, enable_cooling=True, 
                            enable_omni=False, enable_ignition=False)
        elif scenario == "Generative":
            # Stochastic Profile
            return AdamVCpp(optim_groups, lr=lr, betas=(0.9, 0.999), 
                            bakhshali_threshold=1000.0, enable_brake=False, enable_cooling=False, 
                            enable_omni=False, enable_ignition=False)

# =========================================================
# 3. ARCHITECTURES
# =========================================================
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        import torchvision.models as models
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# NanoGPT
batch_size_nlp = 64
block_size = 128
max_iters_nlp = 1000
eval_interval_nlp = 200
eval_iters_nlp = 50
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

def get_batch_nlp(split, train_data, val_data, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size_nlp,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss_nlp(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters_nlp)
        for k in range(eval_iters_nlp):
            X, Y = get_batch_nlp(split, train_data, val_data, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =========================================================
# 4. DATA DOWNLOADERS
# =========================================================
def get_dataloaders(scenario, seed):
    seed_everything(seed)
    if scenario == "Vision":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        return trainloader, testloader, None
    elif scenario == "Generative":
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        return trainloader, testloader, None
    elif scenario == "NLP":
        if not os.path.exists('input.txt'):
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, 'input.txt')
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s]
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data, vocab_size

# =========================================================
# 5. EXECUTION ENGINE
# =========================================================
def run_global_stress_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 5 Seeds recommended by the Benchmark Specialist
    seeds = [42, 1337, 2024, 3141, 8888]
    scenarios = ["Vision", "Generative", "NLP"]
    optimizers = ["AdamW", "AdamV"]
    
    all_results = []
    final_metrics = {"Vision": {}, "Generative": {}, "NLP": {}}
    
    for seed in seeds:
        print(f"\n{'='*50}\nGLOBAL SEED: {seed}\n{'='*50}")
        
        for scenario in scenarios:
            print(f"\n--- Running Scenario: {scenario} ---")
            
            # Setup Data
            dl1, dl2, vocab_size = get_dataloaders(scenario, seed)
            
            # Create base weights for fair comparison within this seed
            seed_everything(seed)
            if scenario == "Vision":
                base_model = ResNet18(num_classes=10)
            elif scenario == "Generative":
                base_model = VAE()
            elif scenario == "NLP":
                base_model = NanoGPT(vocab_size)
            
            torch.save(base_model.state_dict(), f"base_weights_{scenario}.pt")
            del base_model
            gc.collect()
            
            for opt_name in optimizers:
                print(f"  > Optimizer: {opt_name}")
                seed_everything(seed)
                
                # Load identical weights
                if scenario == "Vision":
                    model = ResNet18(num_classes=10).to(device)
                    total_steps = len(dl1) * 4 # 4 epochs
                elif scenario == "Generative":
                    model = VAE().to(device)
                    total_steps = len(dl1) * 8 # 8 epochs
                elif scenario == "NLP":
                    model = NanoGPT(vocab_size).to(device)
                    total_steps = max_iters_nlp
                    
                model.load_state_dict(torch.load(f"base_weights_{scenario}.pt"))
                
                # Configure LR and WD
                lr = 1e-3
                wd = 0.1 if scenario == "NLP" else (0.1 if scenario == "Vision" else 0.0)
                optimizer = get_optimizer(model, opt_name, scenario, lr=lr, wd=wd)
                
                if opt_name == "AdamV":
                    for group in optimizer.param_groups:
                        group['total_steps'] = total_steps
                    scheduler = None
                else:
                    if scenario == "Vision" or scenario == "Generative":
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
                    else:
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

                # Training Loops
                if scenario == "Vision":
                    step = 0
                    for epoch in range(4):
                        model.train()
                        for inputs, targets in dl1:
                            inputs, targets = inputs.to(device), targets.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            outputs = model(inputs)
                            loss = F.cross_entropy(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            if scheduler: scheduler.step()
                            step += 1
                        
                        # Eval
                        model.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for inputs, targets in dl2:
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()
                        val_acc = 100. * correct / total
                        all_results.append({"Seed": seed, "Scenario": scenario, "Optimizer": opt_name, "Epoch": epoch+1, "Metric": val_acc})
                        print(f"      Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")
                        if epoch == 3:
                            if opt_name not in final_metrics[scenario]: final_metrics[scenario][opt_name] = []
                            final_metrics[scenario][opt_name].append(val_acc)
                        
                elif scenario == "Generative":
                    step = 0
                    for epoch in range(8):
                        model.train()
                        for inputs, _ in dl1:
                            inputs = inputs.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            recon_batch, mu, logvar = model(inputs)
                            loss = vae_loss_function(recon_batch, inputs, mu, logvar)
                            loss.backward()
                            optimizer.step()
                            if scheduler: scheduler.step()
                            step += 1
                        
                        # Eval
                        model.eval()
                        test_loss = 0
                        with torch.no_grad():
                            for inputs, _ in dl2:
                                inputs = inputs.to(device)
                                recon_batch, mu, logvar = model(inputs)
                                test_loss += vae_loss_function(recon_batch, inputs, mu, logvar).item()
                        val_loss = test_loss / len(dl2.dataset)
                        all_results.append({"Seed": seed, "Scenario": scenario, "Optimizer": opt_name, "Epoch": epoch+1, "Metric": val_loss})
                        print(f"      Epoch {epoch+1} | Val Loss (ELBO): {val_loss:.4f}")
                        if epoch == 7:
                            if opt_name not in final_metrics[scenario]: final_metrics[scenario][opt_name] = []
                            final_metrics[scenario][opt_name].append(val_loss)
                        
                elif scenario == "NLP":
                    for step in range(max_iters_nlp + 1):
                        if step % 200 == 0 or step == max_iters_nlp:
                            losses = estimate_loss_nlp(model, dl1, dl2, device)
                            all_results.append({"Seed": seed, "Scenario": scenario, "Optimizer": opt_name, "Epoch": step, "Metric": losses['val']})
                            print(f"      Step {step:04d} | Val Loss: {losses['val']:.4f}")
                            if step == max_iters_nlp:
                                if opt_name not in final_metrics[scenario]: final_metrics[scenario][opt_name] = []
                                final_metrics[scenario][opt_name].append(losses['val'])

                        if step < max_iters_nlp:
                            xb, yb = get_batch_nlp('train', dl1, dl2, device)
                            logits, loss = model(xb, yb)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            if scheduler: scheduler.step()

                # CRITICAL MEMORY CLEANUP
                del model, optimizer, scheduler
                gc.collect()
                torch.cuda.empty_cache()
            
            # Clean up dataloaders
            del dl1, dl2
            gc.collect()
            torch.cuda.empty_cache()

    # =========================================================
    # 6. AGGREGATION, STATISTICS & PLOTTING (MIN-MAX SHADING)
    # =========================================================
    print("\n=== STATISTICAL SIGNIFICANCE (p-values) ===")
    for scenario in scenarios:
        adamw_vals = final_metrics[scenario]["AdamW"]
        adamv_vals = final_metrics[scenario]["AdamV"]
        t_stat, p_val = ttest_ind(adamw_vals, adamv_vals)
        print(f"[{scenario}] AdamW Mean: {np.mean(adamw_vals):.4f} | AdamV Mean: {np.mean(adamv_vals):.4f}")
        print(f"[{scenario}] Welch's t-test p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("  -> Result is STATISTICALLY SIGNIFICANT!")
        else:
            print("  -> Result is NOT statistically significant (overlap in distributions).")

    df = pd.DataFrame(all_results)
    df.to_csv("global_stress_results.csv", index=False)
    print("\nSaved global_stress_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Vision (ResNet-18) [Val Acc %]", "Generative (VAE) [Val ELBO Loss]", "NLP (NanoGPT) [Val CrossEntropy]"]
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        scenario_df = df[df["Scenario"] == scenario]
        
        for opt, color in [("AdamW", "#1f77b4"), ("AdamV", "#ff7f0e")]:
            # Adjust legend to reflect Fair Fight Protocol
            label = f"{opt} (Flat LR)" if opt == "AdamV" else f"{opt} (with Scheduler)"
            opt_df = scenario_df[scenario_df["Optimizer"] == opt]
            if opt_df.empty: continue
            
            # Group by Epoch for Min-Max Shading
            agg_df = opt_df.groupby("Epoch")["Metric"].agg(["mean", "min", "max"]).reset_index()
            
            ax.plot(agg_df["Epoch"], agg_df["mean"], label=label, color=color, marker='o')
            ax.fill_between(agg_df["Epoch"], agg_df["min"], agg_df["max"], color=color, alpha=0.2)
            
        ax.set_title(titles[i])
        ax.set_xlabel("Epoch / Step")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Invert y-axis for Loss metrics so "Better" is always visually upwards or clearly marked
        if scenario != "Vision":
            ax.invert_yaxis()
            
    plt.tight_layout()
    plt.savefig("global_stress_plot.png", dpi=300)
    print("Saved global_stress_plot.png")

if __name__ == "__main__":
    run_global_stress_test()
