#include <torch/extension.h>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fused CPU Kernel for AdamV (Adam Vedic) Optimizer
void adamv_step_cpu(
    at::Tensor p,
    at::Tensor grad,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float progresso,
    float bakh_thresh,
    int step,
    int D) 
{
    float bias_correction1 = 1.0f - std::pow(beta1, step);
    float bias_correction2 = 1.0f - std::pow(beta2, step);
    
    // Contiguous memory pointers
    auto p_acc = p.contiguous().data_ptr<float>();
    auto grad_acc = grad.contiguous().data_ptr<float>();
    auto exp_avg_acc = exp_avg.contiguous().data_ptr<float>();
    auto exp_avg_sq_acc = exp_avg_sq.contiguous().data_ptr<float>();
    
    int numel = p.numel();
    
    // --- Pass 1: Compute M, V, and Direction Norm ---
    float norm_dir_sq = 0.0f;
    std::vector<float> direcao(numel); 
    
    for (int i = 0; i < numel; ++i) {
        float g = grad_acc[i];
        
        float m = exp_avg_acc[i] * beta1 + g * (1.0f - beta1);
        float v = exp_avg_sq_acc[i] * beta2 + (g * g) * (1.0f - beta2);
        
        exp_avg_acc[i] = m;
        exp_avg_sq_acc[i] = v;
        
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        
        float dir = m_hat / (std::sqrt(v_hat) + eps);
        direcao[i] = dir;
        norm_dir_sq += dir * dir;
    }
    
    float norm_dir = std::sqrt(norm_dir_sq);
    float norm_dir_padrao = norm_dir / std::sqrt((float)D);
    
    // Ramanujan Envelope Scalar Math
    float envelope = 1.0f / (progresso + norm_dir_padrao + eps);
    float cosine_fator = 0.5f * (1.0f + std::cos(M_PI * progresso));
    float lr_efetivo = lr * std::min((float)(envelope * cosine_fator), 1.5f);
    
    // --- Pass 2: Bakhshali Gating and Weights Update ---
    for (int i = 0; i < numel; ++i) {
        float g = grad_acc[i];
        float v_hat = exp_avg_sq_acc[i] / bias_correction2;
        float dir = direcao[i];
        float a = lr_efetivo * dir;
        
        float w_val = p_acc[i];
        
        // Bakhshali Brake Condition
        bool explosao = std::abs(g) > (bakh_thresh * std::sqrt(v_hat));
        float step_size = a;
        
        if (explosao) {
            float denom = std::abs(w_val) + std::abs(a) + eps;
            float correction = (a * a) / (2.0f * denom);
            float sign_a = (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f);
            step_size = a - sign_a * correction;
        }
        
        // Weight Decay (AdamW style)
        if (weight_decay > 0.0f) {
            w_val = w_val * (1.0f - lr_efetivo * weight_decay);
        }
        
        p_acc[i] = w_val - step_size;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamv_step_cpu", &adamv_step_cpu, "AdamV Optimizer Fused CPU Kernel");
}
