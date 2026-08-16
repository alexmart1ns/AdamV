#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fused CPU Kernel for AdamV (Adam Vedic) Optimizer
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
void adamv_step_cpu_template(
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
    float bakh_thresh,
    int step,
    int D,
    bool enable_cooling,
    bool enable_brake,
    bool omni_triggered,
    int mask_val)
{
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    
    auto p_acc = p.data_ptr<scalar_t>();
    auto grad_acc = grad.data_ptr<scalar_t>();
    auto exp_avg_acc = exp_avg.data_ptr<float>();
    auto exp_avg_sq_acc = exp_avg_sq.data_ptr<float>();
    auto direcao_acc = direcao.data_ptr<float>();
    
    int numel = p.numel();
    
    float old_norm_sq = direcao_acc[0];
    direcao_acc[0] = 0.0f;
    
    float norm_dir = std::sqrt(old_norm_sq);
    float norm_dir_padrao = norm_dir / std::sqrt(static_cast<float>(D));
    
    float lr_efetivo = lr;
    if (enable_cooling) {
        float envelope = (1.0f + progresso) / (progresso + norm_dir_padrao + eps);
        float cooling_factor;
        if (progresso < 0.1f) {
            cooling_factor = 0.01f + (progresso / 0.1f) * 0.99f;
        } else {
            cooling_factor = 1.0f;
        }
        lr_efetivo = lr * std::min(envelope * cooling_factor, 1.5f);
    }
    
    double norm_dir_sq_new = 0.0;
    float wd_factor = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progresso));
    
    #pragma omp parallel for reduction(+:norm_dir_sq_new)
    for (int i = 0; i < numel; ++i) {
        float g = static_cast<float>(grad_acc[i]);
        
        float m = exp_avg_acc[i];
        float v = exp_avg_sq_acc[i];
        
        v = v * beta2 + (1.0f - beta2) * (g * g);
        exp_avg_sq_acc[i] = v;
        
        float v_hat = v / bias_correction2;
        float sqrt_v_hat = std::sqrt(v_hat);
        
        float delta = std::max(0.0f, std::abs(g) - 1.5f * sqrt_v_hat);
        float denom_brcm = sqrt_v_hat + delta + eps;
        float bakh_residual = (delta * delta) / (2.0f * denom_brcm);
        float curvature_shift = bakh_residual / (sqrt_v_hat + eps);
        float beta1_eff = beta1 * std::exp(-0.03f * curvature_shift);
        
        m = beta1_eff * m + (1.0f - beta1_eff) * g;
        exp_avg_acc[i] = m;
        
        float dir = m / (sqrt_v_hat + eps);
        norm_dir_sq_new += static_cast<double>(dir * dir);
        
        float a = lr_efetivo * dir;
        float w_val = static_cast<float>(p_acc[i]);
        float step_size = a;
        
        if (enable_brake) {
            bool explosao = std::abs(g) > (bakh_thresh * sqrt_v_hat);
            if (explosao) {
                float denom = sqrt_v_hat + std::abs(a) + eps;
                float correction = (a * a) / (2.0f * denom);
                float sign_a = std::copysign(1.0f, a);
                step_size = a - sign_a * correction;
            }
        }
        
        if (weight_decay != 0.0f) {
            w_val = w_val * (1.0f - lr * weight_decay * wd_factor);
        }
        
        p_acc[i] = static_cast<scalar_t>(w_val - step_size);
        
        if (omni_triggered) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                float w_new = static_cast<float>(p_acc[i]);
                uint32_t p_bits;
                std::memcpy(&p_bits, &w_new, 4);
                uint32_t mant = p_bits & 0x007FFFFF;
                mant = (((mant + 1) * 31337U) & 0x007FFFFF) & static_cast<uint32_t>(mask_val);
                p_bits = (p_bits & 0xFF800000U) | mant;
                std::memcpy(&w_new, &p_bits, 4);
                p_acc[i] = static_cast<scalar_t>(w_new);
            }
            exp_avg_acc[i] = 0.0f;
            exp_avg_sq_acc[i] *= 0.1f;
        }
    }
    
    direcao_acc[0] = static_cast<float>(norm_dir_sq_new);
}

void adamv_step_cpu(
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
    float bakh_thresh,
    int step,
    int D,
    bool enable_cooling,
    bool enable_brake,
    bool omni_triggered,
    int mask_val) 
{
    CHECK_INPUT(p);
    CHECK_INPUT(grad);
    CHECK_INPUT(exp_avg);
    CHECK_INPUT(exp_avg_sq);
    CHECK_INPUT(direcao);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, p.scalar_type(), "adamv_step_cpu", ([&] {
        adamv_step_cpu_template<scalar_t>(
            p, grad, exp_avg, exp_avg_sq, direcao,
            lr, beta1, beta2, eps, weight_decay, progresso, bakh_thresh, step, D, enable_cooling, enable_brake, omni_triggered, mask_val);
    }));
}

#ifdef WITH_CUDA
void adamv_step_cuda(
    at::Tensor p, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor direcao,
    float lr, float beta1, float beta2, float eps, float weight_decay, float progresso,
    float bakh_thresh_eff, int step, int D, bool enable_cooling, bool enable_brake, bool omni_triggered, int64_t punning_mask);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamv_step_cpu", &adamv_step_cpu, "AdamV Optimizer Fused CPU Kernel");
#ifdef WITH_CUDA
    m.def("adamv_step_cuda", &adamv_step_cuda, "AdamV Optimizer Fused CUDA Kernel");
#endif
}
