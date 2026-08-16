#include <torch/extension.h>
#include <cmath>
#include <vector>

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
    bool enable_brake)
{
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    
    auto p_acc = p.data_ptr<scalar_t>();
    auto grad_acc = grad.data_ptr<scalar_t>();
    auto exp_avg_acc = exp_avg.data_ptr<float>();
    auto exp_avg_sq_acc = exp_avg_sq.data_ptr<float>();
    auto direcao_acc = direcao.data_ptr<float>();
    
    int numel = p.numel();
    double norm_dir_sq = 0.0;
    
    #pragma omp parallel for reduction(+:norm_dir_sq)
    for (int i = 0; i < numel; ++i) {
        float g = static_cast<float>(grad_acc[i]);
        
        float m = exp_avg_acc[i];
        float v = exp_avg_sq_acc[i];
        
        v = v * beta2 + (1.0f - beta2) * (g * g);
        exp_avg_sq_acc[i] = v;
        
        float v_hat = v / bias_correction2;
        float sqrt_v_hat = std::sqrt(v_hat);
        
        // BRCM: Excess Shock Isolator (delta)
        float delta = std::max(0.0f, std::abs(g) - sqrt_v_hat);
        float denom_brcm = sqrt_v_hat + delta + eps;
        float bakh_residual = (delta * delta) / (2.0f * denom_brcm);
        float curvature_shift = bakh_residual / (sqrt_v_hat + eps);
        float beta1_eff = beta1 * std::exp(-curvature_shift);
        
        float bias_correction1_dynamic = 1.0f - std::pow(beta1_eff, static_cast<float>(step));
        
        m = beta1_eff * m + (1.0f - beta1_eff) * g;
        exp_avg_acc[i] = m;
        
        float dir = (m / bias_correction1_dynamic) / (sqrt_v_hat + eps);
        direcao_acc[i] = dir;
        norm_dir_sq += static_cast<double>(dir * dir);
    }
    
    float norm_dir = std::sqrt(static_cast<float>(norm_dir_sq));
    float norm_dir_padrao = norm_dir / std::sqrt(static_cast<float>(D));
    
    float lr_efetivo = lr;
    if (enable_cooling) {
        float envelope = (1.0f + progresso) / (progresso + norm_dir_padrao + eps);
        float cooling_factor;
        if (progresso < 0.1f) {
            cooling_factor = 0.01f + (progresso / 0.1f) * 0.99f;
        } else {
            float cos_progresso = (progresso - 0.1f) / 0.9f;
            cooling_factor = 1.0f;
        }
        lr_efetivo = lr * std::min(envelope * cooling_factor, 1.5f);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < numel; ++i) {
        float g = static_cast<float>(grad_acc[i]);
        float v_hat = exp_avg_sq_acc[i] / bias_correction2;
        float dir = direcao_acc[i];
        float a = lr_efetivo * dir;
        
        float w_val = static_cast<float>(p_acc[i]);
        
        float step_size = a;
        
        if (enable_brake) {
            bool explosao = std::abs(g) > (bakh_thresh * std::sqrt(v_hat));
            if (explosao) {
                // Dimensional Mismatch Fix: parameter absolute magnitude
                float denom = std::abs(w_val) + (std::abs(a) * 2.0f) + eps;
                float correction = (a * a) / (2.0f * denom);
                float sign_a = std::copysign(1.0f, a);
                step_size = a - sign_a * correction;
            }
        }
        
        // Orthogonal Truncated Levy Flight Injection
        uint32_t seed = (static_cast<uint32_t>(step) * 31337) ^ static_cast<uint32_t>(i);
        seed = (seed * 1664525) + 1013904223;
        float u = (static_cast<float>(seed) / 4294967296.0f) * 0.999f + 0.0005f;
        float cauchy = std::tan(static_cast<float>(M_PI) * (u - 0.5f));
        cauchy = std::max(-100.0f, std::min(100.0f, cauchy));
        float sqrt_v_local = std::sqrt(v_hat);
        float levy_scale = sqrt_v_local / (sqrt_v_local + 1.0f);
        float levy_mult = 0.001f * std::max(0.0f, 1.0f - progresso);
        float levy_jump = lr_efetivo * levy_mult * cauchy * levy_scale;
        
        step_size -= levy_jump;
        
        if (weight_decay != 0.0f) {
            float wd_factor = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progresso));
            w_val = w_val * (1.0f - lr * weight_decay * wd_factor);
        }
        
        p_acc[i] = static_cast<scalar_t>(w_val - step_size);
    }
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
    bool enable_brake) 
{
    CHECK_INPUT(p);
    CHECK_INPUT(grad);
    CHECK_INPUT(exp_avg);
    CHECK_INPUT(exp_avg_sq);
    CHECK_INPUT(direcao);

    AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "adamv_step_cpu", ([&] {
        adamv_step_cpu_template<scalar_t>(
            p, grad, exp_avg, exp_avg_sq, direcao,
            lr, beta1, beta2, eps, weight_decay, progresso, bakh_thresh, step, D, enable_cooling, enable_brake);
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
