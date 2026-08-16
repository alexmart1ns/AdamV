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
    const scalar_t* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ direcao_buffer,
    float beta1, float beta2, float bias_correction2, float eps, 
    int numel) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        float g = static_cast<float>(grad[idx]);
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        v = beta2 * v + (1.0f - beta2) * g * g;
        exp_avg_sq[idx] = v;

        float v_hat = v / bias_correction2;
        float sqrt_v_hat = sqrtf(v_hat);

        float delta = fmaxf(0.0f, std::abs(g) - 1.5f * sqrt_v_hat);
        float denom_brcm = sqrt_v_hat + delta + eps;
        float bakh_residual = (delta * delta) / (2.0f * denom_brcm);
        
        float curvature_shift = bakh_residual / (sqrt_v_hat + eps);
        float beta1_eff = beta1 * expf(-0.03f * curvature_shift);
        
        m = beta1_eff * m + (1.0f - beta1_eff) * g;
        exp_avg[idx] = m;
        
        direcao_buffer[idx] = m / (sqrt_v_hat + eps);
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
    float bias_correction2, float eps, 
    float progresso, float cooling_factor, float bakh_thresh_eff, 
    float wd_factor, float lr_max, float weight_decay, int numel, int D,
    bool enable_cooling, bool enable_brake,
    bool omni_triggered, uint32_t punning_mask) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dir = direcao_buffer[idx];
        scalar_t p = params[idx];
        float g = static_cast<float>(grad[idx]);
        float v = exp_avg_sq[idx];
        
        float lr_efetivo = lr_max;
        if (enable_cooling) {
            float norm = *norm_tensor_ptr;
            float norm_dir = norm / sqrtf(static_cast<float>(D));
            float envelope = (1.0f + progresso) / (progresso + norm_dir + eps);
            lr_efetivo = lr_max * fminf(envelope * cooling_factor, 1.5f);
        }
        
        float a = lr_efetivo * dir;
        float v_hat = v / bias_correction2;
        float sqrt_v = sqrtf(v_hat);
        
        float step_size = a;
        if (enable_brake) {
            bool explosao_mask = std::abs(g) > (bakh_thresh_eff * sqrt_v);
            if (explosao_mask) {
                float denom = sqrt_v + std::abs(a) + eps;
                float correction = (a * a) / (2.0f * denom);
                float bakhshali_brake = a - copysignf(1.0f, a) * correction;
                step_size = bakhshali_brake;
            }
        }
        
        if (weight_decay != 0.0f) {
            p = static_cast<scalar_t>(static_cast<float>(p) * (1.0f - lr_max * weight_decay * wd_factor));
        }
        
        params[idx] = static_cast<scalar_t>(static_cast<float>(p) - step_size);
        
        if (omni_triggered) {
            if constexpr (std::is_same<scalar_t, float>::value) {
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
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, p.scalar_type(), "adamv_prepare", [&] {
        adamv_prepare_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            direcao.data_ptr<float>(),
            beta1, beta2, bias_correction2, eps, numel
        );
    });

    at::Tensor norm_tensor = at::linalg_norm(direcao);

    float cooling_factor;
    if (progresso < 0.1f) {
        cooling_factor = 0.01f + (progresso / 0.1f) * 0.99f;
    } else {
        cooling_factor = 1.0f;
    }
    float wd_factor = 0.5f * (1.0f + std::cos(M_PI * progresso));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, p.scalar_type(), "adamv_update", [&] {
        adamv_update_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            p.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            direcao.data_ptr<float>(),
            norm_tensor.data_ptr<float>(),
            bias_correction2, eps, 
            progresso, cooling_factor, bakh_thresh_eff, wd_factor, lr, weight_decay, numel, D,
            enable_cooling, enable_brake,
            omni_triggered, static_cast<uint32_t>(punning_mask)
        );
    });
}
