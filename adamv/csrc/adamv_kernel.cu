#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int BLOCK_SIZE = 256;

__global__ void manage_norm_buffer(float* buffer, float* old_norm) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *old_norm = *buffer;
        *buffer = 0.0f;
    }
}

template <typename scalar_t>
__global__ void adamv_fused_kernel(
    scalar_t* __restrict__ params,
    const scalar_t* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    float* __restrict__ norm_sq_buffer,
    const float* __restrict__ old_norm_sq_ptr,
    float beta1, float beta2, float bias_correction2, float eps, 
    float progresso, float cooling_factor, float bakh_thresh_eff, 
    float wd_factor, float lr_max, float weight_decay, int numel, int D, int step,
    bool enable_cooling, bool enable_brake,
    bool omni_triggered, uint32_t punning_mask) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float dir = 0.0f;
    
    if (idx < numel) {
        float g = static_cast<float>(grad[idx]);
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        v = beta2 * v + (1.0f - beta2) * g * g;
        exp_avg_sq[idx] = v;

        float v_hat = v / bias_correction2;
        float sqrt_v_hat = sqrt(v_hat);

        float delta = fmaxf(0.0f, std::abs(g) - 1.5f * sqrt_v_hat);
        float denom_brcm = sqrt_v_hat + delta + eps;
        float bakh_residual = (delta * delta) / (2.0f * denom_brcm);
        
        float curvature_shift = bakh_residual / (sqrt_v_hat + eps);
        float beta1_dynamic = beta1 * std::exp(-0.03f * curvature_shift);
        
        m = beta1_dynamic * m + (1.0f - beta1_dynamic) * g;
        exp_avg[idx] = m;
        
        float bias_correction1_dynamic = 1.0f - std::pow(beta1_dynamic, static_cast<float>(step));
        float m_hat = m / bias_correction1_dynamic;
        
        dir = m_hat / (sqrt_v_hat + eps);
    }
    
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float dir_sq = (idx < numel) ? (dir * dir) : 0.0f;
    float block_sum = BlockReduce(temp_storage).Sum(dir_sq);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(norm_sq_buffer, block_sum);
    }
    
    if (idx < numel) {
        scalar_t p = params[idx];
        float g = static_cast<float>(grad[idx]);
        float v = exp_avg_sq[idx];
        
        float lr_efetivo = lr_max;
        if (enable_cooling) {
            float old_norm_sq = *old_norm_sq_ptr;
            float norm_dir = sqrt(old_norm_sq) / sqrt(static_cast<float>(D));
            float envelope = (1.0f + progresso) / (progresso + norm_dir + eps);
            lr_efetivo = lr_max * min(envelope * cooling_factor, 1.5f);
        }
        
        lr_efetivo = lr_efetivo * (1.0f + 0.01f * cosf(M_PI * 4.0f * progresso));
        
        float a = lr_efetivo * dir;
        float v_hat = v / bias_correction2;
        float sqrt_v = sqrt(v_hat);
        
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
            p = static_cast<scalar_t>(static_cast<float>(p) * (1.0f - lr_max * weight_decay));
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
    
    at::Tensor old_norm_tensor = at::empty({1}, p.options().dtype(at::kFloat));
    manage_norm_buffer<<<1, 1, 0, stream>>>(direcao.data_ptr<float>(), old_norm_tensor.data_ptr<float>());

    float cooling_factor;
    if (progresso < 0.1f) {
        cooling_factor = 0.01f + (progresso / 0.1f) * 0.99f;
    } else {
        cooling_factor = 1.0f;
    }
    float wd_factor = 0.5f * (1.0f + std::cos(M_PI * progresso));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, p.scalar_type(), "adamv_fused", [&] {
        adamv_fused_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
            p.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            direcao.data_ptr<float>(),
            old_norm_tensor.data_ptr<float>(),
            beta1, beta2, bias_correction2, eps, 
            progresso, cooling_factor, bakh_thresh_eff, wd_factor, lr, weight_decay, numel, D, step,
            enable_cooling, enable_brake,
            omni_triggered, static_cast<uint32_t>(punning_mask)
        );
    });
}
