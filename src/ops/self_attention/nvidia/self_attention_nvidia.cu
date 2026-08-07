#include "self_attention_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__device__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    } else {
        return __bfloat162float(value);
    }
}

template <typename T>
__device__ T fromFloat(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half_rn(value);
    } else {
        return __float2bfloat16_rn(value);
    }
}

template <typename T>
__global__ void scoreSoftmaxKernel(
    float *scores,
    const T *q,
    const T *k,
    size_t q_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t q_dim,
    float scale
) {
    __shared__ float reduced[256];

    const size_t pair = blockIdx.x;
    const size_t qi = pair / n_heads;
    const size_t qh = pair % n_heads;
    const size_t tid = threadIdx.x;

    const size_t group_size = n_heads / n_kv_heads;
    const size_t kvh = qh / group_size;
    const size_t last_key = kv_len - q_len + qi;

    const size_t q_base = (qi * n_heads + qh) * q_dim;
    const size_t score_base = pair * kv_len;

    float local_max = -INFINITY;

    for (size_t ki = tid; ki <= last_key; ki += blockDim.x) {
        const size_t k_base = (ki * n_kv_heads + kvh) * q_dim;
        float score = 0.0f;

        for (size_t d = 0; d < q_dim; ++d) {
            score += toFloat(q[q_base + d])
                   * toFloat(k[k_base + d]);
        }

        score *= scale;
        scores[score_base + ki] = score;
        local_max = fmaxf(local_max, score);
    }

    reduced[tid] = local_max;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduced[tid] = fmaxf(reduced[tid], reduced[tid + stride]);
        }
        __syncthreads();
    }

    const float max_score = reduced[0];
    float local_sum = 0.0f;

    for (size_t ki = tid; ki <= last_key; ki += blockDim.x) {
        const float value = expf(scores[score_base + ki] - max_score);
        scores[score_base + ki] = value;
        local_sum += value;
    }

    reduced[tid] = local_sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduced[tid] += reduced[tid + stride];
        }
        __syncthreads();
    }

    const float sum = reduced[0];

    for (size_t ki = tid; ki <= last_key; ki += blockDim.x) {
        scores[score_base + ki] /= sum;
    }
}

template <typename T>
__global__ void weightedValueKernel(
    T *out,
    const float *scores,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t v_dim
) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = q_len * n_heads * v_dim;

    if (index >= numel) {
        return;
    }

    const size_t d = index % v_dim;
    const size_t pair = index / v_dim;
    const size_t qi = pair / n_heads;
    const size_t qh = pair % n_heads;

    const size_t group_size = n_heads / n_kv_heads;
    const size_t kvh = qh / group_size;
    const size_t last_key = kv_len - q_len + qi;
    const size_t score_base = pair * kv_len;

    float result = 0.0f;

    for (size_t ki = 0; ki <= last_key; ++ki) {
        const size_t v_index = (ki * n_kv_heads + kvh) * v_dim + d;

        result += scores[score_base + ki] * toFloat(v[v_index]);
    }

    out[index] = fromFloat<T>(result);
}

template <typename T>
void launch(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t q_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t q_dim,
    size_t v_dim,
    float scale,
    cudaStream_t stream
) {
    const size_t score_count = q_len * n_heads * kv_len;
    float *scores = nullptr;

    auto error = cudaMallocAsync(
        reinterpret_cast<void **>(&scores),
        score_count * sizeof(float),
        stream
    );
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));

    const unsigned int score_blocks =
        static_cast<unsigned int>(q_len * n_heads);

    scoreSoftmaxKernel<T><<<score_blocks, 256, 0, stream>>>(
        scores,
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        q_len, kv_len, n_heads, n_kv_heads, q_dim, scale
    );

    const size_t output_numel = q_len * n_heads * v_dim;
    const unsigned int value_blocks =
        static_cast<unsigned int>((output_numel + 255) / 256);

    weightedValueKernel<T><<<value_blocks, 256, 0, stream>>>(
        reinterpret_cast<T *>(out),
        scores,
        reinterpret_cast<const T *>(v),
        q_len, kv_len, n_heads, n_kv_heads, v_dim
    );

    error = cudaGetLastError();
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));

    error = cudaFreeAsync(scores, stream);
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));
}

void self_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t q_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t q_dim,
    size_t v_dim,
    float scale,
    llaisysStream_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(
            out, q, k, v, q_len, kv_len, n_heads,
            n_kv_heads, q_dim, v_dim, scale, cuda_stream
        );
    case LLAISYS_DTYPE_F16:
        return launch<__half>(
            out, q, k, v, q_len, kv_len, n_heads,
            n_kv_heads, q_dim, v_dim, scale, cuda_stream
        );
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(
            out, q, k, v, q_len, kv_len, n_heads,
            n_kv_heads, q_dim, v_dim, scale, cuda_stream
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
