#include "rope_nvidia.cuh"
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
__global__ void ropeKernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t n_heads,
    size_t head_dim,
    float theta,
    size_t pair_count
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pair_count) {
        return;
    }

    const size_t half_dim = head_dim / 2;
    const size_t j = i % half_dim;
    const size_t head = (i / half_dim) % n_heads;
    const size_t seq = i / (half_dim * n_heads);

    const size_t base = (seq * n_heads + head) * head_dim;
    const size_t a_index = base + j;
    const size_t b_index = base + half_dim + j;

    const float position = static_cast<float>(pos_ids[seq]);
    const float exponent =
        2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
    const float angle = position / powf(theta, exponent);

    const float sin_value = sinf(angle);
    const float cos_value = cosf(angle);
    const float a = toFloat(in[a_index]);
    const float b = toFloat(in[b_index]);

    out[a_index] = fromFloat<T>(
        a * cos_value - b * sin_value
    );
    out[b_index] = fromFloat<T>(
        b * cos_value + a * sin_value
    );
}

template <typename T>
void launch(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta,
    cudaStream_t stream
) {
    const size_t pair_count = seq_len * n_heads * (head_dim / 2);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pair_count + threads - 1) /threads);

    ropeKernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        n_heads,
        head_dim,
        theta,
        pair_count
    );
}

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta,
    llaisysStream_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch<float>(out, in, pos_ids, seq_len, n_heads, head_dim, theta, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launch<__half>(out, in, pos_ids, seq_len, n_heads, head_dim, theta, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launch<__nv_bfloat16>(out, in, pos_ids, seq_len, n_heads, head_dim, theta, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const auto error = cudaGetLastError();
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia
