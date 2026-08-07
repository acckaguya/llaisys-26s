#include "rms_norm_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t cols,
    float eps
) {
    __shared__ float values[256];

    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t row_start = row * cols;

    float sum = 0.0f;

    for (size_t col = tid; col < cols; col += blockDim.x) {
        const float value = toFloat(in[row_start + col]);
        sum += value * value;
    }

    values[tid] = sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        values[0] = rsqrtf(values[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const float inverse_rms = values[0];

    for (size_t col = tid; col < cols; col += blockDim.x) {
        const float value = toFloat(in[row_start + col]);
        const float scale = toFloat(weight[col]);

        out[row_start + col] = fromFloat<T>(value * inverse_rms * scale);
    }
}

template <typename T>
void launch(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t cols,
    float eps,
    cudaStream_t stream
) {
    constexpr int threads = 256;

    rmsNormKernel<T><<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        cols,
        eps
    );
}

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t cols,
    float eps,
    llaisysStream_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch<float>(out, in, weight, rows, cols, eps, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launch<__half>(out, in, weight, rows, cols, eps, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launch<__nv_bfloat16>(out, in, weight, rows, cols, eps, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const auto error = cudaGetLastError();
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia
