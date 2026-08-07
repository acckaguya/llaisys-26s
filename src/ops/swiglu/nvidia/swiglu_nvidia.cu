#include "swiglu_nvidia.cuh"
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
__global__ void swigluKernel(
    T *out,
    const T *gate,
    const T *up,
    size_t numel
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numel) {
        const float g = toFloat(gate[i]);
        const float u = toFloat(up[i]);
        const float sigmoid = 1.0f / (1.0f + expf(-g));

        out[i] = fromFloat<T>(u * g * sigmoid);
    }
}

template <typename T>
void launch(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);

    swigluKernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel
    );
}

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream
) {
    if (numel == 0) {
        return;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch<float>(out, gate, up, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launch<__half>(out, gate, up, numel, cuda_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launch<__nv_bfloat16>(out, gate, up, numel, cuda_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const auto error = cudaGetLastError();
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia