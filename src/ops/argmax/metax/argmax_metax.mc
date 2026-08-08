#include "argmax_metax.hpp"
#include "../../../utils.hpp"

#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace llaisys::ops::metax {

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
__global__ void argmaxKernel(
    int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel
) {
    __shared__ float shared_values[256];
    __shared__ int64_t shared_indices[256];

    const size_t tid = threadIdx.x;
    float best_value = -INFINITY;
    int64_t best_index = static_cast<int64_t>(numel);

    for (size_t i = tid; i < numel; i += blockDim.x) {
        const float value = toFloat(vals[i]);

        if (value > best_value
            ||(best_value == value
                && static_cast<int64_t>(i) < best_index)) {
            best_value = value;
            best_index = static_cast<int64_t>(i);
        }
    }

    shared_values[tid] = best_value;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            const float other_value = shared_values[tid + stride];
            const int64_t other_index = shared_indices[tid + stride];

            if (other_value > shared_values[tid]
                || (other_value == shared_values[tid]
                    && other_index < shared_indices[tid])) {
                shared_values[tid] = other_value;
                shared_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = shared_indices[0];
        *max_val = fromFloat<T>(shared_values[0]);
    }
}

template <typename T>
void launch(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel,
    mcStream_t stream
) {
    argmaxKernel<T><<<1, 256, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel
    );
}

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream
) {
    CHECK_ARGUMENT(numel > 0, "argmax input cannot be empty");

    auto mc_stream = reinterpret_cast<mcStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch<float>(max_idx, max_val, vals, numel, mc_stream);
        break;
    case LLAISYS_DTYPE_F16:
        launch<__half>(max_idx, max_val, vals, numel, mc_stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launch<maca_bfloat16>(max_idx, max_val, vals, numel, mc_stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    const auto error = mcGetLastError();
    ASSERT(error == mcSuccess, mcGetErrorString(error));
}

} // namespace llaisys::ops::metax
