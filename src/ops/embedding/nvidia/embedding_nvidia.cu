#include "embedding_nvidia.cuh"
#include "../../../utils.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embeddingKernel(
    T *out,
    const int64_t *indices,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = num_indices * embedding_dim;

    if (i >= numel) {
        return;
    }

    const size_t token = i / embedding_dim;
    const size_t column = i % embedding_dim;
    const int64_t row = indices[token];

    if (row < 0 || static_cast<size_t>(row) >= num_embeddings) {
        return;
    }

    out[i] = weight[
        static_cast<size_t>(row) * embedding_dim + column
    ];
}

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    llaisysStream_t stream
) {
    const size_t numel = num_indices * embedding_dim;
    if (numel == 0) {
        return;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto indices = reinterpret_cast<const int64_t *>(index);

#define LAUNCH_EMBEDDING(T) \
    embeddingKernel<T><<<blocks, threads, 0, cuda_stream>>>( \
        reinterpret_cast<T *>(out), \
        indices, \
        reinterpret_cast<const T *>(weight), \
        num_indices, \
        num_embeddings, \
        embedding_dim)

    switch (type) {
    case LLAISYS_DTYPE_F32:
        LAUNCH_EMBEDDING(float);
        break;
    case LLAISYS_DTYPE_F16:
        LAUNCH_EMBEDDING(__half);
        break;
    case LLAISYS_DTYPE_BF16:
        LAUNCH_EMBEDDING(__nv_bfloat16);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

#undef LAUNCH_EMBEDDING

    const auto error = cudaGetLastError();
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia
