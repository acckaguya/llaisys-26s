#include "linear_nvidia.cuh"
#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <climits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

static void checkCublas(cublasStatus_t status) {
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed");
}

static cublasHandle_t getHandle(cudaStream_t stream) {
    static thread_local cublasHandle_t handle = [] {
        cublasHandle_t value = nullptr;
        checkCublas(cublasCreate(&value));
        return value;
    } ();

    checkCublas(cublasSetStream(handle, stream));
    return handle;
}

static cudaDataType_t getCudaType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

void linearGemm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream
) {
    CHECK_ARGUMENT(
        m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
        "linear dimension exceed cuBLAS limits"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto data_type = getCudaType(type);

    checkCublas(cublasGemmEx(
        getHandle(stream),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        weight,
        data_type,
        static_cast<int>(k),
        in,
        data_type,
        static_cast<int>(k),
        &beta,
        out,
        data_type,
        static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

template <typename T>
__global__ void addBiasKernel(
    T *out,
    const T *bias,
    size_t numel,
    size_t n
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numel) {
        const float value = toFloat(out[i]);
        const float bias_value = toFloat(bias[i % n]);

        out[i] = fromFloat<T>(value + bias_value);
    }
}

template <typename T>
void launchBias(
    std::byte *out,
    const std::byte *bias,
    size_t numel,
    size_t n,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);

    addBiasKernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        n
    );
}

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k,
    llaisysStream_t stream
) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    linearGemm(out, in, weight, type, m, n, k, cuda_stream);

    if (bias != nullptr) {
        const size_t numel = m * n;

        switch (type) {
        case LLAISYS_DTYPE_F32:
            launchBias<float>(out, bias, numel, n, cuda_stream);
            break;
        case LLAISYS_DTYPE_F16:
            launchBias<__half>(out, bias, numel, n, cuda_stream);
            break;
        case LLAISYS_DTYPE_BF16:
            launchBias<__nv_bfloat16>(out, bias, numel, n, cuda_stream);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }

    const auto error = cudaGetLastError();
    ASSERT(error == cudaSuccess, cudaGetErrorString(error));
}

} // namespace llaisys::ops::nvidia