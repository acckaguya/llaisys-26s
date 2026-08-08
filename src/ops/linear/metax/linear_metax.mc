#include "linear_metax.hpp"
#include "../../../utils.hpp"

#include <mc_runtime.h>
#include <mcblas.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <climits>
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

static void checkMcblas(mcblasStatus_t status) {
    ASSERT(status == MCBLAS_STATUS_SUCCESS, "mcBLAS call failed");
}

static mcblasHandle_t getHandle(mcStream_t stream) {
    static thread_local mcblasHandle_t handle = [] {
        mcblasHandle_t value = nullptr;
        checkMcblas(mcblasCreate(&value));
        return value;
    } ();

    checkMcblas(mcblasSetStream(handle, stream));
    return handle;
}

static macaDataType getMcType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return MACA_R_32F;
    case LLAISYS_DTYPE_F16:
        return MACA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return MACA_R_16BF;
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
    mcStream_t stream
) {
    CHECK_ARGUMENT(
        m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
        "linear dimension exceed mcBLAS limits"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto data_type = getMcType(type);

    checkMcblas(mcblasGemmEx(
        getHandle(stream),
        MCBLAS_OP_T,
        MCBLAS_OP_N,
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
        MCBLAS_COMPUTE_32F,
        MCBLAS_GEMM_DEFAULT
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
    mcStream_t stream
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
    auto mc_stream = reinterpret_cast<mcStream_t>(stream);

    linearGemm(out, in, weight, type, m, n, k, mc_stream);

    if (bias != nullptr) {
        const size_t numel = m * n;

        switch (type) {
        case LLAISYS_DTYPE_F32:
            launchBias<float>(out, bias, numel, n, mc_stream);
            break;
        case LLAISYS_DTYPE_F16:
            launchBias<__half>(out, bias, numel, n, mc_stream);
            break;
        case LLAISYS_DTYPE_BF16:
            launchBias<maca_bfloat16>(out, bias, numel, n, mc_stream);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }

    const auto error = mcGetLastError();
    ASSERT(error == mcSuccess, mcGetErrorString(error));
}

} // namespace llaisys::ops::metax
