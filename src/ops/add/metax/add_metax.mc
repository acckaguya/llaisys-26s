#include "add_metax.hpp"
#include "../../../utils.hpp"

#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

namespace llaisys::ops::metax {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        c[i] = a[i] + b[i];
    }
}

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream
) {
    if (numel == 0) {
        return;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto mc_stream = reinterpret_cast<mcStream_t>(stream);

#define LAUNCH_ADD(T) \
    addKernel<T><<<blocks, threads, 0, mc_stream>>>( \
    reinterpret_cast<T *>(c), \
    reinterpret_cast<const T *>(a), \
    reinterpret_cast<const T *>(b), \
    numel)

    switch (type) {
    case LLAISYS_DTYPE_F32:
        LAUNCH_ADD(float);
        break;
    case LLAISYS_DTYPE_F16:
        LAUNCH_ADD(__half);
        break;
    case LLAISYS_DTYPE_BF16:
        LAUNCH_ADD(maca_bfloat16);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
#undef LAUNCH_ADD

    const auto error = mcGetLastError();
    ASSERT(error == mcSuccess, mcGetErrorString(error));
}

} // namespace llaisys::ops::metax
