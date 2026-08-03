#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace {

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m,
    size_t n,
    size_t k
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;

            if (bias != nullptr) {
                sum = llaisys::utils::cast<float>(bias[j]);
            }

            for (size_t p = 0; p < k; ++p) {
                float x = llaisys::utils::cast<float>(in[i * k + p]);
                float w = llaisys::utils::cast<float>(weight[j * k + p]);

                sum += x * w;
            }

            out[i * n + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

} // namespace


namespace llaisys::ops::cpu {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            m,
            n,
            k
        );

    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            reinterpret_cast<const fp16_t *>(bias),
            m,
            n,
            k
        );

    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            reinterpret_cast<const bf16_t *>(bias),
            m,
            n,
            k
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
