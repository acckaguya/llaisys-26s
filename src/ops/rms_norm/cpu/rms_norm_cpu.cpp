#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void rms_norm_(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t cols,
    float eps
) {
    for (size_t i = 0; i < rows; ++i) {
        float sum_squares = 0.0f;

        for (size_t j = 0; j < cols; ++j) {
            float value = llaisys::utils::cast<float>(in[i * cols + j]);
            sum_squares += value * value;
        }

        float mean_square = sum_squares / static_cast<float>(cols);
        float inverse_rms = 1.0f / std::sqrt(mean_square + eps);

        for (size_t j = 0; j < cols; ++j) {
            float value = llaisys::utils::cast<float>(in[i * cols + j]);
            float scale = llaisys::utils::cast<float>(weight[j]);

            out[i * cols + j] = llaisys::utils::cast<T>(value * inverse_rms * scale);
        }
    }
}

} // namespace


namespace llaisys::ops::cpu {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t cols,
    float eps
) {
    CHECK_ARGUMENT(rows > 0 && cols > 0, "rms_norm input cannot be empty");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            rows,
            cols,
            eps
        );

    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            rows,
            cols,
            eps
        );

    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            rows,
            cols,
            eps
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
