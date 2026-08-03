#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace {

template <typename T>
void rope_(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta
) {
    const size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; ++s) {
        const float position = static_cast<float>(pos_ids[s]);

        for (size_t j = 0; j < half_dim; ++j) {
            const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
            const float angle = position / std::pow(theta, exponent);
            const float sin_value = std::sin(angle);
            const float cos_value = std::cos(angle);

            for (size_t h = 0; h < n_heads; ++h) {
                const size_t base = (s * n_heads + h) * head_dim;
                const size_t a_index = base + j;
                const size_t b_index = base + half_dim + j;

                const float a = llaisys::utils::cast<float>(in[a_index]);
                const float b = llaisys::utils::cast<float>(in[b_index]);

                out[a_index] = llaisys::utils::cast<T>(
                    a * cos_value - b * sin_value
                );
                out[b_index] = llaisys::utils::cast<T>(
                    b * cos_value + a * sin_value
                );
            }
        }
    }
}

} // namespace


namespace llaisys::ops::cpu {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta
) {
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            positions, seq_len, n_heads, head_dim, theta
        );

    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            positions, seq_len, n_heads, head_dim, theta
        );

    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            positions, seq_len, n_heads, head_dim, theta
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
