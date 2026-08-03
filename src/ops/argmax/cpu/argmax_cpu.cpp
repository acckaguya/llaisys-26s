#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

namespace {

template <typename T>
void argmax_(
    int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel
) {
    size_t best_index = 0;
    float best_value = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        float current_value = llaisys::utils::cast<float>(vals[i]);

        if (current_value > best_value) {
            best_value = current_value;
            best_index = i;
        }
    }

    *max_idx = static_cast<int64_t>(best_index);
    *max_val = llaisys::utils::cast<T>(best_value);
}

} // namespace


namespace llaisys::ops::cpu {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel
) {
    CHECK_ARGUMENT(numel > 0, "argmax input cannot be empty");

    auto *idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(
            idx_ptr,
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel
        );

    case LLAISYS_DTYPE_F16:
        return argmax_(
            idx_ptr,
            reinterpret_cast<fp16_t *>(max_val),
            reinterpret_cast<const fp16_t *>(vals),
            numel
        );

    case LLAISYS_DTYPE_BF16:
        return argmax_(
            idx_ptr,
            reinterpret_cast<bf16_t *>(max_val),
            reinterpret_cast<const bf16_t *>(vals),
            numel
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
