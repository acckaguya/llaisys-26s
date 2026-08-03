#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void self_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t q_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t q_dim,
    size_t v_dim,
    float scale
);

} // namespace llaisys::ops::cpu