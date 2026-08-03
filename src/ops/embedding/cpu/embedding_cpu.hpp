#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void embedding(
    std::byte *out,
    const std::byte *index,     // index 中的元素数量
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,      // weight 的行数
    size_t embedding_dim,       // weight 的列数
    size_t element_size         // 每个元素的字节数
);

} // namespace llaisys::ops::cpu
