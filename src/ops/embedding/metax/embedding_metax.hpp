#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::metax {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    llaisysStream_t stream
);

} // namespace llaisys::ops::metax
