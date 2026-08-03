#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {

void embedding(
    std::byte *out,
    const std::byte *index,     // index 中的元素数量
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,      // weight 的行数
    size_t embedding_dim,       // weight 的列数
    size_t element_size         // 每个元素的字节数
) {
    const auto *indices = reinterpret_cast<const int64_t *>(index);

    // 行大小
    size_t row_bytes = embedding_dim * element_size;

    for (size_t i = 0; i < num_indices; ++i) {
        int64_t row = indices[i];

        CHECK_ARGUMENT(row >= 0, "embedding index cannot be negative");
        CHECK_ARGUMENT(static_cast<size_t>(row) < num_embeddings, "embedding index out of range");

        const std::byte *source = weight + static_cast<size_t>(row) * row_bytes;

        std::byte *destination = out + i * row_bytes;

        std::memcpy(destination, source, row_bytes);
    }
}

} // namespace llaisys::ops::cpu
