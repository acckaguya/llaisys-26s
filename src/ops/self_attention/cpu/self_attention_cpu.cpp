#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename T>
void self_attention_(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t q_dim,
    size_t v_dim,
    float scale
) {
    const size_t group_size = n_heads / n_kv_heads;
    std::vector<float> scores(kv_len);

    for (size_t qi = 0; qi < q_len; ++qi) {
        const size_t last_key = kv_len - q_len + qi;

        for (size_t qh = 0; qh < n_heads; ++qh) {
            const size_t kvh = qh / group_size;
            const size_t q_base = (qi * n_heads + qh) * q_dim;

            float max_score = std::numeric_limits<float>::lowest();

            // 计算 QK^T * scale, 并寻找 softmax 最大值
            for (size_t ki = 0; ki <= last_key; ++ki) {
                const size_t k_base = (ki * n_kv_heads + kvh) * q_dim;
                float score = 0.0f;

                for (size_t d = 0; d < q_dim; ++d) {
                    const float q_value = llaisys::utils::cast<float>(q[q_base + d]);
                    const float k_value = llaisys::utils::cast<float>(k[k_base + d]);

                    score += q_value * k_value;
                }

                score *= scale;
                scores[ki] = score;
                max_score = std::max(max_score, score);
            }

            // 数值稳定的 softmax
            float sum_exp = 0.0f;

            for (size_t ki = 0; ki <= last_key; ++ki) {
                scores[ki] = std::exp(scores[ki] - max_score);
                sum_exp += scores[ki];
            }

            for (size_t ki = 0; ki <= last_key; ++ki) {
                scores[ki] /= sum_exp;
            }

            // softmax(QK^T) * V
            const size_t out_base = (qi * n_heads + qh) * v_dim;

            for (size_t d = 0; d < v_dim; ++d) {
                float result = 0.0f;

                for (size_t ki = 0; ki <= last_key; ++ki) {
                    const size_t v_index = (ki * n_kv_heads + kvh) * v_dim + d;
                    const float v_value = llaisys::utils::cast<float>(v[v_index]);

                    result += scores[ki] * v_value;
                }

                out[out_base + d] = llaisys::utils::cast<T>(result);
            }
        }
    }
}

} // namespace


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
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            q_len, kv_len, n_heads, n_kv_heads,
            q_dim, v_dim, scale
        );

    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(q),
            reinterpret_cast<const fp16_t *>(k),
            reinterpret_cast<const fp16_t *>(v),
            q_len, kv_len, n_heads, n_kv_heads,
            q_dim, v_dim, scale
        );

    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(q),
            reinterpret_cast<const bf16_t *>(k),
            reinterpret_cast<const bf16_t *>(v),
            q_len, kv_len, n_heads, n_kv_heads,
            q_dim, v_dim, scale
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu