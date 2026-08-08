#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/self_attention_metax.hpp"
#endif

namespace llaisys::ops {

void self_attention(
    tensor_t attn_val,
    tensor_t q,
    tensor_t k,
    tensor_t v,
    float scale
) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(
        attn_val->dtype(), q->dtype(), k->dtype(), v->dtype()
    );

    CHECK_ARGUMENT(attn_val->ndim() == 3, "attention output must be 3D");
    CHECK_ARGUMENT(q->ndim() == 3, "attention query must be 3D");
    CHECK_ARGUMENT(k->ndim() == 3, "attention key must be 3D");
    CHECK_ARGUMENT(v->ndim() == 3, "attention value must be 3D");

    const size_t q_len = q->shape()[0];
    const size_t kv_len = k->shape()[0];
    const size_t n_heads = q->shape()[1];
    const size_t n_kv_heads = k->shape()[1];
    const size_t q_dim = q->shape()[2];
    const size_t v_dim = v->shape()[2];

    CHECK_ARGUMENT(
        q_len > 0 && kv_len > 0,
        "attention sequence lengths must be positive"
    );
    CHECK_ARGUMENT(
        n_heads > 0 && n_kv_heads > 0,
        "attention head counts must be positive"
    );
    CHECK_ARGUMENT(
        q_dim > 0 && v_dim > 0,
        "attention head dimensions must be positive"
    );
    CHECK_ARGUMENT(
        kv_len >= q_len,
        "attention KV length must be at least query length"
    );
    CHECK_ARGUMENT(
        k->shape()[0] == v->shape()[0],
        "attention key and value lengths do not match"
    );
    CHECK_ARGUMENT(
        k->shape()[1] == v->shape()[1],
        "attention key and value head counts do not match"
    );
    CHECK_ARGUMENT(
        q_dim == k->shape()[2],
        "attention query and key dimensions do not match"
    );
    CHECK_ARGUMENT(
        n_heads % n_kv_heads == 0,
        "query head count must be divisible by KV head count"
    );
    CHECK_ARGUMENT(
        attn_val->shape()[0] == q_len
            && attn_val->shape()[1] == n_heads
            && attn_val->shape()[2] == v_dim,
        "attention output shape is incorrect"
    );

    ASSERT(
        attn_val->isContiguous()
            && q->isContiguous()
            && k->isContiguous()
            && v->isContiguous(),
        "SelfAttention: all tensors must be contiguous."
    );

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(),
            q_len, kv_len, n_heads, n_kv_heads,
            q_dim, v_dim, scale
        );
    }

    core::context().setDevice(
        attn_val->deviceType(),
        attn_val->deviceId()
    );

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(),
            q_len, kv_len, n_heads, n_kv_heads,
            q_dim, v_dim, scale
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q_len,
            kv_len,
            n_heads,
            n_kv_heads,
            q_dim,
            v_dim,
            scale,
            core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q_len,
            kv_len,
            n_heads,
            n_kv_heads,
            q_dim,
            v_dim,
            scale,
            core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
