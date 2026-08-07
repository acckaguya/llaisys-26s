#include "model.hpp"

#include "../../utils.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"

#include <cmath>
#include <vector>

namespace llaisys::models {

void Qwen2Model::addWeight(
    const std::string &name,
    const std::vector<size_t> &shape,
    llaisysDeviceType_t device,
    int device_id
) {
    // 存入权重
    auto [_, inserted] = _weights.emplace(
        name,
        Tensor::create(shape, _meta.dtype, device, device_id)
    );
    CHECK_ARGUMENT(inserted, "duplicate Qwen2 weight name");
}

Qwen2Model::Qwen2Model(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    int device_id
) : _meta(meta),
    _device(device),
    _device_id(device_id) {
    CHECK_ARGUMENT(meta.nlayer > 0, "Qwen2 layer count must be positive");
    CHECK_ARGUMENT(meta.nh > 0, "Qwen2 head count must be positive");
    CHECK_ARGUMENT(meta.nkvh > 0, "Qwen2 KV head count must be positive");
    CHECK_ARGUMENT(meta.dh > 0, "Qwen2 head dimension must be positive");
    CHECK_ARGUMENT(meta.hs == meta.nh * meta.dh, "invalid Qwen2 head dimensions");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0, "invalid Qwen2 KV head count");

    addWeight("model.embed_tokens.weight", {meta.voc, meta.hs}, device, device_id);
    addWeight("model.norm.weight", {meta.hs}, device, device_id);
    addWeight("lm_head.weight", {meta.voc, meta.hs}, device, device_id);

    const size_t q_size = meta.nh * meta.dh;
    const size_t kv_size = meta.nkvh * meta.dh;

    for (size_t i = 0; i < meta.nlayer; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";

        addWeight(p + "input_layernorm.weight", {meta.hs}, device, device_id);
        addWeight(p + "self_attn.q_proj.weight", {q_size, meta.hs}, device, device_id);
        addWeight(p + "self_attn.q_proj.bias", {q_size}, device, device_id);
        addWeight(p + "self_attn.k_proj.weight", {kv_size, meta.hs}, device, device_id);
        addWeight(p + "self_attn.k_proj.bias", {kv_size}, device, device_id);
        addWeight(p + "self_attn.v_proj.weight", {kv_size, meta.hs}, device, device_id);
        addWeight(p + "self_attn.v_proj.bias", {kv_size}, device, device_id);
        addWeight(p + "self_attn.o_proj.weight", {meta.hs, q_size}, device, device_id);
        addWeight(p + "post_attention_layernorm.weight", {meta.hs}, device, device_id);
        addWeight(p + "mlp.gate_proj.weight", {meta.di, meta.hs}, device, device_id);
        addWeight(p + "mlp.up_proj.weight", {meta.di, meta.hs}, device, device_id);
        addWeight(p + "mlp.down_proj.weight", {meta.hs, meta.di}, device, device_id);
    }
}

void Qwen2Model::loadWeight(
    const std::string &name,
    const void *data,
    size_t nbytes
) {
    auto it = _weights.find(name);
    CHECK_ARGUMENT(it != _weights.end(), "unknown Qwen2 weight");
    CHECK_ARGUMENT(!_loaded.count(name), "Qwen2 weight loaded twice");

    const size_t expected =
        it->second->numel() * it->second->elementSize();

    CHECK_ARGUMENT(nbytes == expected, "Qwen2 weight size mismatch");

    // tensor::load()
    it->second->load(data);
    _loaded.insert(name);
}

tensor_t Qwen2Model::weight(const std::string &name) const {
    auto it = _weights.find(name);
    CHECK_ARGUMENT(it != _weights.end(), "unknown Qwen2 weight");
    return it->second;
}

size_t Qwen2Model::loadedWeightCount() const {
    return _loaded.size();
}

size_t Qwen2Model::expectedWeightCount() const {
    return _weights.size();
}

void Qwen2Model::resetCache(size_t capacity) {
    CHECK_ARGUMENT(capacity > 0, "KV Cache capacity must be positive");
    CHECK_ARGUMENT(capacity <= _meta.maxseq, "KV Cache capacity exceeds model limit");

    _k_cache.clear();
    _v_cache.clear();

    _k_cache.reserve(_meta.nlayer);
    _v_cache.reserve(_meta.nlayer);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        _k_cache.emplace_back(Tensor::create(
            {capacity, _meta.nkvh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id
        ));

        _v_cache.emplace_back(Tensor::create(
            {capacity, _meta.nkvh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id
        ));
    }

    _cache_len = 0;
    _cache_capacity = capacity;
}

int64_t Qwen2Model::infer(
    const int64_t *token_ids,
    size_t ntoken
) {
    CHECK_ARGUMENT(token_ids != nullptr, "token IDs are null");
    CHECK_ARGUMENT(ntoken > 0, "token count must be positive");
    CHECK_ARGUMENT(
        loadedWeightCount() == expectedWeightCount(),
        "Qwen2 weights are incomplete"
    );
    CHECK_ARGUMENT(_cache_capacity > 0, "KV Cache is not initialized");
    const size_t cache_start = _cache_len;
    const size_t cache_end = cache_start + ntoken;
    CHECK_ARGUMENT(cache_end <= _cache_capacity, "KV Cache capacity is insufficient");

    auto embedding_weight = weight("model.embed_tokens.weight");

    auto ids = Tensor::create(
        {ntoken},
        LLAISYS_DTYPE_I64,
        embedding_weight->deviceType(),
        embedding_weight->deviceId()
    );
    ids->load(token_ids);

    auto hidden = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        embedding_weight->deviceType(),
        embedding_weight->deviceId()
    );

    ops::embedding(hidden, ids, embedding_weight);

    const auto device = hidden->deviceType();
    const auto device_id = hidden->deviceId();

    core::context().setDevice(device, device_id);
    auto &runtime = core::context().runtime();
    const auto cache_copy_kind =
        device == LLAISYS_DEVICE_CPU
            ? LLAISYS_MEMCPY_H2H
            : LLAISYS_MEMCPY_D2D;

    const size_t q_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;
    const float scale = 1.0f / std::sqrt(
        static_cast<float>(_meta.dh)
    );

    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        positions[i] = static_cast<int64_t>(cache_start + i);
    }

    auto pos_ids = Tensor::create(
        {ntoken},
        LLAISYS_DTYPE_I64,
        device,
        device_id
    );
    pos_ids->load(positions.data());


    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        const std::string prefix =
            "model.layers." + std::to_string(layer) + ".";

        auto norm = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            device,
            device_id
        );

        ops::rms_norm(
            norm,
            hidden,
            weight(prefix + "input_layernorm.weight"),
            _meta.epsilon
        );

        auto q = Tensor::create(
            {ntoken, q_size},
            _meta.dtype,
            device,
            device_id
        );

        auto k = Tensor::create(
            {ntoken, kv_size},
            _meta.dtype,
            device,
            device_id
        );

        auto v = Tensor::create(
            {ntoken, kv_size},
            _meta.dtype,
            device,
            device_id
        );

        ops::linear(
            q,
            norm,
            weight(prefix + "self_attn.q_proj.weight"),
            weight(prefix + "self_attn.q_proj.bias")
        );

        ops::linear(
            k,
            norm,
            weight(prefix + "self_attn.k_proj.weight"),
            weight(prefix + "self_attn.k_proj.bias")
        );

        ops::linear(
            v,
            norm,
            weight(prefix + "self_attn.v_proj.weight"),
            weight(prefix + "self_attn.v_proj.bias")
        );

        auto q_heads = q->view({
            ntoken,
            _meta.nh,
            _meta.dh
        });

        auto k_heads = k->view({
            ntoken,
            _meta.nkvh,
            _meta.dh
        });

        auto v_heads = v->view({
            ntoken,
            _meta.nkvh,
            _meta.dh
        });

        auto q_rope = Tensor::create(
            {ntoken, _meta.nh, _meta.dh},
            _meta.dtype,
            device,
            device_id
        );

        auto k_rope = Tensor::create(
            {ntoken, _meta.nkvh, _meta.dh},
            _meta.dtype,
            device,
            device_id
        );

        ops::rope(q_rope, q_heads, pos_ids, _meta.theta);
        ops::rope(k_rope, k_heads, pos_ids, _meta.theta);

        // KV Cache
        auto k_write = _k_cache[layer]->slice(0, cache_start, cache_end);
        auto v_write = _v_cache[layer]->slice(0, cache_start, cache_end);

        const size_t cache_bytes = ntoken * kv_size * k_rope->elementSize();

        runtime.api()->memcpy_async(
            k_write->data(),
            k_rope->data(),
            cache_bytes,
            cache_copy_kind,
            runtime.stream()
        );
        runtime.api()->memcpy_async(
            v_write->data(),
            v_heads->data(),
            cache_bytes,
            cache_copy_kind,
            runtime.stream()
        );

        auto k_all = _k_cache[layer]->slice(0, 0, cache_end)->contiguous();
        auto v_all = _v_cache[layer]->slice(0, 0, cache_end)->contiguous();

        auto attention = Tensor::create(
            {ntoken, _meta.nh, _meta.dh},
            _meta.dtype,
            device,
            device_id
        );

        ops::self_attention(
            attention,
            q_rope,
            k_all,
            v_all,
            scale
        );

        // 将多头结果恢复为二维
        auto attention_flat = attention->view({
            ntoken,
            q_size
        });

        // 执行 Attention 输出投影
        auto attention_proj = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            device,
            device_id
        );

        ops::linear(
            attention_proj,
            attention_flat,
            weight(prefix + "self_attn.o_proj.weight"),
            nullptr
        );

        // 执行残差连接
        auto hidden_after_attention = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            device,
            device_id
        );

        ops::add(
            hidden_after_attention,
            hidden,
            attention_proj
        );

        auto mlp_norm = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            device,
            device_id
        );

        ops::rms_norm(
            mlp_norm,
            hidden_after_attention,
            weight(prefix + "post_attention_layernorm.weight"),
            _meta.epsilon
        );

        // 创建 gate 和 up
        auto gate = Tensor::create(
            {ntoken, _meta.di},
            _meta.dtype,
            device,
            device_id
        );

        auto up = Tensor::create(
            {ntoken, _meta.di},
            _meta.dtype,
            device,
            device_id
        );

        ops::linear(
            gate,
            mlp_norm,
            weight(prefix + "mlp.gate_proj.weight"),
            nullptr
        );

        ops::linear(
            up,
            mlp_norm,
            weight(prefix + "mlp.up_proj.weight"),
            nullptr
        );

        // 执行 SwiGLU
        auto activated = Tensor::create(
            {ntoken, _meta.di},
            _meta.dtype,
            device,
            device_id
        );

        ops::swiglu(activated, gate, up);

        // 执行 down projection
        auto mlp_output = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            device,
            device_id
        );

        ops::linear(
            mlp_output,
            activated,
            weight(prefix + "mlp.down_proj.weight"),
            nullptr
        );

        // 加入第二个残差
        auto layer_output = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            device,
            device_id
        );

        ops::add(
            layer_output,
            hidden_after_attention,
            mlp_output
        );

        hidden = layer_output;
    }

    _cache_len = cache_end;

    // 最终 RMSNorm
    auto final_norm = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        device,
        device_id
    );

    ops::rms_norm(
        final_norm,
        hidden,
        weight("model.norm.weight"),
        _meta.epsilon
    );

    // 只取最后一个 token
    auto last_hidden = final_norm->slice(
        0,
        ntoken - 1,
        ntoken
    );

    // 计算词表 logits
    auto logits = Tensor::create(
        {1, _meta.voc},
        _meta.dtype,
        device,
        device_id
    );

    ops::linear(
        logits,
        last_hidden,
        weight("lm_head.weight"),
        nullptr
    );

    // 创建 argmax 输出
    auto max_idx = Tensor::create(
        {1},
        LLAISYS_DTYPE_I64,
        device,
        device_id
    );

    auto max_value = Tensor::create(
        {1},
        _meta.dtype,
        device,
        device_id
    );

    ops::argmax(
        max_idx,
        max_value,
        logits
    );

    // 把 token ID 返回 CPU
    auto host_idx = max_idx->to(
        LLAISYS_DEVICE_CPU,
        0
    );

    const int64_t next_token =
        *reinterpret_cast<const int64_t *>(host_idx->data());

    return next_token;
}
} // namespace llaisys::models
