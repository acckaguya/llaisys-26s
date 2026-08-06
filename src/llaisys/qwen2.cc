#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/model.hpp"
#include "../utils.hpp"

#include <memory>
#include <string>
#include <vector>

struct LlaisysQwen2Model {
    static constexpr size_t LAYER_WEIGHT_GROUPS = 12;

    llaisys::models::Qwen2Model impl;
    LlaisysQwen2Weights weights{};
    std::vector<std::unique_ptr<LlaisysTensor>> tensor_handles;
    std::vector<llaisysTensor_t> layer_handles;

    LlaisysQwen2Model(
        const LlaisysQwen2Meta &meta,
        llaisysDeviceType_t device,
        int device_id
    ) : impl(meta, device, device_id),
        layer_handles(LAYER_WEIGHT_GROUPS * meta.nlayer) {
        initializeWeightHandles(meta);
    }

private:
    llaisysTensor_t makeTensorHandle(const std::string &name) {
        auto handle = std::make_unique<LlaisysTensor>(
            LlaisysTensor{impl.weight(name)}
        );
        auto result = handle.get();
        tensor_handles.emplace_back(std::move(handle));
        return result;
    }

    void initializeWeightHandles(const LlaisysQwen2Meta &meta) {
        tensor_handles.reserve(3 + LAYER_WEIGHT_GROUPS * meta.nlayer);

        weights.in_embed = makeTensorHandle("model.embed_tokens.weight");
        weights.out_embed = makeTensorHandle("lm_head.weight");
        weights.out_norm_w = makeTensorHandle("model.norm.weight");

        const size_t n = meta.nlayer;
        weights.attn_norm_w = layer_handles.data();
        weights.attn_q_w = layer_handles.data() + n;
        weights.attn_q_b = layer_handles.data() + 2 * n;
        weights.attn_k_w = layer_handles.data() + 3 * n;
        weights.attn_k_b = layer_handles.data() + 4 * n;
        weights.attn_v_w = layer_handles.data() + 5 * n;
        weights.attn_v_b = layer_handles.data() + 6 * n;
        weights.attn_o_w = layer_handles.data() + 7 * n;
        weights.mlp_norm_w = layer_handles.data() + 8 * n;
        weights.mlp_gate_w = layer_handles.data() + 9 * n;
        weights.mlp_up_w = layer_handles.data() + 10 * n;
        weights.mlp_down_w = layer_handles.data() + 11 * n;

        for (size_t i = 0; i < n; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            weights.attn_norm_w[i] = makeTensorHandle(p + "input_layernorm.weight");
            weights.attn_q_w[i] = makeTensorHandle(p + "self_attn.q_proj.weight");
            weights.attn_q_b[i] = makeTensorHandle(p + "self_attn.q_proj.bias");
            weights.attn_k_w[i] = makeTensorHandle(p + "self_attn.k_proj.weight");
            weights.attn_k_b[i] = makeTensorHandle(p + "self_attn.k_proj.bias");
            weights.attn_v_w[i] = makeTensorHandle(p + "self_attn.v_proj.weight");
            weights.attn_v_b[i] = makeTensorHandle(p + "self_attn.v_proj.bias");
            weights.attn_o_w[i] = makeTensorHandle(p + "self_attn.o_proj.weight");
            weights.mlp_norm_w[i] = makeTensorHandle(p + "post_attention_layernorm.weight");
            weights.mlp_gate_w[i] = makeTensorHandle(p + "mlp.gate_proj.weight");
            weights.mlp_up_w[i] = makeTensorHandle(p + "mlp.up_proj.weight");
            weights.mlp_down_w[i] = makeTensorHandle(p + "mlp.down_proj.weight");
        }
    }

};

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice
) {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2 meta is null");
    CHECK_ARGUMENT(device_ids != nullptr, "Qwen2 device IDs are null");
    CHECK_ARGUMENT(ndevice == 1, "only one device is supported");

    return new LlaisysQwen2Model(*meta, device, device_ids[0]);
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    LlaisysQwen2Model *model
) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model is null");
    return &model->weights;
}

void llaisysQwen2ModelLoadWeight(
    LlaisysQwen2Model *model,
    const char *name,
    const void *data,
    size_t nbytes
) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model is null");
    CHECK_ARGUMENT(name != nullptr, "Qwen2 weight name is null");
    model->impl.loadWeight(name, data, nbytes);
}

size_t llaisysQwen2ModelLoadWeightCount(
    const LlaisysQwen2Model *model
) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model is null");
    return model->impl.loadedWeightCount();
}

int64_t llaisysQwen2ModelInfer(
    LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken
) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model is null");
    CHECK_ARGUMENT(token_ids != nullptr, "token IDs are null");

    return model->impl.infer(token_ids, ntoken);
}

void llaisysQwen2ModelResetCache(
    LlaisysQwen2Model *model,
    size_t capacity
) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model is null");
    model->impl.resetCache(capacity);
}

}
