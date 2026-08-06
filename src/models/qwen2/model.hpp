#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llaisys::models {

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    std::unordered_map<std::string, tensor_t> _weights;
    std::unordered_set<std::string> _loaded;

    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

    size_t _cache_len = 0;
    size_t _cache_capacity = 0;

    void addWeight(
        const std::string &name,
        const std::vector<size_t> &shape,
        llaisysDeviceType_t device,
        int device_id
    );

public:
    Qwen2Model(
        const LlaisysQwen2Meta &meta,
        llaisysDeviceType_t device,
        int device_id
    );

    void loadWeight(
        const std::string &name,
        const void *data,
        size_t nbytes
    );

    tensor_t weight(const std::string &name) const;
    size_t loadedWeightCount() const;
    size_t expectedWeightCount() const;

    int64_t infer(const int64_t *token_ids, size_t ntoken);

    void resetCache(size_t capacity);
};

} // namespace llaisys::models
