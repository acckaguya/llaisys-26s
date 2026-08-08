#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/embedding_metax.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(
        index->ndim() == 1,
        "embedding index must be a 1D tensor"
    );
    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "embedding weight must be a 2D tensor"
    );
    CHECK_ARGUMENT(
    out->ndim() == 2,
    "embedding output must be a 2D tensor"
    );
    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64,
        "embedding index must use int64 dtype"
    );
    CHECK_SAME_DTYPE(
        out->dtype(),
        weight->dtype()
    );
    CHECK_ARGUMENT(
        out->shape()[0] == index->shape()[0],
        "embedding output row count does not match index count"
    );
    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[1],
        "embedding output dimension does not match weight"
    );
    ASSERT(
        out->isContiguous() && index->isContiguous() && weight->isContiguous(),
        "embedding all tensors must be contiguous."
    );

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            weight->elementSize()
        );
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            weight->elementSize()
        );

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            core::context().runtime().stream()
        );
#endif

#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            core::context().runtime().stream()
        );
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
