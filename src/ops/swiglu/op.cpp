#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(
        out->dtype(),
        gate->dtype(),
        up->dtype()
    );
    CHECK_SAME_SHAPE(
        out->shape(),
        gate->shape(),
        up->shape()
    );

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "swiglu output must be a 2D tensor"
    );
    CHECK_ARGUMENT(
        gate->ndim() == 2,
        "swiglu gate must be a 2D tensor"
    );
    CHECK_ARGUMENT(
        up->ndim() == 2,
        "swiglu up must be a 2D tensor"
    );

    ASSERT(
        out->isContiguous()
            && gate->isContiguous()
            && up->isContiguous(),
        "SwiGLU: all tensors must be contiguous."
    );

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel()
        );
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel(),
            core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops