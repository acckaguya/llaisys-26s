#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_metax.hpp"
#endif


namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "linear output must be a 2D tensor"
    );
    CHECK_ARGUMENT(
        in->ndim() == 2,
        "linear input must be a 2D tensor"
    );
    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "linear weight must be a 2D tensor"
    );

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];

    CHECK_ARGUMENT(
        weight->shape()[1] == k,
        "linear input and weight dimensions do not match"
    );

    CHECK_ARGUMENT(
        out->shape()[0] == m && out->shape()[1] == n,
        "linear output shape is incorrect"
    );

    if (bias != nullptr) {
        CHECK_ARGUMENT(
            bias->ndim() == 1,
            "linear bias must be a 1D tensor"
        );

        CHECK_ARGUMENT(
            bias->shape()[0] == n,
            "linear bias shape is incorrect"
        );

        CHECK_ARGUMENT(
            bias->dtype() == out->dtype(),
            "linear bias dtype does not match output"
        );

        CHECK_ARGUMENT(
            bias->deviceType() == out->deviceType()
                && bias->deviceId() == out->deviceId(),
            "linear bias must be on the same device"
        );

        ASSERT(
            bias->isContiguous(),
            "Linear: bias must be contiguous."
        );
    }

    ASSERT(
        out->isContiguous() && in->isContiguous() && weight->isContiguous(),
        "Linear: all tensors must be contiguous."
    );

    const std::byte *bias_data = bias != nullptr ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k
        );
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k
        );

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k,
            core::context().runtime().stream()
        );
#endif

#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k,
            core::context().runtime().stream()
        );
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
