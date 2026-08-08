// MetaX (曦云 C500) 原生 MXMACA runtime API 实现。
// 与 CUDA 实现（nvidia_runtime_api.cu）完全隔离，使用原生 mc_* API。
#include "../runtime_api.hpp"

#include <mc_runtime.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llaisys::device::metax {

namespace runtime_api {
static void checkMc(mcError_t result) {
    ASSERT(result == mcSuccess, mcGetErrorString(result));
}

int getDeviceCount() {
    int count = 0;
    checkMc(mcGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    checkMc(mcSetDevice(device_id));
}

void deviceSynchronize() {
    checkMc(mcDeviceSynchronize());
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    checkMc(mcStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkMc(mcStreamDestroy(
            reinterpret_cast<mcStream_t>(stream)
        ));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    checkMc(mcStreamSynchronize(
        reinterpret_cast<mcStream_t>(stream)
    ));
}

// mcMemcpyKind 与 llaisysMemcpyKind_t 数值完全一致（H2H=0, H2D=1, D2H=2, D2D=3）
static mcMemcpyKind toMcMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return mcMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return mcMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("invalid MC memcpy kind");
    }
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkMc(mcMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkMc(mcFree(ptr));
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkMc(mcMallocHost(&ptr, size, mcMallocHostDefault));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkMc(mcFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkMc(mcMemcpy(dst, src, size, toMcMemcpyKind(kind)));
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream
) {
    checkMc(mcMemcpyAsync(
        dst,
        src,
        size,
        toMcMemcpyKind(kind),
        reinterpret_cast<mcStream_t>(stream)
    ));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
