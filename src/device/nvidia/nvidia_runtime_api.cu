#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace runtime_api {
static void checkCuda(cudaError_t result) {
    ASSERT(result == cudaSuccess, cudaGetErrorString(result));
}

int getDeviceCount() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    checkCuda(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkCuda(cudaStreamDestroy(
            reinterpret_cast<cudaStream_t>(stream)
        ));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(stream)
    ));
}

static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("invalid CUDA memcpy kind");
    }
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkCuda(cudaFree(ptr));
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkCuda(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkCuda(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream
) {
    checkCuda(cudaMemcpyAsync(
        dst,
        src,
        size,
        toCudaMemcpyKind(kind),
        reinterpret_cast<cudaStream_t>(stream)
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
} // namespace llaisys::device::nvidia
