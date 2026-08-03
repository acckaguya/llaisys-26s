#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <functional>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

// 创建一个 Tensor, 参数: shape, dtype, device_type, device_id
// create 只负责创建元数据, 分配内存, 返回 Tensor 对象, 不负责填充数据
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();    // 计算维度
    std::vector<ptrdiff_t> strides(ndim_);  // 创建 strides
    size_t stride = 1;
    // 根据 shape 推断 strides
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides}; // 创建元数据结构体
    size_t total_elems = stride;            // 记录元素总数
    size_t dtype_size = utils::dsize(dtype);// 单个元素大小

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    ptrdiff_t expected = 1;

    for (size_t i = ndim(); i > 0; i--) {
        size_t dim = i - 1;

        if (strides()[dim] != expected) {
            return false;
        }

        expected *= shape()[dim];
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(
        order.size() == this->ndim(),
        "permute order must have the same number of dimensions"
    );

    std::vector<bool> used(this->ndim(), false);

    for (size_t dim: order) {
        CHECK_ARGUMENT(
            dim < this->ndim(),
            "permute out of range"
        );

        CHECK_ARGUMENT(
            !used[dim],
            "permute order contains dulplicate dimensions"
        );

        used[dim] = true;
    }

    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());

    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = this->shape()[order[i]];
        new_strides[i] = this->strides()[order[i]];
    }

    TensorMeta new_meta{
        this->dtype(),
        new_shape,
        new_strides
    };

    return std::shared_ptr<Tensor>(
        new Tensor(
            new_meta,
            this->_storage,
            this->_offset
        )
    );
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(
        this->isContiguous(),
        "view requres a contiguous tensor"
    );

    size_t new_numel = 1;
    for (size_t dim: shape) {
        new_numel *= dim;
    }

    CHECK_ARGUMENT(
        new_numel == this->numel(),
        "view shape does not match number of elements"
    );

    std::vector<ptrdiff_t> new_strides(shape.size());

    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i > 0; i--) {
        size_t dim = i - 1;
        new_strides[dim] = stride;
        stride *= shape[dim];
    }

    TensorMeta new_meta{
        this->dtype(),
        shape,
        new_strides
    };

    return std::shared_ptr<Tensor>(
        new Tensor(
            new_meta,
            this->_storage,
            this->_offset
        )
    );
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice dimension out of range");
    CHECK_ARGUMENT(start <= end, "slice start must not exced end");
    CHECK_ARGUMENT(end <= shape()[dim], "slice end out of range");

    auto new_shape = shape();
    new_shape[dim] = end - start;

    auto new_strides = strides();

    size_t new_offset = _offset + start * strides()[dim] * elementSize();

    TensorMeta new_meta{
        this->dtype(),
        new_shape,
        new_strides
    };

    return std::shared_ptr<Tensor>(
        new Tensor(
            new_meta,
            this->_storage,
            new_offset
        )
    );
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr, "Source data is null");

    size_t bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, bytes);
    } else {
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(), src_, bytes, LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(
            new Tensor(
                this->_meta,
                this->_storage,
                this->_offset
            )
        );
    }

    auto result = Tensor::create(
        this->shape(),
        this->dtype(),
        this->deviceType(),
        this->deviceId()
    );

    size_t ndim = this->ndim();
    size_t element_size = this->elementSize();

    std::function<void(size_t, size_t, size_t)> copy_recursive;

    copy_recursive = [&](size_t dim, size_t src_offset, size_t dst_offset) {
        if (dim == ndim) {
            std::memcpy(
                result->data() + dst_offset * element_size,
                this->data() + src_offset * element_size,
                element_size
            );
            return;
        }

        for (size_t i = 0; i < this->shape()[dim]; ++i) {
            copy_recursive(
                dim + 1,
                src_offset + i * this->strides()[dim],
                dst_offset + i * result->strides()[dim]
            );
        }
    };

    copy_recursive(0, 0, 0);
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;

    for (size_t dim: shape) {
        new_numel *= dim;
    }

    CHECK_ARGUMENT(
        new_numel == this->numel(),
        "reshape shape does not match number of elements"
    );

    if (this->isContiguous()) {
        return this->view(shape);
    }

    auto contiguous = this->contiguous();
    return contiguous->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (device == -1) {
        device = this->deviceId();
    }

    if (
        this->deviceType() == device_type
        && this->deviceId() == device
    ) {
        return std::shared_ptr<Tensor>(
            new Tensor(
                this->_meta,
                this->_storage,
                this->_offset
            )
        );
    }

    auto result = Tensor::create(
        this->shape(),
        this->dtype(),
        device_type,
        device
    );

    size_t bytes = this->numel() * this->elementSize();

    core::context().setDevice(
        device_type,
        device
    );

    auto *api = core::context().runtime().api();

    if(
        this->deviceType() == LLAISYS_DEVICE_CPU
        && device_type != LLAISYS_DEVICE_CPU
    ) {
        api->memcpy_sync(
            result->data(),
            this->data(),
            bytes,
            LLAISYS_MEMCPY_H2D
        );
    } else if (
        this->deviceType() != LLAISYS_DEVICE_CPU
        && device_type == LLAISYS_DEVICE_CPU
    ) {
        api->memcpy_sync(
            result->data(),
            this->data(),
            bytes,
            LLAISYS_MEMCPY_D2H
        );
    } else {
        api->memcpy_sync(
            result->data(),
            this->data(),
            bytes,
            LLAISYS_MEMCPY_D2D
        );
    }

    return result;
}

} // namespace llaisys
