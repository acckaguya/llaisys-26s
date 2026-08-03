#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
// 声明 Tensor 类, 并定义 tensor_t 为 Tensor 类型的智能指针
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

// 张量的描述信息, 包含三个重要的信息: 数据类型, 张量形状, 步长
struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
private:
    // Tensor 的私有成员, 包括描述信息, 存储空间, Tensor 在存储空间的起始位置, 构造函数
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    // 创建一个张量
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // 基本信息接口
    // Info
    std::byte *data();                                  // 返回数据首地址
    const std::byte *data() const;                      // 返回数据首地址
    size_t ndim() const;                                // 返回维度数量
    const std::vector<size_t> &shape() const;           // 返回形状
    const std::vector<ptrdiff_t> &strides() const;      // 返回步长
    llaisysDataType_t dtype() const;                    // 返回数据类型
    llaisysDeviceType_t deviceType() const;             // 返回设备类型
    int deviceId() const;                               // 返回设备编号
    size_t numel() const;                               // 返回元素总数
    size_t elementSize() const;                         // 返回元素大小

    // 调试接口
    std::string info() const;                           // 返回 Tensor 的描述信息
    void debug() const;                                 // 打印 Tensor 的形状及数据内容

    bool isContiguous() const;                          // 判断是否连续

    // Meta Transform
    // 改变视图, 只修改元数据, 不立即复制数据
    tensor_t permute(const std::vector<size_t> &order) const;       // 交换维度
    tensor_t slice(size_t dim, size_t start, size_t end) const;     // 截取一维的某一部分
    tensor_t view(const std::vector<size_t> &shape) const;          // 在总数不变的情况下改变形状

    // Load data from host memory
    // 从外部内存加载数据
    void load(const void *src);

    // Challenging features
    // 进阶操作
    tensor_t contiguous() const;                                            // 把非连续的 Tensor 复制成连续 Tensor
    tensor_t reshape(const std::vector<size_t> &shape) const;               // 改变形状
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;    // 把 Tensor 复制到另一个设备
};

} // namespace llaisys
