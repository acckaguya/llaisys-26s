#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 声明公共的加法接口, add 接收 Tensor 的智能指针.
void add(tensor_t c, tensor_t a, tensor_t b);
}
