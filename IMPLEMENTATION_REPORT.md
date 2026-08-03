# LLAISYS 实现过程与结果报告

## 公共环境与模型

- Conda 环境：`llaisys`
- Python：3.11.15
- PyTorch：2.13.0
- Transformers：5.14.0
- Xmake：3.0.9
- GCC/G++：11.4.0
- 设备：CPU 可用；CUDA 因 NVIDIA 驱动版本与 PyTorch CUDA 版本不匹配，暂不可用。
- 测试模型：`DeepSeek-R1-Distill-Qwen-1.5B`
- 模型权重和 tokenizer 已完整下载，并可由 Transformers 离线加载。

## 作业 0：环境、构建与模型

### 0.1 环境与构建

完成并验证了基础环境与模型加载：

```bash
conda activate llaisys
xmake
python test/test_runtime.py --device cpu
python test/test_infer.py --model <model_path>
```

CPU runtime 测试通过，项目可以正常编译。

### 0.2 测试模型

PyTorch 基准模型可以成功加载权重并完成推理，权重加载进度达到 `339/339`。

### 0.3 作业结果

作业 0 已完成。

## 作业 1：Tensor 基础功能

### 1.1 实现内容

在 `src/tensor/tensor.cpp` 中完成了以下功能：

- `load`
- `isContiguous`
- `view`
- `permute`
- `slice`

实现要点：

- `load` 将外部数据复制到 Tensor 存储空间；
- `isContiguous` 根据 shape 和 strides 判断内存布局；
- `view` 只修改 shape 和 strides，不复制数据；
- `permute` 重新排列 shape 和 strides，共享原 storage；
- `slice` 修改切片维度的 shape，并根据 stride 调整字节 offset，共享原 storage。

### 1.2 验证结果

每次更新共享库后使用以下命令验证：

```bash
xmake
xmake install
conda run -n llaisys python test/test_tensor.py
```

最终结果：

```text
load          通过
isContiguous  通过
view          通过
permute       通过
slice         通过
Test passed!
```

### 1.3 作业结果

作业 1 的 Tensor 基础测试已通过。

## 作业 2：CPU 算子

状态：进行中。2026-07-30 开始实现 CPU 算子，2026-07-31 完成并验证 RMSNorm、RoPE、Self-Attention 和 SwiGLU。

计划实现：`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`。

算子统一参考 `src/ops/add/` 的组织方式：`op.cpp` 负责参数检查和设备调度，`cpu/*.cpp` 负责 CPU 计算，头文件负责声明接口。

### 2.1 Argmax

新增并实现：

- `src/ops/argmax/cpu/argmax_cpu.hpp`
- `src/ops/argmax/cpu/argmax_cpu.cpp`
- `src/ops/argmax/op.cpp`

实现支持 Float32、Float16 和 BFloat16。计算时将输入转换为 Float32 进行比较，同时记录第一个最大值的索引，最后将最大值转换回输入类型，并将索引写入 Int64 输出。

实现过程中修正了 `max_idx` 的指针转换错误：字节地址需要转换为 `int64_t *`，不能转换为普通 `int64_t`。同时将 `max_idx` 的类型检查和浮点输入类型检查分开处理。

验证命令：

```bash
xmake
xmake install
python test/ops/argmax.py
```

结果：Argmax 测试通过。

### 2.2 Embedding

新增并实现：

- `src/ops/embedding/cpu/embedding_cpu.hpp`
- `src/ops/embedding/cpu/embedding_cpu.cpp`
- `src/ops/embedding/op.cpp`

实现按 Int64 `index` 中的行号，从二维 `weight` 中复制整行到二维 `out`。由于 Embedding 只搬运数据，不进行数值运算，因此使用 `std::memcpy` 按行复制，可以同时支持 Float32、Float16 和 BFloat16。

实现过程中检查并修正了 `weight` 的拼写、`elementSize()` 方法名以及 CPU 实现头文件引用问题。验证命令：

```bash
xmake
xmake install
python test/ops/embedding.py
```

结果：小规模与 `index=(50,)`、`weight=(512, 4096)` 用例的 Float32、Float16、BFloat16 测试全部通过。

### 2.3 Linear

新增并实现：

- `src/ops/linear/cpu/linear_cpu.hpp`
- `src/ops/linear/cpu/linear_cpu.cpp`
- `src/ops/linear/op.cpp`

实现目标为：

```text
out = input * weight^T + bias
input  shape = [M, K]
weight shape = [N, K]
out    shape = [M, N]
```

CPU kernel 使用三重循环，以 Float32 累加 Float32、Float16 和 BFloat16 输入，最后转换回输出类型。Bias 设计为可选参数。

实现过程中修正了 `<cstddef>` 头文件拼写、`dtype()` 方法拼写、设备枚举大小写、NVIDIA 枚举拼写、`return;` 语法以及 `const` 拼写。C++ 内核保留可选 Bias 支持；当前 Python 测试使用 Bias 张量。

结果：小规模与 `input=(512, 4096)`、`weight=(4096, 4096)` 用例的 Float32、Float16、BFloat16 测试全部通过。

### 2.4 RMSNorm

新增并实现：

- `src/ops/rms_norm/cpu/rms_norm_cpu.hpp`
- `src/ops/rms_norm/cpu/rms_norm_cpu.cpp`
- `src/ops/rms_norm/op.cpp`

实现沿二维输入的最后一维逐行计算均方值，并使用 Float32 完成平方和、均值、平方根倒数及权重缩放，最后转换回 Float32、Float16 或 BFloat16：

```text
inverse_rms = 1 / sqrt(mean(x * x) + eps)
out = x * inverse_rms * weight
```

实现过程中修正了 CPU 头文件名 `rms_nrom_cpu.hpp` 的拼写、`flaot` 拼写、`f16_t` 类型名以及 `op.cpp` 末尾多余的右花括号。

结果：`test/ops/rms_norm.py` 在 `(1, 4)` 和 `(512, 4096)` 形状下的 Float32、Float16、BFloat16 测试全部通过。

### 2.5 RoPE

新增并实现：

- `src/ops/rope/cpu/rope_cpu.hpp`
- `src/ops/rope/cpu/rope_cpu.cpp`
- `src/ops/rope/op.cpp`

实现将每个注意力头的最后一维拆成前后两半，按照位置编号和频率将第 `j` 个前半元素与第 `j + head_dim / 2` 个后半元素配对旋转。同一 token 的所有注意力头共享由 `pos_ids` 计算的角度。

```text
angle = position / theta^(2 * j / head_dim)
a' = a * cos(angle) - b * sin(angle)
b' = b * cos(angle) + a * sin(angle)
```

实现过程中修正了 `rope_cpu.hpp` 最后一个参数后多余的逗号。CPU 内核使用 Float32 计算三角函数和旋转结果，再转换回目标类型。

结果：`test/ops/rope.py` 在小规模和 `(512, 4, 4096)` 形状下的 Float32、Float16、BFloat16 测试全部通过。

### 2.6 Self-Attention

新增并实现：

- `src/ops/self_attention/cpu/self_attention_cpu.hpp`
- `src/ops/self_attention/cpu/self_attention_cpu.cpp`
- `src/ops/self_attention/op.cpp`

实现依次计算缩放点积、因果 Softmax 和 Value 加权和：

```text
scores = Q * K^T * scale
weights = causal_softmax(scores)
out = weights * V
```

因果边界使用 `last_key = kv_len - q_len + query_index`，使当前 Query 只能访问 KV Cache 中不晚于自身的位置。实现同时支持 GQA，通过 `query_head / (n_heads / n_kv_heads)` 将多个 Query 头映射到同一个 KV 头。Softmax 先减去当前行最大值以避免指数溢出，点积和加权累加均使用 Float32。

结果：`test/ops/self_attention.py` 的普通多头注意力和 `4` 个 Query 头共享 `2` 个 KV 头的 GQA 用例均通过，覆盖 Float32、Float16 和 BFloat16。

### 2.7 SwiGLU

新增并实现：

- `src/ops/swiglu/cpu/swiglu_cpu.hpp`
- `src/ops/swiglu/cpu/swiglu_cpu.cpp`
- `src/ops/swiglu/op.cpp`

实现对连续二维张量执行逐元素门控：

```text
sigmoid = 1 / (1 + exp(-gate))
out = up * gate * sigmoid
```

实现过程中修正了 `sigmoid` 和 `result` 局部变量缺少 `float` 类型的问题。CPU 内核将输入转换为 Float32 计算指数和乘法，最后转换回目标类型。

结果：`test/ops/swiglu.py` 在 `(2, 3)` 和 `(512, 4096)` 形状下的 Float32、Float16、BFloat16 测试全部通过。

### 2.8 本日验证结果

2026-07-31 使用以下流程构建并验证：

```bash
xmake
xmake install
conda run -n llaisys python test/ops/rms_norm.py
conda run -n llaisys python test/ops/rope.py
conda run -n llaisys python test/ops/self_attention.py
conda run -n llaisys python test/ops/swiglu.py
```

四个算子的全部 CPU 测试通过。

2026-08-03 提交前重新执行构建和安装，并逐一运行 `test/test_tensor.py` 以及 `test/ops/` 下的 Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention 和 SwiGLU 测试，所有 CPU 用例均通过。仓库当前没有 README 中提到的 `test/test_ops.py` 聚合脚本，因此采用逐文件测试覆盖全部算子。

## 作业 3：Qwen2 模型推理

状态：未开始。

计划实现模型权重加载、Qwen2 前向推理和 KV Cache，并与 PyTorch 基准结果进行对照。

## 作业 4：CUDA 支持

状态：未开始。

当前环境因 NVIDIA 驱动版本与 PyTorch CUDA 版本不匹配，暂不可用。

## 进阶功能状态

以下函数已有实现草稿并已通过编译检查：

- `contiguous`
- `reshape`
- `to`

其中 CPU 基础路径已经具备，GPU 路径尚未完成完整验证，因为当前环境无法使用 CUDA。后续需要重点检查非连续 GPU Tensor 的复制和设备间内存传输。

## 后续更新方式

每完成一个作业模块，补充实现文件、关键思路、验证命令、测试结果和未解决问题。

## 当前注意事项

- 必须在 `llaisys` Conda 环境中运行测试；
- 修改 C++ 代码后需要先执行 `xmake install`，否则 Python 可能继续加载旧的共享库；
- 作业 2 的八个 CPU 算子均已通过各自测试；仓库当前没有 `test/test_ops.py` 聚合脚本；
- 当前 LLAISYS 的 Qwen2 推理尚未实现，模型推理测试中的 LLAISYS 结果暂为空。
