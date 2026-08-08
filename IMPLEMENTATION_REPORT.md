# LLAISYS 实现过程与结果报告

## 公共环境与模型

- Conda 环境：`llaisys`
- Python：3.11.15
- PyTorch：2.11.0+cu128
- Transformers：5.14.0
- Xmake：3.0.9
- GCC/G++：11.4.0
- 系统 CUDA Toolkit：12.9
- NVIDIA 驱动：575.51.03
- CUDA 测试设备：NVIDIA L20，计算能力 8.9
- 设备支持状态：CPU、NVIDIA CUDA 和 MetaX MXMACA 三种推理后端均已完成；NVIDIA 在 L20 上验证，MetaX 在独立的曦云 C500 环境上验证。
- 测试模型：`DeepSeek-R1-Distill-Qwen-1.5B`
- 模型权重和 tokenizer 已完整下载，并可由 Transformers 离线加载。

## 三种设备的编译与推理命令

以下命令均从项目根目录执行。CPU 和 NVIDIA 环境使用的模型目录为
`/data/home/caiwl/models/DeepSeek-R1-Distill-Qwen-1.5B`；MetaX 验证机使用
`/data/models/qwen15b`。迁移到其他机器时只需将 `MODEL_PATH` 改为实际模型目录。

每次切换设备后必须重新执行 `xmake f -c`。其中 `-c` 会清理旧的 Xmake 配置缓存，
而显式设置 `--nv-gpu` 和 `--mx-gpu` 可以避免误用上一次构建生成的共享库。
构建成功后，`libllaisys.so` 会自动复制到 `python/llaisys/libllaisys/`。

### CPU

CPU 构建不启用任何 GPU 后端，适用于普通 Linux 开发机和 GitHub Actions 的 Ubuntu 作业：

```bash
conda activate llaisys
export PYTHONPATH="$PWD/python"
export MODEL_PATH=/data/home/caiwl/models/DeepSeek-R1-Distill-Qwen-1.5B

xmake f -c --nv-gpu=n --mx-gpu=n
xmake

python test/test_runtime.py --device cpu
python test/test_infer.py \
    --device cpu \
    --model "$MODEL_PATH" \
    --test \
    --max_steps 2
```

### NVIDIA CUDA

NVIDIA 构建要求 CUDA Toolkit、驱动和 cuBLAS 可用。构建配置使用
`native + compute_80`，会为当前可见 GPU 生成原生 SASS，并保留 Ampere PTX 回退：

```bash
conda activate llaisys
export PYTHONPATH="$PWD/python"
export MODEL_PATH=/data/home/caiwl/models/DeepSeek-R1-Distill-Qwen-1.5B

nvidia-smi
xmake f -c --nv-gpu=y --mx-gpu=n
xmake

python test/test_runtime.py --device nvidia
python test/test_infer.py \
    --device nvidia \
    --model "$MODEL_PATH" \
    --test \
    --max_steps 2
```

### MetaX MXMACA

MetaX 构建在曦云 C500 验证机上执行，要求 MACA SDK 位于 `/opt/maca`，并使用
原生 `mxcc`、`libmcruntime.so` 和 `libmcblas.so`。当前验证机以 root 用户构建，
因此需要设置 `XMAKE_ROOT=y`：

```bash
conda activate llaisys
export MACA_PATH=/opt/maca
export PATH=/root/.local/bin:$PATH
export LD_LIBRARY_PATH="$MACA_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/python:$PWD/test"
export MODEL_PATH=/data/models/qwen15b
export XMAKE_ROOT=y

xmake f -c --nv-gpu=n --mx-gpu=y
xmake

python test/test_runtime.py --device metax
python test/test_infer.py \
    --device metax \
    --model "$MODEL_PATH" \
    --test \
    --max_steps 2
```

上述 `--max_steps 2` 是覆盖首次完整前向和第二步 KV Cache 增量推理的快速验证。
需要执行完整生成对拍时，去掉该参数，使用测试脚本默认的生成步数。单算子验证统一使用：

```bash
python test/ops/<operator>.py --device <cpu|nvidia|metax>
```

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

状态：已完成。八个 CPU 算子均已实现并完成验证。

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

### 2.8 验证结果

使用以下流程构建并验证：

```bash
xmake
xmake install
conda run -n llaisys python test/ops/rms_norm.py
conda run -n llaisys python test/ops/rope.py
conda run -n llaisys python test/ops/self_attention.py
conda run -n llaisys python test/ops/swiglu.py
```

四个算子的全部 CPU 测试通过。

提交前重新执行构建和安装，并逐一运行 `test/test_tensor.py` 以及 `test/ops/` 下的 Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention 和 SwiGLU 测试，所有 CPU 用例均通过。仓库当前没有 README 中提到的 `test/test_ops.py` 聚合脚本，因此采用逐文件测试覆盖全部算子。

## 作业 3：Qwen2 模型推理

状态：已完成。模型创建、权重加载、完整前向推理、KV Cache 和增量生成均已实现，并通过与 Hugging Face 的 argmax 生成对照测试。

### 3.1 模型创建与权重加载

新增 Qwen2 C++ 模型、C API 和 ctypes 包装：

- `src/models/qwen2/model.hpp`、`model.cpp`：根据模型元数据创建并管理权重 Tensor；
- `src/llaisys/qwen2.cc`：实现模型创建、销毁、原始权重访问、按名称加载和加载计数接口；
- `python/llaisys/libllaisys/qwen2.py`：声明 Qwen2 元数据、权重结构和 C API；
- `python/llaisys/models/qwen2.py`：读取 `config.json` 和 safetensors，并加载 BF16 权重；
- `xmake.lua`：编译 Qwen2 模型源码，并在构建后更新 Python 包中的共享库。

DeepSeek-R1-Distill-Qwen-1.5B 共创建并加载 339 个权重，其中包括 3 个全局权重和每层 12 个权重。实际模型加载成功，权重名称、形状和字节数检查均通过。

### 3.2 完整前向推理

当前 `Qwen2Model::infer` 已实现以下完整流程：

```text
Token IDs -> Embedding
-> 28 x (RMSNorm -> Q/K/V -> RoPE -> Self-Attention -> O Projection
         -> Residual -> RMSNorm -> Gate/Up -> SwiGLU -> Down -> Residual)
-> Final RMSNorm -> Last Token -> LM Head -> Argmax
```

Q、K、V 分别按 `[seq, 12, 128]`、`[seq, 2, 128]`、`[seq, 2, 128]` 参与注意力计算。实现支持输入完整 token 序列，并根据最后一个位置的 logits 返回 argmax token。

### 3.3 对照结果

使用本地 DeepSeek-R1-Distill-Qwen-1.5B BF16 权重进行验证：

```text
输入 [1]    ：LLAISYS = 22573，Hugging Face = 22573
输入 [1, 2] ：LLAISYS = 3，    Hugging Face = 3
```

第二组输入同时覆盖非零位置 RoPE 和多 token 因果注意力，结果与 Hugging Face 完全一致。

### 3.4 KV Cache 与增量生成

模型为每一层维护独立的 K Cache 和 V Cache，形状均为 `[capacity, num_kv_heads, head_dim]`。每次开始生成前，通过 C API 按提示词长度与最大生成长度重置 Cache。

首次推理输入完整提示词，将各层产生的 K/V 写入 Cache；后续每轮仅输入上一步生成的 token，并使用绝对位置执行 RoPE。新产生的 K/V 追加到已有 Cache，Self-Attention 则读取从起点到当前位置的全部 K/V。所有层完成后统一更新 Cache 长度，避免不同层看到不一致的位置。

Python `generate()` 已实现以下流程：

- 重置 KV Cache；
- 首轮输入完整提示词，后续每轮只输入一个新 token；
- 将 C++ 返回的 argmax token 追加到输出；
- 遇到 EOS 或达到 `max_new_tokens` 时停止。

### 3.5 最终验证

执行以下命令进行短序列生成对照：

```bash
conda run -n llaisys env PYTHONPATH=python python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --test --max_steps 2
```

Hugging Face 和 LLAISYS 均在相同提示词后生成 token `91786`、`0`，解码内容一致，测试输出为 `Test passed!`。这表明权重加载、完整提示词前向计算、KV Cache 追加、增量位置编码和 argmax 生成流程已经贯通。

当前实现以 CPU 正确性为目标，性能仍有优化空间；Cache 读取会生成连续副本，后续可通过让注意力算子直接读取有效 Cache 区间来减少复制。

## 作业 4：CUDA 支持

状态：NVIDIA CUDA 后端已完成。NVIDIA Runtime、八个 CUDA 算子和 Qwen2 增量推理均已接入并完成验证；第二款类 CUDA 平台 MetaX MXMACA 的适配结果记录在作业 5。

### 4.1 实现范围与文件组织

作业四沿用 CPU 算子的分层结构：

- `src/ops/<op>/op.cpp`：执行形状、数据类型、连续性和设备检查，并分发到 CPU 或 NVIDIA 实现；
- `src/ops/<op>/nvidia/*.cuh`：声明 NVIDIA 后端入口；
- `src/ops/<op>/nvidia/*.cu`：实现 CUDA kernel、类型分发和 stream 启动；
- `src/device/nvidia/nvidia_runtime_api.cu`：实现设备、stream、内存和拷贝 Runtime API；
- `xmake/nvidia.lua`：定义 NVIDIA Runtime 与算子静态库的构建规则；
- `xmake.lua`：通过 `nv-gpu` 选项启用 NVIDIA 后端并连接到主库。

当前 CUDA 算子包括：

- Add
- Argmax
- Embedding
- Linear
- RMSNorm
- RoPE
- Self-Attention
- SwiGLU

### 4.2 NVIDIA 构建系统

根构建文件新增 `nv-gpu` 选项。启用后会定义 `ENABLE_NVIDIA_API`，加载 `xmake/nvidia.lua`，并使通用 Runtime 和算子目标依赖 NVIDIA 静态库。

`xmake/nvidia.lua` 定义两个目标：

- `llaisys-device-nvidia`：编译 `src/device/nvidia/*.cu`；
- `llaisys-ops-nvidia`：编译 `src/ops/*/nvidia/*.cu`，并链接 cuBLAS。

两个目标均使用 C++17、`-fPIC` 和 `cuda.rdc=false`。关闭 CUDA Relocatable Device Code 可避免当前静态库链接方式产生 `__cudaRegisterLinkedBinary` 未解析符号。

两个 NVIDIA 构建目标都使用以下架构配置：

```lua
add_cugencodes("native")
add_cugencodes("compute_80")
```

`native` 由 Xmake 在配置阶段查询当前可见 GPU，并为计算性能最高的设备生成原生 SASS。当前 L20 因此自动生成 `sm_89`，不再需要在构建文件中硬编码。`compute_80` 额外保留 Ampere 虚拟架构 PTX，使计算能力不低于 8.0 的更新 NVIDIA GPU 可以在没有匹配 SASS 时由驱动 JIT 编译。

该方案适合在目标机器上现场构建。`native` 只选择当前可见设备中性能最高的一种架构；混合架构多 GPU 机器上的其他设备将依赖 PTX。无 GPU 的编译节点无法生成 native SASS，但仍会保留 `compute_80` PTX，因此正式发布多架构二进制时仍应显式列出需要支持的 `sm_xx`。

CUDA 构建流程：

```bash
xmake f --nv-gpu=y -cv
xmake
```

构建成功后，`libllaisys.so` 会自动复制到 `python/llaisys/libllaisys/`。

### 4.3 NVIDIA Runtime

`nvidia_runtime_api.cu` 将 LLAISYS Runtime API 映射到 CUDA Runtime：

| LLAISYS 功能 | CUDA 实现 | 说明 |
| --- | --- | --- |
| 获取设备数量 | `cudaGetDeviceCount` | 返回当前可见 NVIDIA 设备数 |
| 设置设备 | `cudaSetDevice` | 切换当前线程使用的 GPU |
| 设备同步 | `cudaDeviceSynchronize` | 等待当前设备全部任务完成 |
| 创建/销毁 stream | `cudaStreamCreate` / `cudaStreamDestroy` | 每个 Runtime Resource 持有自己的 stream |
| stream 同步 | `cudaStreamSynchronize` | 等待指定 stream 完成 |
| 设备内存 | `cudaMalloc` / `cudaFree` | Tensor storage 使用的显存 |
| 页锁定主机内存 | `cudaMallocHost` / `cudaFreeHost` | 支持可靠的异步主机与设备拷贝 |
| 同步拷贝 | `cudaMemcpy` | 支持 H2H、H2D、D2H 和 D2D |
| 异步拷贝 | `cudaMemcpyAsync` | 在 LLAISYS stream 上排队执行 |

所有 CUDA Runtime 返回值统一通过 `checkCuda` 检查，并把 CUDA 错误字符串交给 `ASSERT`。空指针释放会直接跳过，避免无意义的 CUDA 调用。

### 4.4 CUDA 算子的共同设计

所有算子都从 `op.cpp` 获取当前 Runtime 的 stream，并使用同一个 stream 启动 kernel 或 cuBLAS 操作，从而保持算子与异步内存拷贝之间的执行顺序。

数值算子统一支持：

- Float32：CUDA `float`；
- Float16：CUDA `__half`；
- BFloat16：CUDA `__nv_bfloat16`。

需要归约、指数、三角函数或矩阵累加的算子会先转成 Float32 计算，再转换回输出类型，以减少 Float16/BFloat16 的累计误差。普通逐元素和纯数据搬运算子直接使用原类型。

每次 kernel 启动后使用 `cudaGetLastError()` 检查启动参数和即时错误。由于 kernel 是异步执行的，运行期错误可能在后续同步、拷贝或测试比较时才被报告。

### 4.5 Add

Add 使用一维网格，每个线程计算一个元素：

```text
c[i] = a[i] + b[i]
```

线程块大小为 256，支持 Float32、Float16 和 BFloat16，并支持输出与输入之一共享底层地址的逐元素原地计算语义。

### 4.6 Embedding

Embedding 将输出展平，每个线程负责一个 `(token, embedding_column)`：

```text
out[token, column] = weight[index[token], column]
```

索引使用 Int64，权重和输出保持原始浮点类型。CUDA kernel 会对越界索引执行保护，但当前不会像 CPU 路径一样在主机端抛出详细异常；模型推理和测试均要求输入索引合法。

### 4.7 SwiGLU

SwiGLU 为逐元素 kernel，使用 Float32 计算 sigmoid：

```text
sigmoid = 1 / (1 + exp(-gate))
out = up * gate * sigmoid
```

每个线程处理一个元素，最后转换回目标数据类型。

### 4.8 RoPE

RoPE 将每个注意力头的前后半维度配对。一个线程负责一对元素，使用 Int64 `pos_ids` 计算绝对位置角度，并用 Float32 完成 `powf`、`sinf`、`cosf` 和旋转。

```text
angle = position / theta^(2 * pair_index / head_dim)
a' = a * cos(angle) - b * sin(angle)
b' = b * cos(angle) + a * sin(angle)
```

该布局与 CPU 路径以及 Qwen2 模型中的 `[seq_len, n_heads, head_dim]` 张量一致。

### 4.9 RMSNorm

RMSNorm 使用一个线程块处理一行，固定 256 个线程。线程分别累加平方和，再通过共享内存进行树形归约：

```text
inverse_rms = rsqrt(sum(x^2) / cols + eps)
out = x * inverse_rms * weight
```

平方和、归约和缩放均使用 Float32。当前实现适合模型隐藏维度，但还没有使用 warp shuffle 或向量化加载。

### 4.10 Argmax

Argmax 使用一个 256 线程的 block 扫描整个输入。每个线程先处理间隔为 `blockDim.x` 的元素，再通过共享内存同时归约最大值与 Int64 索引。遇到相同最大值时选择较小索引，与 CPU 路径保持一致。

当前实现以正确性为主。输入非常大时，单 block 会限制并行度，后续可改为多 block 两阶段归约。

### 4.11 Linear

Linear 的目标计算为：

```text
out[M, N] = input[M, K] * weight[N, K]^T + bias[N]
```

矩阵乘法由 `cublasGemmEx` 完成。由于 LLAISYS 张量是行主序，而 cuBLAS 默认按列主序解释内存，调用通过计算转置后的等价式完成行主序结果：

```text
out^T = weight * input^T
```

Float16 和 BFloat16 输入使用 Float32 compute type。Bias 由独立逐元素 kernel 添加，按 `output_index % N` 选择对应列的 Bias。cuBLAS handle 使用线程局部缓存，并在每次调用前绑定当前 LLAISYS stream。

### 4.12 Self-Attention

Self-Attention 分为两个 kernel，并使用一个 Float32 临时分数缓冲区：

1. `scoreSoftmaxKernel` 计算 `QK^T * scale`，执行因果掩码范围内的稳定 Softmax；
2. `weightedValueKernel` 计算 Softmax 权重与 V 的加权和。

每个 `(query_position, query_head)` 对应一个分数 kernel block。因果边界为：

```text
last_key = kv_len - q_len + query_index
```

因此完整提示词和 KV Cache 增量推理可以共用同一逻辑。GQA 通过以下映射让多个 Query 头共享一个 KV 头：

```text
group_size = n_heads / n_kv_heads
kv_head = query_head / group_size
```

Softmax 的最大值归约、指数和求和均使用 Float32。临时分数缓冲区通过 `cudaMallocAsync` 和 `cudaFreeAsync` 在同一 stream 上管理，空间复杂度为 `q_len * n_heads * kv_len * sizeof(float)`。

### 4.13 测试参考修正

NVIDIA Self-Attention 测试最初在 CUDA 上创建 `attn_bias`，但将 `temp_mask` 留在 CPU，导致 PyTorch 参考实现在调用 LLAISYS 算子前失败。测试已让 mask 跟随 Query 设备：

```python
temp_mask = torch.ones(
    L, S, dtype=torch.bool, device=query.device
).tril(diagonal=S-L)
```

该修改只修正测试参考张量的设备，不改变算子输入、输出或比较标准。

### 4.14 验证流程与结果

NVIDIA Runtime 验证命令：

```bash
conda run -n llaisys env PYTHONPATH=python \
    python test/test_runtime.py --device nvidia
```

逐算子验证命令形式：

```bash
conda run -n llaisys env PYTHONPATH=python \
    python test/ops/<operator>.py --device nvidia
```

Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention 和 SwiGLU 均完成 Float32、Float16、BFloat16 测试。Self-Attention 额外覆盖普通多头注意力、GQA、`q_len < kv_len` 的 KV Cache 形态和因果掩码。

架构配置改为 `native + compute_80` 后，使用 `cuobjdump` 检查 `libllaisys-ops-nvidia.a`：八个算子对象都包含 `sm_89.cubin`，同时包含目标为 `sm_80` 的 PTX。重新运行 NVIDIA Add 测试和两步 Qwen2 推理后均通过，模型 token 仍为 `91786`、`0`，说明自动架构探测与 PTX 回退没有引入运行或数值回归。

### 4.15 Qwen2 CUDA 推理

完成单算子验证后，进一步修改 Qwen2 模型，使权重、临时 Tensor、KV Cache 和最终 token 回传都能在 NVIDIA 设备上正确工作。

模型原先使用 `std::memcpy` 将当前层生成的 K/V 写入 KV Cache。CPU Tensor 的 `data()` 是主机地址，可以直接使用 `std::memcpy`；CUDA Tensor 的 `data()` 是设备地址，主机不能直接解引用。因此模型改为通过当前设备 Runtime 执行复制：

```text
CPU 模型：H2H
CUDA 模型：D2D
```

K/V 写入使用 `memcpy_async`，并传入模型当前 Runtime 的 stream。RoPE、V Projection、KV Cache 写入和后续 Self-Attention 都在同一个 stream 中排队，因此 Self-Attention 开始读取 Cache 前，前面的 K/V 复制一定已经完成，不需要额外的设备级同步。

模型最后通过 `max_idx->to(LLAISYS_DEVICE_CPU)` 把 GPU Argmax 结果返回主机。原 `Tensor::to()` 总是在复制前切换到目标 Runtime，导致 D2H 路径切换到 CPU Runtime 后由 `std::memcpy` 读取显存。修正后的 Runtime 选择原则为：

- H2D 使用目标 GPU Runtime；
- D2H 使用源 GPU Runtime；
- 当前 D2D 路径使用目标 GPU Runtime。

D2H 仍使用同步复制，保证 `Tensor::to()` 返回后 CPU 可以立即读取 token ID。

完整模型对照命令：

```bash
conda run -n llaisys env PYTHONPATH=python \
    python test/test_infer.py \
    --model <model_path> \
    --device nvidia \
    --test \
    --max_steps 2
```

Hugging Face 与 LLAISYS 均生成 token `91786`、`0`，最终输出 `Test passed!`。第一个生成步骤覆盖完整提示词的 CUDA 前向计算和初始 Cache 写入；第二个步骤只输入一个新 token，覆盖 `cache_start > 0`、K/V 追加、完整历史 Cache 读取和增量 RoPE 位置，因此确认 CUDA KV Cache 主流程已经贯通。

### 4.16 跨机器适配清单

后续在不同 GPU 或软件环境上适配时，应按以下顺序检查：

1. 使用 `nvidia-smi` 确认驱动、GPU 型号和设备可见性；
2. 使用 `nvcc --version` 确认构建所用 CUDA Toolkit；
3. 使用 Python 检查 `torch.__version__`、`torch.version.cuda` 和 `torch.cuda.is_available()`；
4. 确认 Xmake 的 `native` 探测结果；无 GPU 编译节点或混合架构机器应显式配置需要支持的 `sm_xx`；
5. 确认目标 GPU 支持所需的 Float16/BFloat16 指令，老架构可能需要禁用 BFloat16 或提供回退；
6. 确认 cuBLAS 可链接，并验证目标版本支持 `cublasGemmEx` 使用的数据类型与 compute type；
7. 确认 CUDA Runtime 支持 `cudaMallocAsync`，否则 Self-Attention 需要改用普通分配或由框架提供可复用 workspace；
8. 在多 GPU 场景检查 cuBLAS handle 与创建设备的绑定关系，必要时按设备分别缓存 handle；
9. 先运行 Runtime 测试，再按 Add、Embedding、SwiGLU、RoPE、RMSNorm、Argmax、Linear、Self-Attention 的顺序定位问题；
10. 最后运行 Qwen2 推理对照，验证所有 kernel 在真实模型形状和 KV Cache 路径下协同正确。

### 4.17 当前限制与后续优化

- `native` 只为当前可见的最快 GPU 生成原生 SASS；现有 `compute_80` PTX 提供向前兼容，但尚未生成覆盖多种旧 GPU 的完整多架构 SASS；
- Self-Attention 每次调用都会申请临时分数缓冲区，长序列下显存占用为二次增长，可进一步使用融合注意力或可复用 workspace；
- Self-Attention 的点积和 Value 加权仍包含线程内循环，性能目标低于 FlashAttention 等成熟实现；
- Argmax 使用单 block，适合当前测试和词表 logits，但可扩展为多阶段归约；
- RoPE 会重复计算同一位置和维度对应的三角函数，可预计算频率或使用缓存；
- Linear 的 cuBLAS handle 当前按主机线程缓存，多设备轮换需要进一步处理；
- `cudaGetLastError` 只检查 kernel 启动错误，异步执行错误仍依赖 stream 同步或后续内存操作暴露；
- CUDA Embedding 对非法索引仅阻止越界访问，尚未提供与 CPU 完全一致的异常信息。

作业四后续新增机器适配、性能优化或新 CUDA 算子时，应继续在本节记录目标设备、计算能力、软件版本、代码路径、验证命令、结果和仍存在的限制。

## 作业 5：MetaX 曦云 C500 原生适配

### 5.1 背景与目标

在保留 CPU / NVIDIA 后端的前提下，将 llaisys 移植到本机 **MetaX 曦云 C500**（沐曦通用 GPGPU）。要求：

- 采用 **C500 原生 MXMACA（`mc_*` API）** 开发，编写对应的 runtime 与算子实现（不使用 cu-bridge CUDA 复用路线）；
- **保持设备间代码隔离**：metax 代码独立于 CPU / NVIDIA，互不干扰；
- 完整打通 构建 → Runtime → 8 个算子 → Qwen2 端到端推理。

平台知识库见 `/root/metax-xuyun-c500/`，移植计划见 `/root/llaisys-metax-porting-plan/`（本节即移植计划与过程的整合记录）。

### 5.2 平台与环境

| 项 | 值 |
|----|----|
| GPU | MetaX C500（曦云 C500）× 1，64 GiB 显存 |
| 驱动 / SDK | KMD 3.8.30 / MACA 3.3.0.15（`/opt/maca`） |
| 原生编译器 | `mxcc`（`/opt/maca/mxgpu_llvm/bin/mxcc`，LLVM 基） |
| 库 | `libmcruntime.so`（≈CUDA Runtime）、`libmcblas.so`（≈cuBLAS） |
| 构建工具 | Xmake 3.1.0（root 下需 `XMAKE_ROOT=y`） |
| Python | torch `2.8.0+metax3.3.0.2`（`torch.cuda` 直接可用，设备名 `MetaX C500`） |
| 测试模型 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`（`/data/models/qwen15b`） |

### 5.3 移植计划（M0–M5）

```
M0 环境与冒烟 ──> M1 设备分发 ──> M2 构建 ──> M3 算子验证 ──> M4 Python 集成 ──> M5 推理对拍
```

| 阶段 | 内容 | 结果 |
|------|------|------|
| M0 | 安装 xmake；原生 MC 冒烟（kernel + `mcMalloc/mcMemcpy` + `mcblasGemmEx`(BF16)） | ✅ 全通过，确定原生路线 |
| M1 | `LLAISYS_DEVICE_METAX` 枚举、runtime 分发、metax runtime/算子目录 | ✅ |
| M2 | `xmake/metax.lua`（mxcc 编译 + 对象合并）、`--mx-gpu=y` | ✅ 100% 构建 |
| M3 | 8 个算子 GPU 对拍（f32/f16/bf16） | ✅ 全通过 |
| M4 | Python `metax` 映射、runtime/tensor/utils 测试 | ✅ |
| M5 | 模型下载、端到端 token 对拍 | ✅ `Test passed!` |

### 5.4 原生 MXMACA 技术要点

编译：

```bash
export MACA_PATH=/opt/maca
mxcc -x maca -std=c++17 -offload-arch native -fPIC \
     -I $MACA_PATH/include/mcr -I $MACA_PATH/include/mcblas \
     -I $MACA_PATH/include/common -I $MACA_PATH/include \
     --maca-path=$MACA_PATH -c src.mc -o src.o
# 链接：-L $MACA_PATH/lib -lmcruntime -lmcblas
```

API 与 CUDA 一一对应：`mcGetDeviceCount/mcSetDevice/mcDeviceSynchronize`、`mcStreamCreate/Destroy/Synchronize`、`mcMalloc/mcFree`、`mcMallocHost/mcFreeHost`、`mcMemcpy/mcMemcpyAsync`、`mcblasGemmEx`。要点：

- `mcMemcpyKind` 与 `llaisysMemcpyKind_t` 数值一致（H2H=0, H2D=1, D2H=2, D2D=3）；
- `MACA_R_32F/16F/16BF` = `0/2/14`，与 CUDA `CUDA_R_*` 数值一致；
- 数据类型/转换函数与 CUDA 同名（`__half`、`maca_bfloat16`、`__half2float`、`__bfloat162float` 等），kernel 逻辑可直接改写；
- 编译要点：`-x maca` 强制语言；`-std=c++17`（否则 `std::byte` 报错）；`-fPIC`（共享库必需）。

冒烟验证：`/root/m0-smoke/smoke_mc.cpp`（kernel + 内存/拷贝 + 流 + `mcblasGemmEx`(BF16)）用 `mxcc` 编译运行**全部通过**，确认原生 MC 编译、链接、kernel 启动、mcBLAS 全链路可用，作为 M0 路线决策依据。

### 5.5 代码改动（设备隔离）

公共层：

- `include/llaisys.h`：`LLAISYS_DEVICE_METAX = 2`；
- `src/device/runtime_api.hpp/.cpp`：`ENABLE_METAX_API` 下声明并分发 `metax::getRuntimeAPI()`；
- `python/llaisys/libllaisys/llaisys_types.py`：`DeviceType.METAX`；
- `test/test_utils.py`：`metax` 设备映射；关闭 TF32（见 5.8）；
- `test/test_runtime.py`、`test/test_infer.py`、`test/ops/*.py`：`--device` 增加 `metax`。

新增（全部隔离在 `metax` 目录，不触碰 nvidia/cpu）：

```
src/device/metax/metax_runtime_api.{hpp,mc}          # 原生 mc_* runtime
src/ops/{add,argmax,embedding,linear,rms_norm,rope,self_attention,swiglu}/metax/*.{hpp,mc}
xmake/metax.lua                                      # 构建目标
```

每个 `op.cpp` 增加与 NVIDIA 对称的分支（`#ifdef ENABLE_METAX_API case LLAISYS_DEVICE_METAX`）。
`linear` 用 `mcblasGemmEx`（`MCBLAS_OP_T`）；`self_attention` 临时分数缓冲用 `mcMalloc/mcFree`
（等价原 `cudaMallocAsync` 语义，兼容性最稳）。

### 5.6 构建系统与踩坑

`xmake.lua` 新增 `--mx-gpu` 选项（`has_config("mx-gpu")` 时 `add_defines("ENABLE_METAX_API")` + include `xmake/metax.lua`）。
`xmake/metax.lua` 的 metax 静态库目标：`on_load` 钩子调用 `mxcc` 增量编译 `.mc` → `build/metax-objs/*.o`，
`target:add("files", obj)` 动态注册，`add_rules("c++")` 提供对象合并；`mcruntime/mcblas` 挂在 metax 静态库上经 `add_deps` 传播链接。

构建踩坑（均有验证记录）：

| 尝试 | 结果 | 结论 |
|------|------|------|
| 自定义 `rule` + `on_build_file` + `set_extensions(".mc")` | `.o` 生成但静态库归档为空（8 字节） | 对象按 sourcekind 归类，`.mc` 无语言定义 |
| `sourcekind="maca"` | `no suitable linker for static.{maca}` | 需注册 ar 工具 |
| `sourcekind="cxx"` | g++ 编译（不识 `__global__`） | sourcekind 决定默认编译器 |
| 自定义 `toolchain("maca")`（`xmake/toolchains/`） | `cannot find known tool script for mxcc` | 需 `find_<name>` 检测器 |
| **`on_load` + `mxcc` 编译 + `target:add("files", obj)` + `add_rules("c++")`** | **✅ 归档成功** | 最终方案 |

其它：共享库需 `-fPIC`；g++ 默认 `--as-needed` 会丢弃 `libmcblas` → 链接经 metax 静态库传播，排在引用它的静态库之后。

构建命令：

```bash
export PATH=/root/.local/bin:$PATH XMAKE_ROOT=y MACA_PATH=/opt/maca
cd /root/llaisys-26s
xmake f --mx-gpu=y -c
xmake
```

产物 `libllaisys.so`（含 22 个 metax 导出符号）自动复制到 `python/llaisys/libllaisys/`。

### 5.7 验证结果

Runtime：

```bash
LD_LIBRARY_PATH=/opt/maca/lib PYTHONPATH=python:test python test/test_runtime.py --device metax
# Found 1 metax devices / Test passed!
```

算子：`add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`
在 f32/f16/bf16 下**全部 Passed**（`python test/ops/<op>.py --device metax`）。

端到端推理对拍：

```bash
LD_LIBRARY_PATH=/opt/maca/lib PYTHONPATH=python:test \
  python test/test_infer.py --device metax --model /data/models/qwen15b --test
# Test passed!（llaisys token 与 HF 完全一致；0.77s vs HF 2.72s）
```

### 5.8 平台差异与解决（TF32）

- **现象**：`linear`/`self_attention` 的 f32 用例对拍失败，差异 ~1e-4；
- **根因**：MetaX 定制 PyTorch 默认 `allow_tf32=True`、`float32_matmul_precision="high"`
  （NVIDIA PyTorch 2.x 默认 False），torch reference 用 TF32 降精度；
- **验证**：llaisys metax kernel 与 CPU float64 参考差异为 **0**，而 torch 结果差 1.86e-4 → 是 torch 侧降精度，非实现错误；
- **修复**：`test/test_utils.py` 顶部统一关闭：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

其它：补充 Python 依赖 `pip install ml_dtypes accelerate`。

### 5.9 与 cu-bridge 路线对比（备选）

本次按用户要求采用**原生 MXMACA** 路线。cu-bridge 路线（CUDA 源码直接 `cucc` 编译）经静态探测同样可行：
`cudaMallocAsync→wcudaMallocAsync`、`cublasGemmEx→mcblasGemmEx`、`CUDA_R_*`/`CUBLAS_COMPUTE_*` 与 MC 枚举数值一致，
可作为后续快速接入现有 CUDA 生态的备选路径。原生路线的优点是直接使用 `mc_*` API、无 wrapper 间接层、便于针对 C500 优化。

### 5.10 当前限制与后续优化

- `self_attention` 临时缓冲为同步分配，可改用流序分配（`mcMallocAsync` 等价物）减少同步开销；
- f32 linear 可对比 `mcblasSgemm` 与 `mcblasGemmEx` 的精度/性能；
- 可引入 `mcTracer`（`/opt/maca/bin/mcTracer`）做算子级性能分析；
- 当前仅单卡支持（`llaisysQwen2ModelCreate` 限制），多卡需扩展；
- 手写 kernel 可进一步替换为 `libmctlassEx` / TileLang 实现以提升性能。

## 进阶功能状态

以下函数已有实现草稿并已通过编译检查：

- `contiguous`
- `reshape`
- `to`

其中 CPU 基础路径已经具备。`to` 的 H2D 和 D2H 路径已在 CUDA 模型权重加载及 token 回传中实际使用；NVIDIA Runtime 已完成设备内存和设备间拷贝支持，但非连续 GPU Tensor 的 `contiguous`、`reshape` 以及跨 GPU 的 `to` 路径仍需要单独验证。

## 后续更新方式

每完成一个作业模块，补充实现文件、关键思路、验证命令、测试结果和未解决问题。

## 当前注意事项

- 必须在 `llaisys` Conda 环境中运行测试；
- 修改 C++ 代码后需要执行 `xmake`；构建完成后共享库会自动复制到 Python 包目录；
- Qwen2 的权重加载、完整前向推理、KV Cache 增量生成和 argmax 对照测试均已通过；
- 作业 4 的 NVIDIA Runtime、八个 CUDA 算子和 Qwen2 两步增量推理均已通过；
- 作业 4 使用 `native + compute_80` 自动生成当前 GPU 的 SASS 并保留 PTX 回退；迁移机器时仍必须检查计算能力与 CUDA/PyTorch/驱动兼容关系；
- 作业 5 的 MetaX 原生后端已在本机 C500 通过 Runtime、八个算子和 Qwen2 端到端推理对拍；构建需 `xmake f --mx-gpu=y`，运行需 `LD_LIBRARY_PATH=/opt/maca/lib`；
- metax 环境注意：torch 默认开启 TF32（测试对拍需关闭）；Qwen2 Python 依赖需 `ml_dtypes`、`accelerate`。
