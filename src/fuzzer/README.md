# 多内核模糊测试框架 (Multi-Kernel Fuzzing Framework)

这是一个用于生成和测试多内核 PyPTO 程序的自动化框架。该框架可以随机生成多个 InCore 内核函数，并通过 Orchestration 函数以不同的模式组合它们。

**注意**：`src/fuzzer` 是一个独立的框架，不依赖 `src/pto_test/fuzzing`。所有必要的代码都包含在此目录中。

---

## 目录

1. [快速开始](#快速开始)
2. [代码结构](#代码结构)
3. [核心概念](#核心概念)
4. [配置指南](#配置指南)
5. [算子规则](#算子规则)
6. [使用示例](#使用示例)
7. [更新日志](#更新日志)

---

## 快速开始

### 基础示例

```bash
# 生成测试用例（使用默认配置）
python src/fuzzer/example_multi_kernel.py

# 生成特定配置的测试用例
python src/fuzzer/example_multi_kernel.py --config-index 0

# 设置误差容忍度
python src/fuzzer/example_multi_kernel.py --atol 1e-3 --rtol 1e-3

# 运行测试（只生成代码）
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v --codegen-only

# 查看生成的 C++ 代码
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v --codegen-only --save-kernels --kernels-dir=/tmp/kernels
```

### 命令行参数

```bash
python src/fuzzer/example_multi_kernel.py [选项]

选项:
  --config-index N  指定配置索引（从0开始），不指定则使用所有配置
  --output PATH     输出文件路径（默认: src/fuzzer/generated_tests/test_fuzz_multi_kernel.py）
  --atol FLOAT      绝对误差容忍度（默认: 1e-4）
  --rtol FLOAT      相对误差容忍度（默认: 1e-4）
```

---

## 代码结构

### 目录结构

```
src/fuzzer/                          # 独立的模糊测试框架
├── __init__.py                      # 外部接口
├── example_multi_kernel.py          # 使用示例脚本（主入口）
├── conftest.py                      # pytest 配置
├── README.md                        # 本文档
├── src/                             # 内部实现
│   ├── __init__.py
│   ├── fuzzer.py                    # OpFuzzer 核心逻辑
│   ├── kernel_generator.py          # InCore 内核生成器
│   ├── orchestrator_generator.py    # Orchestration 组合函数生成器
│   └── multi_kernel_test_generator.py  # 完整测试用例生成器
└── generated_tests/                 # 生成的测试文件目录
    └── test_fuzz_multi_kernel.py    # 生成的测试文件
```

### 核心模块说明

#### 1. fuzzer.py - OpFuzzer
操作符模糊生成器，负责：
- 定义所有支持的算子（二元、一元、标量、高级算子）
- 随机生成操作链
- 处理数据约束（避免除零、正值约束等）
- 生成 NumPy/PyTorch 参考实现

**主要类**：
- `OpSpec`: 算子规格定义
- `OpFuzzer`: 操作链生成器

#### 2. kernel_generator.py - KernelGenerator
内核生成器，负责：
- 生成单个 InCore 内核函数
- 支持不同数量和维度的输入
- 生成 PyPTO 代码和 PyTorch 参考实现
- 处理形状对齐约束

#### 3. orchestrator_generator.py - OrchestratorGenerator
编排函数生成器，负责：
- 生成 Orchestration 函数
- 支持三种组合模式：sequential、branching、mixed
- 管理内核间的数据流

#### 4. multi_kernel_test_generator.py - MultiKernelTestGenerator
测试用例生成器，负责：
- 生成完整的 PTOTestCase 类
- 集成内核和编排函数
- 生成 PyTorch 参考实现
- 生成测试文件

---

## 核心概念

### 1. 内核生成规则

每个 InCore 内核包含：
- **输入**: 1-3 个 tile 张量，支持不同维度
- **操作链**: 3-10 个随机操作
- **输出**: 1 个 tile 张量

**操作链生成规则**：
1. 从输入张量中随机选择操作数
2. 随机选择一个操作符（根据权重）
3. 执行操作并生成中间结果
4. 中间结果可以被后续操作使用
5. 最后一个操作的结果作为内核输出

**示例**：
```python
@pl.function(type=pl.FunctionType.InCore)
def kernel_0(self, a: pl.Tensor[[128, 128], pl.FP32],
                   b: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
    tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
    tmp_0 = pl.add(tile_b, tile_a)      # 操作1
    tmp_1 = pl.mul(tmp_0, tile_a)       # 操作2
    tmp_2 = pl.sub(tmp_1, tile_b)       # 操作3
    return tmp_2
```

### 2. 内核组合模式

#### Sequential (顺序模式)
内核按顺序执行，每个内核的输出作为下一个内核的输入。

```
input → kernel_0 → kernel_1 → kernel_2 → output
```

#### Branching (分支模式)
多个内核并行执行，使用 merge 内核合并结果。

```
input → kernel_0 ↘
input → kernel_1 → merge → output
input → kernel_2 ↗
```

#### Mixed (混合模式)
结合顺序和分支执行。

```
input → kernel_0 ↘
input → kernel_1 → merge → kernel_2 → kernel_3 → output
```

### 3. 支持的算子

#### 基本算子（默认启用）
- **二元操作**: add, sub, mul, div, maximum, minimum
- **标量操作**: adds, subs, muls, divs
- **一元操作**: sqrt, rsqrt, exp, neg, recip, log, abs, relu

#### 高级算子（需要启用）
- **行广播操作**: row_expand_add, row_expand_sub, row_expand_mul, row_expand_div
- **矩阵操作**: matmul

详细算子规则请参考 [算子规则](#算子规则) 章节。

---

## 配置指南

### 配置结构

所有配置都在 `example_multi_kernel.py` 的 `all_configs` 列表中定义：

```python
all_configs = [
    {
        # 基本信息
        "name": "test_name",              # 测试用例名称（必需）
        "description": "测试描述",         # 测试描述（可选）

        # 生成控制
        "num_instances": 1,               # 从该配置生成的测试实例数量
        "seed": 42,                       # 随机种子

        # 算子配置
        "enable_advanced_ops": False,     # 是否启用高级算子

        # 张量配置
        "tensor_init_type": "constant",   # 张量初始化类型
        "shape": (128, 128),              # 张量形状

        # 内核配置
        "num_kernels": 3,                 # 内核数量
        "mode": "sequential",             # 组合模式
        "num_ops_range": (3, 7),          # 每个内核的操作数量范围
        "input_shapes_list": None,        # 每个内核的输入形状列表（可选）
    },
]
```

### 配置字段详解

#### 1. 基本信息
- **name** (必需): 测试用例的名称
- **description** (可选): 测试用例的描述

#### 2. 生成控制
- **num_instances** (默认: 1): 从该配置生成的测试实例数量
  - 如果设置为 N > 1，将生成 N 个测试用例
  - 每个实例使用不同的随机种子：`seed + instance_index`
  - 实例名称自动添加索引：`name_0`, `name_1`, ..., `name_N-1`

- **seed** (默认: 42): 随机种子，用于可重现性

#### 3. 算子配置
- **enable_advanced_ops** (默认: False): 是否启用高级算子
  - False: 只使用基本算子
  - True: 包含高级算子（row_expand, matmul 等）

#### 4. 张量配置
- **tensor_init_type** (默认: "constant"): 张量初始化类型
  - `"constant"`: 每个张量使用不同的常量值（2.0, 2.5, 3.0, ...）
  - `"random"`: 使用 `torch.randn` 生成随机正态分布值
  - `"range"`: 使用 `torch.rand` 生成 [0, 1) 范围内的随机值
  - `"normal"`: 使用 `torch.randn` 生成标准正态分布值
  - `"ones"`: 所有元素初始化为 1.0
  - `"zeros"`: 所有元素初始化为 0.0

- **shape** (默认: (128, 128)): 张量的形状

#### 5. 内核配置
- **num_kernels** (默认: 3): 生成的内核数量

- **mode** (默认: "sequential"): 内核组合模式
  - `"sequential"`: 顺序执行
  - `"branching"`: 分支执行
  - `"mixed"`: 混合模式

- **num_ops_range** (默认: (3, 7)): 每个内核包含的操作数量范围

- **input_shapes_list** (可选): 每个内核的输入形状列表
  - 如果为 None，则自动生成
  - 示例：`[[(128, 128), (128, 128)], [(128, 128)]]`

### 配置示例

#### 示例 1: 简单顺序执行
```python
{
    "name": "simple_sequential",
    "num_instances": 1,
    "seed": 42,
    "enable_advanced_ops": False,
    "num_kernels": 2,
    "mode": "sequential",
    "shape": (128, 128),
    "num_ops_range": (3, 5),
    "tensor_init_type": "constant",
    "input_shapes_list": [
        [(128, 128), (128, 128)],  # kernel_0: 2个输入
    ],
    "description": "简单顺序执行：2个内核"
}
```

#### 示例 2: 生成多个随机测试实例
```python
{
    "name": "random_tests",
    "num_instances": 5,  # 生成5个测试用例
    "seed": 100,         # 将使用种子 100, 101, 102, 103, 104
    "enable_advanced_ops": False,
    "num_kernels": 3,
    "mode": "branching",
    "shape": (128, 128),
    "num_ops_range": (4, 8),
    "tensor_init_type": "random",
    "input_shapes_list": None,
    "description": "随机分支测试：生成5个不同的测试实例"
}
```

---

## 算子规则

### 形状对齐约束

**重要**: 所有 tensor 创建和 reshape 操作必须满足 32 字节对齐约束。

**规则**:
- 形状的尾轴（最后一个维度，即列数）必须满足：
  1. 尾轴 = 1, 或者
  2. (尾轴 × sizeof(datatype)) % 32 == 0

**FP32 类型的有效尾轴值**:
- 尾轴 = 1（总是有效）
- 尾轴 % 8 == 0（因为 8 × 4 = 32）
- 有效值: 1, 8, 16, 24, 32, 40, 48, 56, 64, ..., 128, ...

**示例**:
```python
# ✓ 有效的形状
pl.tensor.create([128, 1], pl.FP32)      # 尾轴=1
pl.tensor.create([128, 8], pl.FP32)      # 8*4=32, 对齐
pl.tensor.create([128, 128], pl.FP32)    # 128*4=512, 对齐

# ✗ 无效的形状
pl.tensor.create([128, 3], pl.FP32)      # 3*4=12, 不对齐
pl.tensor.create([128, 5], pl.FP32)      # 5*4=20, 不对齐
```

### 算子分类

#### 1. Block Element-wise Binary Operations
| 算子名 | 输入类型 | 输出类型 | 约束 | NumPy等价 |
|--------|----------|----------|------|-----------|
| `block.add` | `tile, tile` | `tile` | 支持广播 | `a + b` |
| `block.sub` | `tile, tile` | `tile` | 支持广播 | `a - b` |
| `block.mul` | `tile, tile` | `tile` | 支持广播 | `a * b` |
| `block.div` | `tile, tile` | `tile` | 避免除零 | `a / b` |
| `block.maximum` | `tile, tile` | `tile` | 支持广播 | `np.maximum(a, b)` |
| `block.minimum` | `tile, tile` | `tile` | 支持广播 | `np.minimum(a, b)` |

#### 2. Block Scalar Operations
| 算子名 | 输入类型 | 输出类型 | NumPy等价 |
|--------|----------|----------|-----------|
| `block.adds` | `tile, scalar` | `tile` | `a + s` |
| `block.subs` | `tile, scalar` | `tile` | `a - s` |
| `block.muls` | `tile, scalar` | `tile` | `a * s` |
| `block.divs` | `tile, scalar` | `tile` | `a / s` |

#### 3. Block Unary Operations
| 算子名 | 输入类型 | 输出类型 | 约束 | NumPy等价 |
|--------|----------|----------|------|-----------|
| `block.neg` | `tile` | `tile` | - | `-a` |
| `block.exp` | `tile` | `tile` | 建议范围 [-10, 10] | `np.exp(a)` |
| `block.recip` | `tile` | `tile` | 避免除零 | `1.0 / a` |
| `block.sqrt` | `tile` | `tile` | 输入 ≥ 0 | `np.sqrt(a)` |
| `block.rsqrt` | `tile` | `tile` | 输入 > 0 | `1.0 / np.sqrt(a)` |
| `block.log` | `tile` | `tile` | 输入 > 0 | `np.log(a)` |
| `block.abs` | `tile` | `tile` | - | `np.abs(a)` |
| `block.relu` | `tile` | `tile` | - | `np.maximum(0, a)` |

#### 4. Block Row/Column Broadcast Operations (高级)
| 算子名 | 输入类型 | 输出类型 | 形状约束 | NumPy等价 |
|--------|----------|----------|----------|-----------|
| `block.row_expand_add` | `tile[M,N], tile[M,1]` | `tile[M,N]` | 第二个输入 [M,1] | `tile + row_vec` |
| `block.row_expand_sub` | `tile[M,N], tile[M,1]` | `tile[M,N]` | 第二个输入 [M,1] | `tile - row_vec` |
| `block.row_expand_mul` | `tile[M,N], tile[M,1]` | `tile[M,N]` | 第二个输入 [M,1] | `tile * row_vec` |
| `block.row_expand_div` | `tile[M,N], tile[M,1]` | `tile[M,N]` | 第二个输入 [M,1]，避免除零 | `tile / row_vec` |

#### 5. Block Matrix Operations (高级)
| 算子名 | 输入类型 | 输出类型 | 形状约束 | NumPy等价 |
|--------|----------|----------|----------|-----------|
| `block.matmul` | `tile, tile` | `tile` | `[M, K] @ [K, N] -> [M, N]` | `a @ b` |

### 数据约束

1. **避免除零**: `div`, `divs`, `recip`, `row_expand_div`
   - 确保分母绝对值 ≥ 0.01

2. **正值约束**: `sqrt`, `rsqrt`, `log`
   - 确保输入 > 0 或使用 `abs(x) + 1e-6`

3. **范围约束**: `exp`
   - 建议输入范围 [-10, 10] 避免溢出

### 常见算子组合模式

#### Softmax 组件
```python
# Step 1: Row max reduction
max_vals = pl.row_max(tile, tmp_tile)  # [M,N] -> [M,1]

# Step 2: Subtract max (数值稳定性)
centered = pl.row_expand_sub(tile, max_vals)

# Step 3: Exponential
exp_vals = pl.exp(centered)

# Step 4: Row sum
sum_vals = pl.row_sum(exp_vals, tmp_tile)

# Step 5: Normalize
output = pl.row_expand_div(exp_vals, sum_vals)
```

#### ReLU 及变体
```python
# ReLU
output = pl.relu(tile)

# LeakyReLU (alpha=0.01)
neg_part = pl.muls(tile, 0.01)
output = pl.maximum(tile, neg_part)
```

---

## 使用示例

### 生成测试用例

```bash
# 使用默认配置生成所有测试用例
python src/fuzzer/example_multi_kernel.py

# 只生成第一个配置
python src/fuzzer/example_multi_kernel.py --config-index 0

# 设置误差容忍度
python src/fuzzer/example_multi_kernel.py --atol 1e-3 --rtol 1e-3

# 指定输出文件
python src/fuzzer/example_multi_kernel.py --output my_tests.py

# 组合使用
python src/fuzzer/example_multi_kernel.py --config-index 1 --atol 1e-4 --rtol 1e-4 --output my_test.py
```

### 运行测试

```bash
# 运行所有测试
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v

# 只生成代码，不执行
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v --codegen-only

# 保存生成的 C++ 代码
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v --codegen-only --save-kernels --kernels-dir=/tmp/kernels

# 运行特定测试
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py::TestMultiKernelFuzzing::test_fuzz_sequential_simple -v
```

### 生成的代码结构

```python
class TestFuzzSequentialSimple(PTOTestCase):
    rows = 128
    cols = 128

    def __init__(self):
        super().__init__()
        self.config.atol = 1e-4
        self.config.rtol = 1e-4

    def get_name(self):
        return "fuzz_sequential_simple"

    def define_tensors(self):
        return [
            TensorSpec('a', [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec('b', [128, 128], DataType.FP32, init_value=2.5),
            TensorSpec('output', [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_0(self, a, b):
                # 内核实现
                pass

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a, b):
                # 组合逻辑
                pass

        return Program

    def compute_expected(self, tensors, params=None):
        # PyTorch 参考实现
        pass
```

---

## 更新日志

### 最新更新

#### 新增功能
- 支持多种张量初始化类型：constant, random, range, normal, ones, zeros
- 支持从单个配置生成多个测试实例（通过 `num_instances` 字段）
- 新增 `--config-index` 命令行参数，可以指定只生成某个配置的测试用例
- 新增 `--atol` 和 `--rtol` 命令行参数，支持设置误差容忍度
- 将所有配置参数移至 `all_configs` 结构，统一管理

#### 重要变更
- 将所有 golden 数据生成从 NumPy 替换为 PyTorch
  - `_generate_numpy_reference` 重命名为 `_generate_torch_reference`
  - `_get_numpy_operation` 重命名为 `_get_torch_operation`
  - 所有中间计算使用 PyTorch 张量操作

- 简化命令行参数
  - 移除 `--num-cases`、`--seed`、`--enable-advanced-ops`、`--tensor-init` 参数
  - 所有配置现在通过 `all_configs` 结构管理
  - 保留 `--output`、`--config-index`、`--atol`、`--rtol` 参数

#### 修复
- 修复生成的 golden.py 文件中缺少 torch 导入的问题

---

## 注意事项

1. **32字节对齐约束**: 所有 tensor 创建和 reshape 操作的形状必须满足32字节对齐
   - 形状尾轴（列数）必须是 1，或 `(cols * sizeof(dtype)) % 32 == 0`
   - FP32 类型有效的列数: 1, 8, 16, 24, 32, 40, 48, 56, 64, ..., 128, ...
   - Fuzzer 会自动验证并修正不对齐的形状

2. **张量形状**: 支持不同维度的输入张量，可以在配置中指定每个内核的输入形状

3. **数据类型**: 当前仅支持 FP32 类型

4. **操作约束**: 框架自动处理除零、负数开方等约束

5. **ISA 支持**: 确保添加的操作在目标硬件的 ISA 中有对应实现

6. **输入数量**: 每个内核支持 1-3 个输入张量，可以在配置中指定

---

## 扩展框架

### 添加新操作符

编辑 `fuzzer.py` 的 `OpFuzzer.__init__` 方法：

```python
# 在 OpFuzzer.__init__ 中
custom_ops = [
    OpSpec("block.custom_op", ["tile", "tile"], "tile", {},
           lambda a, b: custom_numpy_impl(a, b)),
]
self.ops = self.ops + custom_ops
```

### 添加新组合模式

在 `orchestrator_generator.py` 中添加新的生成方法。

---

## 参考文件

- [tests/test_cases/test_matmul.py](../../tests/test_cases/test_matmul.py): PTOTestCase 使用模式
- [src/fuzzer/src/fuzzer.py](src/fuzzer.py): OpFuzzer 操作生成逻辑和操作符定义
- [example_multi_kernel.py](example_multi_kernel.py): 配置示例
