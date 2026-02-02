# 多内核模糊测试框架

这是一个用于生成和测试多内核 PyPTO 程序的自动化框架。该框架可以随机生成多个 InCore 内核函数，并通过 Orchestration 函数以不同的模式组合它们。

**注意**：`src/fuzzer` 是一个独立的框架，不依赖 `src/pto_test/fuzzing`。所有必要的代码都包含在此目录中。

## 快速开始

```bash
# 生成1个测试用例
python src/fuzzer/example_multi_kernel.py --num-cases 1

# 生成5个测试用例
python src/fuzzer/example_multi_kernel.py --num-cases 5

# 运行测试（只生成代码）
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v --codegen-only

# 查看生成的 C++ 代码
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py -v --codegen-only --save-kernels --kernels-dir=/tmp/kernels
```

## 目录结构

```
src/fuzzer/                          # 独立的模糊测试框架
├── __init__.py                      # 外部接口
├── example_multi_kernel.py          # 使用示例脚本
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

## Op 组合规则

### 1. 操作符定义

操作符在 [fuzzer.py](fuzzer.py) 的 `OpFuzzer.__init__` 方法中定义。

**当前支持的操作**：
- **二元操作**: `block.add`, `block.sub`, `block.mul`, `block.div`, `block.maximum`
- **标量操作**: `block.adds`, `block.subs`, `block.muls`, `block.divs`
- **一元操作**: `block.sqrt`, `block.rsqrt`, `block.exp`, `block.neg`, `block.recip`

**添加新操作**：
```python
# 在 fuzzer.py 的 OpFuzzer.__init__ 中修改
self.ops = self.BLOCK_BINARY_OPS + self.BLOCK_SCALAR_OPS + self.BLOCK_UNARY_OPS

# 或者只使用基础操作
basic_ops = [
    OpSpec("block.add", ["tile", "tile"], "tile", {}, lambda a, b: a + b),
    OpSpec("block.sub", ["tile", "tile"], "tile", {}, lambda a, b: a - b),
    OpSpec("block.mul", ["tile", "tile"], "tile", {}, lambda a, b: a * b),
    OpSpec("block.div", ["tile", "tile"], "tile", {"avoid_zero": True}, lambda a, b: a / b),
]
self.ops = basic_ops
```

### 2. 内核生成规则

每个 InCore 内核包含：
- **输入**: 1-3 个 tile 张量，**支持不同维度**
- **操作链**: 1-10 个随机操作
- **输出**: 1 个 tile 张量

**输入张量配置**：
- 可以指定每个内核的输入数量和维度
- 不同内核可以有不同数量的输入（1-3个）
- 每个输入可以有不同的形状（如 128x128, 64x64, 256x256）
- 如果不指定，框架会随机生成输入配置

**示例配置**：
```python
# 在 example_multi_kernel.py 中配置
{
    "name": "test_case_name",
    "num_kernels": 3,
    "input_shapes_list": [
        [(128, 128), (64, 64)],           # kernel_0: 2个不同维度的输入
        [(128, 128), (128, 128), (256, 256)],  # kernel_1: 3个不同维度的输入
        [(256, 256)],                     # kernel_2: 1个输入
    ],
}
```

**操作链生成规则**：
1. 从输入张量中随机选择操作数
2. 随机选择一个操作符（add/sub/mul/div）
3. 执行操作并生成中间结果
4. 中间结果可以被后续操作使用
5. 最后一个操作的结果作为内核输出

**示例**：
```python
# 生成的内核代码 - 不同维度的输入
@pl.function(type=pl.FunctionType.InCore)
def kernel_0(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    tile_a = pl.op.block.load(a, 0, 0, 128, 128)
    tile_b = pl.op.block.load(b, 0, 0, 128, 128)  # 加载到输出大小
    tmp_0 = pl.op.block.add(tile_b, tile_a)      # 操作1: b + a
    tmp_1 = pl.op.block.mul(tmp_0, tile_a)       # 操作2: tmp_0 * a
    tmp_2 = pl.op.block.sub(tmp_1, tile_b)       # 操作3: tmp_1 - b
    return tmp_2
```

### 3. 内核组合模式

**Sequential (顺序模式)**：
- 内核按顺序执行
- 每个内核的输出作为下一个内核的输入
```
input → kernel_0 → kernel_1 → kernel_2 → output
```

**Branching (分支模式)**：
- 多个内核并行执行
- 使用 merge 内核合并结果
```
input → kernel_0 ↘
input → kernel_1 → merge → output
input → kernel_2 ↗
```

**Mixed (混合模式)**：
- 结合顺序和分支执行
```
input → kernel_0 ↘
input → kernel_1 → merge → kernel_2 → kernel_3 → output
```

### 4. 带参数的操作符

框架支持带参数的操作符（如 transpose, reduce, reshape）：

```python
# 在 fuzzer.py 中添加
OpSpec(
    "block.transpose",
    ["tile"], "tile", {},
    lambda a, dims: np.transpose(a, dims),
    shape_transform=lambda shapes, params: tuple(shapes[0][i] for i in params['dims']),
    param_generator=lambda shapes, rng: {'dims': (1, 0)},
    requires_params=True
)
```

**OpSpec 参数说明**：
- `name`: 操作名称（PyPTO API）
- `input_types`: 输入类型列表
- `output_type`: 输出类型
- `constraints`: 约束条件（如 `avoid_zero`, `positive_only`）
- `np_equivalent`: NumPy 参考实现
- `shape_transform`: shape 变换函数（可选）
- `param_generator`: 参数生成函数（可选）
- `requires_params`: 是否需要参数

## 命令行参数

### 生成测试用例

```bash
python src/fuzzer/example_multi_kernel.py [选项]

选项:
  --num-cases N    生成的测试用例数量 (1-5，默认: 1)
  --output PATH    输出文件路径
  --seed N         随机种子
```

### 运行测试

```bash
pytest src/fuzzer/generated_tests/test_fuzz_multi_kernel.py [选项]

常用选项:
  -v                    显示详细输出
  -s                    显示 print 输出
  --codegen-only        只生成代码，不执行
  --platform=PLATFORM   指定平台（如 a2a3sim）
  --device=N            指定设备编号
  --save-kernels        保存生成的 C++ 代码
  --kernels-dir=DIR     指定保存目录
  --dump-passes         打印编译器优化 pass
```

## 生成的代码结构

```python
class TestFuzzSequentialSimple(PTOTestCase):
    def get_name(self):
        return "fuzz_sequential_simple"

    def define_tensors(self):
        return [
            TensorSpec('a', [128, 128], DataType.FP32, is_input=True),
            TensorSpec('b', [128, 128], DataType.FP32, is_input=True),
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
        # NumPy 参考实现
        pass
```

## 扩展框架

### 添加新操作符

编辑 [fuzzer.py](fuzzer.py) 的 `OpFuzzer.__init__` 方法：

```python
# 在 OpFuzzer.__init__ 中
self.ops = self.BLOCK_BINARY_OPS + self.BLOCK_SCALAR_OPS + self.BLOCK_UNARY_OPS

# 或者自定义操作集合
custom_ops = [
    OpSpec("block.add", ["tile", "tile"], "tile", {}, lambda a, b: a + b),
    OpSpec("block.maximum", ["tile", "tile"], "tile", {}, lambda a, b: np.maximum(a, b)),
    OpSpec("block.sqrt", ["tile"], "tile", {"positive_only": True}, lambda a: np.sqrt(a)),
]
self.ops = custom_ops
```

### 添加新组合模式

在 [orchestrator_generator.py](orchestrator_generator.py) 中添加新的生成方法。

## 注意事项

1. **张量形状**: 支持不同维度的输入张量，可以在配置中指定每个内核的输入形状
2. **数据类型**: 当前仅支持 FP32 类型
3. **操作约束**: 框架自动处理除零、负数开方等约束
4. **ISA 支持**: 确保添加的操作在目标硬件的 ISA 中有对应实现
5. **输入数量**: 每个内核支持 1-3 个输入张量，可以在配置中指定

## 参考文件

- [tests/test_cases/test_matmul.py](../../tests/test_cases/test_matmul.py): PTOTestCase 使用模式
- [src/fuzzer/src/fuzzer.py](src/fuzzer.py): OpFuzzer 操作生成逻辑和操作符定义
- [example_multi_kernel.py](example_multi_kernel.py): 配置示例，包括如何指定不同维度的输入
