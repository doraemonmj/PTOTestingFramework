"""
自动生成的多内核模糊测试用例

该文件由 MultiKernelTestGenerator 自动生成。
包含多个测试用例，每个测试用例包含多个 InCore 内核和一个 Orchestration 函数。
"""

import sys
from pathlib import Path
from typing import Any, List

import torch
import pytest

from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# 添加 pypto 到路径
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
if _PYPTO_ROOT.exists() and str(_PYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYPTO_ROOT))


class TestFuzzSequentialSimple(PTOTestCase):
    """
    测试用例: fuzz_sequential_simple
    组合模式: sequential
    内核数量: 2
    """

    rows = 128
    cols = 128

    def __init__(self):
        super().__init__()
        self.config.atol = 0.0001
        self.config.rtol = 0.0001

    def get_name(self) -> str:
        return 'fuzz_sequential_simple'

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec('a', [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec('b', [128, 128], DataType.FP32, init_value=2.5),
            TensorSpec('output', [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class FuzzSequentialSimpleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_0(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tmp_0 = pl.subs(tile_b, 1.0)
                tmp_1 = pl.mul(tile_a, tile_a)
                tmp_2 = pl.subs(tmp_1, 1.0)
                tmp_3 = pl.add(tmp_0, tmp_2)
                result = pl.store(tmp_3, offsets=[0, 0], shapes=[128, 128], output_tensor=output)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_1(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32], output: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 128])
                tmp_0 = pl.div(tile_a, tile_b)
                tmp_1 = pl.muls(tmp_0, 0.5)
                tmp_2 = pl.rsqrt(tmp_1)
                tmp_3 = pl.exp(tmp_0)
                tmp_4 = pl.add(tmp_2, tmp_3)
                result = pl.store(tmp_4, offsets=[0, 0], shapes=[128, 128], output_tensor=output)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                result_0 = self.kernel_0(a, b)
                result_1 = self.kernel_1(result_0, b)
                return result_1

        return FuzzSequentialSimpleProgram

    def compute_expected(self, tensors, params=None):
        """使用 Torch 计算期望输出"""
        # 将 numpy 数组转换为 torch 张量（仅在输入边界）
        torch_tensors = {name: torch.from_numpy(arr) for name, arr in tensors.items() if not name.endswith('output')}

        def _torch_kernel_0(a, b):
            """Torch 实现: kernel_0"""
            # 创建变量环境
            env = {}
            env['tile_a'] = a.clone()
            env['tile_b'] = b.clone()

            # 执行操作链
            env['tmp_0'] = env['tile_b'] - 1.0
            env['tmp_1'] = env['tile_a'] * env['tile_a']
            env['tmp_2'] = env['tmp_1'] - 1.0
            env['tmp_3'] = env['tmp_0'] + env['tmp_2']
            return env['tmp_3']

        def _torch_kernel_1(a, b):
            """Torch 实现: kernel_1"""
            # 创建变量环境
            env = {}
            env['tile_a'] = a.clone()
            env['tile_b'] = b.clone()

            # 执行操作链
            env['tile_a'] = torch.where(torch.abs(env['tile_a']) < 0.01, torch.tensor(1.0), env['tile_a'])
            env['tile_b'] = torch.where(torch.abs(env['tile_b']) < 0.01, torch.tensor(1.0), env['tile_b'])
            env['tmp_0'] = env['tile_a'] / env['tile_b']
            env['tmp_1'] = env['tmp_0'] * 0.5
            env['tmp_1'] = torch.abs(env['tmp_1']) + 1e-6
            env['tmp_2'] = torch.rsqrt(env['tmp_1'])
            env['tmp_3'] = torch.exp(torch.clamp(env['tmp_0'], -10, 10))
            env['tmp_4'] = env['tmp_2'] + env['tmp_3']
            return env['tmp_4']


        # 顺序执行模式
        result_0 = _torch_kernel_0(torch_tensors['a'], torch_tensors['b'])
        result_1 = _torch_kernel_1(result_0, torch_tensors['b'])
        # 将结果转换回 numpy 并写入输出
        tensors['output'][:] = result_1.numpy()



class TestMultiKernelFuzzing:
    """多内核模糊测试套件"""

    def test_fuzz_sequential_simple(self, test_runner):
        """测试 fuzz_sequential_simple"""
        test_case = TestFuzzSequentialSimple()
        result = test_runner.run(test_case)
        assert result.passed, f"测试失败: {result.error}"

