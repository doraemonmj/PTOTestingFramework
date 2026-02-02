"""
自动生成的多内核模糊测试用例

该文件由 MultiKernelTestGenerator 自动生成。
包含多个测试用例，每个测试用例包含多个 InCore 内核和一个 Orchestration 函数。
"""

import sys
from pathlib import Path
from typing import Any, List

import numpy as np
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = 128
        self.cols = 128

    def get_name(self) -> str:
        return 'fuzz_sequential_simple'

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec('a', [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec('b', [128, 128], DataType.FP32, init_value=2.5),
            TensorSpec('c', [256, 256], DataType.FP32, init_value=3.0),
            TensorSpec('output', [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class FuzzSequentialSimpleProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_0(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, 0, 0, 128, 128)
                tile_b = pl.op.block.load(b, 0, 0, 128, 128)
                tmp_0 = pl.op.block.div(tile_b, tile_a)
                tmp_1 = pl.op.block.sub(tmp_0, tile_a)
                tmp_2 = pl.op.block.div(tmp_1, tile_b)
                return tmp_2

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_1(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32], c: pl.Tensor[[256, 256], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.op.block.load(a, 0, 0, 128, 128)
                tile_b = pl.op.block.load(b, 0, 0, 128, 128)
                tile_c = pl.op.block.load(c, 0, 0, 128, 128)
                tmp_0 = pl.op.block.add(tile_a, tile_c)
                tmp_1 = pl.op.block.neg(tile_b)
                tmp_2 = pl.op.block.maximum(tmp_1, tmp_1)
                tmp_3 = pl.op.block.rsqrt(tmp_0)
                tmp_4 = pl.op.block.add(tmp_2, tmp_3)
                return tmp_4

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(self, a: pl.Tensor[[128, 128], pl.FP32], b: pl.Tensor[[128, 128], pl.FP32], c: pl.Tensor[[256, 256], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                result_0 = self.kernel_0(a, b)
                result_1 = self.kernel_1(result_0, b, c)
                return result_1

        return FuzzSequentialSimpleProgram

    def compute_expected(self, tensors, params=None):
        """使用 NumPy 计算期望输出"""
    def _numpy_kernel_0(self, a, b):
        """NumPy 实现: kernel_0"""
        # 创建变量环境
        env = {}
        env['tile_a'] = a.copy()
        env['tile_b'] = b.copy()

        # 执行操作链
        env['tile_b'] = np.where(np.abs(env['tile_b']) < 0.01, 1.0, env['tile_b'])
        env['tile_a'] = np.where(np.abs(env['tile_a']) < 0.01, 1.0, env['tile_a'])
        env['tmp_0'] = env['tile_b'] / env['tile_a']
        env['tmp_1'] = env['tmp_0'] - env['tile_a']
        env['tmp_1'] = np.where(np.abs(env['tmp_1']) < 0.01, 1.0, env['tmp_1'])
        env['tile_b'] = np.where(np.abs(env['tile_b']) < 0.01, 1.0, env['tile_b'])
        env['tmp_2'] = env['tmp_1'] / env['tile_b']
        return env['tmp_2']

    def _numpy_kernel_1(self, a, b, c):
        """NumPy 实现: kernel_1"""
        # 创建变量环境
        env = {}
        env['tile_a'] = a.copy()
        env['tile_b'] = b.copy()
        env['tile_c'] = c.copy()

        # 执行操作链
        env['tmp_0'] = env['tile_a'] + env['tile_c']
        env['tmp_1'] = -env['tile_b']
        env['tmp_2'] = np.maximum(env['tmp_1'], env['tmp_1'])
        env['tmp_0'] = np.abs(env['tmp_0']) + 1e-6
        env['tmp_3'] = 1.0 / np.sqrt(env['tmp_0'])
        env['tmp_4'] = env['tmp_2'] + env['tmp_3']
        return env['tmp_4']


        # 顺序执行模式
        result_0 = self._numpy_kernel_0(tensors['a'], tensors['b'])
        result_1 = self._numpy_kernel_1(result_0, tensors['b'], tensors['c'])
        tensors['output'][:] = result_1


class TestMultiKernelFuzzing:
    """多内核模糊测试套件"""

    def test_fuzz_sequential_simple(self, test_runner):
        """测试 fuzz_sequential_simple"""
        test_case = TestFuzzSequentialSimple()
        result = test_runner.run(test_case)
        assert result.passed, f"测试失败: {result.error}"

