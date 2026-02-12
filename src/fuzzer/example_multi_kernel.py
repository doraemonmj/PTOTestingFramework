"""
多内核模糊测试框架使用示例

该脚本演示如何使用多内核测试生成器创建测试用例。
支持通过命令行参数控制生成的测试用例数量和配置。

使用方法:
    python example_multi_kernel.py --num-cases 5
"""

import argparse
import sys
from pathlib import Path

# 添加当前目录到路径
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.multi_kernel_test_generator import MultiKernelTestGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成多内核模糊测试用例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置生成测试用例
  python example_multi_kernel.py

  # 指定配置索引（从0开始）
  python example_multi_kernel.py --config-index 0

  # 指定输出文件
  python example_multi_kernel.py --output custom_test.py

  # 设置误差容忍度
  python example_multi_kernel.py --atol 1e-3 --rtol 1e-3

  # 组合使用
  python example_multi_kernel.py --config-index 1 --atol 1e-4 --rtol 1e-4 --output my_test.py
        """
    )

    parser.add_argument(
        "--config-index",
        type=int,
        default=0,
        help="指定要使用的配置索引（从0开始），如果不指定则使用所有配置"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: src/fuzzer/generated_tests/test_fuzz_multi_kernel.py)"
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="绝对误差容忍度 (默认: 1e-5)"
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="相对误差容忍度 (默认: 1e-5)"
    )

    args = parser.parse_args()

    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = str(_SCRIPT_DIR / "generated_tests" / "test_fuzz_multi_kernel.py")

    # 定义不同配置的测试用例
    # 每个配置可以生成多个测试实例（通过 num_instances 控制）
    all_configs = [
        {
            "name": "fuzz_sequential_simple",
            "num_instances": 1,  # 从这个配置生成1个测试用例
            "seed": 4,
            "enable_advanced_ops": False,
            "num_kernels": 2,
            "mode": "sequential",
            "shape": (128, 128),
            "num_ops_range": (3, 5),
            "tensor_init_type": "constant",
            "input_shapes_list": [
                [(128, 128), (128, 128)],  # kernel_0: 2个相同维度的输入
            ],
            "description": "简单顺序执行：2个内核，相同维度输入"
        },
        # {
        #     "name": "fuzz_branching_parallel",
        #     "num_instances": 1,  # 从这个配置生成2个测试用例
        #     "seed": 42,
        #     "enable_advanced_ops": False,
        #     "num_kernels": 3,
        #     "mode": "branching",
        #     "shape": (128, 128),
        #     "num_ops_range": (4, 6),
        #     "tensor_init_type": "random",
        #     "input_shapes_list": [
        #         [(128, 128), (128, 128)],  # kernel_0: 2个相同维度
        #         [(128, 128), (128, 128)],  # kernel_1: 2个相同维度
        #         [(128, 128)],              # kernel_2: 1个输入
        #     ],
        #     "description": "分支并行执行：3个内核，相同维度输入"
        # },
        # {
        #     "name": "fuzz_mixed_complex",
        #     "num_instances": 1,
        #     "seed": 100,
        #     "enable_advanced_ops": False,
        #     "num_kernels": 4,
        #     "mode": "mixed",
        #     "shape": (128, 128),
        #     "num_ops_range": (5, 8),
        #     "tensor_init_type": "range",
        #     "input_shapes_list": None,  # 使用随机生成
        #     "description": "混合模式：前2个并行，后2个顺序，随机输入"
        # },
        # {
        #     "name": "fuzz_sequential_deep",
        #     "num_instances": 1,
        #     "seed": 200,
        #     "enable_advanced_ops": False,
        #     "num_kernels": 5,
        #     "mode": "sequential",
        #     "shape": (128, 128),
        #     "num_ops_range": (6, 10),
        #     "tensor_init_type": "normal",
        #     "input_shapes_list": None,  # 使用随机生成
        #     "description": "深度顺序执行：5个内核链式调用，随机输入"
        # },
        # {
        #     "name": "fuzz_branching_wide",
        #     "num_instances": 1,
        #     "seed": 300,
        #     "enable_advanced_ops": False,
        #     "num_kernels": 4,
        #     "mode": "branching",
        #     "shape": (128, 128),
        #     "num_ops_range": (4, 7),
        #     "tensor_init_type": "ones",
        #     "input_shapes_list": [
        #         [(128, 128), (128, 128), (128, 128)],  # kernel_0: 3个相同维度
        #         [(128, 128)],                          # kernel_1: 1个输入
        #         [(128, 128), (128, 128)],              # kernel_2: 2个相同维度
        #         [(128, 128), (128, 128)],              # kernel_3: 2个相同维度
        #     ],
        #     "description": "宽分支执行：4个内核，统一维度输入"
        # },
    ]

    # 根据 config_index 选择配置
    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(all_configs):
            print(f"错误: 配置索引 {args.config_index} 超出范围 (0-{len(all_configs)-1})")
            return
        selected_configs = [all_configs[args.config_index]]
    else:
        selected_configs = all_configs

    # 计算总测试用例数
    total_test_cases = sum(config.get("num_instances", 1) for config in selected_configs)

    print(f"多内核模糊测试生成器")
    print(f"=" * 60)
    print(f"配置数量: {len(selected_configs)}")
    print(f"总测试用例数: {total_test_cases}")
    print(f"输出文件: {output_path}")
    print(f"绝对误差容忍度 (atol): {args.atol}")
    print(f"相对误差容忍度 (rtol): {args.rtol}")
    print(f"=" * 60)
    print()

    print("将生成以下测试用例:")
    print()
    test_case_num = 1
    for config_idx, config in enumerate(selected_configs):
        num_instances = config.get("num_instances", 1)
        print(f"配置 {config_idx}: {config['name']}")
        print(f"   {config['description']}")
        print(f"   实例数量: {num_instances}")
        print(f"   随机种子: {config.get('seed', 42)}")
        print(f"   启用高级算子: {'是' if config.get('enable_advanced_ops', False) else '否'}")
        print(f"   张量初始化: {config.get('tensor_init_type', 'constant')}")
        if num_instances > 1:
            print(f"   将生成测试用例: {test_case_num} - {test_case_num + num_instances - 1}")
        else:
            print(f"   将生成测试用例: {test_case_num}")
        test_case_num += num_instances
        print()

    # 展开配置，为每个实例创建一个测试用例
    expanded_test_configs = []
    for config in selected_configs:
        num_instances = config.get("num_instances", 1)
        base_seed = config.get("seed", 42)

        for instance_idx in range(num_instances):
            # 为每个实例创建一个配置副本
            test_config = config.copy()

            # 如果有多个实例，在名称后添加索引
            if num_instances > 1:
                test_config["name"] = f"{config['name']}_{instance_idx}"
                # 每个实例使用不同的种子
                test_config["seed"] = base_seed + instance_idx

            expanded_test_configs.append(test_config)

    # 生成测试文件
    print("正在生成测试文件...")

    # 为每个配置创建独立的生成器（使用各自的种子和配置）
    all_test_cases = []
    for test_config in expanded_test_configs:
        generator = MultiKernelTestGenerator(
            seed=test_config.get("seed", 42),
            enable_advanced_ops=test_config.get("enable_advanced_ops", False),
            tensor_init_type=test_config.get("tensor_init_type", "constant")
        )

        test_code = generator.generate_test_case(
            test_name=test_config["name"],
            num_kernels=test_config.get("num_kernels", 3),
            orchestration_mode=test_config.get("mode", "sequential"),
            shape=test_config.get("shape", (128, 128)),
            num_ops_range=test_config.get("num_ops_range", (3, 7)),
            input_shapes_list=test_config.get("input_shapes_list"),
            tensor_init_type=test_config.get("tensor_init_type"),
            atol=args.atol,
            rtol=args.rtol,
        )
        all_test_cases.append(test_code)

    # 生成文件头部
    file_header = '''"""
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


'''

    # 生成测试套件类
    test_suite = f'''
class TestMultiKernelFuzzing:
    """多内核模糊测试套件"""

'''

    # 为每个测试用例添加测试方法
    for test_config in expanded_test_configs:
        test_name = test_config["name"]
        test_suite += f'''    def test_{test_name}(self, test_runner):
        """测试 {test_name}"""
        test_case = Test{test_name.title().replace("_", "")}()
        result = test_runner.run(test_case)
        assert result.passed, f"测试失败: {{result.error}}"

'''

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(file_header)
        f.write('\n\n'.join(all_test_cases))
        f.write('\n\n')
        f.write(test_suite)

    print()
    print(f"✓ 成功生成 {len(expanded_test_configs)} 个测试用例")
    print(f"✓ 输出文件: {output_path}")
    print()
    print("运行测试:")
    print(f"  pytest {output_path}")
    print()


if __name__ == "__main__":
    main()
