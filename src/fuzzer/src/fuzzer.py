"""
Operator fuzzer for generating random operator combinations.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # Used in lambda functions for op equivalents


@dataclass
class OpSpec:
    """Operator specification for fuzzing.

    Attributes:
        name: Operator name (e.g., "block.add")
        input_types: List of input types (e.g., ["tile", "tile"])
        output_type: Output type (e.g., "tile")
        constraints: Additional constraints (e.g., {"min_shape": [64, 64]})
        np_equivalent: NumPy equivalent function for golden reference
        shape_transform: Optional callable that computes output shape from input shapes
        param_generator: Optional callable that generates operator parameters
        requires_params: Whether this operator requires parameters (default: False)
    """
    name: str
    input_types: List[str]
    output_type: str
    constraints: Dict[str, Any]
    np_equivalent: Optional[Any] = None
    shape_transform: Optional[Any] = None
    param_generator: Optional[Any] = None
    requires_params: bool = False

    def compute_output_shape(self, input_shapes: List[Tuple[int, int]], params: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
        """Compute output shape from input shapes."""
        if self.shape_transform:
            import inspect
            sig = inspect.signature(self.shape_transform)
            if len(sig.parameters) >= 2 and params is not None:
                return self.shape_transform(input_shapes, params)
            else:
                return self.shape_transform(input_shapes)
        return input_shapes[0] if input_shapes else (128, 128)

    def generate_params(self, input_shapes: List[Tuple[int, int]], rng) -> Dict[str, Any]:
        """Generate operator parameters based on input shapes."""
        if self.param_generator and self.requires_params:
            return self.param_generator(input_shapes, rng)
        return {}


class OpFuzzer:
    """Generates random operator combinations for fuzzing."""

    # Block-level binary operators
    BLOCK_BINARY_OPS = [
        OpSpec("block.add", ["tile", "tile"], "tile", {}, lambda a, b: a + b),
        OpSpec("block.sub", ["tile", "tile"], "tile", {}, lambda a, b: a - b),
        OpSpec("block.mul", ["tile", "tile"], "tile", {}, lambda a, b: a * b),
        OpSpec("block.div", ["tile", "tile"], "tile", {"avoid_zero": True}, lambda a, b: a / b),
        OpSpec("block.maximum", ["tile", "tile"], "tile", {}, lambda a, b: np.maximum(a, b)),
    ]

    # Block-level scalar operators
    BLOCK_SCALAR_OPS = [
        OpSpec("block.adds", ["tile", "scalar"], "tile", {}, lambda a, s: a + s),
        OpSpec("block.subs", ["tile", "scalar"], "tile", {}, lambda a, s: a - s),
        OpSpec("block.muls", ["tile", "scalar"], "tile", {}, lambda a, s: a * s),
        OpSpec("block.divs", ["tile", "scalar"], "tile", {"avoid_zero": True}, lambda a, s: a / s),
    ]

    # Block-level unary operators
    BLOCK_UNARY_OPS = [
        OpSpec("block.sqrt", ["tile"], "tile", {"positive_only": True}, lambda a: np.sqrt(a)),
        OpSpec("block.rsqrt", ["tile"], "tile", {"positive_only": True}, lambda a: 1.0 / np.sqrt(a)),
        OpSpec("block.exp", ["tile"], "tile", {}, lambda a: np.exp(np.clip(a, -10, 10))),
        OpSpec("block.neg", ["tile"], "tile", {}, lambda a: -a),
        OpSpec("block.recip", ["tile"], "tile", {"avoid_zero": True}, lambda a: 1.0 / a),
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize fuzzer with optional seed for reproducibility."""
        self.rng = random.Random(seed)
        # 使用所有操作符
        self.ops = self.BLOCK_BINARY_OPS + self.BLOCK_SCALAR_OPS + self.BLOCK_UNARY_OPS

    def generate_op_chain(
        self,
        num_ops: int = 5,
        input_count: int = 2,
        allow_scalars: bool = True,
        track_shapes: bool = False,
        default_shape: Tuple[int, int] = (128, 128),
    ) -> List[Dict[str, Any]]:
        """Generate a chain of operator calls.

        All input tensors and intermediate results are guaranteed to contribute
        to the final output through smart generation and post-processing.
        """
        # Initialize available variables
        available_tiles = [f"tile_{chr(97 + i)}" for i in range(input_count)]
        available_scalars = ["1.0", "2.0", "0.5"] if allow_scalars else []

        # Track which initial inputs have been used
        initial_inputs = set(available_tiles)
        used_inputs = set()

        # Track usage count for each variable
        variable_usage_count = {tile: 0 for tile in available_tiles}

        # Shape tracking (optional)
        variable_shapes = {}
        if track_shapes:
            for tile in available_tiles:
                variable_shapes[tile] = default_shape

        operations = []

        for i in range(num_ops):
            # Calculate urgency for using unused inputs
            unused_count = len(initial_inputs - used_inputs)
            remaining_ops = num_ops - i

            # Dynamic priority
            use_unused_priority = 0.7
            if unused_count > 0:
                if unused_count >= remaining_ops:
                    use_unused_priority = 1.0
                elif remaining_ops > 0:
                    use_unused_priority = min(0.9, 0.7 + 0.3 * (unused_count / remaining_ops))

            # Select eligible operators
            eligible_ops = self._get_eligible_ops(
                available_tiles,
                available_scalars,
                allow_scalars,
                variable_shapes if track_shapes else None,
            )

            if not eligible_ops:
                break

            # Prioritize binary ops if we need to use unused inputs
            if unused_count > 0 and use_unused_priority >= 0.9:
                binary_ops = [op for op in eligible_ops if sum(1 for t in op.input_types if t == "tile") >= 2]
                if binary_ops:
                    eligible_ops = binary_ops

            op = self.rng.choice(eligible_ops)

            # Select inputs
            inputs = []
            scalar_value = None

            for input_type in op.input_types:
                if input_type == "tile":
                    candidate_tiles = available_tiles

                    if track_shapes:
                        candidate_tiles = [
                            t for t in candidate_tiles
                            if self._is_shape_compatible(op, t, variable_shapes)
                        ]
                        if not candidate_tiles:
                            continue

                    # Smart selection: prioritize unused inputs
                    unused_initial_inputs = {
                        t for t in candidate_tiles
                        if t in initial_inputs and t not in used_inputs
                    }

                    candidate_scores = []
                    for t in candidate_tiles:
                        score = 0

                        if t in unused_initial_inputs:
                            score += 50
                            if use_unused_priority >= 0.9:
                                score += 30

                        usage = variable_usage_count.get(t, 0)
                        score += max(0, 20 - usage * 5)

                        if t.startswith("tmp_"):
                            score += 5

                        candidate_scores.append((t, score))

                    if candidate_scores:
                        max_score = max(score for _, score in candidate_scores)

                        if max_score >= 40:
                            threshold = max(max_score * 0.6, 30)
                            top_candidates = [t for t, score in candidate_scores if score >= threshold]

                            if top_candidates and self.rng.random() < 0.85:
                                candidate_tiles = top_candidates
                        else:
                            min_score_needed = max(max_score * 0.7, 10)
                            preferred = [t for t, score in candidate_scores if score >= min_score_needed]
                            if preferred and self.rng.random() < 0.75:
                                candidate_tiles = preferred

                    selected_input = self.rng.choice(candidate_tiles)
                    inputs.append(selected_input)

                    variable_usage_count[selected_input] = variable_usage_count.get(selected_input, 0) + 1

                    if selected_input in initial_inputs:
                        used_inputs.add(selected_input)

                elif input_type == "scalar":
                    if self.rng.random() < 0.5 and available_scalars:
                        scalar_value = self.rng.choice(available_scalars)
                    else:
                        scalar_value = f"{self.rng.uniform(0.1, 10.0):.2f}"
                    inputs.append(scalar_value)

            output = f"tmp_{i}"

            # Generate operator parameters if required
            params = None
            if op.requires_params:
                input_shapes = [variable_shapes[inp] for inp in inputs if inp in variable_shapes]
                if input_shapes:
                    params = op.generate_params(input_shapes, self.rng)

            op_dict = {
                "op": op,
                "inputs": inputs,
                "output": output,
                "scalar_value": scalar_value,
                "params": params,
            }

            # Compute output shape if tracking
            if track_shapes:
                input_shapes = [variable_shapes[inp] for inp in inputs if inp in variable_shapes]
                output_shape = op.compute_output_shape(input_shapes, params)
                op_dict["output_shape"] = output_shape
                variable_shapes[output] = output_shape

            operations.append(op_dict)
            available_tiles.append(output)
            variable_usage_count[output] = 0

        # Ensure all initial inputs are used
        unused_inputs = initial_inputs - used_inputs
        if unused_inputs:
            add_op = next((op for op in self.BLOCK_BINARY_OPS if op.name == "block.add"), None)

            for unused_input in unused_inputs:
                if operations:
                    current_final = operations[-1]["output"]
                    output = f"tmp_{len(operations)}"

                    op_dict = {
                        "op": add_op,
                        "inputs": [unused_input, current_final],
                        "output": output,
                        "scalar_value": None,
                        "params": None,
                    }

                    if track_shapes:
                        input_shapes = [
                            variable_shapes.get(unused_input, default_shape),
                            variable_shapes.get(current_final, default_shape)
                        ]
                        output_shape = add_op.compute_output_shape(input_shapes)
                        op_dict["output_shape"] = output_shape
                        variable_shapes[output] = output_shape

                    operations.append(op_dict)
                    available_tiles.append(output)
                    used_inputs.add(unused_input)
                    variable_usage_count[output] = 0
                    variable_usage_count[unused_input] = variable_usage_count.get(unused_input, 0) + 1
                    variable_usage_count[current_final] = variable_usage_count.get(current_final, 0) + 1

        # Ensure all intermediate results contribute to the final output
        if operations:
            final_output = operations[-1]["output"]
            unused_intermediates = []

            for var_name, usage_count in variable_usage_count.items():
                if var_name.startswith("tmp_") and usage_count == 0 and var_name != final_output:
                    unused_intermediates.append(var_name)

            if unused_intermediates:
                add_op = next((op for op in self.BLOCK_BINARY_OPS if op.name == "block.add"), None)

                for unused_var in unused_intermediates:
                    current_final = operations[-1]["output"]
                    output = f"tmp_{len(operations)}"

                    op_dict = {
                        "op": add_op,
                        "inputs": [unused_var, current_final],
                        "output": output,
                        "scalar_value": None,
                        "params": None,
                    }

                    if track_shapes:
                        input_shapes = [
                            variable_shapes.get(unused_var, default_shape),
                            variable_shapes.get(current_final, default_shape)
                        ]
                        output_shape = add_op.compute_output_shape(input_shapes)
                        op_dict["output_shape"] = output_shape
                        variable_shapes[output] = output_shape

                    operations.append(op_dict)
                    available_tiles.append(output)
                    variable_usage_count[output] = 0
                    variable_usage_count[unused_var] = variable_usage_count.get(unused_var, 0) + 1
                    variable_usage_count[current_final] = variable_usage_count.get(current_final, 0) + 1

        return operations

    def _get_eligible_ops(
        self,
        available_tiles: List[str],
        available_scalars: List[str],
        allow_scalars: bool,
        variable_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> List[OpSpec]:
        """Get operators that can be applied with current variables."""
        eligible = []

        for op in self.ops:
            tile_inputs = sum(1 for t in op.input_types if t == "tile")
            scalar_inputs = sum(1 for t in op.input_types if t == "scalar")

            has_tiles = len(available_tiles) >= tile_inputs
            has_scalars = (scalar_inputs == 0) or (allow_scalars and
                          (len(available_scalars) >= scalar_inputs or scalar_inputs > 0))

            if has_tiles and has_scalars:
                eligible.append(op)

        return eligible

    def _is_shape_compatible(
        self,
        op: OpSpec,
        var: str,
        variable_shapes: Dict[str, Tuple[int, int]]
    ) -> bool:
        """Check if a variable's shape is compatible with an operator."""
        if var not in variable_shapes:
            return True
        return True  # All current ops are compatible with any shape

    def generate_numpy_reference(
        self,
        op_chain: List[Dict[str, Any]],
        input_tensors: Dict[str, Any],
    ) -> Any:
        """Generate NumPy golden reference from operation chain."""
        import numpy as np

        # Create variable environment
        env = {}
        for name, tensor in input_tensors.items():
            env[f"tile_{name}"] = tensor.copy()

        # Execute operations
        for op_dict in op_chain:
            op = op_dict["op"]
            inputs = op_dict["inputs"]
            output = op_dict["output"]
            params = op_dict.get("params")

            # Get input values
            input_vals = []
            for inp in inputs:
                if inp in env:
                    val = env[inp]
                else:
                    val = float(inp)
                input_vals.append(val)

            # Apply constraints
            if "avoid_zero" in op.constraints and op.constraints["avoid_zero"]:
                for i, val in enumerate(input_vals):
                    if isinstance(val, np.ndarray):
                        input_vals[i] = np.where(np.abs(val) < 0.01, 1.0, val)

            if "positive_only" in op.constraints and op.constraints["positive_only"]:
                for i, val in enumerate(input_vals):
                    if isinstance(val, np.ndarray):
                        input_vals[i] = np.abs(val) + 1e-6

            # Execute operation
            if op.np_equivalent:
                import inspect
                sig = inspect.signature(op.np_equivalent)
                if params and len(sig.parameters) > len(input_vals):
                    result = op.np_equivalent(*input_vals, params)
                else:
                    result = op.np_equivalent(*input_vals)
                env[output] = result

        # Return final result
        if op_chain:
            return env[op_chain[-1]["output"]]
        else:
            return input_tensors[list(input_tensors.keys())[0]]
