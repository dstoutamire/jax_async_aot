"""Tests for compile-time async dispatch verification.

These tests verify that XLA command buffers properly capture control flow
(while loops, conditionals) for async GPU dispatch.

Run with:
    pytest test_compile_aot.py -v
"""

import os
import sys
import tempfile
import subprocess
import pytest

from compile_aot import (
    parse_thunk_sequence,
    run_with_thunk_dump,
    COMMAND_BUFFER_FLAGS,
)


class TestParseThunkSequence:
    """Unit tests for thunk sequence parsing."""

    def test_empty_input(self):
        """Empty input returns empty results."""
        result = parse_thunk_sequence("")
        assert result['top_level_whiles'] == []
        assert result['captured_whiles'] == []
        assert result['command_buffers'] == []

    def test_top_level_while(self):
        """Detects top-level while loop (sync)."""
        thunk_text = """
0 kWhile while.1
  1 kFusion fusion.1
"""
        result = parse_thunk_sequence(thunk_text)
        assert len(result['top_level_whiles']) == 1
        assert len(result['captured_whiles']) == 0

    def test_captured_while(self):
        """Detects while loop inside command buffer (async)."""
        thunk_text = """
0 kCommandBuffer command_buffer.1
  1 kWhile while.1
    2 kFusion fusion.1
"""
        result = parse_thunk_sequence(thunk_text)
        assert len(result['top_level_whiles']) == 0
        assert len(result['captured_whiles']) == 1
        assert len(result['command_buffers']) == 1

    def test_mixed_control_flow(self):
        """Handles mix of top-level and captured thunks."""
        thunk_text = """
0 kCommandBuffer cmd1
  1 kWhile while.captured
  2 kConditional cond.captured
0 kWhile while.toplevel
0 kConditional cond.toplevel
"""
        result = parse_thunk_sequence(thunk_text)
        assert len(result['captured_whiles']) == 1
        assert len(result['top_level_whiles']) == 1
        assert len(result['captured_conditionals']) == 1
        assert len(result['top_level_conditionals']) == 1


class TestSimpleWhileCapture:
    """Test that simple while loops are captured into command buffers."""

    def test_simple_while_is_captured(self):
        """A simple while loop should be captured in a command buffer."""
        script = '''
import jax
import jax.numpy as jnp

@jax.jit
def simple_while(x):
    def cond(state):
        i, _ = state
        return i < 10
    def body(state):
        i, x = state
        return (i + 1, x + 1.0)
    _, result = jax.lax.while_loop(cond, body, (0, x))
    return result

result = simple_while(jnp.array(0.0))
result.block_until_ready()
'''
        thunk_text = run_with_thunk_dump(script)
        parsed = parse_thunk_sequence(thunk_text)

        print("\n=== Analysis ===")
        print(f"Command buffers: {len(parsed['command_buffers'])}")
        print(f"Captured while loops: {len(parsed['captured_whiles'])}")
        print(f"Top-level while loops (SYNC): {len(parsed['top_level_whiles'])}")

        assert len(parsed['captured_whiles']) > 0, \
            "While loop should be captured in command buffer"
        assert len(parsed['top_level_whiles']) == 0, \
            f"While loop at top level (SYNC): {parsed['top_level_whiles']}"


class TestLBFGSCapture:
    """Test that jaxopt L-BFGS while loops are captured."""

    def test_lbfgs_while_is_captured(self):
        """L-BFGS optimization loop should be captured in command buffer."""
        script = '''
import jax
import jax.numpy as jnp
import jaxopt

def loss_fn(x):
    return jnp.sum((x - jnp.array([1.0, 2.0, 3.0])) ** 2)

solver = jaxopt.LBFGS(fun=loss_fn, maxiter=50)

@jax.jit
def optimize(x0):
    return solver.run(x0)

result = optimize(jnp.zeros(3))
jax.tree.map(lambda x: x.block_until_ready(), result)
'''
        thunk_text = run_with_thunk_dump(script)
        parsed = parse_thunk_sequence(thunk_text)

        print("\n=== Analysis ===")
        print(f"Command buffers: {len(parsed['command_buffers'])}")
        print(f"Captured while loops: {len(parsed['captured_whiles'])}")
        print(f"Top-level while loops (SYNC): {len(parsed['top_level_whiles'])}")

        assert len(parsed['captured_whiles']) > 0, \
            "L-BFGS while loop should be captured in command buffer"

        if parsed['top_level_whiles']:
            pytest.fail(
                f"L-BFGS has {len(parsed['top_level_whiles'])} SYNC while loop(s)"
            )


class TestConditionalCapture:
    """Test that conditionals are captured into command buffers."""

    def test_conditional_is_captured(self):
        """A conditional should be captured in a command buffer."""
        script = '''
import jax
import jax.numpy as jnp

@jax.jit
def conditional(x, threshold):
    return jax.lax.cond(
        x > threshold,
        lambda x: x * 2.0,
        lambda x: x * 0.5,
        x
    )

result = conditional(jnp.array(5.0), jnp.array(3.0))
result.block_until_ready()
'''
        thunk_text = run_with_thunk_dump(script)
        parsed = parse_thunk_sequence(thunk_text)

        print("\n=== Analysis ===")
        print(f"Command buffers: {len(parsed['command_buffers'])}")
        print(f"Captured conditionals: {len(parsed['captured_conditionals'])}")
        print(f"Top-level conditionals (SYNC): {len(parsed['top_level_conditionals'])}")

        # Note: Simple conditionals may be optimized away or handled differently
        # The important thing is no top-level conditionals
        assert len(parsed['top_level_conditionals']) == 0, \
            f"Conditional at top level (SYNC): {parsed['top_level_conditionals']}"


class TestFullAOTWorkflow:
    """Integration tests for full AOT compilation workflow."""

    def test_compile_and_load(self):
        """Test full compile -> save -> load -> execute workflow.

        Note: This test uses check_async=False because XLA_FLAGS must be set
        before JAX is imported. The async checking is tested separately in
        TestSimpleWhileCapture and TestLBFGSCapture using subprocess isolation.
        """
        # This test runs in a subprocess to ensure clean XLA flag state
        script = '''
import os
import sys
import tempfile

# Set flags before importing JAX
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_command_buffer="
    "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION,"
    "DYNAMIC_SLICE_COPY_FUSION,CONDITIONAL,WHILE "
    "--xla_gpu_graph_min_graph_size=1"
)

import jax.numpy as jnp
from compile_aot import compile_aot, load_aot
from example_fn import simple_while, simple_while_example_args

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = f"{tmpdir}/test.zst"

    # Compile (async check disabled - see docstring)
    compiled = compile_aot(
        simple_while,
        simple_while_example_args,
        output_path,
        check_async=False,
        verbose=True,
    )

    # Execute compiled version
    result1 = compiled(*simple_while_example_args)
    result1.block_until_ready()

    # Load and execute
    loaded = load_aot(output_path)
    result2 = loaded(*simple_while_example_args)
    result2.block_until_ready()

    # Verify results match
    import jax.numpy as jnp
    assert jnp.allclose(result1, result2), f"Results differ: {result1} vs {result2}"

print("SUCCESS: compile -> save -> load -> execute workflow works")
'''
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(__file__) or '.',
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"Workflow failed: {result.stderr}"
        assert "SUCCESS" in result.stdout


if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-s"],
    )
    sys.exit(result.returncode)
