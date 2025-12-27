#!/usr/bin/env python3
"""
Benchmark: While loop performance with/without command buffers.

This benchmark demonstrates the performance difference when using
XLA command buffers for while loops with dynamic bounds.

Run without command buffers (sync):
    python3 bench_command_buffer.py

Run with command buffers (async):
    XLA_FLAGS="--xla_gpu_enable_command_buffer=CONDITIONAL,WHILE" python3 bench_command_buffer.py

Expected results:
- Without command buffers: dispatch ratio ~100% (sync - blocks until done)
- With command buffers: dispatch ratio <20% (async - returns immediately)
"""

import os
import time
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np


def check_command_buffer_enabled():
    """Check if command buffer flags are set."""
    flags = os.environ.get("XLA_FLAGS", "")
    has_conditional = "CONDITIONAL" in flags
    has_while = "WHILE" in flags
    has_fusion = "FUSION" in flags
    has_min_size = "graph_min_graph_size" in flags
    return has_conditional and has_while, has_fusion, has_min_size


def create_while_loop_fn(iterations: int):
    """Create a function with a dynamic-bound while loop."""

    @jax.jit
    def fn(x, max_iter):
        """While loop that runs until condition met or max_iter reached."""
        def cond(state):
            i, val = state
            return i < max_iter

        def body(state):
            i, val = state
            # Do some computation each iteration
            val = val @ val.T
            val = val / jnp.max(jnp.abs(val))  # Normalize
            return (i + 1, val)

        init_state = (jnp.array(0, dtype=jnp.int32), x)
        final_i, final_val = lax.while_loop(cond, body, init_state)
        return final_val

    return fn


def measure_dispatch_ratio(compiled_fn, *args, warmup=3, trials=10):
    """Measure dispatch time vs total time."""
    # Warmup
    for _ in range(warmup):
        result = compiled_fn(*args)
        result.block_until_ready()

    dispatch_times = []
    total_times = []

    for _ in range(trials):
        start = time.perf_counter()
        result = compiled_fn(*args)
        dispatch_time = time.perf_counter() - start
        result.block_until_ready()
        total_time = time.perf_counter() - start

        dispatch_times.append(dispatch_time)
        total_times.append(total_time)

    return {
        "dispatch_ms": np.mean(dispatch_times) * 1000,
        "dispatch_std": np.std(dispatch_times) * 1000,
        "total_ms": np.mean(total_times) * 1000,
        "total_std": np.std(total_times) * 1000,
        "ratio": np.mean(dispatch_times) / np.mean(total_times),
    }


def run_benchmark():
    """Run the while loop benchmark."""
    print("=" * 70)
    print("BENCHMARK: While Loop Performance with Command Buffers")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    cmd_buf_enabled, has_fusion, has_min_size = check_command_buffer_enabled()
    if cmd_buf_enabled:
        print("MODE: Command buffers ENABLED")
        flags_info = ["CONDITIONAL", "WHILE"]
        if has_fusion:
            flags_info.append("FUSION")
        if has_min_size:
            flags_info.append("graph_min_graph_size=1")
        print(f"       Flags: {', '.join(flags_info)}")
        if not has_fusion or not has_min_size:
            print("       TIP: For best results, use:")
            print("       XLA_FLAGS=\"--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1\"")
    else:
        print("MODE: Command buffers DISABLED (baseline)")
        print("       To enable:")
        print("       XLA_FLAGS=\"--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1\"")
    print()

    # Test parameters - use same size as comparison for consistency
    matrix_size = 256
    iterations = 50

    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"While loop iterations: {iterations}")
    print()

    # Create function and compile
    fn = create_while_loop_fn(iterations)
    x = jax.random.normal(jax.random.key(42), (matrix_size, matrix_size))
    max_iter = jnp.array(iterations, dtype=jnp.int32)

    print("Compiling...")
    lowered = fn.lower(x, max_iter)
    compiled = lowered.compile()
    print("Compilation complete.")
    print()

    # Run benchmark
    print("Running benchmark (10 trials)...")
    results = measure_dispatch_ratio(compiled, x, max_iter)

    print()
    print("Results:")
    print(f"  Dispatch time: {results['dispatch_ms']:.2f} +/- {results['dispatch_std']:.2f} ms")
    print(f"  Total time:    {results['total_ms']:.2f} +/- {results['total_std']:.2f} ms")
    print(f"  Dispatch ratio: {results['ratio']:.1%}")
    print()

    # Interpretation
    if results['ratio'] < 0.5:
        print("RESULT: ASYNC dispatch detected!")
        print(f"        Dispatch returns in {results['ratio']:.0%} of total time.")
        print("        GPU computation continues asynchronously.")
    else:
        print("RESULT: SYNC dispatch detected.")
        print(f"        Dispatch blocks for {results['ratio']:.0%} of total time.")
        if not cmd_buf_enabled:
            print("        Enable command buffers to achieve async dispatch.")

    print()
    print("=" * 70)

    return results


def run_comparison():
    """Compare different iteration counts."""
    print()
    print("=" * 70)
    print("COMPARISON: Dispatch ratio vs iteration count")
    print("=" * 70)
    print()

    matrix_size = 256
    iteration_counts = [10, 25, 50, 100]

    x = jax.random.normal(jax.random.key(42), (matrix_size, matrix_size))

    print(f"{'Iterations':<12} {'Dispatch (ms)':<15} {'Total (ms)':<15} {'Ratio':<10}")
    print("-" * 52)

    for iters in iteration_counts:
        fn = create_while_loop_fn(iters)
        max_iter = jnp.array(iters, dtype=jnp.int32)
        compiled = fn.lower(x, max_iter).compile()

        results = measure_dispatch_ratio(compiled, x, max_iter, warmup=2, trials=5)

        print(f"{iters:<12} {results['dispatch_ms']:<15.2f} {results['total_ms']:<15.2f} {results['ratio']:<10.1%}")

    print()


if __name__ == "__main__":
    run_benchmark()
    run_comparison()
