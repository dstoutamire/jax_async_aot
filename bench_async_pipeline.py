#!/usr/bin/env python3
"""
Benchmark: Async dispatch enables pipelining multiple operations.

This benchmark demonstrates that async dispatch allows overlapping
multiple GPU operations, achieving better throughput than serial execution.

Run with command buffers for async dispatch:
    XLA_FLAGS="--xla_gpu_enable_command_buffer=CONDITIONAL,WHILE" python3 bench_async_pipeline.py

Expected results:
- With sync dispatch: N operations take N * single_op_time
- With async dispatch: N operations take less than N * single_op_time
  because dispatch overhead is hidden by GPU execution
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
    return has_conditional and has_while  # Returns boolean


def create_workload():
    """Create a while-loop workload suitable for pipelining."""

    @jax.jit
    def workload(x, max_iter):
        """Iterative computation with dynamic bound."""
        def cond(state):
            i, val = state
            return i < max_iter

        def body(state):
            i, val = state
            val = val @ val.T
            val = val / jnp.max(jnp.abs(val))
            return (i + 1, val)

        init = (jnp.array(0, dtype=jnp.int32), x)
        _, result = lax.while_loop(cond, body, init)
        return result

    return workload


def measure_serial_execution(compiled_fn, inputs, max_iter, trials=5):
    """Measure time for serial execution: dispatch, wait, dispatch, wait, ..."""
    times = []

    for _ in range(trials):
        start = time.perf_counter()
        for x in inputs:
            result = compiled_fn(x, max_iter)
            result.block_until_ready()  # Wait before next dispatch
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def measure_pipelined_execution(compiled_fn, inputs, max_iter, trials=5):
    """Measure time for pipelined execution: dispatch all, then wait for all."""
    times = []

    for _ in range(trials):
        start = time.perf_counter()

        # Dispatch all operations (should return immediately if async)
        results = [compiled_fn(x, max_iter) for x in inputs]

        # Wait for all to complete
        for r in results:
            r.block_until_ready()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def measure_single_operation(compiled_fn, x, max_iter, warmup=3, trials=10):
    """Measure single operation time for reference."""
    # Warmup
    for _ in range(warmup):
        r = compiled_fn(x, max_iter)
        r.block_until_ready()

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        r = compiled_fn(x, max_iter)
        r.block_until_ready()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def run_benchmark():
    """Run the pipelining benchmark."""
    print("=" * 70)
    print("BENCHMARK: Async Dispatch Enables Pipelining")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    cmd_buf_enabled = check_command_buffer_enabled()
    if cmd_buf_enabled:
        print("MODE: Command buffers ENABLED (async dispatch expected)")
    else:
        print("MODE: Command buffers DISABLED (sync dispatch expected)")
        print("       To enable: XLA_FLAGS=\"--xla_gpu_enable_command_buffer=CONDITIONAL,WHILE\"")
    print()

    # Parameters
    matrix_size = 256
    iterations = 30
    batch_sizes = [1, 2, 4, 8]

    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"While loop iterations: {iterations}")
    print()

    # Create and compile workload
    workload = create_workload()
    key = jax.random.key(42)
    x_sample = jax.random.normal(key, (matrix_size, matrix_size))
    max_iter = jnp.array(iterations, dtype=jnp.int32)

    print("Compiling...")
    compiled = workload.lower(x_sample, max_iter).compile()
    print("Compilation complete.")
    print()

    # Measure single operation time
    single_time, single_std = measure_single_operation(compiled, x_sample, max_iter)
    print(f"Single operation time: {single_time*1000:.2f} +/- {single_std*1000:.2f} ms")
    print()

    # Run scaling test
    print("Pipelining Scaling Test:")
    print("-" * 70)
    print(f"{'Batch':<8} {'Serial (ms)':<15} {'Pipelined (ms)':<17} {'Speedup':<10} {'Time Saved':<10}")
    print("-" * 70)

    for batch_size in batch_sizes:
        # Create batch of inputs
        keys = jax.random.split(key, batch_size)
        inputs = [jax.random.normal(k, (matrix_size, matrix_size)) for k in keys]

        # Warmup
        for x in inputs:
            r = compiled(x, max_iter)
            r.block_until_ready()

        # Measure serial execution
        serial_time, serial_std = measure_serial_execution(compiled, inputs, max_iter)

        # Measure pipelined execution
        pipeline_time, pipeline_std = measure_pipelined_execution(compiled, inputs, max_iter)

        # Calculate metrics
        speedup = serial_time / pipeline_time
        # Overlap: how much time was saved by pipelining
        time_saved = serial_time - pipeline_time
        overlap_pct = (time_saved / serial_time * 100) if serial_time > 0 else 0

        print(f"{batch_size:<8} {serial_time*1000:<15.2f} {pipeline_time*1000:<17.2f} {speedup:<10.2f}x {overlap_pct:<10.1f}%")

    print("-" * 70)
    print()

    # Interpretation
    print("Interpretation:")
    print("- Speedup > 1: Pipelining provides benefit (async dispatch working)")
    print("- Speedup ~ 1: No pipelining benefit (sync dispatch)")
    print("- Time Saved: Percentage of time saved by overlapping operations")
    print()

    if not cmd_buf_enabled:
        print("TIP: Enable command buffers to see async dispatch benefits:")
        print("     XLA_FLAGS=\"--xla_gpu_enable_command_buffer=CONDITIONAL,WHILE\"")

    print("=" * 70)


def run_latency_vs_throughput():
    """Compare latency-optimized vs throughput-optimized execution."""
    print()
    print("=" * 70)
    print("COMPARISON: Latency vs Throughput Optimization")
    print("=" * 70)
    print()

    matrix_size = 256
    iterations = 30
    num_ops = 8

    workload = create_workload()
    key = jax.random.key(42)
    x_sample = jax.random.normal(key, (matrix_size, matrix_size))
    max_iter = jnp.array(iterations, dtype=jnp.int32)

    compiled = workload.lower(x_sample, max_iter).compile()

    # Create inputs
    keys = jax.random.split(key, num_ops)
    inputs = [jax.random.normal(k, (matrix_size, matrix_size)) for k in keys]

    # Warmup
    for x in inputs:
        r = compiled(x, max_iter)
        r.block_until_ready()

    # Measure latency-optimized (serial)
    start = time.perf_counter()
    for x in inputs:
        result = compiled(x, max_iter)
        result.block_until_ready()
    serial_total = time.perf_counter() - start

    # Measure throughput-optimized (pipelined)
    start = time.perf_counter()
    results = [compiled(x, max_iter) for x in inputs]
    dispatch_done = time.perf_counter() - start
    for r in results:
        r.block_until_ready()
    pipeline_total = time.perf_counter() - start

    print(f"Processing {num_ops} operations:")
    print()
    print("Serial execution (latency-optimized):")
    print(f"  Total time: {serial_total*1000:.2f} ms")
    print(f"  Per-op latency: {serial_total/num_ops*1000:.2f} ms")
    print()
    print("Pipelined execution (throughput-optimized):")
    print(f"  Dispatch time: {dispatch_done*1000:.2f} ms")
    print(f"  Total time: {pipeline_total*1000:.2f} ms")
    print(f"  Dispatch overhead: {dispatch_done/num_ops*1000:.2f} ms per op")
    print()
    print(f"Throughput improvement: {serial_total/pipeline_total:.2f}x")
    print()


if __name__ == "__main__":
    run_benchmark()
    run_latency_vs_throughput()
