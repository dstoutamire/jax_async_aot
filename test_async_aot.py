#!/usr/bin/env python3
"""
Test async AOT dispatch workaround.

This script tests whether the async AOT workaround actually provides
non-blocking dispatch compared to standard AOT dispatch.
"""

import time
import sys
sys.path.insert(0, '/home/dps/jax_async_aot')

import jax
import jax.numpy as jnp
import numpy as np

from async_aot import make_async_aot_caller, check_async_compatible


def test_basic_functionality():
    """Test that async caller produces correct results."""
    print("=" * 60)
    print("TEST: Basic Functionality")
    print("=" * 60)

    @jax.jit
    def matmul(x):
        return x @ x.T

    # Create test data
    x = jnp.ones((100, 100))

    # AOT compile
    lowered = jax.jit(matmul).lower(x)
    compiled = lowered.compile()

    # Check compatibility
    info = check_async_compatible(compiled)
    print(f"Compatibility check: {info}")

    # Create async caller
    async_caller = make_async_aot_caller(compiled)

    # Compare results
    result_standard = compiled(x)
    result_async = async_caller(x)

    # Block to get actual values
    result_standard.block_until_ready()
    result_async.block_until_ready()

    # Verify correctness
    np.testing.assert_allclose(
        np.array(result_standard),
        np.array(result_async),
        rtol=1e-5
    )
    print("PASS: Results match between standard and async AOT")
    print()


def test_async_behavior():
    """Test that async caller returns before computation completes."""
    print("=" * 60)
    print("TEST: Async Behavior (Non-blocking dispatch)")
    print("=" * 60)

    @jax.jit
    def heavy_compute(x):
        # Do some heavy computation
        for _ in range(10):
            x = x @ x.T
            x = x / jnp.max(x)  # Normalize to prevent overflow
        return x

    # Large matrix for measurable computation time
    size = 2000
    x = jax.random.normal(jax.random.key(0), (size, size))

    # Warmup and AOT compile
    lowered = jax.jit(heavy_compute).lower(x)
    compiled = lowered.compile()

    # Create async caller
    async_caller = make_async_aot_caller(compiled)

    # Warmup both paths
    _ = compiled(x)
    _.block_until_ready()
    _ = async_caller(x)
    _.block_until_ready()

    print(f"Matrix size: {size}x{size}")
    print()

    # Test standard AOT dispatch time (time until function returns)
    start = time.perf_counter()
    result_std = compiled(x)
    dispatch_time_std = time.perf_counter() - start
    result_std.block_until_ready()
    total_time_std = time.perf_counter() - start

    print(f"Standard AOT:")
    print(f"  Dispatch time (until return): {dispatch_time_std*1000:.3f} ms")
    print(f"  Total time (with block):      {total_time_std*1000:.3f} ms")
    print()

    # Test async AOT dispatch time
    start = time.perf_counter()
    result_async = async_caller(x)
    dispatch_time_async = time.perf_counter() - start
    result_async.block_until_ready()
    total_time_async = time.perf_counter() - start

    print(f"Async AOT:")
    print(f"  Dispatch time (until return): {dispatch_time_async*1000:.3f} ms")
    print(f"  Total time (with block):      {total_time_async*1000:.3f} ms")
    print()

    # Analysis
    print("Analysis:")
    if dispatch_time_async < dispatch_time_std * 0.5:
        print(f"  PASS: Async dispatch is {dispatch_time_std/dispatch_time_async:.1f}x faster")
    else:
        print(f"  NOTE: Dispatch times similar (async: {dispatch_time_async*1000:.3f}ms, std: {dispatch_time_std*1000:.3f}ms)")
        print(f"        This may indicate both are already async, or computation is very fast")

    # Verify results match
    np.testing.assert_allclose(
        np.array(result_std),
        np.array(result_async),
        rtol=1e-4
    )
    print("  PASS: Results match")
    print()


def test_jit_vs_aot_dispatch():
    """Compare JIT dispatch vs AOT dispatch timing."""
    print("=" * 60)
    print("TEST: JIT vs AOT Dispatch Comparison")
    print("=" * 60)

    @jax.jit
    def matmul(x):
        for _ in range(5):
            x = x @ x.T
            x = x / jnp.max(x)
        return x

    size = 2000
    x = jax.random.normal(jax.random.key(0), (size, size))

    # Warmup JIT
    jit_fn = jax.jit(matmul)
    _ = jit_fn(x)
    _.block_until_ready()

    # AOT compile
    lowered = jax.jit(matmul).lower(x)
    compiled = lowered.compile()
    _ = compiled(x)
    _.block_until_ready()

    # Create async caller
    async_caller = make_async_aot_caller(compiled)
    _ = async_caller(x)
    _.block_until_ready()

    # Measure JIT dispatch
    start = time.perf_counter()
    result_jit = jit_fn(x)
    jit_dispatch = time.perf_counter() - start
    result_jit.block_until_ready()
    jit_total = time.perf_counter() - start

    # Measure standard AOT dispatch
    start = time.perf_counter()
    result_aot = compiled(x)
    aot_dispatch = time.perf_counter() - start
    result_aot.block_until_ready()
    aot_total = time.perf_counter() - start

    # Measure async AOT dispatch
    start = time.perf_counter()
    result_async = async_caller(x)
    async_dispatch = time.perf_counter() - start
    result_async.block_until_ready()
    async_total = time.perf_counter() - start

    print(f"Matrix size: {size}x{size}")
    print()
    print(f"{'Method':<20} {'Dispatch (ms)':<15} {'Total (ms)':<15}")
    print("-" * 50)
    print(f"{'JIT':<20} {jit_dispatch*1000:<15.3f} {jit_total*1000:<15.3f}")
    print(f"{'AOT (standard)':<20} {aot_dispatch*1000:<15.3f} {aot_total*1000:<15.3f}")
    print(f"{'AOT (async)':<20} {async_dispatch*1000:<15.3f} {async_total*1000:<15.3f}")
    print()

    # Check if async AOT is closer to JIT behavior
    if async_dispatch < aot_dispatch * 0.8:
        print(f"SUCCESS: Async AOT dispatch ({async_dispatch*1000:.3f}ms) is faster than standard AOT ({aot_dispatch*1000:.3f}ms)")
    print()


def test_overlapping_computation():
    """Test that we can overlap computation with other work."""
    print("=" * 60)
    print("TEST: Overlapping Computation")
    print("=" * 60)

    @jax.jit
    def compute(x):
        for _ in range(10):
            x = x @ x.T
            x = x / jnp.max(x)
        return x

    size = 1500
    x = jax.random.normal(jax.random.key(0), (size, size))

    # AOT compile
    lowered = jax.jit(compute).lower(x)
    compiled = lowered.compile()
    async_caller = make_async_aot_caller(compiled)

    # Warmup
    _ = async_caller(x)
    _.block_until_ready()

    # Test: Launch async, do other work, then wait
    other_work_time = 0.05  # 50ms of "other work"

    # With async dispatch - should be able to overlap
    start = time.perf_counter()
    result = async_caller(x)
    dispatch_done = time.perf_counter() - start

    # Simulate other work
    time.sleep(other_work_time)
    work_done = time.perf_counter() - start

    result.block_until_ready()
    total = time.perf_counter() - start

    print(f"Async AOT with overlapping work:")
    print(f"  Dispatch time:     {dispatch_done*1000:.3f} ms")
    print(f"  After other work:  {work_done*1000:.3f} ms")
    print(f"  Total time:        {total*1000:.3f} ms")
    print(f"  Other work time:   {other_work_time*1000:.3f} ms")

    # If async is working, total time should be less than compute_time + other_work_time
    # because they overlap
    compute_only_start = time.perf_counter()
    result2 = async_caller(x)
    result2.block_until_ready()
    compute_only = time.perf_counter() - compute_only_start

    print(f"  Compute only time: {compute_only*1000:.3f} ms")

    if total < compute_only + other_work_time * 0.9:
        overlap = (compute_only + other_work_time - total) / other_work_time * 100
        print(f"  PASS: Achieved ~{overlap:.0f}% overlap with other work")
    else:
        print(f"  NOTE: Limited overlap detected")
    print()


def main():
    print("JAX Async AOT Dispatch Tests")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    test_basic_functionality()
    test_async_behavior()
    test_jit_vs_aot_dispatch()
    test_overlapping_computation()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
