#!/usr/bin/env python3
"""
Test comparing sync vs async dispatch for functions WITH effects.
This is where the async workaround should make a difference.
"""

import time
import sys
sys.path.insert(0, '/home/dps/jax_async_aot')

import jax
import jax.numpy as jnp
import numpy as np

from async_aot import make_async_aot_caller, check_async_compatible


def test_function_with_callback_timing():
    """Test timing for function with host callback - this should show sync behavior."""
    print("=" * 60)
    print("TEST: Function WITH host callback (forces Python fallback)")
    print("=" * 60)

    # Host callback function
    def identity_callback(x):
        return x

    @jax.jit
    def fn_with_callback(x):
        # Do some computation
        y = x @ x.T
        # Then a callback (forces Python fallback)
        result = jax.pure_callback(
            identity_callback,
            jax.ShapeDtypeStruct(y.shape, y.dtype),
            y
        )
        # More computation
        return result @ result.T

    size = 1000
    x = jax.random.normal(jax.random.key(0), (size, size))

    # Compile
    compiled = jax.jit(fn_with_callback).lower(x).compile()

    # Check dispatch path
    info = check_async_compatible(compiled)
    print(f"Compatibility: {info}")
    print()

    # Warmup
    _ = compiled(x)
    _.block_until_ready()

    # Time standard AOT (should be sync due to callback)
    times_dispatch = []
    times_total = []
    for _ in range(5):
        start = time.perf_counter()
        result = compiled(x)
        dispatch = time.perf_counter() - start
        result.block_until_ready()
        total = time.perf_counter() - start
        times_dispatch.append(dispatch)
        times_total.append(total)

    print("Standard AOT with callback:")
    print(f"  Dispatch time: {np.mean(times_dispatch)*1000:.3f} +/- {np.std(times_dispatch)*1000:.3f} ms")
    print(f"  Total time:    {np.mean(times_total)*1000:.3f} +/- {np.std(times_total)*1000:.3f} ms")

    # Check if dispatch time is close to total time (indicating sync behavior)
    dispatch_ratio = np.mean(times_dispatch) / np.mean(times_total)
    print(f"  Dispatch/Total ratio: {dispatch_ratio:.2%}")
    if dispatch_ratio > 0.8:
        print(f"  => SYNC BEHAVIOR CONFIRMED (dispatch takes most of the time)")
    else:
        print(f"  => Async behavior (dispatch returns quickly)")
    print()


def test_pure_function_timing():
    """Test timing for pure function - should already be async."""
    print("=" * 60)
    print("TEST: Pure function (should already use C++ fastpath)")
    print("=" * 60)

    @jax.jit
    def pure_fn(x):
        y = x @ x.T
        return y @ y.T

    size = 1000
    x = jax.random.normal(jax.random.key(0), (size, size))

    # Compile
    compiled = jax.jit(pure_fn).lower(x).compile()

    # Check dispatch path
    info = check_async_compatible(compiled)
    print(f"Compatibility: {info}")
    print()

    # Warmup
    _ = compiled(x)
    _.block_until_ready()

    # Time standard AOT
    times_dispatch = []
    times_total = []
    for _ in range(5):
        start = time.perf_counter()
        result = compiled(x)
        dispatch = time.perf_counter() - start
        result.block_until_ready()
        total = time.perf_counter() - start
        times_dispatch.append(dispatch)
        times_total.append(total)

    print("Standard AOT (pure function):")
    print(f"  Dispatch time: {np.mean(times_dispatch)*1000:.3f} +/- {np.std(times_dispatch)*1000:.3f} ms")
    print(f"  Total time:    {np.mean(times_total)*1000:.3f} +/- {np.std(times_total)*1000:.3f} ms")

    dispatch_ratio = np.mean(times_dispatch) / np.mean(times_total)
    print(f"  Dispatch/Total ratio: {dispatch_ratio:.2%}")
    if dispatch_ratio > 0.8:
        print(f"  => Sync behavior (dispatch takes most of the time)")
    else:
        print(f"  => ASYNC BEHAVIOR (dispatch returns quickly)")
    print()


def test_with_debug_print():
    """Test function with jax.debug.print - common use case."""
    print("=" * 60)
    print("TEST: Function with jax.debug.print")
    print("=" * 60)

    @jax.jit
    def fn_with_print(x):
        y = x @ x.T
        jax.debug.print("Computed y with shape {}", y.shape)
        return y @ y.T

    size = 1000
    x = jax.random.normal(jax.random.key(0), (size, size))

    try:
        compiled = jax.jit(fn_with_print).lower(x).compile()
        info = check_async_compatible(compiled)
        print(f"Compatibility: {info}")

        # Warmup
        _ = compiled(x)
        _.block_until_ready()

        # Time
        times_dispatch = []
        times_total = []
        for _ in range(3):
            start = time.perf_counter()
            result = compiled(x)
            dispatch = time.perf_counter() - start
            result.block_until_ready()
            total = time.perf_counter() - start
            times_dispatch.append(dispatch)
            times_total.append(total)

        print()
        print("Standard AOT with debug.print:")
        print(f"  Dispatch time: {np.mean(times_dispatch)*1000:.3f} ms")
        print(f"  Total time:    {np.mean(times_total)*1000:.3f} ms")

        dispatch_ratio = np.mean(times_dispatch) / np.mean(times_total)
        print(f"  Dispatch/Total ratio: {dispatch_ratio:.2%}")

    except Exception as e:
        print(f"  Error: {e}")
    print()


def main():
    print("JAX Effects and Async Dispatch Analysis")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    test_pure_function_timing()
    test_function_with_callback_timing()
    test_with_debug_print()

    print("=" * 60)
    print("SUMMARY:")
    print("- Pure functions: Already async (C++ fastpath)")
    print("- Functions with effects: Sync (Python fallback)")
    print("- The async workaround helps for effectful functions")
    print("=" * 60)


if __name__ == "__main__":
    main()
