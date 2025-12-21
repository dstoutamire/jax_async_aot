#!/usr/bin/env python3
"""
Test to determine which dispatch path is being used and when blocking occurs.
"""

import time
import sys
sys.path.insert(0, '/home/dps/jax_async_aot')

import jax
import jax.numpy as jnp
import numpy as np

from async_aot import check_async_compatible


def check_dispatch_path(compiled):
    """Check which dispatch path an AOT compiled function will use."""
    executable = compiled._executable

    # Check if C++ fastpath would be available based on conditions
    info = check_async_compatible(compiled)

    # The conditions for C++ fastpath (from pxla.py:3269-3272)
    would_use_cpp = (
        info.get('compatible', False) and
        not info.get('has_unordered_effects', True) and
        not info.get('has_host_callbacks', True) and
        not info.get('ordered_effects', [])
    )

    print(f"  Would use C++ fastpath: {would_use_cpp}")
    print(f"  has_unordered_effects: {info.get('has_unordered_effects', 'N/A')}")
    print(f"  ordered_effects: {info.get('ordered_effects', 'N/A')}")
    print(f"  has_host_callbacks: {info.get('has_host_callbacks', 'N/A')}")

    return would_use_cpp


def test_pure_function():
    """Test a pure function (should use C++ fastpath)."""
    print("=" * 60)
    print("TEST: Pure Function (no effects)")
    print("=" * 60)

    @jax.jit
    def pure_fn(x):
        return x @ x.T

    x = jnp.ones((1000, 1000))
    compiled = jax.jit(pure_fn).lower(x).compile()

    uses_cpp = check_dispatch_path(compiled)
    print(f"  Uses C++ fastpath: {uses_cpp}")
    print()


def test_function_with_print():
    """Test a function with print (has effects)."""
    print("=" * 60)
    print("TEST: Function with jax.debug.print (has effects)")
    print("=" * 60)

    @jax.jit
    def fn_with_print(x):
        jax.debug.print("x shape: {}", x.shape)
        return x @ x.T

    x = jnp.ones((100, 100))

    try:
        compiled = jax.jit(fn_with_print).lower(x).compile()
        uses_cpp = check_dispatch_path(compiled)
        print(f"  Uses C++ fastpath: {uses_cpp}")
    except Exception as e:
        print(f"  Error: {e}")
    print()


def test_function_with_callback():
    """Test a function with host callback."""
    print("=" * 60)
    print("TEST: Function with io_callback (has host callbacks)")
    print("=" * 60)

    def host_fn(x):
        return x * 2

    @jax.jit
    def fn_with_callback(x):
        result = jax.pure_callback(host_fn, jax.ShapeDtypeStruct(x.shape, x.dtype), x)
        return result

    x = jnp.ones((100, 100))

    try:
        compiled = jax.jit(fn_with_callback).lower(x).compile()
        uses_cpp = check_dispatch_path(compiled)
        print(f"  Uses C++ fastpath: {uses_cpp}")
    except Exception as e:
        print(f"  Error: {e}")
    print()


def test_timing_with_effects():
    """Compare dispatch timing between pure and effectful functions."""
    print("=" * 60)
    print("TEST: Timing comparison - Pure vs Effectful")
    print("=" * 60)

    size = 2000
    x = jax.random.normal(jax.random.key(0), (size, size))

    # Pure function
    @jax.jit
    def pure_fn(x):
        for _ in range(5):
            x = x @ x.T
            x = x / jnp.max(x)
        return x

    compiled_pure = jax.jit(pure_fn).lower(x).compile()

    # Warmup
    _ = compiled_pure(x)
    _.block_until_ready()

    # Time pure function
    times_pure = []
    for _ in range(10):
        start = time.perf_counter()
        result = compiled_pure(x)
        dispatch_time = time.perf_counter() - start
        result.block_until_ready()
        times_pure.append(dispatch_time)

    print(f"Pure function dispatch times:")
    print(f"  Mean: {np.mean(times_pure)*1000:.3f} ms")
    print(f"  Std:  {np.std(times_pure)*1000:.3f} ms")
    print(f"  Uses C++ fastpath: {check_dispatch_path(compiled_pure)}")
    print()


def test_internal_call_attribute():
    """Check the internal _call attribute to see what's being used."""
    print("=" * 60)
    print("TEST: Internal _call attribute inspection")
    print("=" * 60)

    @jax.jit
    def fn(x):
        return x @ x.T

    x = jnp.ones((100, 100))
    compiled = jax.jit(fn).lower(x).compile()

    # Before first call
    print(f"Before first call:")
    print(f"  compiled._call: {compiled._call}")

    # After first call
    _ = compiled(x)
    print(f"After first call:")
    print(f"  compiled._call: {compiled._call}")
    print(f"  _call type: {type(compiled._call)}")

    # Check if it's the C++ pjit or Python fallback
    call_str = str(compiled._call)
    if 'PjitFunction' in call_str:
        print(f"  Using: C++ PjitFunction dispatcher")
    elif 'cpp_call_fallback' in call_str or 'function' in call_str.lower():
        print(f"  Using: Python fallback dispatcher")
    else:
        print(f"  Using: {call_str}")
    print()


def main():
    print("JAX Dispatch Path Analysis")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    test_pure_function()
    test_function_with_print()
    test_function_with_callback()
    test_timing_with_effects()
    test_internal_call_attribute()


if __name__ == "__main__":
    main()
