# JAX Async AOT Dispatch Investigation

## Problem Statement

JAX's AOT (Ahead-of-Time) compiled functions appear to dispatch synchronously, while `@jit` decorated functions dispatch asynchronously. This is a showstopper for performance-critical applications that need to overlap computation with other work.

## Key Finding

After thorough investigation of the JAX codebase, we discovered:

**Pure AOT functions are already async!** The synchronous behavior only occurs when functions have **effects**.

### Test Results (1000x1000 matrix, CUDA)

| Function Type | Dispatch Time | Total Time | Dispatch Ratio | Behavior |
|--------------|---------------|------------|----------------|----------|
| Pure function (no effects) | 0.14 ms | 0.37 ms | 38% | **ASYNC** |
| Function with `jax.pure_callback` | 2.07 ms | 2.12 ms | 98% | **SYNC** |
| Function with `jax.debug.print` | 0.95 ms | 0.99 ms | 96% | **SYNC** |

A low dispatch/total ratio means the function returns quickly (async), while a high ratio means it blocks until completion (sync).

## Root Cause Analysis

### The Dispatch Path Decision

In `jax/_src/stages.py:876-885`, when you call an AOT compiled function:

```python
def __call__(self, *args, **kwargs):
    if self._call is None:
        self._call = self._executable.create_cpp_call(self._params)
        if self._call is None:
            # Python fallback - SYNC
            params = self._params
            def cpp_call_fallback(*args, **kwargs):
                outs, _, _ = Compiled.call(params, *args, **kwargs)
                return outs
            self._call = cpp_call_fallback
    return self._call(*args, **kwargs)
```

### The C++ Fastpath Condition

In `jax/_src/interpreters/pxla.py:3268-3272`:

```python
def create_cpp_call(self, params: stages.CompiledCallParams):
    # C++ fastpath is DISABLED if ANY of these are true:
    if not (isinstance(self.unsafe_call, ExecuteReplicated) and
            not self.unsafe_call.has_unordered_effects and
            not self.unsafe_call.has_host_callbacks):
        return None  # Falls back to Python dispatch (SYNC)

    # Otherwise, use C++ pjit dispatcher (ASYNC)
    return xc._xla.pjit(...)
```

### Effects That Cause Sync Dispatch

| Effect | `has_unordered_effects` | `has_host_callbacks` | Result |
|--------|------------------------|---------------------|--------|
| `jax.debug.print()` | True | True | SYNC |
| `jax.pure_callback()` | False | True | SYNC |
| `jax.experimental.io_callback()` | False | True | SYNC |
| Ordered effects (I/O) | False | Varies | SYNC |
| Pure computation | False | False | ASYNC |

## How to Check Your Function

```python
from async_aot import check_async_compatible
import jax
import jax.numpy as jnp

@jax.jit
def my_func(x):
    return x @ x.T

x = jnp.ones((1000, 1000))
compiled = jax.jit(my_func).lower(x).compile()

info = check_async_compatible(compiled)
print(info)
# Output:
# {'compatible': True,
#  'has_unordered_effects': False,
#  'ordered_effects': [],
#  'has_host_callbacks': False}
```

If `compatible` is `True`, your function is already async!

## Solutions

### For Pure Functions (No Effects)

**No workaround needed!** Pure AOT functions already use the C++ fastpath.

```python
@jax.jit
def pure_fn(x):
    return x @ x.T

compiled = jax.jit(pure_fn).lower(x).compile()
result = compiled(x)  # Already async!
# Do other work here...
result.block_until_ready()  # Block only when needed
```

### For Functions With Effects

If your function has effects and you need async dispatch:

#### Option 1: Remove Effects for Production

```python
# Development version with debug prints
@jax.jit
def my_func_debug(x):
    jax.debug.print("x shape: {}", x.shape)
    return x @ x.T

# Production version without debug prints
@jax.jit
def my_func_prod(x):
    return x @ x.T

# Use prod version for async dispatch
compiled = jax.jit(my_func_prod).lower(x).compile()
```

#### Option 2: Separate Effectful and Pure Parts

```python
# Instead of:
@jax.jit
def combined(x):
    result = x @ x.T
    jax.debug.print("result: {}", result)
    return result

# Do this:
@jax.jit
def pure_compute(x):
    return x @ x.T

compiled = jax.jit(pure_compute).lower(x).compile()
result = compiled(x)  # Async!
print(f"result: {result}")  # Print after
```

#### Option 3: Use the Workaround Module

```python
from async_aot import make_async_aot_caller

compiled = jax.jit(my_func).lower(x).compile()
async_caller = make_async_aot_caller(compiled)

result = async_caller(x)  # Bypasses Python fallback
result.block_until_ready()
```

**Note:** The workaround only works for pure functions. For effectful functions, the effects inherently require synchronization.

## Files in This Directory

| File | Description |
|------|-------------|
| `async_aot.py` | Compatibility checker and workaround utilities |
| `test_async_aot.py` | Basic functionality and timing tests |
| `test_dispatch_path.py` | Tests which dispatch path (C++ vs Python) is used |
| `test_effects_comparison.py` | Timing comparison between pure and effectful functions |

## Running the Tests

```bash
cd /home/dps/jax_async_aot
python3 test_async_aot.py
python3 test_dispatch_path.py
python3 test_effects_comparison.py
```

## Key Source Files in JAX

| File | Lines | Description |
|------|-------|-------------|
| `jax/_src/stages.py` | 876-885 | Dispatch decision point (`__call__`) |
| `jax/_src/interpreters/pxla.py` | 3268-3308 | `create_cpp_call` and C++ dispatcher creation |
| `jax/_src/interpreters/pxla.py` | 1352-1406 | `ExecuteReplicated.__call__` (Python fallback) |
| `jax/_src/pjit.py` | 187-227 | `_get_fastpath_data` conditions |
| `jaxlib/pjit.cc` | 814-821 | C++ async execution with GIL release |

## Technical Details

### Why C++ Fastpath is Async

In `jaxlib/pjit.cc:814-821`:

```cpp
{
    nb::gil_scoped_release gil_release;  // Release Python GIL
    TF_ASSIGN_OR_RETURN(auto result,
        cache_entry->executable->ifrt_executable()->Execute(...));
    output_arrays = std::move(result.outputs);
}
// Returns immediately with async array handles
```

### Why Python Fallback is Sync

The Python fallback path in `ExecuteReplicated.__call__` still calls `execute_sharded()` which releases the GIL, but:

1. More Python overhead before/after execution
2. Effect token handling adds synchronization points
3. Result processing happens in Python with GIL held

## Conclusion

The "AOT is sync" issue is actually "**effectful AOT is sync**".

- **Pure functions**: Already async, no workaround needed
- **Effectful functions**: Inherently sync due to effect handling

If you're seeing sync behavior, run `check_async_compatible()` to identify the cause:

```python
from async_aot import check_async_compatible
info = check_async_compatible(compiled)
if not info['compatible']:
    print(f"Sync due to: {info['issues']}")
```
