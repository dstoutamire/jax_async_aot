# JAX Async AOT Dispatch

Tools and techniques for ensuring async GPU dispatch in JAX's AOT (Ahead-of-Time) compiled functions.

## The Problem

JAX's AOT compiled functions can dispatch synchronously, blocking until GPU computation completes. This prevents overlapping computation with other work - a showstopper for performance-critical applications.

**Two distinct causes require different solutions:**

| Cause | Symptom | Solution |
|-------|---------|----------|
| While loops with dynamic bounds | ~90% dispatch ratio | XLA command buffers |
| Host effects (debug.print, callbacks) | Falls back to Python path | Remove effects from hot path |

## Quick Start

Enable XLA command buffers before importing JAX:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1"
```

This reduces while loop dispatch ratio from ~90% to ~20%, enabling async execution.

## Compile-Time Verification

The most reliable way to verify async compatibility is to parse XLA's thunk dumps during compilation. This definitively shows whether control flow is captured into command buffers.

### How It Works

When XLA compiles a JAX function, it generates "thunks" - executable operations. Control flow can appear as:

- **Top-level thunks**: SYNCHRONOUS - CPU must check loop conditions each iteration
- **Captured in command buffers**: ASYNC - GPU handles everything without CPU round-trips

The `compile_aot.py` module parses thunk dumps to verify all control flow is captured.

### Usage

```python
from compile_aot import compile_aot, load_aot
import jax.numpy as jnp

def my_optimization(x0):
    # ... contains while loops, conditionals ...
    return result

# Compile with async verification
compiled = compile_aot(
    my_optimization,
    example_args=(jnp.zeros(3),),
    output_path="model.zst",  # optional: save to file
    check_async=True,         # verify async compatibility
)

# Use immediately or load later
result = compiled(jnp.zeros(3))
# or
loaded = load_aot("model.zst")
result = loaded(jnp.zeros(3))
```

### CLI

```bash
# Compile example function
python compile_aot.py example_fn:optimize --output model.zst

# Skip async check (not recommended)
python compile_aot.py example_fn:optimize --no-check
```

### What the Output Means

```
Compiling optimize...
  Lowering and compiling...
  Checking async dispatch compatibility...
  Async dispatch check PASSED - all control flow captured in command buffers
    Command buffers: 2
    Captured while loops: 1
    Captured conditionals: 0
  Serializing executable...
  Compressing with zstd (level 3)...
Successfully compiled to model.zst
```

If verification fails:

```
ASYNC DISPATCH CHECK FAILED
Found 1 SYNC while loop(s) at top level
This means CPU-GPU synchronization will occur during execution.
```

## Thunk Sequence Analysis

For debugging, you can inspect thunk sequences directly:

```python
from compile_aot import run_with_thunk_dump, parse_thunk_sequence

script = '''
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    def cond(state):
        i, _ = state
        return i < 10
    def body(state):
        i, x = state
        return (i + 1, x + 1.0)
    _, result = jax.lax.while_loop(cond, body, (0, x))
    return result

result = my_fn(jnp.array(0.0))
result.block_until_ready()
'''

thunk_text = run_with_thunk_dump(script)
parsed = parse_thunk_sequence(thunk_text)

print(f"Command buffers: {len(parsed['command_buffers'])}")
print(f"Captured while loops: {len(parsed['captured_whiles'])}")
print(f"Top-level (SYNC) while loops: {len(parsed['top_level_whiles'])}")
```

### Example Thunk Sequence

**Good (async)** - while loop inside command buffer:
```
0 kCommandBuffer command_buffer.1
  1 kWhile while.1
    2 kFusion fusion.body
```

**Bad (sync)** - while loop at top level:
```
0 kWhile while.1
  1 kFusion fusion.body
```

## Root Cause 1: While Loops

XLA's `WhileThunk` calls `stream.BlockHostUntilDone()` on every loop iteration when bounds are dynamic. The host must read the loop condition back from GPU to decide whether to continue.

### Solution: CUDA Graph Command Buffers

CUDA 12.3+ introduced conditional nodes in CUDA graphs. XLA's command buffers use these to execute loops entirely on GPU:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1"
```

| Flag | Purpose |
|------|---------|
| `FUSION` | Captures fused operations into CUDA graphs |
| `CONDITIONAL` | Enables `lax.cond` on GPU via graph conditional nodes |
| `WHILE` | Enables `lax.while_loop` on GPU via graph while-loop nodes |
| `graph_min_graph_size=1` | Captures even small operations (default threshold is higher) |

### Requirements

**CUDA 12.9+** is required for full support:

| Pattern | CUDA 12.8 | CUDA 12.9+ |
|---------|-----------|------------|
| Dynamic-bound while loops | Works | Works |
| Static-bound loops (unrolled) | Crashes | Works |
| Nested while loops | Crashes | Works |
| jaxopt L-BFGS | Crashes | Works |

**XLA Bug Fix Required:** Older XLA versions have a bug where `CreateConditionalNode` fails to set the `parent_` pointer on nested command buffers, causing crashes on second execution of while loops.

- **Fixed in:** [PR #36135](https://github.com/openxla/xla/pull/36135) (merged January 2026, expected in JAX >= 0.8.3)

## Root Cause 2: Host Effects

Functions with effects (`jax.debug.print`, `jax.pure_callback`, etc.) fall back to a Python dispatch path that handles effect tokens synchronously.

| Function Type | Dispatch Ratio | Behavior |
|--------------|----------------|----------|
| Pure function | ~20-38% | Async |
| With `jax.pure_callback` | ~98% | Sync |
| With `jax.debug.print` | ~96% | Sync |

### Checking for Effects

```python
from async_aot import check_async_compatible

compiled = jax.jit(my_func).lower(x).compile()
info = check_async_compatible(compiled)

if not info['compatible']:
    print(f"Sync due to: {info['issues']}")
```

### Solutions

**Remove effects in production:**
```python
# Keep debug prints in development version only
compiled = jax.jit(pure_compute).lower(x).compile()
result = compiled(x)  # Async!
print(f"result: {result}")  # Print after, outside JAX
```

## Performance Results

| Metric | Without Command Buffers | With Command Buffers |
|--------|------------------------|---------------------|
| While loop dispatch ratio | ~90% (sync) | ~20% (async) |
| L-BFGS dispatch ratio | ~90% | ~17% |
| Pipeline throughput (8 ops) | 1.0x | 1.5-1.7x |

With async dispatch, you can pipeline operations:

```python
# Launch all (returns immediately with async handles)
futures = [compiled(x) for x in batch]
# GPU executes while CPU prepares next batch
results = [f.block_until_ready() for f in futures]
```

## Files

| File | Description |
|------|-------------|
| `compile_aot.py` | **AOT compilation with async verification** - thunk parsing, serialize/compress |
| `async_aot.py` | Runtime compatibility checking (effects inspection) |
| `example_fn.py` | Example functions for testing (L-BFGS, while loops, conditionals) |
| `test_compile_aot.py` | Tests for compile-time async verification |
| `bench_command_buffer.py` | While loop dispatch ratio benchmark |
| `bench_async_pipeline.py` | Pipelining throughput benchmark |

## Running Tests

```bash
# All tests
pytest -v

# Compile-time verification tests
pytest test_compile_aot.py -v

# Benchmarks
XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1" \
    python bench_command_buffer.py
```

## Technical Deep Dive

### XLA Thunk Types

| Thunk | Behavior | Async Compatible |
|-------|----------|------------------|
| `kFusion` | Fused kernel launch | Yes |
| `kWhile` (top-level) | CPU checks condition each iteration | No |
| `kWhile` (in command buffer) | GPU evaluates condition | Yes |
| `kConditional` (top-level) | CPU selects branch | No |
| `kConditional` (in command buffer) | GPU selects branch | Yes |
| `kCommandBuffer` | CUDA graph container | Yes |
| `kFft` | FFT operation | No (not yet supported) |

### Dispatch Path Decision

In `jax/_src/stages.py`, calling an AOT compiled function checks if it can use the C++ fastpath:

```python
def __call__(self, *args, **kwargs):
    if self._call is None:
        self._call = self._executable.create_cpp_call(self._params)
        if self._call is None:
            # Python fallback - SYNC
            self._call = cpp_call_fallback
    return self._call(*args, **kwargs)
```

The C++ fastpath releases the GIL and returns immediately with async array handles.

### Key XLA Source Files

| File | Purpose |
|------|---------|
| `xla/backends/gpu/runtime/while_thunk.cc` | Standard while loop (sync) |
| `xla/backends/gpu/runtime/command_buffer_cmd.cc` | `WhileCmd` implementation (async) |
| `xla/stream_executor/cuda/cuda_command_buffer.cc` | CUDA graph creation |
