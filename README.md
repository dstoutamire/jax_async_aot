# JAX Async AOT Dispatch Investigation

## Problem

JAX's AOT (Ahead-of-Time) compiled functions can dispatch synchronously, blocking until GPU computation completes. This prevents overlapping computation with other work—a showstopper for performance-critical applications.

After investigating JAX and XLA internals, we found **two distinct causes** of synchronous dispatch, each with different solutions.

## Quick Start

For while loops (the most common cause), enable XLA command buffers:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1"
```

This reduces dispatch ratio from ~90% to ~20%, enabling async execution and 1.5x+ throughput improvements through pipelining.

## Root Cause 1: While Loops with Dynamic Bounds

This is the **major cause** of sync dispatch in practice. XLA's `WhileThunk` calls `stream.BlockHostUntilDone()` on every loop iteration when bounds are dynamic. The host must read the loop condition back from GPU memory to decide whether to continue—forcing a sync on each iteration.

**Affected operations:** `jax.lax.while_loop`, `jaxopt.LBFGS`, and any iterative algorithm with a data-dependent termination condition.

### The Solution: CUDA Graph Command Buffers

CUDA 12.3+ introduced conditional nodes in CUDA graphs, enabling on-device control flow without CPU round-trips. XLA implements this through command buffers, controlled by `XLA_FLAGS`:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1"
```

**What each flag does:**

| Flag | Purpose |
|------|---------|
| `FUSION` | Captures fused operations (matmul, elementwise ops) into CUDA graphs for reduced kernel launch overhead |
| `CONDITIONAL` | Enables `lax.cond` and `lax.switch` to execute entirely on GPU using CUDA graph conditional nodes |
| `WHILE` | Enables `lax.while_loop` to execute entirely on GPU using CUDA graph while-loop nodes—the key fix for async dispatch |
| `graph_min_graph_size=1` | Captures even small operations into graphs (default threshold is higher, which can miss simple loops) |

### Requirements

**CUDA 12.9+** is required for full support. Earlier versions have limitations:

| Pattern | CUDA 12.8 | CUDA 12.9+ |
|---------|-----------|------------|
| Dynamic-bound while loops | Works | Works |
| Static-bound loops (unrolled) | Crashes | Works |
| Nested while loops | Crashes | Works |
| jaxopt L-BFGS | Crashes | Works |

**XLA Bug Fix:** Versions before JAX 0.8.3 have a bug in `cuda_command_buffer.cc` where `CreateConditionalNode` fails to set the `parent_` pointer on nested command buffers, causing crashes on the second execution. This is fixed in our local XLA build (`../xla_repo`).

### Performance Results

| Metric | Without Command Buffers | With Command Buffers |
|--------|------------------------|---------------------|
| While loop dispatch ratio | ~90% (sync) | ~20% (async) |
| L-BFGS dispatch ratio | ~90% | ~17% |
| Pipeline throughput (8 ops) | 1.0x | 1.5-1.7x |

With async dispatch, you can pipeline multiple operations:

```python
# Launch all operations (returns immediately with async handles)
futures = [compiled(x) for x in batch]
# GPU executes while CPU prepares next batch
results = [f.block_until_ready() for f in futures]
```

## Root Cause 2: Host Effects

Pure AOT functions are already async. Sync behavior occurs when functions have **effects** that require host interaction: `jax.debug.print`, `jax.pure_callback`, `jax.experimental.io_callback`, etc.

When effects are present, JAX falls back to a Python dispatch path that handles effect tokens synchronously. This is inherent to how effects work—they require coordination with the host.

| Function Type | Dispatch Ratio | Behavior |
|--------------|----------------|----------|
| Pure function | ~38% | Async |
| With `jax.pure_callback` | ~98% | Sync |
| With `jax.debug.print` | ~96% | Sync |

### Checking Your Function

```python
from async_aot import check_async_compatible

compiled = jax.jit(my_func).lower(x).compile()
info = check_async_compatible(compiled)

if not info['compatible']:
    print(f"Sync due to: {info['issues']}")
```

### Solutions for Effectful Functions

**Remove effects in production:** Keep debug prints in a separate development version.

**Separate effectful and pure parts:** Move effects outside the compiled function:

```python
# Instead of printing inside the function...
compiled = jax.jit(pure_compute).lower(x).compile()
result = compiled(x)  # Async!
print(f"result: {result}")  # Print after, outside JAX
```

## Running the Benchmarks

```bash
cd /home/dps/jax_async_aot
source venv/bin/activate

# Command buffer benchmark (while loop dispatch ratio)
python3 bench_command_buffer.py  # Baseline: ~90% ratio

XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1" \
    python3 bench_command_buffer.py  # With fix: ~20% ratio

# Pipelining benchmark (throughput scaling)
XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1" \
    python3 bench_async_pipeline.py  # Shows 1.5x speedup with 8 ops
```

## Files

| File | Description |
|------|-------------|
| `async_aot.py` | Compatibility checker and workaround utilities |
| `bench_command_buffer.py` | While loop dispatch ratio benchmark |
| `bench_async_pipeline.py` | Pipelining throughput benchmark |
| `test_*.py` | Unit tests for dispatch path behavior |

## Technical Deep Dive

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

The C++ fastpath (`jaxlib/pjit.cc`) releases the GIL and returns immediately with async array handles. The Python fallback holds the GIL longer and processes effect tokens synchronously.

### Why Command Buffers Fix While Loops

Without command buffers, XLA's `WhileThunk` executes:
1. Run loop body on GPU
2. Copy condition result to CPU (`BlockHostUntilDone`)
3. Check condition on CPU
4. Repeat

With command buffers (`WhileCmd`), the entire loop is captured as a CUDA graph with conditional nodes. The GPU evaluates the condition and branches without CPU involvement. The host dispatches the graph once and returns immediately.

### Key XLA Source Files

| File | Purpose |
|------|---------|
| `xla/backends/gpu/runtime/while_thunk.cc` | Standard while loop execution (sync) |
| `xla/backends/gpu/runtime/command_buffer_cmd.cc` | `WhileCmd` implementation (async) |
| `xla/stream_executor/cuda/cuda_command_buffer.cc` | CUDA graph creation, including the `parent_` bug fix |
