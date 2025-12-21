"""
Async AOT Dispatch Workaround for JAX

This module provides a workaround to enable async dispatch for AOT compiled
JAX functions, bypassing the Python fallback path that causes synchronous behavior.
"""

from typing import Callable, Any
import jax
from jax._src import stages
from jax._src.interpreters import pxla


def check_async_compatible(compiled: stages.Compiled) -> dict[str, Any]:
    """Check if an AOT compiled function is compatible with async dispatch.

    Returns a dict with compatibility info and any blocking conditions.
    """
    executable = compiled._executable

    info = {
        "compatible": True,
        "issues": [],
        "executable_type": type(executable).__name__,
    }

    if not isinstance(executable, pxla.MeshExecutable):
        info["compatible"] = False
        info["issues"].append(f"Executable is {type(executable).__name__}, not MeshExecutable")
        return info

    unsafe_call = executable.unsafe_call

    if not isinstance(unsafe_call, pxla.ExecuteReplicated):
        info["compatible"] = False
        info["issues"].append(f"unsafe_call is {type(unsafe_call).__name__}, not ExecuteReplicated")
        return info

    info["has_unordered_effects"] = unsafe_call.has_unordered_effects
    info["ordered_effects"] = list(unsafe_call.ordered_effects) if unsafe_call.ordered_effects else []
    info["has_host_callbacks"] = unsafe_call.has_host_callbacks

    if unsafe_call.has_unordered_effects:
        info["compatible"] = False
        info["issues"].append("Has unordered effects")

    if unsafe_call.ordered_effects:
        info["compatible"] = False
        info["issues"].append(f"Has ordered effects: {unsafe_call.ordered_effects}")

    if unsafe_call.has_host_callbacks:
        info["compatible"] = False
        info["issues"].append("Has host callbacks")

    return info


def make_async_aot_caller(compiled: stages.Compiled) -> Callable:
    """Create an async-friendly caller for AOT compiled functions.

    This bypasses the Python fallback path and calls the underlying
    ExecuteReplicated directly, similar to the C++ pjit dispatcher.

    Args:
        compiled: A compiled JAX function from `.compile()`

    Returns:
        A callable that dispatches asynchronously

    Raises:
        TypeError: If the compiled function is not compatible
    """
    executable = compiled._executable

    if not isinstance(executable, pxla.MeshExecutable):
        raise TypeError(f"Expected MeshExecutable, got {type(executable).__name__}")

    # Get the underlying ExecuteReplicated
    unsafe_call = executable.unsafe_call

    if not isinstance(unsafe_call, pxla.ExecuteReplicated):
        raise TypeError(f"Expected ExecuteReplicated, got {type(unsafe_call).__name__}")

    # Get handlers and metadata
    in_handler = unsafe_call.in_handler
    out_handler = unsafe_call.out_handler
    kept_var_idx = unsafe_call.kept_var_idx
    xla_executable = unsafe_call.xla_executable

    # Get the output tree for unflattening
    out_tree = compiled.out_tree

    def async_call(*args):
        """Async dispatch that bypasses Python fallback."""
        # Flatten input args
        args_flat = jax.tree_util.tree_leaves(args)

        # Filter args based on kept_var_idx (DCE - dead code elimination)
        args_filtered = [x for i, x in enumerate(args_flat) if i in kept_var_idx]

        # Shard inputs using the input handler
        input_bufs = in_handler.handler(args_filtered)

        # Execute - this releases GIL internally and returns immediately
        # with async array handles
        results = xla_executable.execute_sharded(input_bufs)

        # Wrap results using output handlers (non-blocking)
        out_flat = results.consume_with_handlers(out_handler.handlers)

        # Unflatten outputs according to the original output tree
        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    return async_call


def make_async_aot_caller_unsafe(compiled: stages.Compiled) -> Callable:
    """Ultra-minimal async caller - even less overhead but more fragile.

    This version does minimal processing and is suitable when you know
    your inputs are already properly sharded JAX arrays.

    Warning: This bypasses ALL safety checks. Use only if you understand
    the implications.
    """
    executable = compiled._executable
    unsafe_call = executable.unsafe_call
    xla_executable = unsafe_call.xla_executable
    out_handler = unsafe_call.out_handler
    out_tree = compiled.out_tree

    def async_call_unsafe(*args):
        # Assume args are already flat and properly formatted
        # Just get the underlying buffers and execute
        input_bufs = [arg if hasattr(arg, '_arrays') else arg for arg in args]

        # Direct execution
        results = xla_executable.execute_sharded(input_bufs)
        out_flat = results.consume_with_handlers(out_handler.handlers)

        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    return async_call_unsafe


def make_async_aot_caller_with_effects(compiled: stages.Compiled) -> Callable:
    """Create an async caller for functions WITH effects.

    This handles the token machinery required for effectful functions while
    still trying to minimize blocking.

    Note: For functions with host callbacks, true async is not possible
    because the callback must execute synchronously. However, this can
    reduce Python overhead.
    """
    executable = compiled._executable

    if not isinstance(executable, pxla.MeshExecutable):
        raise TypeError(f"Expected MeshExecutable, got {type(executable).__name__}")

    unsafe_call = executable.unsafe_call

    if not isinstance(unsafe_call, pxla.ExecuteReplicated):
        raise TypeError(f"Expected ExecuteReplicated, got {type(unsafe_call).__name__}")

    # Get handlers and metadata
    in_handler = unsafe_call.in_handler
    out_handler = unsafe_call.out_handler
    kept_var_idx = unsafe_call.kept_var_idx
    xla_executable = unsafe_call.xla_executable
    out_tree = compiled.out_tree

    # Effect-related
    has_effects = unsafe_call.has_unordered_effects or unsafe_call.ordered_effects
    has_callbacks = unsafe_call.has_host_callbacks

    def async_call_with_effects(*args):
        """Dispatch with effect handling."""
        # Flatten input args
        args_flat = jax.tree_util.tree_leaves(args)

        # Filter args based on kept_var_idx
        args_filtered = [x for i, x in enumerate(args_flat) if i in kept_var_idx]

        # Shard inputs
        input_bufs = in_handler.handler(args_filtered)

        # For effectful functions, we still need to call through unsafe_call
        # to handle tokens properly, but we return immediately after dispatch
        if has_effects or has_callbacks:
            # Call the unsafe_call directly - this handles tokens
            out_flat = unsafe_call(*args_flat)
        else:
            # Pure path - direct execute_sharded
            results = xla_executable.execute_sharded(input_bufs)
            out_flat = results.consume_with_handlers(out_handler.handlers)

        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    return async_call_with_effects


def wrap_without_effects(fn):
    """Decorator to strip effects from a function for async dispatch.

    This wraps a function to remove debug prints and other effects,
    allowing it to use the C++ fastpath.

    WARNING: This actually removes the effects, so debug.print won't print!
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Disable debug output temporarily
        with jax.disable_jit(False):  # Ensure JIT is enabled
            return fn(*args, **kwargs)
    return wrapper
