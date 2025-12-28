#!/usr/bin/env python
"""AOT compilation with async dispatch verification.

This module provides ahead-of-time compilation for JAX functions with
compile-time verification that control flow is captured into XLA command
buffers for async GPU dispatch.

The key insight: XLA's thunk dumps show whether while loops and conditionals
are at "top level" (synchronous - CPU must check loop condition) or "captured"
inside command buffers (asynchronous - GPU handles everything).

Usage:
    # Programmatic
    from compile_aot import compile_aot
    compiled = compile_aot(my_fn, example_args, "model.zst")

    # CLI
    python compile_aot.py example_fn:optimize --output model.zst
"""

import argparse
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Any

import zstandard as zstd

# XLA flags for command buffer capture - must be set BEFORE JAX import
# Includes all default types plus CONDITIONAL and WHILE for full async dispatch
COMMAND_BUFFER_FLAGS = (
    "--xla_gpu_enable_command_buffer="
    "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION,"
    "DYNAMIC_SLICE_COPY_FUSION,CONDITIONAL,WHILE "
    "--xla_gpu_graph_min_graph_size=1"
)

# Compression level (3 = good balance of speed and ratio)
ZSTD_LEVEL = 3


def parse_thunk_sequence(thunk_text: str) -> dict:
    """Parse XLA thunk sequence to identify sync vs async control flow.

    XLA compiles JAX functions into "thunks" - executable operations.
    Control flow (while loops, conditionals) can appear as:
    - Top-level thunks: SYNCHRONOUS - CPU must check conditions each iteration
    - Captured in command buffers: ASYNC - GPU handles everything

    Args:
        thunk_text: Contents of thunk_sequence_after_thunk_passes.txt

    Returns:
        Dict with lists of found thunks by category:
        - top_level_whiles: SYNC while loops (bad for async)
        - captured_whiles: ASYNC while loops (good)
        - top_level_conditionals: SYNC conditionals (bad)
        - captured_conditionals: ASYNC conditionals (good)
        - command_buffers: All command buffer thunks found
    """
    result = {
        'top_level_whiles': [],
        'captured_whiles': [],
        'top_level_conditionals': [],
        'captured_conditionals': [],
        'command_buffers': [],
    }

    lines = thunk_text.split('\n')
    in_command_buffer = False
    indent_level = 0

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue

        current_indent = len(line) - len(stripped)

        # Track when we enter/exit command buffers
        if 'kCommandBuffer' in stripped:
            result['command_buffers'].append(stripped)
            in_command_buffer = True
            indent_level = current_indent
        elif current_indent <= indent_level and in_command_buffer and stripped.startswith('0'):
            # Exited command buffer (back to top level)
            in_command_buffer = False

        # Categorize while loops
        if 'kWhile' in stripped:
            if in_command_buffer:
                result['captured_whiles'].append(stripped)
            else:
                result['top_level_whiles'].append(stripped)

        # Categorize conditionals
        if 'kConditional' in stripped:
            if in_command_buffer:
                result['captured_conditionals'].append(stripped)
            else:
                result['top_level_conditionals'].append(stripped)

    return result


def check_async_dispatch(dump_dir: Path, verbose: bool = True) -> dict:
    """Analyze XLA thunk dumps for async dispatch compatibility.

    Args:
        dump_dir: Directory containing XLA dump files
        verbose: Print analysis results

    Returns:
        Dict with analysis results:
        - passed: True if all control flow is captured (async compatible)
        - parsed: Full parse results from parse_thunk_sequence
        - fft_modules: List of modules containing FFT (informational)
        - error_message: Description of failure (if any)

    Raises:
        RuntimeError: If sync control flow is detected (unless check-only mode)
    """
    result = {
        'passed': True,
        'parsed': None,
        'fft_modules': [],
        'error_message': None,
    }

    # Check for FFT in any module (informational - FFT can't be command-buffered yet)
    for f in dump_dir.iterdir():
        if 'thunk_metadata.txt' in f.name:
            if 'kFft' in f.read_text():
                result['fft_modules'].append(f.name)

    # Find largest thunk sequence file (main module)
    thunk_files = sorted(
        [f for f in dump_dir.iterdir() if 'thunk_sequence_after_thunk_passes.txt' in f.name],
        key=lambda x: x.stat().st_size,
        reverse=True
    )

    if not thunk_files:
        result['error_message'] = "No thunk sequence files found in dump directory"
        result['passed'] = False
        return result

    # Parse the main thunk sequence
    parsed = parse_thunk_sequence(thunk_files[0].read_text())
    result['parsed'] = parsed

    has_sync_whiles = len(parsed['top_level_whiles']) > 0
    has_sync_conds = len(parsed['top_level_conditionals']) > 0

    if has_sync_whiles or has_sync_conds:
        result['passed'] = False
        msg_parts = ["ASYNC DISPATCH CHECK FAILED"]
        if has_sync_whiles:
            msg_parts.append(f"Found {len(parsed['top_level_whiles'])} SYNC while loop(s) at top level")
        if has_sync_conds:
            msg_parts.append(f"Found {len(parsed['top_level_conditionals'])} SYNC conditional(s) at top level")
        if result['fft_modules']:
            msg_parts.append(f"Note: FFT found in {len(result['fft_modules'])} module(s) - may block capture")
        msg_parts.append("This means CPU-GPU synchronization will occur during execution.")
        result['error_message'] = '\n'.join(msg_parts)

    if verbose:
        if result['passed']:
            if result['fft_modules']:
                print(f"  Note: FFT in {len(result['fft_modules'])} module(s) (OK - outside control flow)")
            print("  Async dispatch check PASSED - all control flow captured in command buffers")
            print(f"    Command buffers: {len(parsed['command_buffers'])}")
            print(f"    Captured while loops: {len(parsed['captured_whiles'])}")
            print(f"    Captured conditionals: {len(parsed['captured_conditionals'])}")
        else:
            print(result['error_message'])

    return result


def compile_aot(
    fn: Callable,
    example_args: tuple,
    output_path: Path | str | None = None,
    check_async: bool = True,
    verbose: bool = True,
) -> Any:
    """Compile a JAX function to AOT format with async dispatch verification.

    This function:
    1. Sets XLA flags for command buffer capture and dump generation
    2. JIT compiles and lowers the function
    3. Parses XLA thunk dumps to verify async compatibility
    4. Optionally serializes and compresses to file

    Args:
        fn: JAX function to compile
        example_args: Example arguments for shape/dtype inference
        output_path: Path to save compressed executable (optional)
        check_async: Whether to verify async dispatch compatibility
        verbose: Print progress messages

    Returns:
        The compiled function (can be called directly or loaded later)

    Raises:
        RuntimeError: If async check fails and check_async=True
    """
    # Create temp dir for XLA dumps
    dump_dir = Path(tempfile.mkdtemp(prefix='xla_compile_dump_'))

    # Set XLA flags BEFORE importing JAX
    xla_flags = f"{COMMAND_BUFFER_FLAGS} --xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
    os.environ["XLA_FLAGS"] = xla_flags

    try:
        # Import JAX after setting flags
        import jax
        import jax.numpy as jnp
        from jax.experimental.serialize_executable import serialize, deserialize_and_load

        if verbose:
            print(f"Compiling {fn.__name__}...")

        # JIT compile
        jitted_fn = jax.jit(fn)

        # Lower and compile (triggers XLA compilation)
        if verbose:
            print("  Lowering and compiling...", flush=True)
        lowered = jitted_fn.lower(*example_args)
        compiled = lowered.compile()

        # Check async dispatch compatibility
        if check_async:
            if verbose:
                print("  Checking async dispatch compatibility...", flush=True)
            result = check_async_dispatch(dump_dir, verbose=verbose)
            if not result['passed']:
                raise RuntimeError(result['error_message'])

        # Serialize and compress if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            if verbose:
                print("  Serializing executable...", flush=True)
            serialized_bytes, in_tree, out_tree = serialize(compiled)

            if verbose:
                print(f"  Compressing with zstd (level {ZSTD_LEVEL})...", flush=True)
            pickled = pickle.dumps({
                'executable': serialized_bytes,
                'in_tree': in_tree,
                'out_tree': out_tree,
            })
            cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
            compressed = cctx.compress(pickled)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(compressed)

            uncompressed_mb = len(pickled) / 1024 / 1024
            compressed_mb = output_path.stat().st_size / 1024 / 1024

            if verbose:
                print(f"Successfully compiled to {output_path}")
                print(f"  Uncompressed: {uncompressed_mb:.1f} MB")
                print(f"  Compressed:   {compressed_mb:.1f} MB")

        return compiled

    finally:
        # Clean up dump directory
        shutil.rmtree(dump_dir, ignore_errors=True)


def load_aot(path: Path | str) -> Any:
    """Load a compiled AOT executable from file.

    Args:
        path: Path to .zst file created by compile_aot

    Returns:
        Compiled executable that can be called like a regular function
    """
    import jax
    from jax.experimental.serialize_executable import deserialize_and_load

    path = Path(path)
    dctx = zstd.ZstdDecompressor()

    with open(path, 'rb') as f:
        compressed = f.read()

    pickled = dctx.decompress(compressed)
    data = pickle.loads(pickled)

    compiled = deserialize_and_load(
        data['executable'],
        data['in_tree'],
        data['out_tree'],
    )

    return compiled


def run_with_thunk_dump(script: str) -> str:
    """Run a script with XLA flags and return thunk sequence.

    XLA_FLAGS must be set before JAX imports, so we use a subprocess.
    This is useful for testing thunk capture behavior.

    Args:
        script: Python script to execute

    Returns:
        Contents of the largest thunk_sequence_after_thunk_passes.txt file
    """
    import subprocess

    dump_dir = tempfile.mkdtemp(prefix='xla_thunk_check_')

    try:
        full_script = f'''
import os
os.environ["XLA_FLAGS"] = "{COMMAND_BUFFER_FLAGS} --xla_dump_to={dump_dir} --xla_dump_hlo_as_text"

{script}
'''
        result = subprocess.run(
            [sys.executable, '-c', full_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Find thunk sequence files
        thunk_files = []
        if os.path.exists(dump_dir):
            for f in os.listdir(dump_dir):
                if 'thunk_sequence_after_thunk_passes.txt' in f:
                    thunk_files.append(os.path.join(dump_dir, f))

        if result.returncode != 0 and not thunk_files:
            raise RuntimeError(f"Script failed: {result.stderr[-500:]}")

        if not thunk_files:
            files = os.listdir(dump_dir) if os.path.exists(dump_dir) else []
            raise RuntimeError(f"No thunk sequence files. Contents: {files[:10]}")

        # Return largest file (main module)
        thunk_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        with open(thunk_files[0], 'r') as f:
            return f.read()

    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def main():
    """CLI for AOT compilation."""
    parser = argparse.ArgumentParser(
        description="Compile JAX function to AOT format with async verification"
    )
    parser.add_argument(
        "function",
        help="Function to compile as module:function (e.g., example_fn:optimize)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("model.zst"),
        help="Output path (default: model.zst)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip async dispatch verification",
    )
    args = parser.parse_args()

    # Parse module:function
    if ':' not in args.function:
        print(f"Error: function must be module:function, got {args.function}", file=sys.stderr)
        return 1

    module_name, func_name = args.function.split(':', 1)

    try:
        # Import the module
        import importlib
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name)

        # Get example args from module if available
        example_args_name = f"{func_name}_example_args"
        if hasattr(module, example_args_name):
            example_args = getattr(module, example_args_name)
        else:
            print(f"Error: {module_name} must define {example_args_name}", file=sys.stderr)
            return 1

        compile_aot(
            fn,
            example_args,
            args.output,
            check_async=not args.no_check,
        )
        return 0

    except Exception as e:
        print(f"Compilation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
