"""Example JAX functions for AOT compilation testing.

These functions demonstrate patterns that require async dispatch checking:
- While loops (via jaxopt L-BFGS)
- Conditionals (via lax.cond)

Each function has a corresponding _example_args tuple for compilation.
"""

import jax
import jax.numpy as jnp
import jaxopt


# =============================================================================
# L-BFGS Optimization (while loop)
# =============================================================================

def optimize(x0: jnp.ndarray) -> tuple:
    """Minimize a quadratic loss using L-BFGS.

    This function contains a while loop (from jaxopt.LBFGS) that should
    be captured into an XLA command buffer for async dispatch.

    Args:
        x0: Initial parameter values, shape (3,)

    Returns:
        Tuple of (optimized params, optimizer state)
    """
    def loss_fn(x):
        target = jnp.array([1.0, 2.0, 3.0])
        return jnp.sum((x - target) ** 2)

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=50)
    return solver.run(x0)


# Example args for compile_aot
optimize_example_args = (jnp.zeros(3),)


# =============================================================================
# Simple While Loop
# =============================================================================

def simple_while(x: jnp.ndarray) -> jnp.ndarray:
    """Simple while loop that increments until condition met.

    Args:
        x: Initial value

    Returns:
        Final value after loop
    """
    def cond(state):
        i, val = state
        return i < 10

    def body(state):
        i, val = state
        return (i + 1, val + 1.0)

    _, result = jax.lax.while_loop(cond, body, (0, x))
    return result


simple_while_example_args = (jnp.array(0.0),)


# =============================================================================
# Conditional
# =============================================================================

def conditional(x: jnp.ndarray, threshold: jnp.ndarray) -> jnp.ndarray:
    """Conditional execution based on input.

    Args:
        x: Input value
        threshold: Threshold for branching

    Returns:
        Result of selected branch
    """
    return jax.lax.cond(
        x > threshold,
        lambda x: x * 2.0,   # true branch
        lambda x: x * 0.5,   # false branch
        x
    )


conditional_example_args = (jnp.array(5.0), jnp.array(3.0))


# =============================================================================
# Combined Example
# =============================================================================

def optimize_with_cond(x0: jnp.ndarray, use_l2: jnp.ndarray) -> tuple:
    """Optimization with conditional loss selection.

    Combines while loop (L-BFGS) with conditional (loss selection).

    Args:
        x0: Initial parameters
        use_l2: Boolean - if True use L2 loss, else L1

    Returns:
        Tuple of (optimized params, optimizer state)
    """
    target = jnp.array([1.0, 2.0, 3.0])

    def loss_fn(x):
        diff = x - target
        return jax.lax.cond(
            use_l2,
            lambda d: jnp.sum(d ** 2),   # L2 loss
            lambda d: jnp.sum(jnp.abs(d)),  # L1 loss
            diff
        )

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=50)
    return solver.run(x0)


optimize_with_cond_example_args = (jnp.zeros(3), jnp.array(True))


# =============================================================================
# Pure computation (no control flow) - for comparison
# =============================================================================

def matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Simple matrix multiplication - no control flow.

    This should always be async-compatible as there's no
    while loop or conditional to worry about.
    """
    return a @ b


matmul_example_args = (jnp.zeros((512, 512)), jnp.zeros((512, 512)))
