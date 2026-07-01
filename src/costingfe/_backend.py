"""Backend resolver: jax if available (or forced), else numpy.

Selecting numpy drops the jax/jaxlib dependency entirely for the forward
cost model and finite-difference sensitivities.
"""

import os

_forced = os.environ.get("COSTINGFE_BACKEND", "").strip().lower()

if _forced == "numpy":
    HAS_JAX = False
elif _forced == "jax":
    HAS_JAX = True  # import below raises if jax is genuinely absent
else:
    try:
        import jax as _jax_probe  # noqa: F401

        HAS_JAX = True
    except ImportError:
        HAS_JAX = False

if HAS_JAX:
    import jax
    import jax.lax
    import jax.numpy as xp  # noqa: F401 - exported

    Tracer = jax.core.Tracer
    grad = jax.grad
    vmap = jax.vmap
    fori_loop = jax.lax.fori_loop
    optimization_barrier = jax.lax.optimization_barrier
else:
    import numpy as xp  # noqa: F401 - exported

    class Tracer:  # noqa: D401 - sentinel; nothing is ever an instance
        """Placeholder so isinstance(x, Tracer) is always False under numpy."""

    def fori_loop(lo, hi, body, init):
        val = init
        for i in range(int(lo), int(hi)):
            val = body(i, val)
        return val

    def optimization_barrier(x):
        return x

    grad = None
    vmap = None
