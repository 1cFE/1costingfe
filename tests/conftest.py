"""Shared test configuration.

Enables the JAX persistent compilation cache so repeated jit compilations
of near-identical cost graphs are served from disk across runs and across
pytest-xdist workers. The cache lives outside the repo (no working-tree
pollution); JAX manages eviction.
"""

import os

import jax

_CACHE_DIR = os.path.expanduser("~/.cache/costingfe-jax")
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
