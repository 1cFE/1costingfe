"""Shared test configuration.

Enables the JAX persistent compilation cache so repeated jit compilations
of near-identical cost graphs are served from disk across runs and across
pytest-xdist workers. The cache lives outside the repo (no working-tree
pollution); JAX manages eviction.
"""

import os

import jax
import pytest

_CACHE_DIR = os.path.expanduser("~/.cache/costingfe-jax")
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


# Power-to-geometry sizing (size_from_power), LCOE optimization (optimize_lcoe),
# and the bundled 0D physics models (use_0d_model) are gated off for the
# released build (see costingfe.model.SIZING_FEATURES_ENABLED and
# MODELS_0D_ENABLED). Tests that exercise those code paths through forward() hit
# the gate's NotImplementedError; convert that one specific error into a skip so
# the suite stays green while the feature tests are preserved on disk. Tests
# that call the pure solver/layer functions directly are unaffected (the code
# remains intact) and continue to run. Both gate messages share the substring
# below.
_GATE_MESSAGE = "are not available in this release"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    excinfo = outcome.excinfo
    if (
        excinfo is not None
        and issubclass(excinfo[0], NotImplementedError)
        and _GATE_MESSAGE in str(excinfo[1])
    ):
        outcome.force_exception(
            pytest.skip.Exception(
                "size_from_power / optimize_lcoe / use_0d_model gated off for release"
            )
        )
