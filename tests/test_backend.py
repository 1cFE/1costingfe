import importlib


def _reload_backend():
    import costingfe._backend as b

    return importlib.reload(b)


def test_numpy_backend_surface(monkeypatch):
    monkeypatch.setenv("COSTINGFE_BACKEND", "numpy")
    b = _reload_backend()
    assert b.HAS_JAX is False
    # xp is numpy
    assert b.xp.array([1.0, 2.0]).sum() == 3.0
    # fori_loop accumulates like a python loop
    assert b.fori_loop(0, 5, lambda i, acc: acc + i, 0) == 10
    # optimization_barrier is identity
    assert b.optimization_barrier(7.0) == 7.0
    # nothing is a Tracer in numpy mode
    assert isinstance(3.0, b.Tracer) is False
    assert b.grad is None and b.vmap is None


def test_force_jax_when_present(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        import pytest

        pytest.skip("jax not installed")
    monkeypatch.setenv("COSTINGFE_BACKEND", "jax")
    b = _reload_backend()
    assert b.HAS_JAX is True
    assert callable(b.grad) and callable(b.vmap)
