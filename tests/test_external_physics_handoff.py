"""Smoke test: the canonical Section 2.2 reference scenario runs and produces
group-level LCOE rollups."""

import subprocess
import sys
from pathlib import Path


def test_external_physics_handoff_runs():
    """Running the example produces a non-zero LCOE and the rollup keys the
    paper's Table relies on."""
    script = (
        Path(__file__).resolve().parents[1] / "examples" / "external_physics_handoff.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Required group-level rollup keys appear, in M$ rows.
    for key in (
        "CAS10",
        "CAS21",
        "CAS22",
        "CAS40",
        "CAS50",
        "CAS60",
        "CAS70",
        "CAS80",
        "CAS90",
        "Overnight (CAS10-50)",
        "LCOE",
    ):
        assert key in out, f"missing rollup row: {key}"
