"""Smoke tests for the paper's benchmarking scripts.

Each script exposes a `run(output_dir)` (or `run(input_dir, figure_dir)`
for the figure script) function that writes a single output file into
the given directory and returns its payload (or output path). The tests
check the payload structure, not the exact numbers (those depend on
framework defaults and are validated by reading the JSON manually).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PAPER_SCRIPTS = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "papers"
    / "1costingfe_paper"
    / "scripts"
)


@pytest.fixture(autouse=True)
def _add_scripts_to_path():
    sys.path.insert(0, str(PAPER_SCRIPTS))
    yield
    sys.path.remove(str(PAPER_SCRIPTS))


def test_benchmark_arc_produces_expected_keys(tmp_path):
    import benchmark_arc

    payload = benchmark_arc.run(tmp_path)

    assert (tmp_path / "arc.json").exists()
    on_disk = json.loads((tmp_path / "arc.json").read_text())
    assert payload == on_disk

    expected_top_level = {
        "reactor",
        "fuel",
        "concept",
        "inputs",
        "predicted_overnight_musd",
        "predicted_overnight_per_kwe_usd",
        "predicted_lcoe_usd_per_mwh",
        "fusion_power_mw",
        "net_electric_mw",
        "cas",
        "cas22_detail",
    }
    assert expected_top_level.issubset(payload.keys()), (
        f"missing keys: {expected_top_level - set(payload.keys())}"
    )
    assert payload["reactor"] == "ARC"
    assert payload["fuel"] == "DT"
    assert payload["concept"] == "TOKAMAK"


def test_benchmark_aries_at_produces_expected_keys(tmp_path):
    import benchmark_aries_at

    payload = benchmark_aries_at.run(tmp_path)

    assert (tmp_path / "aries_at.json").exists()
    on_disk = json.loads((tmp_path / "aries_at.json").read_text())
    assert payload == on_disk

    expected_top_level = {
        "reactor",
        "fuel",
        "concept",
        "inputs",
        "predicted_overnight_musd",
        "predicted_overnight_per_kwe_usd",
        "predicted_lcoe_usd_per_mwh",
        "fusion_power_mw",
        "net_electric_mw",
        "cas",
        "cas22_detail",
    }
    assert expected_top_level.issubset(payload.keys())
    assert payload["reactor"] == "ARIES-AT"
    assert payload["fuel"] == "DT"
    assert payload["concept"] == "TOKAMAK"
    assert "cas22" in payload["cas"]
    assert "C220103" in payload["cas22_detail"]
