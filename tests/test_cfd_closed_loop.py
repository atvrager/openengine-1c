"""Unit tests for 3D-to-0D closed-loop thermodynamic calibration."""

import json
from pathlib import Path
import tempfile
import pytest

from openengine.cfd.closed_loop import (
    ClosedLoopCalibrationResult,
    ClosedLoopConfig,
    CycleComparisonMetrics,
    run_closed_loop_calibration,
)


def test_closed_loop_calibration_defaults(tmp_path: Path):
  """Verify closed-loop calibration runs with defaults even if files are missing."""
  cfg = ClosedLoopConfig(
      combustion_results_path=tmp_path / "nonexistent_comb.json",
      port_flow_results_path=tmp_path / "nonexistent_port.json",
      cht_results_path=tmp_path / "nonexistent_cht.json",
  )
  out_json = tmp_path / "test_closed_loop.json"
  res = run_closed_loop_calibration(config=cfg, output_json_path=out_json)

  assert isinstance(res, ClosedLoopCalibrationResult)
  assert out_json.exists()

  # Baseline validation
  assert res.baseline.imep_bar > 10.0
  assert res.baseline.indicated_power_kw > 8.0
  assert res.baseline.thermal_efficiency > 0.40

  # Calibrated validation
  assert res.calibrated.imep_bar > 5.0
  assert res.calibrated.indicated_power_kw > 5.0
  assert res.calibrated.thermal_efficiency > 0.30

  # Comparison deltas
  expected_delta_imep = ((res.calibrated.imep_bar - res.baseline.imep_bar) / res.baseline.imep_bar) * 100.0
  assert pytest.approx(res.delta_imep_percent, rel=1e-3) == expected_delta_imep
  assert res.solve_time_s > 0.0


def test_closed_loop_calibration_with_cfd_outputs():
  """Verify closed-loop calibration ingests generated CFD simulation JSON results."""
  res = run_closed_loop_calibration()

  # Check Wiebe parameters ingested from combustion_results.json (87-AKI E10 fit)
  comb_json = Path("outputs/cfd/combustion_results.json")
  if comb_json.exists():
    wiebe_ref = json.loads(comb_json.read_text())["wiebe"]
    assert res.calibrated_wiebe_a == pytest.approx(wiebe_ref["a"], abs=0.01)
    assert res.calibrated_wiebe_m == pytest.approx(wiebe_ref["m"], abs=0.01)
    assert res.calibrated_wiebe_duration_deg == pytest.approx(wiebe_ref["duration_deg"], abs=0.5)
  assert 1.2 <= res.calibrated_wiebe_a <= 3.0
  assert 2.0 <= res.calibrated_wiebe_m <= 4.0
  assert 60.0 <= res.calibrated_wiebe_duration_deg <= 90.0
  assert res.calibrated_wiebe_soc_deg == pytest.approx(-18.0, abs=0.1)

  # Check Port Flow Curtain Cd from port_flow_results.json
  assert 0.15 <= res.calibrated_intake_cd <= 0.65

  # Check CHT Wall Temperature from cht_cooling_results.json
  assert res.calibrated_wall_temp_k == pytest.approx(443.0, abs=1.0)

  # Physics validation: realistic combustion duration yields lower peak pressure than instantaneous burn
  assert res.calibrated.p_max_bar < res.baseline.p_max_bar
  assert res.calibrated.imep_bar > 8.0  # realistic indicated work


def test_closed_loop_json_serialization(tmp_path: Path):
  """Verify JSON serialization and deserialization integrity."""
  out_json = tmp_path / "closed_loop.json"
  res = run_closed_loop_calibration(output_json_path=out_json)

  with open(out_json, "r") as f:
    data = json.load(f)

  assert "config" in data
  assert "baseline" in data
  assert "calibrated" in data
  assert "delta_imep_percent" in data
  assert "calibrated_wiebe_a" in data
  assert data["calibrated"]["imep_bar"] == pytest.approx(res.calibrated.imep_bar)
