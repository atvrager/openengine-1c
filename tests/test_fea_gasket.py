"""Unit tests for Cylinder Head Bolt Preload and Gasket Fire-Ring Sealing."""

import json
from pathlib import Path
import pytest

from openengine.fea.head_gasket import (
    HeadGasketFeaConfig,
    HeadGasketSummaryResult,
    run_head_gasket_fea,
)


def test_head_gasket_sealing_physics(tmp_path: Path):
  """Verify bolted joint clamping, combustion separation, and fire-ring sealing."""
  cfg = HeadGasketFeaConfig(
      num_bolts=4,
      bolt_preload_kn=22.0,
      peak_pressure_mpa=7.80,
  )
  out_dir = tmp_path / "fea_gasket"

  res = run_head_gasket_fea(config=cfg, output_dir=out_dir)

  assert isinstance(res, HeadGasketSummaryResult)
  assert res.total_clamp_force_kn == 88.0
  assert res.combustion_separation_force_kn == pytest.approx(28.33, abs=0.1)

  # Fire-ring pressure criteria
  assert res.initial_contact_pressure_mpa > 70.0
  assert res.residual_contact_pressure_mpa >= cfg.min_sealing_pressure_mpa
  assert res.sealing_safety_factor >= 1.0
  assert res.is_sealing_safe is True

  # Bolt stresses
  assert res.bolt_peak_stress_mpa < 940.0  # < Grade 10.9 yield strength
  assert res.bolt_yield_safety_factor >= cfg.min_bolt_safety_factor
  assert res.is_bolt_safe is True
  assert res.is_structural_safe is True

  # JSON persistence
  json_file = out_dir / "head_gasket_fea_results.json"
  assert json_file.exists()
  with open(json_file, "r") as f:
    data = json.load(f)
  assert data["is_sealing_safe"] is True
  assert data["residual_contact_pressure_mpa"] == pytest.approx(res.residual_contact_pressure_mpa)
