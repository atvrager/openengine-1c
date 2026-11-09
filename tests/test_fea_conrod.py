"""Unit tests for Connecting Rod FEA (LC-3A compression, LC-3B tension, buckling, fatigue)."""

import json
from pathlib import Path
import pytest

from openengine.fea.connecting_rod import (
    ConrodFeaConfig,
    ConrodFeaSummaryResult,
    compute_column_buckling_load,
    generate_conrod_mesh,
    run_conrod_fea,
)


def test_conrod_meshing_and_surfaces(tmp_path: Path):
  """Verify Gmsh quadratic C3D10 meshing and node set identification on connecting rod."""
  cad_step = Path("outputs/cad/connecting_rod.step")
  mesh_inp = tmp_path / "conrod_test_mesh.inp"

  res = generate_conrod_mesh(
      cad_step_path=cad_step,
      output_inp_path=mesh_inp,
      mesh_size_min=3.5,
      mesh_size_max=7.0,
  )

  assert res.num_nodes > 1000
  assert res.num_elements > 500
  assert res.element_type == "C3D10"
  assert "NSMALL_PIN" in res.node_sets
  assert "NBIG_CRANK" in res.node_sets
  assert len(res.node_sets["NSMALL_PIN"]) > 100
  assert len(res.node_sets["NBIG_CRANK"]) > 100


def test_conrod_column_buckling():
  """Verify analytical Euler critical buckling formula for I-beam shank."""
  p_cr = compute_column_buckling_load()
  assert p_cr > 200000.0  # > 200 kN critical buckling load
  f_firing = 7.80e6 * (3.14159 / 4.0) * (0.068**2)  # ~28.3 kN
  sf_buckle = p_cr / f_firing
  assert sf_buckle >= 2.50  # Passes acceptance criterion


def test_conrod_fea_solve(tmp_path: Path):
  """Verify complete LC-3A, LC-3B, buckling, and Goodman fatigue analysis."""
  cfg = ConrodFeaConfig(
      peak_pressure_mpa=7.80,
      redline_rpm=7000.0,
      mesh_size_min_mm=3.5,
      mesh_size_max_mm=7.0,
  )
  out_dir = tmp_path / "fea_conrod"

  res = run_conrod_fea(
      config=cfg,
      output_dir=out_dir,
      cad_step_path="outputs/cad/connecting_rod.step",
  )

  assert isinstance(res, ConrodFeaSummaryResult)
  assert res.total_solve_time_s < 30.0

  # LC-3A Firing Compression
  assert res.lc3a_compression.applied_force_kn > 25.0
  assert res.lc3a_compression.yield_safety_factor >= 2.0
  assert res.lc3a_compression.is_pass is True

  # LC-3B Redline Inertia Pull
  assert res.lc3b_tension.applied_force_kn > 2.0
  assert res.lc3b_tension.yield_safety_factor >= 10.0
  assert res.lc3b_tension.is_pass is True

  # Stability & Fatigue
  assert res.buckling_safety_factor >= 2.50
  assert res.is_buckling_safe is True
  assert res.fatigue_safety_factor >= 1.50
  assert res.is_fatigue_safe is True
  assert res.is_structural_safe is True

  # JSON serialization
  json_file = out_dir / "conrod_fea_results.json"
  assert json_file.exists()
  with open(json_file, "r") as f:
    data = json.load(f)
  assert data["is_structural_safe"] is True
  assert data["buckling_safety_factor"] == pytest.approx(res.buckling_safety_factor)
