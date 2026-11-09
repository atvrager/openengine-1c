"""Unit tests for Piston FEA (LC-1 mechanical & LC-2 thermo-mechanical)."""

import json
from pathlib import Path
import pytest

from openengine.fea.piston import (
    PistonFeaConfig,
    PistonFeaSummaryResult,
    generate_piston_mesh,
    run_piston_fea,
)


def test_piston_meshing_and_surfaces(tmp_path: Path):
  """Verify Gmsh quadratic C3D10 meshing and node set identification on piston."""
  cad_step = Path("outputs/cad/piston.step")
  mesh_inp = tmp_path / "piston_test_mesh.inp"

  res = generate_piston_mesh(
      cad_step_path=cad_step,
      output_inp_path=mesh_inp,
      mesh_size_min=3.5,
      mesh_size_max=7.0,
  )

  assert res.num_nodes > 1000
  assert res.num_elements > 500
  assert res.element_type == "C3D10"
  assert "NCROWN" in res.node_sets
  assert "NPIN_BORE" in res.node_sets
  assert len(res.node_sets["NCROWN"]) > 50
  assert len(res.node_sets["NPIN_BORE"]) > 20
  assert mesh_inp.exists()


def test_piston_fea_solve(tmp_path: Path):
  """Verify complete LC-1 and LC-2 CalculiX FEA execution on the piston."""
  cfg = PistonFeaConfig(
      peak_pressure_mpa=7.80,
      mesh_size_min_mm=3.5,
      mesh_size_max_mm=7.0,
  )
  out_dir = tmp_path / "fea_piston"

  res = run_piston_fea(
      config=cfg,
      output_dir=out_dir,
      cad_step_path="outputs/cad/piston.step",
  )

  assert isinstance(res, PistonFeaSummaryResult)
  assert res.total_solve_time_s < 30.0  # Fast solve within budget

  # LC-1 Mechanical Validation
  assert res.lc1_mechanical.max_von_mises_mpa > 100.0
  assert res.lc1_mechanical.p95_von_mises_mpa < 250.0
  assert res.lc1_mechanical.yield_safety_factor >= 1.60
  assert res.lc1_mechanical.max_displacement_mm < 0.50

  # LC-2 Thermo-Mechanical Validation
  assert res.hot_skirt_clearance_mm >= cfg.min_hot_clearance_mm
  assert res.is_clearance_safe is True
  assert res.is_structural_safe is True

  # JSON persistence
  json_file = out_dir / "piston_fea_results.json"
  assert json_file.exists()
  with open(json_file, "r") as f:
    data = json.load(f)
  assert data["is_structural_safe"] is True
  assert data["lc1_mechanical"]["yield_safety_factor"] == pytest.approx(res.lc1_mechanical.yield_safety_factor)


def test_piston_acceptance_criteria(tmp_path: Path):
  """Verify piston strictly satisfies Section 8 acceptance criteria (SF >= 1.60)."""
  cfg = PistonFeaConfig(
      peak_pressure_mpa=7.80,
      mesh_size_min_mm=2.5,
      mesh_size_max_mm=5.0,
  )
  out_dir = tmp_path / "fea_piston_strict"
  res = run_piston_fea(
      config=cfg,
      output_dir=out_dir,
      cad_step_path="outputs/cad/piston.step",
  )
  assert res.is_structural_safe is True
  assert res.lc1_mechanical.yield_safety_factor >= 1.60
