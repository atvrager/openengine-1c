"""Unit tests for Conjugate Heat Transfer (CHT) air-fin thermal cooling (OpenFOAM laplacianFoam)."""

import json
from pathlib import Path
import tempfile
import pytest

from openengine.cfd.cht_cooling import (
    ChtCoolingConfig,
    ChtCoolingResult,
    generate_finned_block_mesh,
    run_cht_simulation,
    setup_cht_case,
)
from openengine.infra.podman import PodmanRunner


def test_cht_config_defaults():
  """Verify thermal simulation default properties and boundary parameters."""
  cfg = ChtCoolingConfig()
  assert cfg.ambient_temp_k == 300.0
  assert cfg.liner_inner_temp_k == 443.0
  assert cfg.fin_h_coeff == 75.0
  assert cfg.thermal_conductivity == 160.0
  assert cfg.heat_budget_w == 1300.0
  assert cfg.num_fins == 8


def test_generate_finned_block_mesh():
  """Verify Gmsh parametric OCC finned block mesh generation and boundary tagging."""
  with tempfile.TemporaryDirectory() as td:
    msh_path = Path(td) / "test_block.msh"
    cfg = ChtCoolingConfig(
        mesh_size_min_m=0.010,
        mesh_size_max_m=0.015,
        num_fins=4,
    )

    num_cells, ext_area = generate_finned_block_mesh(cfg, output_msh_path=msh_path)
    assert msh_path.exists()
    assert msh_path.stat().st_size > 0
    assert num_cells > 500
    assert ext_area > 0.05

    # Check physical groups in MSH 2.2 format
    with open(msh_path, "r") as f:
      content = f.read(2048)
    assert "$PhysicalNames" in content
    assert "liner" in content
    assert "fins" in content
    assert "topDeck" in content
    assert "baseFlange" in content
    assert "internal" in content


def test_setup_cht_case():
  """Verify OpenFOAM case directory preparation, transport properties, and Robin boundary condition."""
  with tempfile.TemporaryDirectory() as td:
    case_path = Path(td) / "case_cht_test"
    cfg = ChtCoolingConfig(
        ambient_temp_k=295.0,
        liner_inner_temp_k=440.0,
        fin_h_coeff=80.0,
    )

    setup_cht_case(case_path, cfg)

    assert (case_path / "constant" / "transportProperties").exists()
    assert (case_path / "system" / "controlDict").exists()
    assert (case_path / "system" / "fvSchemes").exists()
    assert (case_path / "system" / "fvSolution").exists()

    t_txt = (case_path / "0" / "T").read_text()
    assert "value           uniform 440.0;" in t_txt
    assert "type            mixed;" in t_txt
    assert "refValue        uniform 295.0;" in t_txt

    tp_txt = (case_path / "constant" / "transportProperties").read_text()
    assert "DT [0 2 -1 0 0 0 0] 6.58e-05;" in tp_txt


@pytest.mark.skipif(
    not PodmanRunner().is_available,
    reason="Podman solver container is required for OpenFOAM CFD execution.",
)
def test_fast_cht_thermal_solve():
  """Execute rapid end-to-end CHT simulation and verify thermal safety metrics."""
  with tempfile.TemporaryDirectory() as td:
    cfg = ChtCoolingConfig(
        mesh_size_min_m=0.008,
        mesh_size_max_m=0.014,
        end_time_s=10.0,
        delta_t_s=0.5,
    )

    res = run_cht_simulation(cfg, case_dir=td)
    assert isinstance(res, ChtCoolingResult)
    assert res.total_cells > 2000
    assert res.solve_time_s < 15.0

    # Physical verification:
    # 1. Inner liner temperature must be below oil breakdown/scuffing limit (180 C = 453.15 K)
    assert res.max_liner_temp_k <= 453.15

    # 2. Block and fin temperatures must be between ambient and liner
    assert cfg.ambient_temp_k <= res.min_fin_temp_k <= res.max_liner_temp_k
    assert cfg.ambient_temp_k <= res.mean_block_temp_k <= res.max_liner_temp_k

    # 3. Heat dissipation must be positive and verify cooling capacity
    assert res.heat_dissipated_w > 500.0


def test_cht_results_json_deliverable():
  """Verify standard CHT results artifact file and continuous heat rejection budget."""
  repo_root = Path(__file__).resolve().parent.parent
  json_path = repo_root / "outputs" / "cfd" / "cht_cooling_results.json"
  if not json_path.exists():
    pytest.skip("outputs/cfd/cht_cooling_results.json does not exist yet.")

  with open(json_path) as f:
    data = json.load(f)

  assert "config" in data
  assert "heat_dissipated_w" in data
  assert "thermal_margin_percent" in data
  assert "is_thermal_safe" in data

  # Heat dissipation must shed combustion heat rejection (>= 1300 W)
  assert data["heat_dissipated_w"] >= 1300.0
  assert data["is_thermal_safe"] is True
  assert data["max_liner_temp_k"] <= 453.15
  assert data["thermal_margin_percent"] > 0.0
