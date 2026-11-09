"""Unit tests for the intake port flow bench simulation module (OpenFOAM simpleFoam)."""

import json
from pathlib import Path
import tempfile
import pytest

from openengine.cad.cylinder_head import CylinderHeadParams
from openengine.cfd.port_bench import (
    LiftPointResult,
    PortFlowConfig,
    PortFlowSweepResult,
    generate_port_geometry_at_lift,
    run_port_flow_sweep,
    setup_port_flow_case,
    solve_port_flow_at_lift,
)
from openengine.infra.podman import PodmanRunner
from openengine.meshing.gmsh_cfd import mesh_port_fluid_domain


def test_port_geometry_generation():
  """Verify CAD negative fluid volume creation with poppet valve subtraction."""
  geom_2mm = generate_port_geometry_at_lift(2.0)
  geom_5mm = generate_port_geometry_at_lift(5.0)

  assert geom_2mm.is_valid
  assert geom_5mm.is_valid
  assert geom_2mm.volume > 150000.0  # mm^3

  # Zero or negative lift should raise ValueError
  with pytest.raises(ValueError, match="Valve lift must be > 0.05 mm"):
    generate_port_geometry_at_lift(0.0)


def test_gmsh_cfd_meshing():
  """Verify automated boundary tagging and MSH 2.2 generation via Gmsh."""
  with tempfile.TemporaryDirectory() as td:
    geom = generate_port_geometry_at_lift(5.0)
    msh_path = Path(td) / "test_port.msh"

    stats = mesh_port_fluid_domain(
        geometry=geom,
        output_msh=msh_path,
        mesh_size_min=3.0,
        mesh_size_max=6.0,
    )

    assert msh_path.exists()
    assert stats["num_tets"] > 3000
    assert stats["inlet_surfaces"] == 1
    assert stats["outlet_surfaces"] == 1
    assert stats["wall_surfaces"] >= 15
    assert stats["wall_time_s"] < 5.0


def test_case_setup_and_pressure_injection():
  """Verify OpenFOAM case directory preparation and kinematic pressure scaling."""
  with tempfile.TemporaryDirectory() as td:
    case_path = Path(td) / "case_test"
    setup_port_flow_case(
        case_dir=case_path,
        delta_p_pa=7000.0,
        air_density=1.205,
        max_iterations=50,
    )

    p_file = case_path / "0" / "p"
    assert p_file.exists()
    p_text = p_file.read_text()
    assert "uniform 5809.13;" in p_text

    ctrl_file = case_path / "system" / "controlDict"
    assert ctrl_file.exists()
    ctrl_text = ctrl_file.read_text()
    assert "endTime         50;" in ctrl_text


@pytest.mark.skipif(
    not PodmanRunner().is_available,
    reason="Podman solver container is required for OpenFOAM CFD execution.",
)
def test_solve_single_lift_point():
  """Execute end-to-end simpleFoam simulation and verify physical aerodynamic metrics."""
  with tempfile.TemporaryDirectory() as td:
    case_path = Path(td) / "run_8_5mm"
    cfg = PortFlowConfig(
        max_iterations=60,
        mesh_size_min=2.5,
        mesh_size_max=5.0,
    )

    res = solve_port_flow_at_lift(
        lift_mm=8.5,
        workdir=case_path,
        config=cfg,
    )

    assert isinstance(res, LiftPointResult)
    assert res.lift_mm == 8.5
    assert res.mesh_cells > 3000
    assert res.mass_flow_kg_s > 0.015  # ~21.4 g/s at peak lift under 7 kPa
    assert 0.15 <= res.cd_seat <= 0.60  # Seat discharge coefficient (~0.23)
    assert 0.20 <= res.cd_curtain <= 0.75  # Curtain discharge coefficient (~0.29)
    assert abs(res.tumble_ratio) >= 0.50  # Strong tumble vortex induced by pent-roof (~1.05)
    assert abs(res.swirl_ratio) <= 0.20  # Negligible swirl for symmetric pent-roof (~0.002)
    assert res.solve_time_s < 25.0  # Rapid solve budget


@pytest.mark.skipif(
    not PodmanRunner().is_available,
    reason="Podman solver container is required for OpenFOAM CFD execution.",
)
def test_port_flow_sweep_execution():
  """Execute a 2-point lift sweep, verify JSON export and monotonic flow increase."""
  with tempfile.TemporaryDirectory() as td:
    out_dir = Path(td) / "sweep_out"
    cfg = PortFlowConfig(
        max_iterations=50,
        mesh_size_min=2.5,
        mesh_size_max=5.0,
    )

    sweep = run_port_flow_sweep(
        lifts_mm=[5.0, 8.5],
        output_dir=out_dir,
        config=cfg,
    )

    assert isinstance(sweep, PortFlowSweepResult)
    assert len(sweep.points) == 2

    # Mass flow must increase with valve lift
    assert sweep.points[1].mass_flow_kg_s > sweep.points[0].mass_flow_kg_s
    assert sweep.points[1].cd_seat > sweep.points[0].cd_seat

    # Verify JSON artifact
    json_path = out_dir / "port_flow_results.json"
    assert json_path.exists()
    with open(json_path) as f:
      data = json.load(f)
    assert len(data["points"]) == 2
    assert "cd_seat" in data["points"][0]
