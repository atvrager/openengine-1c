"""Unit tests for 3D moving-mesh in-cylinder combustion simulation (XiEngineFoam)."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from openengine.cfd.combustion import (
    CombustionConfig,
    CombustionResult,
    CombustionStepPoint,
    WiebeFitResult,
    calibrate_wiebe_from_trajectory,
    compute_clearance_height,
    generate_combustion_mesh,
    parse_combustion_stdout,
    run_combustion_simulation,
    setup_combustion_case,
)
from openengine.infra.podman import PodmanRunner
from openengine.thermo.cycle import wiebe_burned_fraction


def test_compute_clearance_height():
  """Verify clearance height calculation from stroke and compression ratio."""
  # S = 54.0 mm, r_c = 9.80:1
  # c = S / (r_c - 1) = 0.054 / 8.80 = 0.00613636 m (~6.136 mm)
  c = compute_clearance_height(0.054, 9.80)
  assert abs(c - 0.00613636) < 1e-6
  assert abs(c * 1000.0 - 6.136) < 0.01

  # Test r_c <= 1.0 error handling
  with pytest.raises(ValueError, match="Compression ratio must be > 1.0"):
    compute_clearance_height(0.054, 1.0)


def test_generate_combustion_mesh():
  """Verify automated 3D cylinder tetrahedral mesh creation with boundary physical groups."""
  with tempfile.TemporaryDirectory() as td:
    msh_path = Path(td) / "test_cylinder.msh"
    # Fast coarse test mesh
    num_cells = generate_combustion_mesh(
        bore_m=0.068,
        stroke_m=0.054,
        clearance_height_m=0.006136,
        mesh_size_min_m=0.008,
        mesh_size_max_m=0.012,
        output_msh_path=msh_path,
    )
    assert msh_path.exists()
    assert msh_path.stat().st_size > 0
    assert num_cells > 300

    # Read MSH header and verify physical group names
    with open(msh_path, "r") as f:
      content = f.read(2048)
    assert "$PhysicalNames" in content
    assert "piston" in content
    assert "cylinderHead" in content
    assert "liner" in content
    assert "internal" in content


def test_setup_combustion_case():
  """Verify OpenFOAM case directory template population and dictionary parametrization."""
  with tempfile.TemporaryDirectory() as td:
    case_path = Path(td) / "case_test"
    cfg = CombustionConfig(
        bore_m=0.068,
        stroke_m=0.054,
        start_crank_angle_deg=-180.0,
        end_crank_angle_deg=-160.0,
        spark_crank_angle_deg=-25.0,
        spark_duration_deg=18.0,
    )

    setup_combustion_case(case_path, cfg)

    # Verify constant directory
    assert (case_path / "constant" / "engineGeometry").exists()
    assert (case_path / "constant" / "combustionProperties").exists()
    assert (case_path / "constant" / "thermophysicalProperties").exists()
    assert (case_path / "constant" / "turbulenceProperties").exists()

    # Check parameterized engine geometry
    geom_txt = (case_path / "constant" / "engineGeometry").read_text()
    assert "conRodLength [0 1 0 0 0 0 0] 0.108;" in geom_txt
    assert "bore [0 1 0 0 0 0 0] 0.068;" in geom_txt
    assert "stroke [0 1 0 0 0 0 0] 0.054;" in geom_txt
    assert "rpm [0 0 -1 0 0 0 0] 5500;" in geom_txt

    # Check spark location and timing
    comb_txt = (case_path / "constant" / "combustionProperties").read_text()
    assert "start           -25.0;" in comb_txt
    assert "duration        18.0;" in comb_txt

    # Check controlDict
    ctrl_txt = (case_path / "system" / "controlDict").read_text()
    assert "startTime       -180;" in ctrl_txt
    assert "endTime         -160;" in ctrl_txt

    # Check initial conditions directory matching startTime
    init_p = case_path / "-180" / "p"
    assert init_p.exists()
    assert "uniform 100000.0;" in init_p.read_text()


def test_calibrate_wiebe_from_synthetic_trajectory():
  """Verify non-linear curve fitting accurately recovers known Wiebe parameters."""
  theta = np.linspace(-22.0, 24.0, 100)
  # True parameters: a=4.5, m=2.2, duration=48.0 deg
  xb_true = wiebe_burned_fraction(theta, theta_soc_deg=-22.0, duration_deg=48.0, a=4.5, m=2.2)

  # 1. Fit with fixed a=4.5 to test parameter identification of m and duration
  fit_fixed = calibrate_wiebe_from_trajectory(theta, xb_true, theta_spark_deg=-22.0, a_fixed=4.5)
  assert fit_fixed is not None
  assert isinstance(fit_fixed, WiebeFitResult)
  assert fit_fixed.a == 4.5
  assert abs(fit_fixed.m - 2.2) < 0.1
  assert abs(fit_fixed.duration_deg - 48.0) < 0.5
  assert fit_fixed.r_squared > 0.999

  # 2. Fit with unconstrained (a, m, duration) and verify high goodness-of-fit
  fit_free = calibrate_wiebe_from_trajectory(theta, xb_true, theta_spark_deg=-22.0)
  assert fit_free is not None
  assert fit_free.r_squared > 0.999


def test_parse_combustion_stdout_format():
  """Verify regex parsing of XiEngineFoam stdout stream."""
  sample_stdout = """
Courant Number mean: 0.000273322 max: 0.00362594
Crank angle = -180 CA-deg
deltaZ = 7.51729e-06
clearance: 0.060136
Piston speed = 0.992282 m/s
Mean pressure:100000
Mean temperature:373.0
Mean u':1.61122

Courant Number mean: 0.00544023 max: 0.0569268
Crank angle = -175 CA-deg
deltaZ = 0.000229337
clearance: 0.059800
Piston speed = 2.19782 m/s
Mean pressure:101500
Mean temperature:375.2
Mean u':1.65000

Combustion progress = 2.50%
Courant Number mean: 0.0123 max: 0.150
Crank angle = -10 CA-deg
deltaZ = 0.000100
clearance: 0.007500
Piston speed = 5.20000 m/s
Mean pressure:1450000
Mean temperature:710.5
Mean u':2.40000
"""
  points, p_max, th_p, t_max, th_t = parse_combustion_stdout(sample_stdout)
  assert len(points) == 3
  assert points[0].crank_angle_deg == -180.0
  assert points[0].mean_pressure_pa == 100000.0
  assert points[1].crank_angle_deg == -175.0
  assert points[1].mean_pressure_pa == 101500.0
  assert points[2].crank_angle_deg == -10.0
  assert points[2].combustion_progress == 0.025
  assert p_max == 1450000.0
  assert th_p == -10.0
  assert t_max == 710.5
  assert th_t == -10.0


@pytest.mark.skipif(
    not PodmanRunner().is_available,
    reason="Podman solver container is required for OpenFOAM CFD execution.",
)
def test_fast_moving_mesh_kinematics():
  """Execute a rapid moving-mesh simulation (-180 to -175 CA) to verify dynamic mesh motion."""
  with tempfile.TemporaryDirectory() as td:
    cfg = CombustionConfig(
        start_crank_angle_deg=-180.0,
        end_crank_angle_deg=-175.0,
        mesh_size_min_m=0.006,
        mesh_size_max_m=0.009,
    )
    res = run_combustion_simulation(cfg, case_dir=td)

    assert isinstance(res, CombustionResult)
    assert len(res.points) >= 2
    # Piston must compress fluid: clearance must decrease, pressure must rise
    assert res.points[-1].clearance_m < res.points[0].clearance_m
    assert res.points[-1].mean_pressure_pa > res.points[0].mean_pressure_pa
    assert res.solve_time_s < 10.0  # Ultra-rapid kinematic check


def test_combustion_results_json_deliverable():
  """Verify standard combustion results artifact file and thermodynamic criteria."""
  repo_root = Path(__file__).resolve().parent.parent
  json_path = repo_root / "outputs" / "cfd" / "combustion_results.json"
  if not json_path.exists():
    pytest.skip("outputs/cfd/combustion_results.json does not exist yet.")

  with open(json_path) as f:
    data = json.load(f)

  assert "config" in data
  assert "p_max_pa" in data
  assert "wiebe" in data
  assert "points" in data
  assert len(data["points"]) > 100

  # Physical combustion metrics: peak pressure ~2 MPa (20 bar), ignition rise
  assert data["p_max_pa"] > 1.8e6  # > 18 bar
  assert 0.0 <= data["theta_p_max_deg"] <= 10.0  # Peak pressure near TDC
  assert data["t_max_k"] > 750.0  # Gas heating during compression and early flame
  assert data["burn_progress_at_end"] > 0.02  # Mass fraction burned advanced

  # Wiebe calibration quality
  wiebe = data["wiebe"]
  assert wiebe is not None
  assert wiebe["r_squared"] > 0.99
  assert 1.0 <= wiebe["a"] <= 5.0
  assert 1.5 <= wiebe["m"] <= 4.0

