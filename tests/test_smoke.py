"""Smoke tests for OpenEngine-1C core modules."""

import os
import tempfile
import numpy as np
from openengine.cad.connecting_rod import generate_connecting_rod
from openengine.cad.piston import generate_piston
from openengine.dynamics.slider_crank import SliderCrankDynamics
from openengine.infra.podman import PodmanRunner
from openengine.thermo.cycle import (
    ClosedCycleSimulator,
    EngineGeometry,
    wiebe_burned_fraction,
)
import pytest


@pytest.fixture
def ref_geom():
  return EngineGeometry(
      bore=0.068,
      stroke=0.054,
      con_rod_length=0.108,
      compression_ratio=9.8,
  )


def test_geometry_kinematics(ref_geom):
  # Displacement should be ~196 cm^3
  disp_cm3 = ref_geom.displacement * 1.0e6
  assert 195.0 < disp_cm3 < 197.0

  # Volume at TDC (theta=0) should equal clearance volume
  v_tdc = ref_geom.volume(0.0)
  assert np.isclose(v_tdc, ref_geom.clearance_volume, rtol=1e-5)

  # Volume at BDC (theta=pi) should equal clearance volume + displacement
  v_bdc = ref_geom.volume(np.pi)
  assert np.isclose(
      v_bdc, ref_geom.clearance_volume + ref_geom.displacement, rtol=1e-5
  )


def test_wiebe_burned_fraction():
  # Before start of combustion (-15 deg)
  assert wiebe_burned_fraction(-20.0) == 0.0

  # After combustion ends (-15 + 42 = +27 deg)
  assert wiebe_burned_fraction(30.0) == 1.0

  # Mid-combustion fraction should be between 0 and 1
  mid_xb = wiebe_burned_fraction(0.0)
  assert 0.2 < mid_xb < 0.8


def test_closed_cycle_thermodynamics(ref_geom):
  sim = ClosedCycleSimulator(ref_geom)
  res = sim.solve_cycle()

  # Firing pressure should be realistic for NA gasoline engine (5.0 - 9.0 MPa)
  assert 5.0e6 < res.p_max_pa < 9.0e6
  assert 2000.0 < res.t_max_k < 3500.0
  assert res.imep_pa > 7.0e5  # > 7 bar IMEP
  assert 0.25 < res.thermal_efficiency < 0.60


def test_mechanism_dynamics_and_balancing(ref_geom):
  sim = ClosedCycleSimulator(ref_geom)
  res = sim.solve_cycle()

  dyn = SliderCrankDynamics(ref_geom)
  forces = dyn.solve_forces(
      rpm=5500.0,
      theta_deg=res.theta_deg,
      pressure_pa=res.pressure_pa,
  )

  # Peak gas force should match P_max * Area
  piston_area = np.pi * (ref_geom.bore / 2.0) ** 2
  expected_gas_force = res.p_max_pa * piston_area
  assert np.isclose(np.max(forces.gas_force_n), expected_gas_force, rtol=1e-3)

  # Size flywheel
  balancer = dyn.size_balancer_and_flywheel(rpm=5500.0, forces=forces)
  assert balancer.counterweight_mass_kg > 0.05  # > 50 grams
  assert balancer.flywheel_inertia_kg_m2 > 1.0e-4
  assert 1.0 < balancer.flywheel_mass_kg < 10.0


def test_podman_runner():
  runner = PodmanRunner()
  # Verify runner can be instantiated and returns a boolean for availability
  assert isinstance(runner.is_available, bool)


def test_cad_connecting_rod_generation():
  rod = generate_connecting_rod()
  assert rod is not None
  # Solid must have non-zero positive volume
  assert rod.volume > 0.0
  # Check bounding box envelope (length ~ 108 mm between centers)
  bbox = rod.bounding_box()
  assert bbox.size.Y > 100.0


def test_cad_piston_generation():
  piston = generate_piston()
  assert piston is not None
  assert piston.volume > 0.0
  bbox = piston.bounding_box()
  # Diameter should be ~68 mm
  assert np.isclose(bbox.size.X, 68.0, atol=1.0)
  assert np.isclose(bbox.size.Y, 68.0, atol=1.0)


def test_container_openfoam_solvers():
  runner = PodmanRunner()
  if not runner.is_available:
    pytest.skip("Podman/Docker is not available")

  for solver in ["simpleFoam", "XiEngineFoam", "engineFoam", "blockMesh"]:
    res = runner.run_command(f"{solver} -help", os.getcwd())
    assert res.returncode == 0
    assert len(res.stdout) > 100


def test_container_calculix_solve():
  runner = PodmanRunner()
  if not runner.is_available:
    pytest.skip("Podman/Docker is not available")

  inp_content = """*HEADING
Single element cube under tension
*NODE
1, 0., 0., 0.
2, 1., 0., 0.
3, 1., 1., 0.
4, 0., 1., 0.
5, 0., 0., 1.
6, 1., 0., 1.
7, 1., 1., 1.
8, 0., 1., 1.
*ELEMENT, TYPE=C3D8, ELSET=Eall
1, 1, 2, 3, 4, 5, 6, 7, 8
*MATERIAL, NAME=STEEL
*ELASTIC
210000., 0.3
*SOLID SECTION, ELSET=Eall, MATERIAL=STEEL
*BOUNDARY
1, 1, 3
2, 2, 3
4, 1
*STEP
*STATIC
*CLOAD
5, 3, 100.
6, 3, 100.
7, 3, 100.
8, 3, 100.
*NODE FILE
U
*EL FILE
S
*END STEP
"""
  with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
    rel_tmp = os.path.relpath(tmpdir, os.getcwd())
    inp_path = os.path.join(tmpdir, "cube.inp")
    with open(inp_path, "w") as f:
      f.write(inp_content)

    res = runner.run_command(f"cd {rel_tmp} && ccx cube", os.getcwd())
    assert res.returncode == 0
    frd_file = os.path.join(tmpdir, "cube.frd")
    assert os.path.exists(frd_file)
    assert os.path.getsize(frd_file) > 1000


def test_container_gmsh_mesh():
  runner = PodmanRunner()
  if not runner.is_available:
    pytest.skip("Podman/Docker is not available")

  geo_content = """Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
"""
  with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
    rel_tmp = os.path.relpath(tmpdir, os.getcwd())
    geo_path = os.path.join(tmpdir, "box.geo")
    with open(geo_path, "w") as f:
      f.write(geo_content)

    res = runner.run_command(f"cd {rel_tmp} && gmsh box.geo -2 -o box.msh", os.getcwd())
    assert res.returncode == 0
    msh_file = os.path.join(tmpdir, "box.msh")
    assert os.path.exists(msh_file)
    assert os.path.getsize(msh_file) > 200
