"""Intake port and valve aerodynamic flow bench simulation via OpenFOAM simpleFoam.

Computes the mass flow rate, discharge coefficient Cd(h_v), and tumble/swirl
ratios across parametric valve lift positions.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from build123d import Location, Part, Shape, Vector
import numpy as np

from openengine.cad.cylinder_head import (
    CylinderHeadParams,
    extract_intake_port_fluid_domain,
    generate_poppet_valve,
)
from openengine.infra.podman import PodmanRunner
from openengine.meshing.gmsh_cfd import mesh_port_fluid_domain


@dataclass
class PortFlowConfig:
  """Configuration parameters for the steady intake port flow bench."""

  delta_p_pa: float = 6973.7  # Standard flow bench depression: 28 in H2O (~7.0 kPa)
  air_density: float = 1.205  # kg/m^3 at 20 deg C, 1 atm
  kinematic_viscosity: float = 1.5e-5  # m^2/s
  rated_rpm: float = 5500.0  # Reference engine speed for non-dimensional tumble index
  max_iterations: int = 100  # SIMPLE iterations (sufficient for sub-5 second convergence)
  mesh_size_min: float = 2.0  # mm minimum cell size near valve seat
  mesh_size_max: float = 4.5  # mm maximum cell size in barrel
  cylinder_depth_m: float = 0.040  # 40 mm cylinder extension below deck face


@dataclass
class LiftPointResult:
  """Aerodynamic flow bench simulation results for a single valve lift position."""

  lift_mm: float
  lift_over_diameter: float
  mass_flow_kg_s: float
  volume_flow_m3_s: float
  cd_seat: float
  cd_curtain: float
  tumble_angular_momentum_n_m_s: float
  tumble_angular_velocity_rad_s: float
  tumble_ratio: float
  swirl_angular_momentum_n_m_s: float
  swirl_angular_velocity_rad_s: float
  swirl_ratio: float
  iterations: int
  solve_time_s: float
  mesh_cells: int


@dataclass
class PortFlowSweepResult:
  """Aggregated flow bench results across a parametric valve lift sweep."""

  config: PortFlowConfig
  points: List[LiftPointResult]
  total_solve_time_s: float

  def to_dict(self) -> Dict[str, Any]:
    return {
        "config": asdict(self.config),
        "total_solve_time_s": self.total_solve_time_s,
        "points": [asdict(pt) for pt in self.points],
    }

  def to_json(self, filepath: Union[str, Path]) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
      json.dump(self.to_dict(), f, indent=2)


def generate_port_geometry_at_lift(
    lift_mm: float,
    params: Optional[CylinderHeadParams] = None,
    cylinder_depth_m: float = 0.040,
) -> Part:
  """Construct the negative air volume of the intake port with the valve open at lift_mm.

  Displaces the intake poppet valve along its stem axis by lift_mm into the cylinder
  chamber and subtracts the valve body from the fluid volume.

  Args:
    lift_mm: Valve opening lift in millimeters (must be > 0.05 mm).
    params: Cylinder head geometric parameters.
    cylinder_depth_m: Depth of cylinder fluid domain below deck face.

  Returns:
    Watertight build123d Part representing the negative fluid domain.
  """
  if lift_mm <= 0.05:
    raise ValueError(f"Valve lift must be > 0.05 mm to permit flow (got {lift_mm} mm).")

  p = params or CylinderHeadParams()
  fluid_base = extract_intake_port_fluid_domain(p, cylinder_depth_m=cylinder_depth_m)
  valve = generate_poppet_valve(p.intake_valve_diam_m, p.valve_stem_diam_m)

  slope = p.valve_angle_deg
  axis_dir = Vector(np.sin(np.radians(slope)), 0.0, np.cos(np.radians(slope)))
  p_seat = Vector(16.0, 0.0, 6.0)

  # Valve opening moves down into the chamber (-axis_dir)
  pos = p_seat - axis_dir * lift_mm
  valve_pos = Location(pos, (0.0, slope, 0.0)) * valve

  # Subtract valve from fluid domain
  fluid_domain = fluid_base - valve_pos
  return fluid_domain


def parse_openfoam_field(file_path: Union[str, Path]) -> np.ndarray:
  """Parse a non-uniform ASCII OpenFOAM scalar or vector field file into a NumPy array."""
  p = Path(file_path)
  if not p.exists():
    raise FileNotFoundError(f"OpenFOAM field file not found: {p}")

  with open(p, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

  # Match list block: count \n ( \n ... \n )
  match = re.search(r"nonuniform\s+List<[^>]+>\s*\n(\d+)\s*\n\((.*?)\n\)", content, re.DOTALL)
  if not match:
    # Try uniform field format
    match_uni = re.search(r"internalField\s+uniform\s+([^;]+);", content)
    if match_uni:
      val_str = match_uni.group(1).strip()
      if val_str.startswith("(") and val_str.endswith(")"):
        return np.array([[float(x) for x in val_str[1:-1].split()]])
      return np.array([float(val_str)])
    raise ValueError(f"Could not parse internalField from {file_path}")

  body = match.group(2).strip()
  lines = [line.strip() for line in body.splitlines() if line.strip()]

  data = []
  for line in lines:
    if line.startswith("(") and line.endswith(")"):
      data.append([float(x) for x in line[1:-1].split()])
    else:
      try:
        data.append(float(line))
      except ValueError:
        pass

  return np.array(data)


def parse_outlet_flow_rate(case_dir: Union[str, Path]) -> Tuple[int, float]:
  """Extract the converged volumetric flow rate sum(phi) from surfaceFieldValue dat file."""
  post_dir = Path(case_dir) / "postProcessing" / "outletFlow"
  dat_files = list(post_dir.glob("*/surfaceFieldValue*.dat"))
  if not dat_files:
    raise FileNotFoundError(f"No surfaceFieldValue dat file found in {post_dir}")

  # Use the latest file
  latest_file = sorted(dat_files, key=lambda f: f.stat().st_mtime)[-1]
  with open(latest_file, "r") as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

  if not lines:
    raise ValueError(f"No data lines found in {latest_file}")

  last_line = lines[-1].split()
  time_step = int(float(last_line[0]))
  sum_phi = float(last_line[1])
  return time_step, abs(sum_phi)


def compute_tumble_and_swirl(
    case_dir: Union[str, Path],
    latest_time: int,
    air_density: float = 1.205,
    rated_rpm: float = 5500.0,
) -> Dict[str, float]:
  """Compute cylinder tumble and swirl angular momentum and non-dimensional ratios."""
  time_dir = Path(case_dir) / str(latest_time)
  u_path = time_dir / "U"
  c_path = time_dir / "C"
  v_path = time_dir / "V"

  u_field = parse_openfoam_field(u_path)
  c_field = parse_openfoam_field(c_path)
  v_field = parse_openfoam_field(v_path)

  # Only consider cells inside the cylinder (Z <= 0.0)
  cyl_mask = c_field[:, 2] <= 0.0
  if np.sum(cyl_mask) == 0:
    cyl_mask = np.ones(len(c_field), dtype=bool)

  v_cyl = v_field[cyl_mask]
  c_cyl = c_field[cyl_mask]
  u_cyl = u_field[cyl_mask]

  # Centroid along cylinder axis (Z)
  z_c = float(np.average(c_cyl[:, 2], weights=v_cyl))

  # Coordinates relative to cylinder bore axis (X=0, Y=0)
  x = c_cyl[:, 0]
  y = c_cyl[:, 1]
  dz = c_cyl[:, 2] - z_c

  # Tumble angular momentum (rotation around transverse Y axis)
  # Ly = sum(rho * V * ((z - z_c) * Ux - x * Uz))
  ly = float(np.sum(air_density * v_cyl * (dz * u_cyl[:, 0] - x * u_cyl[:, 2])))
  iy = float(np.sum(air_density * v_cyl * (x**2 + dz**2)))
  omega_tumble = ly / iy if iy > 1e-12 else 0.0

  # Swirl angular momentum (rotation around cylinder bore axis Z at X=0, Y=0)
  # Lz = sum(rho * V * (x * Uy - y * Ux))
  lz = float(np.sum(air_density * v_cyl * (x * u_cyl[:, 1] - y * u_cyl[:, 0])))
  iz = float(np.sum(air_density * v_cyl * (x**2 + y**2)))
  omega_swirl = lz / iz if iz > 1e-12 else 0.0

  omega_eng = 2.0 * np.pi * rated_rpm / 60.0
  tumble_ratio = omega_tumble / omega_eng
  swirl_ratio = omega_swirl / omega_eng

  return {
      "tumble_angular_momentum_n_m_s": ly,
      "tumble_angular_velocity_rad_s": omega_tumble,
      "tumble_ratio": tumble_ratio,
      "swirl_angular_momentum_n_m_s": lz,
      "swirl_angular_velocity_rad_s": omega_swirl,
      "swirl_ratio": swirl_ratio,
  }


def setup_port_flow_case(
    case_dir: Path,
    template_dir: Optional[Path] = None,
    delta_p_pa: float = 6973.7,
    air_density: float = 1.205,
    max_iterations: int = 100,
) -> None:
  """Prepare OpenFOAM case directory by copying templates and updating pressure drop."""
  if template_dir is None:
    # Default to cases/openfoam_port_flow in repository
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    template_dir = repo_root / "cases" / "openfoam_port_flow"

  if not template_dir.exists():
    raise FileNotFoundError(f"Template directory does not exist: {template_dir}")

  # Clean destination
  if case_dir.exists():
    shutil.rmtree(case_dir)
  shutil.copytree(template_dir, case_dir)

  # Update inlet total pressure p0 = delta_p / rho
  p0_kinematic = delta_p_pa / air_density
  p_file = case_dir / "0" / "p"
  p_content = p_file.read_text()
  p_content = re.sub(r"p0\s+uniform\s+[\d\.]+;", f"p0              uniform {p0_kinematic:.2f};", p_content)
  p_content = re.sub(r"value\s+uniform\s+[\d\.]+;", f"value           uniform {p0_kinematic:.2f};", p_content)
  p_file.write_text(p_content)

  # Update max iterations and writeInterval in controlDict
  ctrl_file = case_dir / "system" / "controlDict"
  ctrl_content = ctrl_file.read_text()
  ctrl_content = re.sub(r"endTime\s+\d+;", f"endTime         {max_iterations};", ctrl_content)
  ctrl_content = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {max_iterations};", ctrl_content)
  ctrl_file.write_text(ctrl_content)


def solve_port_flow_at_lift(
    lift_mm: float,
    workdir: Union[str, Path],
    config: Optional[PortFlowConfig] = None,
    head_params: Optional[CylinderHeadParams] = None,
    runner: Optional[PodmanRunner] = None,
) -> LiftPointResult:
  """Execute an end-to-end CAD -> Mesh -> simpleFoam -> Post-process flow bench simulation.

  Args:
    lift_mm: Valve opening lift in mm.
    workdir: Working directory where simulation files and mesh are created.
    config: Port flow benchmark configuration.
    head_params: Geometric parameters for the cylinder head and valve.
    runner: Podman container execution runner.

  Returns:
    LiftPointResult containing aerodynamic flow metrics.
  """
  cfg = config or PortFlowConfig()
  hp = head_params or CylinderHeadParams()
  sol_runner = runner or PodmanRunner()
  work_path = Path(workdir)

  if not sol_runner.is_available:
    raise RuntimeError("Podman / Docker container runtime is not available.")

  t_start = time.perf_counter()
  work_path.mkdir(parents=True, exist_ok=True)

  # 1. Generate CAD fluid domain
  fluid_domain = generate_port_geometry_at_lift(
      lift_mm=lift_mm,
      params=hp,
      cylinder_depth_m=cfg.cylinder_depth_m,
  )

  # 2. Setup OpenFOAM case from template
  setup_port_flow_case(
      case_dir=work_path,
      delta_p_pa=cfg.delta_p_pa,
      air_density=cfg.air_density,
      max_iterations=cfg.max_iterations,
  )

  # 3. Generate Gmsh mesh with adaptive refinement near valve curtain gap
  actual_mesh_min = min(cfg.mesh_size_min, max(0.6, lift_mm * 0.45))
  actual_mesh_max = max(actual_mesh_min * 2.0, cfg.mesh_size_max)
  msh_file = work_path / "port.msh"
  mesh_stats = mesh_port_fluid_domain(
      geometry=fluid_domain,
      output_msh=msh_file,
      mesh_size_min=actual_mesh_min,
      mesh_size_max=actual_mesh_max,
      cylinder_depth_mm=cfg.cylinder_depth_m * 1000.0,
      inlet_x_mm=(hp.head_width_m * 1000.0) * 0.5,
      inlet_z_mm=34.0,
  )

  # 4. Convert Gmsh mesh to OpenFOAM polyMesh, scale to meters, and fix wall patch type
  cmd_mesh = (
      'gmshToFoam port.msh && '
      'transformPoints -scale "(0.001 0.001 0.001)" && '
      "sed -i '/walls/,/}/ s/type[ \\t]*patch;/type            wall;/' constant/polyMesh/boundary"
  )
  sol_runner.run_command(cmd_mesh, str(work_path), timeout_sec=60)

  # 5. Execute steady-state RANS simpleFoam
  sol_runner.run_command("simpleFoam", str(work_path), timeout_sec=120)

  # Determine latest output time directory
  time_dirs = [
      int(d.name)
      for d in work_path.iterdir()
      if d.is_dir() and d.name.isdigit() and int(d.name) > 0
  ]
  latest_time = max(time_dirs) if time_dirs else cfg.max_iterations

  # 6. Extract cell centers and volumes for aerodynamic momentum integration
  cmd_post = (
      f"postProcess -func writeCellCentres -time {latest_time} && "
      f"postProcess -func writeCellVolumes -time {latest_time}"
  )
  sol_runner.run_command(cmd_post, str(work_path), timeout_sec=60)

  # 7. Post-process mass flow rate and discharge coefficients
  iters, volume_flow_m3_s = parse_outlet_flow_rate(work_path)
  mass_flow_kg_s = cfg.air_density * volume_flow_m3_s

  # Theoretical Bernoulli ideal velocity
  v_theor = np.sqrt(2.0 * cfg.delta_p_pa / cfg.air_density)

  # Reference geometric areas
  valve_d = hp.intake_valve_diam_m
  a_seat = (np.pi / 4.0) * (valve_d**2)
  # 45-degree seat curtain area: pi * D_v * h_v * cos(45 deg)
  a_curtain = np.pi * valve_d * (lift_mm * 1e-3) * np.cos(np.radians(45.0))

  cd_seat = mass_flow_kg_s / (cfg.air_density * a_seat * v_theor)
  cd_curtain = mass_flow_kg_s / (cfg.air_density * a_curtain * v_theor) if a_curtain > 1e-8 else 0.0

  # Compute in-cylinder angular momentum and tumble/swirl ratios
  aero = compute_tumble_and_swirl(
      case_dir=work_path,
      latest_time=latest_time,
      air_density=cfg.air_density,
      rated_rpm=cfg.rated_rpm,
  )

  t_solve = time.perf_counter() - t_start

  return LiftPointResult(
      lift_mm=lift_mm,
      lift_over_diameter=lift_mm / (valve_d * 1000.0),
      mass_flow_kg_s=mass_flow_kg_s,
      volume_flow_m3_s=volume_flow_m3_s,
      cd_seat=cd_seat,
      cd_curtain=cd_curtain,
      tumble_angular_momentum_n_m_s=aero["tumble_angular_momentum_n_m_s"],
      tumble_angular_velocity_rad_s=aero["tumble_angular_velocity_rad_s"],
      tumble_ratio=aero["tumble_ratio"],
      swirl_angular_momentum_n_m_s=aero["swirl_angular_momentum_n_m_s"],
      swirl_angular_velocity_rad_s=aero["swirl_angular_velocity_rad_s"],
      swirl_ratio=aero["swirl_ratio"],
      iterations=iters,
      solve_time_s=t_solve,
      mesh_cells=int(mesh_stats["num_tets"]),
  )


def run_port_flow_sweep(
    lifts_mm: Optional[List[float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[PortFlowConfig] = None,
    head_params: Optional[CylinderHeadParams] = None,
    runner: Optional[PodmanRunner] = None,
) -> PortFlowSweepResult:
  """Run a parametric valve lift sweep on the intake port flow bench.

  Default fast profile executes 3 representative lift points [2.0, 5.0, 8.5] mm
  well within the 60-second execution envelope.
  """
  cfg = config or PortFlowConfig()
  lifts = lifts_mm or [2.0, 5.0, 8.5]
  out_dir = Path(output_dir or "outputs/cfd")
  out_dir.mkdir(parents=True, exist_ok=True)

  sol_runner = runner or PodmanRunner()
  t0 = time.perf_counter()
  results: List[LiftPointResult] = []

  for lift in lifts:
    case_path = out_dir / f"case_lift_{lift:.1f}mm"
    res = solve_port_flow_at_lift(
        lift_mm=lift,
        workdir=case_path,
        config=cfg,
        head_params=head_params,
        runner=sol_runner,
    )
    results.append(res)

  total_time = time.perf_counter() - t0
  sweep_res = PortFlowSweepResult(
      config=cfg,
      points=results,
      total_solve_time_s=total_time,
  )

  # Save results JSON
  sweep_res.to_json(out_dir / "port_flow_results.json")
  return sweep_res
