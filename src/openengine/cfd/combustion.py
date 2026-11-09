"""3D Moving-Mesh In-Cylinder Combustion simulation via OpenFOAM XiEngineFoam.

Models the 4-stroke single-cylinder compression stroke, spark ignition, and
turbulent flame propagation using the Weller b-Xi model and layered moving mesh
kinematics. Calibrates 0D/1D Wiebe combustion parameters directly against the
resolved 3D mass fraction burned curve.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gmsh
import numpy as np
from scipy.optimize import curve_fit

from openengine.infra.podman import PodmanRunner
from openengine.thermo.cycle import wiebe_burned_fraction


@dataclass
class CombustionConfig:
  """Configuration parameters for 3D in-cylinder combustion simulation."""

  bore_m: float = 0.068  # 68.0 mm cylinder bore
  stroke_m: float = 0.054  # 54.0 mm piston stroke
  con_rod_length_m: float = 0.108  # 108.0 mm connecting rod
  compression_ratio: float = 9.80  # 9.80:1 nominal geometric compression ratio
  rated_rpm: float = 5500.0  # 5500 RPM rated operating speed
  start_crank_angle_deg: float = -180.0  # BDC start of compression stroke
  end_crank_angle_deg: float = 5.0  # Crank angle to stop simulation (past TDC)
  spark_crank_angle_deg: float = -22.0  # Ignition timing (-22 deg BTDC)
  spark_duration_deg: float = 20.0  # Spark plasma kernel duration in CA degrees
  mesh_size_min_m: float = 0.0035  # 3.5 mm minimum tetrahedral cell size
  mesh_size_max_m: float = 0.0050  # 5.0 mm maximum tetrahedral cell size
  max_co: float = 0.2  # Max Courant number for adaptive time-stepping
  max_delta_t_deg: float = 0.5  # Max allowable crank angle step size
  initial_pressure_pa: float = 100000.0  # 1.0 bar cylinder intake pressure at BDC
  initial_temperature_k: float = 373.0  # 373 K initial gas charge temperature


@dataclass
class CombustionStepPoint:
  """Thermodynamic and kinematic state at a single crank angle step."""

  crank_angle_deg: float
  mean_pressure_pa: float
  mean_temperature_k: float
  mean_u_prime_m_s: float
  combustion_progress: float  # Mass fraction burned x_b (0.0 to 1.0)
  clearance_m: float
  piston_speed_m_s: float


@dataclass
class WiebeFitResult:
  """Calibrated 0D Wiebe function parameters extracted from 3D CFD."""

  a: float  # Wiebe efficiency parameter (~2.0 - 5.0)
  m: float  # Wiebe form / shape factor (~1.5 - 3.0)
  duration_deg: float  # Total combustion duration in CA degrees (10-90% or full)
  theta_soc_deg: float  # Start of combustion / spark timing in CA degrees
  r_squared: float  # Coefficient of determination (fit quality)


@dataclass
class CombustionResult:
  """Aggregated 3D moving-mesh combustion simulation results."""

  config: CombustionConfig
  p_max_pa: float
  theta_p_max_deg: float
  t_max_k: float
  theta_t_max_deg: float
  burn_progress_at_end: float
  points: List[CombustionStepPoint]
  wiebe: Optional[WiebeFitResult]
  solve_time_s: float
  total_cells: int

  def to_dict(self) -> Dict[str, Any]:
    """Serialize combustion results to a dictionary."""
    return {
        "config": asdict(self.config),
        "p_max_pa": self.p_max_pa,
        "theta_p_max_deg": self.theta_p_max_deg,
        "t_max_k": self.t_max_k,
        "theta_t_max_deg": self.theta_t_max_deg,
        "burn_progress_at_end": self.burn_progress_at_end,
        "total_cells": self.total_cells,
        "solve_time_s": self.solve_time_s,
        "wiebe": asdict(self.wiebe) if self.wiebe else None,
        "points": [asdict(pt) for pt in self.points],
    }

  def to_json(self, filepath: Union[str, Path]) -> None:
    """Save combustion results to a JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
      json.dump(self.to_dict(), f, indent=2)


def compute_clearance_height(stroke_m: float, compression_ratio: float) -> float:
  """Compute equivalent clearance height c [m] for a flat cylinder from stroke and r_c.

  For an engine with displacement V_d and clearance volume V_c:
    r_c = (V_d + V_c) / V_c  =>  V_c = V_d / (r_c - 1)
  Since V = A_bore * h, the equivalent clearance height is:
    c = stroke / (compression_ratio - 1)

  Args:
    stroke_m: Piston stroke in meters.
    compression_ratio: Geometric compression ratio r_c (> 1.0).

  Returns:
    Clearance height c in meters.
  """
  if compression_ratio <= 1.0:
    raise ValueError(f"Compression ratio must be > 1.0, got {compression_ratio}")
  return stroke_m / (compression_ratio - 1.0)


def generate_combustion_mesh(
    bore_m: float,
    stroke_m: float,
    clearance_height_m: float,
    mesh_size_min_m: float = 0.0035,
    mesh_size_max_m: float = 0.0050,
    output_msh_path: Union[str, Path] = "engine.msh",
) -> int:
  """Generate a 3D cylinder tetrahedral mesh for OpenFOAM layeredEngineMesh.

  The geometry is a cylinder of radius R = bore/2 and height H = clearance + stroke
  representing the cylinder volume at Bottom Dead Center (BDC, -180 deg CA).
  Surfaces are tagged with OpenFOAM standard engine boundary names:
  - piston: bottom flat surface at Z = 0
  - cylinderHead: top flat surface at Z = H
  - liner: cylindrical side wall
  - internal: 3D volume domain

  Args:
    bore_m: Cylinder bore diameter in meters.
    stroke_m: Piston stroke in meters.
    clearance_height_m: Clearance height at TDC in meters.
    mesh_size_min_m: Minimum mesh element size in meters.
    mesh_size_max_m: Maximum mesh element size in meters.
    output_msh_path: Path to write the Gmsh MSH 2.2 mesh file.

  Returns:
    Total number of 3D tetrahedral elements generated.
  """
  radius_m = bore_m / 2.0
  height_m = clearance_height_m + stroke_m

  gmsh.initialize()
  try:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("engine_cylinder")

    # Cylinder aligned along Z-axis from Z=0 to Z=height_m
    gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, height_m, radius_m)
    gmsh.model.occ.synchronize()

    # Classify surfaces by Z center of mass
    surfaces = gmsh.model.getEntities(dim=2)
    piston_tags: List[int] = []
    head_tags: List[int] = []
    liner_tags: List[int] = []

    for dim, s_tag in surfaces:
      com = gmsh.model.occ.getCenterOfMass(dim, s_tag)
      if abs(com[2] - 0.0) < 1e-4:
        piston_tags.append(s_tag)
      elif abs(com[2] - height_m) < 1e-4:
        head_tags.append(s_tag)
      else:
        liner_tags.append(s_tag)

    if not piston_tags or not head_tags or not liner_tags:
      raise RuntimeError(
          f"Failed to identify all boundary patches: piston={piston_tags}, "
          f"head={head_tags}, liner={liner_tags}"
      )

    gmsh.model.addPhysicalGroup(2, piston_tags, name="piston")
    gmsh.model.addPhysicalGroup(2, head_tags, name="cylinderHead")
    gmsh.model.addPhysicalGroup(2, liner_tags, name="liner")

    volumes = gmsh.model.getEntities(dim=3)
    vol_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, vol_tags, name="internal")

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min_m)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max_m)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(3)

    out_p = Path(output_msh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(out_p))

    elements = gmsh.model.mesh.getElements(dim=3)
    total_cells = len(elements[1][0]) if elements[1] else 0
    return total_cells
  finally:
    gmsh.finalize()


def setup_combustion_case(
    case_dir: Path,
    config: CombustionConfig,
    template_dir: Optional[Path] = None,
) -> None:
  """Prepare OpenFOAM case directory for XiEngineFoam moving-mesh combustion.

  Copies case templates, configures engine kinematics, ignition parameters,
  and initial thermodynamic boundary fields.

  Args:
    case_dir: Target case directory.
    config: Combustion configuration parameters.
    template_dir: Template directory (defaults to cases/openfoam_combustion).
  """
  if template_dir is None:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    template_dir = repo_root / "cases" / "openfoam_combustion"

  if not template_dir.exists():
    raise FileNotFoundError(f"Template directory does not exist: {template_dir}")

  if case_dir.exists():
    shutil.rmtree(case_dir)

  case_dir.mkdir(parents=True, exist_ok=True)

  # Copy constant and system directories
  shutil.copytree(template_dir / "constant", case_dir / "constant")
  shutil.copytree(template_dir / "system", case_dir / "system")

  # Format start crank angle string
  start_ca = config.start_crank_angle_deg
  start_str = str(int(start_ca) if start_ca.is_integer() else start_ca)

  # Set up initial condition directory matching startTime
  init_dir = case_dir / start_str
  shutil.copytree(template_dir / "0", init_dir)

  # Calculate geometry
  clearance_m = compute_clearance_height(config.stroke_m, config.compression_ratio)
  head_z_m = clearance_m + config.stroke_m
  spark_z_m = head_z_m - 0.001  # 1 mm below cylinder head deck inside fluid

  rpm_str = str(int(config.rated_rpm) if config.rated_rpm.is_integer() else config.rated_rpm)

  # Update constant/engineGeometry
  geom_file = case_dir / "constant" / "engineGeometry"
  geom_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      engineGeometry;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

engineMesh      layered;

conRodLength    conRodLength [0 1 0 0 0 0 0] {config.con_rod_length_m};

bore            bore [0 1 0 0 0 0 0] {config.bore_m};

stroke          stroke [0 1 0 0 0 0 0] {config.stroke_m};

clearance       clearance [0 1 0 0 0 0 0] {clearance_m:.6g};

rpm             rpm [0 0 -1 0 0 0 0] {rpm_str};

// ************************************************************************* //
"""
  with open(geom_file, "w") as f:
    f.write(geom_content)

  # Update constant/combustionProperties spark location & timing
  comb_file = case_dir / "constant" / "combustionProperties"
  with open(comb_file, "r") as f:
    comb_txt = f.read()

  comb_txt = re.sub(
      r"location\s+\([^)]+\);",
      f"location        (0 0 {spark_z_m:.6f});",
      comb_txt,
  )
  comb_txt = re.sub(
      r"start\s+[-+]?[0-9]*\.?[0-9]+;",
      f"start           {config.spark_crank_angle_deg};",
      comb_txt,
  )
  comb_txt = re.sub(
      r"duration\s+[-+]?[0-9]*\.?[0-9]+;",
      f"duration        {config.spark_duration_deg};",
      comb_txt,
  )
  with open(comb_file, "w") as f:
    f.write(comb_txt)

  # Update system/controlDict
  end_ca = config.end_crank_angle_deg
  end_str = str(int(end_ca) if end_ca.is_integer() else end_ca)

  ctrl_file = case_dir / "system" / "controlDict"
  with open(ctrl_file, "r") as f:
    ctrl_txt = f.read()

  ctrl_txt = re.sub(r"startTime\s+[-+]?[0-9]*\.?[0-9]+;", f"startTime       {start_str};", ctrl_txt)
  ctrl_txt = re.sub(r"endTime\s+[-+]?[0-9]*\.?[0-9]+;", f"endTime         {end_str};", ctrl_txt)
  ctrl_txt = re.sub(r"maxCo\s+[-+]?[0-9]*\.?[0-9]+;", f"maxCo           {config.max_co};", ctrl_txt)
  ctrl_txt = re.sub(r"maxDeltaT\s+[-+]?[0-9]*\.?[0-9]+;", f"maxDeltaT       {config.max_delta_t_deg};", ctrl_txt)

  with open(ctrl_file, "w") as f:
    f.write(ctrl_txt)

  # Update initial pressure and temperature in field files
  p_file = init_dir / "p"
  if p_file.exists():
    with open(p_file, "r") as f:
      p_txt = f.read()
    p_txt = re.sub(
        r"internalField\s+uniform\s+[-+]?[0-9]*\.?[0-9]+;",
        f"internalField   uniform {config.initial_pressure_pa};",
        p_txt,
    )
    with open(p_file, "w") as f:
      f.write(p_txt)

  for t_name in ["T", "Tu"]:
    t_file = init_dir / t_name
    if t_file.exists():
      with open(t_file, "r") as f:
        t_txt = f.read()
      t_txt = re.sub(
          r"internalField\s+uniform\s+[-+]?[0-9]*\.?[0-9]+;",
          f"internalField   uniform {config.initial_temperature_k};",
          t_txt,
      )
      with open(t_file, "w") as f:
        f.write(t_txt)


def parse_combustion_stdout(
    stdout: str,
) -> Tuple[List[CombustionStepPoint], float, float, float, float]:
  """Parse XiEngineFoam stdout log to extract trajectory of crank angle and properties.

  Args:
    stdout: Full text output of XiEngineFoam.

  Returns:
    Tuple of (points, p_max_pa, theta_p_max, t_max_k, theta_t_max).
  """
  re_ca = re.compile(r"Crank angle = ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?) CA-deg")
  re_p = re.compile(r"Mean pressure:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
  re_t = re.compile(r"Mean temperature:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
  re_u = re.compile(r"Mean u':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
  re_prog = re.compile(r"Combustion progress = ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)%")
  re_clr = re.compile(r"clearance: ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
  re_spd = re.compile(r"Piston speed = ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?) m/s")

  points: List[CombustionStepPoint] = []
  lines = stdout.splitlines()

  cur_ca: Optional[float] = None
  cur_clr: float = 0.0
  cur_spd: float = 0.0
  cur_prog: float = 0.0
  cur_p: Optional[float] = None
  cur_t: Optional[float] = None
  cur_u: float = 0.0

  for line in lines:
    m_ca = re_ca.search(line)
    if m_ca:
      if cur_ca is not None and cur_p is not None and cur_t is not None:
        points.append(
            CombustionStepPoint(
                crank_angle_deg=cur_ca,
                mean_pressure_pa=cur_p,
                mean_temperature_k=cur_t,
                mean_u_prime_m_s=cur_u,
                combustion_progress=cur_prog,
                clearance_m=cur_clr,
                piston_speed_m_s=cur_spd,
            )
        )
      cur_ca = float(m_ca.group(1))
      cur_p = None
      cur_t = None

    m_clr = re_clr.search(line)
    if m_clr:
      cur_clr = float(m_clr.group(1))

    m_spd = re_spd.search(line)
    if m_spd:
      cur_spd = float(m_spd.group(1))

    m_prog = re_prog.search(line)
    if m_prog:
      cur_prog = float(m_prog.group(1)) / 100.0

    m_p = re_p.search(line)
    if m_p:
      cur_p = float(m_p.group(1))

    m_t = re_t.search(line)
    if m_t:
      cur_t = float(m_t.group(1))

    m_u = re_u.search(line)
    if m_u:
      cur_u = float(m_u.group(1))

  if cur_ca is not None and cur_p is not None and cur_t is not None:
    points.append(
        CombustionStepPoint(
            crank_angle_deg=cur_ca,
            mean_pressure_pa=cur_p,
            mean_temperature_k=cur_t,
            mean_u_prime_m_s=cur_u,
            combustion_progress=cur_prog,
            clearance_m=cur_clr,
            piston_speed_m_s=cur_spd,
        )
    )

  if not points:
    return [], 0.0, 0.0, 0.0, 0.0

  p_vals = np.array([pt.mean_pressure_pa for pt in points])
  t_vals = np.array([pt.mean_temperature_k for pt in points])
  ca_vals = np.array([pt.crank_angle_deg for pt in points])

  p_max_idx = int(np.argmax(p_vals))
  t_max_idx = int(np.argmax(t_vals))

  return (
      points,
      float(p_vals[p_max_idx]),
      float(ca_vals[p_max_idx]),
      float(t_vals[t_max_idx]),
      float(ca_vals[t_max_idx]),
  )


def calibrate_wiebe_from_trajectory(
    crank_angles_deg: np.ndarray,
    combustion_progress: np.ndarray,
    theta_spark_deg: float = -22.0,
    a_fixed: Optional[float] = None,
) -> Optional[WiebeFitResult]:
  """Fit 0D Wiebe function parameters to 3D CFD combustion progress curve x_b(theta).

  Fits:
    x_b(theta) = 1 - exp( -a * ((theta - theta_soc) / duration)**(m + 1) )
  using non-linear least squares (scipy.optimize.curve_fit).

  Args:
    crank_angles_deg: Array of crank angles in degrees.
    combustion_progress: Array of burned mass fractions x_b (0.0 to 1.0).
    theta_spark_deg: Ignition crank angle in degrees.
    a_fixed: Optional fixed Wiebe efficiency parameter (e.g. 5.0). If None, a is fitted.

  Returns:
    WiebeFitResult if successful, or None if combustion did not progress.
  """
  theta = np.asarray(crank_angles_deg)
  xb = np.asarray(combustion_progress)

  # Filter points after spark where progress is positive
  mask = (theta >= theta_spark_deg) & (xb > 1e-5)
  if np.sum(mask) < 4 or np.max(xb[mask]) < 0.005:
    return None

  theta_fit = theta[mask]
  xb_fit = xb[mask]

  try:
    if a_fixed is not None:
      def wiebe_func_fixed(th, m, duration):
        return wiebe_burned_fraction(
            th,
            theta_soc_deg=theta_spark_deg,
            duration_deg=duration,
            a=a_fixed,
            m=m,
        )

      popt, _ = curve_fit(
          wiebe_func_fixed,
          theta_fit,
          xb_fit,
          p0=[2.0, 50.0],
          bounds=([0.5, 15.0], [6.0, 100.0]),
          maxfev=2000,
      )
      a_opt = a_fixed
      m_opt, dur_opt = popt
      y_pred = wiebe_func_fixed(theta_fit, *popt)
    else:
      def wiebe_func(th, a, m, duration):
        return wiebe_burned_fraction(
            th,
            theta_soc_deg=theta_spark_deg,
            duration_deg=duration,
            a=a,
            m=m,
        )

      popt, _ = curve_fit(
          wiebe_func,
          theta_fit,
          xb_fit,
          p0=[3.0, 2.0, 50.0],
          bounds=([0.5, 0.5, 15.0], [12.0, 6.0, 100.0]),
          maxfev=2000,
      )
      a_opt, m_opt, dur_opt = popt
      y_pred = wiebe_func(theta_fit, *popt)

    ss_res = float(np.sum((xb_fit - y_pred) ** 2))
    ss_tot = float(np.sum((xb_fit - np.mean(xb_fit)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return WiebeFitResult(
        a=float(a_opt),
        m=float(m_opt),
        duration_deg=float(dur_opt),
        theta_soc_deg=float(theta_spark_deg),
        r_squared=float(r2),
    )
  except Exception:
    return None


def run_combustion_simulation(
    config: Optional[CombustionConfig] = None,
    case_dir: Optional[Union[str, Path]] = None,
    runner: Optional[PodmanRunner] = None,
) -> CombustionResult:
  """Execute end-to-end 3D moving-mesh in-cylinder combustion simulation.

  1. Computes cylinder clearance height from engine stroke and r_c.
  2. Generates Gmsh 3D cylinder mesh with piston, liner, cylinderHead patches.
  3. Prepares OpenFOAM case directory with geometry, ignition, and initial fields.
  4. Converts mesh via gmshToFoam and configures wall boundary types.
  5. Runs XiEngineFoam inside container.
  6. Parses thermodynamic trajectory and calibrates Wiebe parameters.

  Args:
    config: Simulation parameters (defaults to CombustionConfig baseline).
    case_dir: Working case directory (defaults to a temporary directory).
    runner: Podman container runner instance.

  Returns:
    CombustionResult containing time-history, peak pressure/temperature, and Wiebe calibration.
  """
  cfg = config or CombustionConfig()
  runner = runner or PodmanRunner()

  if not runner.is_available:
    raise RuntimeError("Podman / Docker container runtime is not available.")

  cleanup_temp = False
  if case_dir is None:
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="openengine_combustion_")
    work_path = Path(temp_dir_obj.name)
    cleanup_temp = True
  else:
    work_path = Path(case_dir)
    temp_dir_obj = None

  t0 = time.time()
  try:
    clearance_m = compute_clearance_height(cfg.stroke_m, cfg.compression_ratio)

    # 1. Case setup
    setup_combustion_case(work_path, cfg)

    # 2. Mesh generation
    msh_path = work_path / "engine.msh"
    total_cells = generate_combustion_mesh(
        bore_m=cfg.bore_m,
        stroke_m=cfg.stroke_m,
        clearance_height_m=clearance_m,
        mesh_size_min_m=cfg.mesh_size_min_m,
        mesh_size_max_m=cfg.mesh_size_max_m,
        output_msh_path=msh_path,
    )

    # 3. Mesh conversion and boundary patch correction
    conv_cmd = (
        "rm -rf constant/polyMesh && "
        "gmshToFoam engine.msh && "
        "sed -i 's/type[ \\t]*patch;/type            wall;/g' constant/polyMesh/boundary"
    )
    runner.run_command(conv_cmd, workspace_dir=str(work_path), timeout_sec=60)

    # 4. Solver execution
    solve_cmd = "XiEngineFoam"
    res = runner.run_command(solve_cmd, workspace_dir=str(work_path), timeout_sec=300)
    stdout = res.stdout

    # 5. Parse output
    points, p_max, th_p_max, t_max, th_t_max = parse_combustion_stdout(stdout)
    burn_end = points[-1].combustion_progress if points else 0.0

    # 6. Wiebe calibration
    ca_arr = np.array([pt.crank_angle_deg for pt in points])
    xb_arr = np.array([pt.combustion_progress for pt in points])
    wiebe_res = calibrate_wiebe_from_trajectory(
        ca_arr,
        xb_arr,
        theta_spark_deg=cfg.spark_crank_angle_deg,
    )

    solve_time_s = time.time() - t0

    return CombustionResult(
        config=cfg,
        p_max_pa=p_max,
        theta_p_max_deg=th_p_max,
        t_max_k=t_max,
        theta_t_max_deg=th_t_max,
        burn_progress_at_end=burn_end,
        points=points,
        wiebe=wiebe_res,
        solve_time_s=solve_time_s,
        total_cells=total_cells,
    )
  finally:
    if cleanup_temp and temp_dir_obj is not None:
      temp_dir_obj.cleanup()
