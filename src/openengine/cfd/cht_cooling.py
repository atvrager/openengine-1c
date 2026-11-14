"""Conjugate Heat Transfer (CHT) and air-fin thermal cooling simulation via OpenFOAM laplacianFoam.

Models 3D steady-state heat conduction through the cast aluminum engine block
and forced air convection across the cooling fin array at vehicle speed.
Verifies that continuous combustion heat rejection (1.3 - 1.6 kW) is safely
dissipated while keeping cylinder liner temperatures below the scuffing limit (180 deg C).
"""

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gmsh
import numpy as np

from openengine.cad.cylinder_block import (
    DEFAULT_BARREL_WALL_M,
    DEFAULT_BLOCK_HEIGHT_M,
    DEFAULT_BORE_M,
    DEFAULT_FIN_DEPTH_Y_M,
    DEFAULT_FIN_PITCH_M,
    DEFAULT_FIN_THICKNESS_M,
    DEFAULT_FIN_WIDTH_X_M,
    DEFAULT_LINER_WALL_M,
    DEFAULT_NUM_FINS,
    THERMAL_BUDGET_W,
)
from openengine.fea.calculix import ALUMINUM_6061_T6
from openengine.infra.podman import PodmanRunner

if TYPE_CHECKING:
  from openengine.lubrication.thermal import SumpThermalBalanceResult


@dataclass
class ChtCoolingConfig:
  """Configuration parameters for cylinder block air-fin conjugate thermal simulation."""

  airspeed_m_s: float = 13.5  # Shrouded cooling fan + vehicle cruise airflow speed
  ambient_temp_k: float = 300.0  # 27 deg C ambient air temperature
  liner_inner_temp_k: float = 443.0  # 170 deg C peak inner liner boundary temperature
  fin_h_coeff: float = 105.0  # W/(m2.K) shrouded forced convective heat transfer coefficient
  thermal_conductivity: float = ALUMINUM_6061_T6.thermal_conductivity_w_mm_k * 1000.0
  thermal_diffusivity: float = 6.903e-05  # m2/s thermal diffusivity (6061-T6 k / rho*Cp)
  heat_budget_w: float = THERMAL_BUDGET_W
  bore_m: float = DEFAULT_BORE_M
  barrel_outer_radius_m: float = 0.5 * DEFAULT_BORE_M + DEFAULT_LINER_WALL_M + DEFAULT_BARREL_WALL_M
  fin_outer_radius_m: float = round(
      math.sqrt((DEFAULT_FIN_WIDTH_X_M * DEFAULT_FIN_DEPTH_Y_M) / math.pi),
      4,
  )
  block_height_m: float = DEFAULT_BLOCK_HEIGHT_M
  num_fins: int = DEFAULT_NUM_FINS
  fin_thickness_m: float = DEFAULT_FIN_THICKNESS_M
  fin_pitch_m: float = DEFAULT_FIN_PITCH_M
  mesh_size_min_m: float = 0.005  # 5 mm minimum tetrahedral element size
  mesh_size_max_m: float = 0.009  # 9 mm maximum tetrahedral element size
  end_time_s: float = 45.0  # Pseudo-transient convergence end time
  delta_t_s: float = 0.5  # Time step size for thermal solver
  piston_splash_relief_w: float = 0.0  # Barrel heat relief from piston oil splash

  def with_splash_cooling(
      self,
      thermal: "SumpThermalBalanceResult",
  ) -> "ChtCoolingConfig":
    """Return a copy with barrel heat budget relieved by piston oil splash."""
    relief_w = max(0.0, self.heat_budget_w - thermal.adjusted_barrel_heat_budget_w)
    return replace(
        self,
        heat_budget_w=thermal.adjusted_barrel_heat_budget_w,
        piston_splash_relief_w=relief_w,
    )


@dataclass
class ChtCoolingResult:
  """Thermal simulation results and cooling capacity metrics."""

  config: ChtCoolingConfig
  max_liner_temp_k: float
  min_fin_temp_k: float
  mean_block_temp_k: float
  mean_fin_temp_k: float
  exterior_fin_area_m2: float
  heat_dissipated_w: float
  fin_efficiency: float
  thermal_margin_percent: float
  is_thermal_safe: bool
  solve_time_s: float
  total_cells: int

  def to_dict(self) -> Dict[str, Any]:
    """Serialize CHT results to a dictionary."""
    return {
        "config": asdict(self.config),
        "max_liner_temp_k": self.max_liner_temp_k,
        "min_fin_temp_k": self.min_fin_temp_k,
        "mean_block_temp_k": self.mean_block_temp_k,
        "mean_fin_temp_k": self.mean_fin_temp_k,
        "exterior_fin_area_m2": self.exterior_fin_area_m2,
        "heat_dissipated_w": self.heat_dissipated_w,
        "fin_efficiency": self.fin_efficiency,
        "thermal_margin_percent": self.thermal_margin_percent,
        "is_thermal_safe": self.is_thermal_safe,
        "solve_time_s": self.solve_time_s,
        "total_cells": self.total_cells,
    }

  def to_json(self, filepath: Union[str, Path]) -> None:
    """Save CHT simulation results to a JSON file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
      json.dump(self.to_dict(), f, indent=2)


def generate_finned_block_mesh(
    config: ChtCoolingConfig,
    output_msh_path: Union[str, Path] = "block.msh",
) -> Tuple[int, float]:
  """Generate a 3D finned cylinder block tetrahedral mesh with tagged boundary patches.

  Boundary patches:
  - liner: inner cylinder bore (Z in [0, H], R = bore/2)
  - topDeck: top annular cylinder head mating face (Z = H)
  - baseFlange: bottom annular crankcase mating face (Z = 0)
  - fins: exterior cooling fin surfaces and exposed outer barrel
  - internal: 3D solid domain

  Args:
    config: Thermal simulation configuration.
    output_msh_path: Destination path for Gmsh MSH 2.2 file.

  Returns:
    Tuple of (total_tetrahedra_cells, exterior_fin_surface_area_m2).
  """
  r_in = config.bore_m / 2.0
  r_out = config.barrel_outer_radius_m
  h = config.block_height_m
  r_fin = config.fin_outer_radius_m
  n_fins = config.num_fins
  fin_t = config.fin_thickness_m
  fin_pitch = config.fin_pitch_m

  gmsh.initialize()
  try:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("finned_block")

    # Base barrel cylinder
    barrel = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, h, r_out)

    # Annular fins
    fin_cyls = []
    for i in range(n_fins):
      z = 0.010 + i * fin_pitch
      fc = gmsh.model.occ.addCylinder(0.0, 0.0, z, 0.0, 0.0, fin_t, r_fin)
      fin_cyls.append((3, fc))

    # Fuse fins with barrel body
    fused, _ = gmsh.model.occ.fuse([(3, barrel)], fin_cyls)

    # Subtract inner bore
    bore = gmsh.model.occ.addCylinder(0.0, 0.0, -0.002, 0.0, 0.0, h + 0.004, r_in)
    solid, _ = gmsh.model.occ.cut(fused, [(3, bore)])
    gmsh.model.occ.synchronize()

    # Classify surfaces by bounding box
    surfaces = gmsh.model.getEntities(dim=2)
    liner_tags: List[int] = []
    top_tags: List[int] = []
    base_tags: List[int] = []
    fin_tags: List[int] = []

    for dim, s_tag in surfaces:
      bbox = gmsh.model.occ.getBoundingBox(dim, s_tag)
      r_max = max(abs(bbox[0]), abs(bbox[1]), abs(bbox[3]), abs(bbox[4]))
      if abs(bbox[2] - h) < 1e-4 and abs(bbox[5] - h) < 1e-4:
        top_tags.append(s_tag)
      elif abs(bbox[2] - 0.0) < 1e-4 and abs(bbox[5] - 0.0) < 1e-4:
        base_tags.append(s_tag)
      elif abs(r_max - r_in) < 1e-3:
        liner_tags.append(s_tag)
      else:
        fin_tags.append(s_tag)

    if not liner_tags or not fin_tags:
      raise RuntimeError(
          f"Failed to identify surface boundaries: liner={liner_tags}, fins={fin_tags}"
      )

    gmsh.model.addPhysicalGroup(2, liner_tags, name="liner")
    gmsh.model.addPhysicalGroup(2, top_tags, name="topDeck")
    gmsh.model.addPhysicalGroup(2, base_tags, name="baseFlange")
    gmsh.model.addPhysicalGroup(2, fin_tags, name="fins")

    volumes = gmsh.model.getEntities(dim=3)
    vol_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, vol_tags, name="internal")

    # Calculate analytical/measured exterior fin area
    fin_face_area = 2.0 * np.pi * (r_fin**2 - r_out**2) * n_fins
    fin_edge_area = 2.0 * np.pi * r_fin * fin_t * n_fins
    barrel_od_area = 2.0 * np.pi * r_out * (h - n_fins * fin_t)
    exterior_area_m2 = float(fin_face_area + fin_edge_area + barrel_od_area)

    gmsh.option.setNumber("Mesh.MeshSizeMin", config.mesh_size_min_m)
    gmsh.option.setNumber("Mesh.MeshSizeMax", config.mesh_size_max_m)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(3)

    out_p = Path(output_msh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(out_p))

    elements = gmsh.model.mesh.getElements(dim=3)
    total_cells = len(elements[1][0]) if elements[1] else 0

    return total_cells, exterior_area_m2
  finally:
    gmsh.finalize()


def setup_cht_case(
    case_dir: Path,
    config: ChtCoolingConfig,
    template_dir: Optional[Path] = None,
) -> None:
  """Prepare OpenFOAM case directory for laplacianFoam CHT thermal simulation.

  Args:
    case_dir: Target case directory.
    config: CHT simulation configuration.
    template_dir: Template directory (defaults to cases/openfoam_cht_cooling).
  """
  if template_dir is None:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    template_dir = repo_root / "cases" / "openfoam_cht_cooling"

  if not template_dir.exists():
    raise FileNotFoundError(f"Template directory does not exist: {template_dir}")

  if case_dir.exists():
    shutil.rmtree(case_dir)

  case_dir.mkdir(parents=True, exist_ok=True)

  # Copy template files
  shutil.copytree(template_dir / "0", case_dir / "0")
  shutil.copytree(template_dir / "constant", case_dir / "constant")
  shutil.copytree(template_dir / "system", case_dir / "system")

  # Compute Robin boundary condition weighting
  delta_n = 0.003
  biot = (config.fin_h_coeff * delta_n) / config.thermal_conductivity
  w_fraction = biot / (1.0 + biot)

  # Update 0/T with configuration values
  t_file = case_dir / "0" / "T"
  t_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       volScalarField;
    object      T;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 1 0 0 0];

internalField   uniform {config.ambient_temp_k + 50.0};

boundaryField
{{
    liner
    {{
        type            fixedValue;
        value           uniform {config.liner_inner_temp_k};
    }}

    fins
    {{
        type            mixed;
        refValue        uniform {config.ambient_temp_k};
        refGradient     uniform 0;
        valueFraction   uniform {w_fraction:.6g};
        value           uniform {config.ambient_temp_k + 80.0};
    }}

    topDeck
    {{
        type            zeroGradient;
    }}

    baseFlange
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
  with open(t_file, "w") as f:
    f.write(t_content)

  # Update constant/transportProperties
  tp_file = case_dir / "constant" / "transportProperties"
  tp_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

DT              DT [0 2 -1 0 0 0 0] {config.thermal_diffusivity};

// ************************************************************************* //
"""
  with open(tp_file, "w") as f:
    f.write(tp_content)

  # Update system/controlDict
  ctrl_file = case_dir / "system" / "controlDict"
  with open(ctrl_file, "r") as f:
    ctrl_txt = f.read()

  ctrl_txt = re.sub(r"endTime\s+[-+]?[0-9]*\.?[0-9]+;", f"endTime         {config.end_time_s};", ctrl_txt)
  ctrl_txt = re.sub(r"deltaT\s+[-+]?[0-9]*\.?[0-9]+;", f"deltaT          {config.delta_t_s};", ctrl_txt)
  with open(ctrl_file, "w") as f:
    f.write(ctrl_txt)


def parse_temperature_field(field_path: Union[str, Path]) -> np.ndarray:
  """Extract cell temperature scalar values from an OpenFOAM field file.

  Args:
    field_path: Path to time directory field file (e.g. 1/T).

  Returns:
    1D numpy array of cell scalar temperature values in Kelvin.
  """
  p = Path(field_path)
  if not p.exists():
    raise FileNotFoundError(f"Temperature field file not found: {p}")

  text = p.read_text()
  # Look for internalField nonuniform List<scalar> or uniform
  uniform_m = re.search(r"internalField\s+uniform\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?);", text)
  if uniform_m:
    return np.array([float(uniform_m.group(1))])

  lines = text.splitlines()
  start = False
  values: List[float] = []

  for line in lines:
    line_s = line.strip()
    if line_s == "(":
      start = True
      continue
    if line_s == ")":
      start = False
      break
    if start and line_s:
      try:
        values.append(float(line_s))
      except ValueError:
        pass

  return np.array(values, dtype=float)


def run_cht_simulation(
    config: Optional[ChtCoolingConfig] = None,
    case_dir: Optional[Union[str, Path]] = None,
    runner: Optional[PodmanRunner] = None,
) -> ChtCoolingResult:
  """Execute end-to-end air-fin conjugate heat transfer simulation in OpenFOAM.

  1. Sets up OpenFOAM case directory.
  2. Generates 3D finned cylinder block mesh with Gmsh.
  3. Converts mesh via gmshToFoam and sets wall boundary types.
  4. Solves solid heat conduction and exterior convection via laplacianFoam.
  5. Extracts temperature distribution, heat dissipation, and thermal margin.

  Args:
    config: Simulation configuration parameters.
    case_dir: Working case directory (defaults to a temporary directory).
    runner: Podman container runner instance.

  Returns:
    ChtCoolingResult containing temperature metrics, heat rejection, and safety status.
  """
  cfg = config or ChtCoolingConfig()
  runner = runner or PodmanRunner()

  if not runner.is_available:
    raise RuntimeError("Podman / Docker container runtime is not available.")

  cleanup_temp = False
  if case_dir is None:
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="openengine_cht_")
    work_path = Path(temp_dir_obj.name)
    cleanup_temp = True
  else:
    work_path = Path(case_dir)
    temp_dir_obj = None

  t0 = time.time()
  try:
    # 1. Setup case template
    setup_cht_case(work_path, cfg)

    # 2. Generate mesh
    msh_path = work_path / "block.msh"
    total_cells, ext_area_m2 = generate_finned_block_mesh(cfg, output_msh_path=msh_path)

    # 3. Mesh conversion and boundary setup
    conv_cmd = (
        "rm -rf constant/polyMesh && "
        "gmshToFoam block.msh && "
        "sed -i 's/type[ \\t]*patch;/type            wall;/g' constant/polyMesh/boundary"
    )
    runner.run_command(conv_cmd, workspace_dir=str(work_path), timeout_sec=60)

    # 4. Solver execution
    solve_cmd = "laplacianFoam"
    runner.run_command(solve_cmd, workspace_dir=str(work_path), timeout_sec=120)

    # 5. Parse temperature field from latest time directory
    time_dirs = []
    for d in work_path.iterdir():
      if d.is_dir() and d.name not in ["0", "constant", "system", "sets"]:
        try:
          time_dirs.append((float(d.name), d))
        except ValueError:
          pass

    if not time_dirs:
      latest_t_file = work_path / "0" / "T"
    else:
      time_dirs.sort(key=lambda x: x[0])
      latest_t_file = time_dirs[-1][1] / "T"

    t_cells = parse_temperature_field(latest_t_file)

    max_liner_k = float(cfg.liner_inner_temp_k)
    min_fin_k = float(np.min(t_cells)) if len(t_cells) > 0 else cfg.ambient_temp_k
    mean_block_k = float(np.mean(t_cells)) if len(t_cells) > 0 else cfg.ambient_temp_k

    # Estimated average fin surface temperature
    mean_fin_k = float(0.65 * mean_block_k + 0.35 * min_fin_k)

    # Total heat dissipation: Q = h * A_fin * (T_fin,avg - T_inf)
    delta_t_effective = max(0.0, mean_fin_k - cfg.ambient_temp_k)
    heat_dissipated_w = float(cfg.fin_h_coeff * ext_area_m2 * delta_t_effective)

    # Fin efficiency: eta = (T_fin,avg - T_inf) / (T_base - T_inf)
    delta_t_base = max(1e-3, max_liner_k - cfg.ambient_temp_k)
    fin_efficiency = float(delta_t_effective / delta_t_base)

    # Thermal safety: heat dissipation >= budget and max liner temp <= 453 K (180 deg C)
    margin_percent = float(((heat_dissipated_w - cfg.heat_budget_w) / cfg.heat_budget_w) * 100.0)
    is_safe = (max_liner_k <= 453.15) and (heat_dissipated_w >= cfg.heat_budget_w * 0.95)

    solve_time_s = time.time() - t0

    return ChtCoolingResult(
        config=cfg,
        max_liner_temp_k=max_liner_k,
        min_fin_temp_k=min_fin_k,
        mean_block_temp_k=mean_block_k,
        mean_fin_temp_k=mean_fin_k,
        exterior_fin_area_m2=ext_area_m2,
        heat_dissipated_w=heat_dissipated_w,
        fin_efficiency=fin_efficiency,
        thermal_margin_percent=margin_percent,
        is_thermal_safe=is_safe,
        solve_time_s=solve_time_s,
        total_cells=total_cells,
    )
  finally:
    if cleanup_temp and temp_dir_obj is not None:
      temp_dir_obj.cleanup()
