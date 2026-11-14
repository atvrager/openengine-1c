"""Coupled thermo-mechanical FEA for the parametric piston (CalculiX ccx).

Analyzes two critical load cases:
- LC-1: Peak firing gas pressure (mechanical stress & yield safety factor)
- LC-2: Coupled thermo-mechanical expansion (thermal growth & hot skirt clearance)
"""

from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
  from openengine.lubrication.thermal import SumpThermalBalanceResult

from openengine.cad.cylinder_block import DEFAULT_BORE_M
from openengine.cad.piston import generate_piston
from openengine.fea.calculix import (
    ALUMINUM_4032_T6,
    DOM_STEEL_A513_T5,
    FeaNodalResults,
    MaterialProperties,
    run_ccx_simulation,
)
from openengine.meshing.gmsh_structural import (
    StructuralMeshResult,
    SurfaceFilter,
    format_nset_cards,
    mesh_structural_part_c3d10,
)


@dataclass
class PistonFeaConfig:
  """Configuration parameters for piston FEA simulations."""

  bore_m: float = DEFAULT_BORE_M
  peak_pressure_mpa: float = 7.80  # Peak gas pressure at firing TDC (7.8 MPa = 78 bar)
  crown_temp_c: float = 250.0  # Crown operating temperature in Celsius
  ring_land_temp_c: float = 190.0  # Ring belt operating temperature in Celsius
  skirt_temp_c: float = 140.0  # Skirt and pin boss operating temperature in Celsius
  liner_temp_c: float = 170.0  # Cylinder liner operating temperature (from CHT)
  ambient_temp_c: float = 20.0  # Cold assembly reference temperature
  liner_thermal_expansion_coeff: float = DOM_STEEL_A513_T5.thermal_expansion_coeff  # DOM A513-T5 steel sleeve (1/K)
  cold_skirt_clearance_mm: float = 0.065  # 65 um nominal cold radial clearance
  mesh_size_min_mm: float = 2.5
  mesh_size_max_mm: float = 5.0
  min_yield_safety_factor: float = 1.60
  min_hot_clearance_mm: float = 0.020
  undercrown_oil_htc_w_m2_k: float = 0.0
  sump_oil_temp_c: float = 100.0

  def with_splash_cooling(
      self,
      thermal: "SumpThermalBalanceResult",
  ) -> "PistonFeaConfig":
    """Return a copy updated with under-crown splash-oil cooled temperatures."""
    return replace(
        self,
        crown_temp_c=thermal.cooled_piston_crown_temp_c,
        ring_land_temp_c=thermal.cooled_piston_ring_land_temp_c,
        skirt_temp_c=thermal.cooled_piston_skirt_temp_c,
        undercrown_oil_htc_w_m2_k=thermal.undercrown_htc_w_m2_k,
        sump_oil_temp_c=thermal.equilibrium_oil_temp_c,
    )


@dataclass
class PistonLoadCaseResult:
  """Metrics for a single piston FEA load case."""

  load_case_id: str
  title: str
  max_von_mises_mpa: float
  p95_von_mises_mpa: float
  max_displacement_mm: float
  max_radial_growth_mm: float
  yield_safety_factor: float
  is_pass: bool
  solve_time_s: float
  details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PistonFeaSummaryResult:
  """Complete structural and thermal evaluation results for the piston."""

  config: PistonFeaConfig
  mesh_stats: Dict[str, Any]
  lc1_mechanical: PistonLoadCaseResult
  lc2_thermo_mechanical: PistonLoadCaseResult
  hot_skirt_clearance_mm: float
  is_clearance_safe: bool
  is_structural_safe: bool
  total_solve_time_s: float

  def to_dict(self) -> Dict[str, Any]:
    return {
        "config": asdict(self.config),
        "mesh_stats": self.mesh_stats,
        "lc1_mechanical": asdict(self.lc1_mechanical),
        "lc2_thermo_mechanical": asdict(self.lc2_thermo_mechanical),
        "hot_skirt_clearance_mm": self.hot_skirt_clearance_mm,
        "is_clearance_safe": self.is_clearance_safe,
        "is_structural_safe": self.is_structural_safe,
        "total_solve_time_s": self.total_solve_time_s,
    }

  def to_json(self, filepath: Union[str, Path]) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
      json.dump(self.to_dict(), f, indent=2)


def generate_piston_mesh(
    cad_step_path: Optional[Union[str, Path, Any]] = None,
    output_inp_path: Union[str, Path] = "outputs/fea/piston/piston_mesh.inp",
    mesh_size_min: float = 2.5,
    mesh_size_max: float = 5.0,
) -> StructuralMeshResult:
  """Mesh the piston STEP geometry and tag boundary node sets."""
  if cad_step_path is None:
    geom: Any = generate_piston()
  elif isinstance(cad_step_path, (str, Path)):
    step_p = Path(cad_step_path)
    if not step_p.exists():
      from build123d import export_step

      step_p.parent.mkdir(parents=True, exist_ok=True)
      piston_part = generate_piston()
      export_step(piston_part, str(step_p))
    geom = step_p
  else:
    geom = cad_step_path

  # Crown face: top surface at z > 20 mm
  crown_filter = SurfaceFilter("NCROWN", lambda com, bbox: com[2] > 20.0)
  # Pin bore: cylindrical contact area at z ~ -3 mm (exclude outer circlip groove)
  pin_filter = SurfaceFilter(
      "NPIN_BORE",
      lambda com, bbox: abs(com[2] - (-3.0)) < 2.0 and (bbox[3] - bbox[0]) < 16.5 and abs(com[1]) < 22.8,
  )
  # Skirt outer contact surface: z < 10 mm, r ~ 34 mm
  skirt_filter = SurfaceFilter("NSKIRT", lambda com, bbox: com[2] < 10.0 and math.sqrt(com[0]**2 + com[1]**2) > 30.0)

  return mesh_structural_part_c3d10(
      geometry=geom,
      output_inp_path=output_inp_path,
      mesh_size_min=mesh_size_min,
      mesh_size_max=mesh_size_max,
      surface_filters=[crown_filter, pin_filter, skirt_filter],
      volume_name="EALL",
  )


def build_piston_lc1_deck(
    mesh_inp_path: Union[str, Path],
    output_deck_path: Union[str, Path],
    config: PistonFeaConfig,
    material: MaterialProperties = ALUMINUM_4032_T6,
) -> None:
  """Build CalculiX input deck for LC-1: Peak Firing Pressure Mechanical Load."""
  with open(mesh_inp_path, "r") as f:
    mesh_content = f.read()

  # Calculate total firing force: F = P_max * Area
  bore_mm = config.bore_m * 1000.0
  area_mm2 = (math.pi / 4.0) * (bore_mm**2)
  total_force_n = config.peak_pressure_mpa * area_mm2  # N

  deck_lines = [
      "*HEADING",
      "Piston LC-1: Firing TDC Peak Gas Pressure Mechanical Stress",
      mesh_content,
      f"*MATERIAL, NAME={material.name}",
      "*ELASTIC",
      f"{material.youngs_modulus_mpa:.1f}, {material.poissons_ratio:.2f}",
      "*DENSITY",
      f"{material.density_tonne_mm3:.3e}",
      f"*SOLID SECTION, ELSET=EALL, MATERIAL={material.name}",
      "*STEP",
      "*STATIC",
      # Constrain wrist pin bore in Y (axial pin axis) and Z (wrist pin vertical reaction)
      "*BOUNDARY",
      "NPIN_BORE, 2, 3, 0.0",
      # Fix one pin node in X to prevent rigid body translation
      "NPIN_BORE, 1, 1, 0.0",
      # Apply pressure to the crown: distributed nodal force
      "*CLOAD",
  ]

  # Distribute total gas force across NCROWN nodes
  # Extract node count for NCROWN from mesh
  ncrown_nodes = []
  in_ncrown = False
  for line in mesh_content.splitlines():
    if line.startswith("*NSET, NSET=NCROWN"):
      in_ncrown = True
      continue
    elif in_ncrown and line.startswith("*"):
      in_ncrown = False
      continue
    elif in_ncrown:
      for token in line.split(","):
        t = token.strip()
        if t.isdigit():
          ncrown_nodes.append(int(t))

  if not ncrown_nodes:
    raise ValueError("No NCROWN nodes found in mesh deck.")

  fz_per_node = -total_force_n / len(ncrown_nodes)
  for nid in ncrown_nodes:
    deck_lines.append(f"{nid}, 3, {fz_per_node:.4f}")

  deck_lines.extend([
      "*NODE FILE",
      "U",
      "*EL FILE",
      "S",
      "*END STEP",
      "",
  ])

  with open(output_deck_path, "w") as f:
    f.write("\n".join(deck_lines))


def build_piston_lc2_deck(
    mesh_inp_path: Union[str, Path],
    output_deck_path: Union[str, Path],
    config: PistonFeaConfig,
    material: MaterialProperties = ALUMINUM_4032_T6,
) -> None:
  """Build CalculiX input deck for LC-2: Coupled Thermo-Mechanical Expansion."""
  with open(mesh_inp_path, "r") as f:
    mesh_content = f.read()

  bore_mm = config.bore_m * 1000.0
  area_mm2 = (math.pi / 4.0) * (bore_mm**2)
  total_force_n = config.peak_pressure_mpa * area_mm2

  # Parse node coordinates from mesh_content to assign temperature gradient based on Z position
  node_coords: Dict[int, Tuple[float, float, float]] = {}
  in_nodes = False
  for line in mesh_content.splitlines():
    if line.startswith("*NODE"):
      in_nodes = True
      continue
    elif in_nodes and line.startswith("*"):
      in_nodes = False
      continue
    elif in_nodes:
      parts = [p.strip() for p in line.split(",") if p.strip()]
      if len(parts) >= 4:
        try:
          nid = int(parts[0])
          x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
          node_coords[nid] = (x, y, z)
        except ValueError:
          pass

  # Interpolate temperature based on Z:
  # z_top = 21.0 mm -> T_crown (250°C)
  # z_ring = 15.0 mm -> T_ring (190°C)
  # z_skirt = -21.0 mm -> T_skirt (140°C)
  t_ref = config.ambient_temp_c
  t_crown = config.crown_temp_c
  t_ring = config.ring_land_temp_c
  t_skirt = config.skirt_temp_c

  node_temperatures: Dict[int, float] = {}
  for nid, (x, y, z) in node_coords.items():
    if z >= 15.0:
      # Upper crown & ring land
      frac = (z - 15.0) / 6.0
      t_val = t_ring + frac * (t_crown - t_ring)
    else:
      # Lower skirt & pin boss
      frac = (z - (-21.0)) / 36.0
      t_val = t_skirt + frac * (t_ring - t_skirt)
    node_temperatures[nid] = t_val

  deck_lines = [
      "*HEADING",
      "Piston LC-2: Coupled Thermo-Mechanical Expansion & Stress",
      mesh_content,
      f"*MATERIAL, NAME={material.name}",
      "*ELASTIC",
      f"{material.youngs_modulus_mpa:.1f}, {material.poissons_ratio:.2f}",
      "*EXPANSION",
      f"{material.thermal_expansion_coeff:.3e}",
      "*DENSITY",
      f"{material.density_tonne_mm3:.3e}",
      f"*SOLID SECTION, ELSET=EALL, MATERIAL={material.name}",
      f"*NSET, NSET=NALL, GENERATE",
      f"1, {len(node_coords)}, 1",
      f"*INITIAL CONDITIONS, TYPE=TEMPERATURE",
      f"NALL, {t_ref:.1f}",
      "*STEP",
      "*STATIC",
      "*BOUNDARY",
      "NPIN_BORE, 2, 3, 0.0",
      "NPIN_BORE, 1, 1, 0.0",
      "*TEMPERATURE",
  ]

  for nid, t_deg in node_temperatures.items():
    deck_lines.append(f"{nid}, {t_deg:.1f}")

  # Add mechanical gas load
  deck_lines.append("*CLOAD")
  ncrown_nodes = []
  in_ncrown = False
  for line in mesh_content.splitlines():
    if line.startswith("*NSET, NSET=NCROWN"):
      in_ncrown = True
      continue
    elif in_ncrown and line.startswith("*"):
      in_ncrown = False
      continue
    elif in_ncrown:
      for token in line.split(","):
        t = token.strip()
        if t.isdigit():
          ncrown_nodes.append(int(t))

  fz_per_node = -total_force_n / max(1, len(ncrown_nodes))
  for nid in ncrown_nodes:
    deck_lines.append(f"{nid}, 3, {fz_per_node:.4f}")

  deck_lines.extend([
      "*NODE FILE",
      "U",
      "*EL FILE",
      "S",
      "*END STEP",
      "",
  ])

  with open(output_deck_path, "w") as f:
    f.write("\n".join(deck_lines))


def run_piston_fea(
    config: Optional[PistonFeaConfig] = None,
    output_dir: Union[str, Path] = "outputs/fea/piston",
    cad_step_path: Optional[Union[str, Path]] = None,
) -> PistonFeaSummaryResult:
  """Execute complete structural & thermal FEA validation for the piston.

  1. Meshes piston CAD into quadratic 10-node tetrahedra (C3D10).
  2. Runs LC-1: Mechanical peak gas pressure.
  3. Runs LC-2: Coupled thermo-mechanical expansion and skirt growth.
  4. Quantifies hot skirt clearance and safety factors against yield.

  Args:
    config: Piston FEA parameters and thermal conditions.
    output_dir: Directory to store mesh, decks, and CalculiX results.
    cad_step_path: Path to piston.step. If None, generates from CAD module.

  Returns:
    PistonFeaSummaryResult containing both load case evaluations.
  """
  cfg = config or PistonFeaConfig()
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  t0_total = time.time()

  # 1. Resolve or generate piston CAD STEP
  if cad_step_path:
    step_file = Path(cad_step_path)
  elif Path("outputs/cad/piston.step").exists():
    step_file = Path("outputs/cad/piston.step")
  else:
    step_file = out_dir / "piston.step"

  if not step_file.exists():
    from build123d import export_step

    step_file.parent.mkdir(parents=True, exist_ok=True)
    piston_part = generate_piston(bore_m=cfg.bore_m)
    export_step(piston_part, str(step_file))

  # 2. Mesh piston geometry
  mesh_inp = out_dir / "piston_mesh.inp"
  mesh_res = generate_piston_mesh(
      cad_step_path=step_file,
      output_inp_path=mesh_inp,
      mesh_size_min=cfg.mesh_size_min_mm,
      mesh_size_max=cfg.mesh_size_max_mm,
  )

  # 3. Execute LC-1: Mechanical Peak Gas Load
  t0_lc1 = time.time()
  deck_lc1 = out_dir / "piston_lc1.inp"
  build_piston_lc1_deck(mesh_inp, deck_lc1, cfg)
  results_lc1 = run_ccx_simulation("piston_lc1", out_dir)
  t_lc1 = time.time() - t0_lc1

  # Forged 4032-T6 yield strength: 315 MPa
  s_yield = ALUMINUM_4032_T6.yield_strength_mpa
  sf_lc1 = s_yield / max(0.1, results_lc1.p95_von_mises_mpa)
  pass_lc1 = sf_lc1 >= cfg.min_yield_safety_factor

  lc1_res = PistonLoadCaseResult(
      load_case_id="LC-1",
      title="Firing TDC Peak Gas Pressure (7.8 MPa)",
      max_von_mises_mpa=results_lc1.max_von_mises_mpa,
      p95_von_mises_mpa=results_lc1.p95_von_mises_mpa,
      max_displacement_mm=results_lc1.max_disp_mm,
      max_radial_growth_mm=results_lc1.max_radial_growth_mm,
      yield_safety_factor=float(sf_lc1),
      is_pass=pass_lc1,
      solve_time_s=t_lc1,
      details={
          "mean_von_mises_mpa": results_lc1.mean_von_mises_mpa,
          "max_principal_stress_mpa": results_lc1.max_principal_stress_mpa,
      },
  )

  # 4. Execute LC-2: Coupled Thermo-Mechanical Expansion
  t0_lc2 = time.time()
  deck_lc2 = out_dir / "piston_lc2.inp"
  build_piston_lc2_deck(mesh_inp, deck_lc2, cfg)
  results_lc2 = run_ccx_simulation("piston_lc2", out_dir)
  t_lc2 = time.time() - t0_lc2

  # Calculate hot skirt clearance considering both piston expansion and cylinder liner thermal growth
  node_coords = {}
  with open(mesh_inp, "r") as f:
    for line in f:
      if line.startswith("*ELEMENT") or line.startswith("*NSET"):
        break
      parts = [p.strip() for p in line.split(",") if p.strip()]
      if len(parts) >= 4 and parts[0].isdigit():
        node_coords[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))

  skirt_radials = []
  for nid, (x, y, z) in node_coords.items():
    if math.sqrt(x**2 + y**2) >= 33.0 and z < 10.0:
      if nid in results_lc2.displacements:
        ux, uy, _ = results_lc2.displacements[nid]
        skirt_radials.append(math.sqrt(ux**2 + uy**2))

  delta_r_skirt = float(np.median(skirt_radials)) if skirt_radials else results_lc2.max_radial_growth_mm
  delta_r_liner = cfg.liner_thermal_expansion_coeff * (cfg.liner_temp_c - cfg.ambient_temp_c) * (cfg.bore_m * 1000.0 / 2.0)
  hot_clearance = cfg.cold_skirt_clearance_mm + delta_r_liner - delta_r_skirt
  is_clearance_safe = hot_clearance >= cfg.min_hot_clearance_mm

  sf_lc2 = ALUMINUM_4032_T6.yield_strength_mpa / max(0.1, results_lc2.p95_von_mises_mpa)
  pass_lc2 = sf_lc2 >= 1.0 and is_clearance_safe

  lc2_res = PistonLoadCaseResult(
      load_case_id="LC-2",
      title="Coupled Thermo-Mechanical Expansion (250°C Crown + 7.8 MPa Gas)",
      max_von_mises_mpa=results_lc2.max_von_mises_mpa,
      p95_von_mises_mpa=results_lc2.p95_von_mises_mpa,
      max_displacement_mm=results_lc2.max_disp_mm,
      max_radial_growth_mm=delta_r_skirt,
      yield_safety_factor=float(sf_lc2),
      is_pass=pass_lc2,
      solve_time_s=t_lc2,
      details={
          "skirt_thermal_growth_um": float(delta_r_skirt * 1000.0),
          "liner_thermal_growth_um": float(delta_r_liner * 1000.0),
          "hot_clearance_um": float(hot_clearance * 1000.0),
      },
  )

  total_time = time.time() - t0_total
  is_structural_safe = pass_lc1 and pass_lc2

  summary = PistonFeaSummaryResult(
      config=cfg,
      mesh_stats={
          "num_nodes": mesh_res.num_nodes,
          "num_elements": mesh_res.num_elements,
          "element_type": mesh_res.element_type,
          "crown_nodes": len(mesh_res.node_sets.get("NCROWN", [])),
          "pin_bore_nodes": len(mesh_res.node_sets.get("NPIN_BORE", [])),
          "meshing_time_s": mesh_res.wall_time_s,
      },
      lc1_mechanical=lc1_res,
      lc2_thermo_mechanical=lc2_res,
      hot_skirt_clearance_mm=float(hot_clearance),
      is_clearance_safe=is_clearance_safe,
      is_structural_safe=is_structural_safe,
      total_solve_time_s=total_time,
  )

  summary.to_json(out_dir / "piston_fea_results.json")
  return summary
