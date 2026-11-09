"""Structural, buckling, and high-cycle fatigue FEA for the connecting rod (CalculiX ccx).

Analyzes:
- LC-3A: Firing TDC peak compressive load (28.3 kN)
- LC-3B: Overlap TDC redline inertia tensile load (2.54 kN @ 7000 RPM)
- Euler-Johnson column buckling critical load & safety factor
- Modified Goodman fatigue safety factor under cyclic tension-compression
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from openengine.cad.connecting_rod import generate_connecting_rod
from openengine.fea.calculix import (
    STEEL_4340,
    FeaNodalResults,
    MaterialProperties,
    run_ccx_simulation,
)
from openengine.meshing.gmsh_structural import (
    StructuralMeshResult,
    SurfaceFilter,
    mesh_structural_part_c3d10,
)


@dataclass
class ConrodFeaConfig:
  """Configuration parameters for connecting rod FEA and fatigue assessment."""

  center_to_center_m: float = 0.108
  bore_m: float = 0.068
  stroke_m: float = 0.054
  reciprocating_mass_kg: float = 0.140  # Piston + pin + rings
  peak_pressure_mpa: float = 7.80  # 78 bar peak cylinder pressure
  redline_rpm: float = 7000.0  # Max engine speed
  mesh_size_min_mm: float = 3.0
  mesh_size_max_mm: float = 6.0
  min_yield_safety_factor: float = 1.60
  min_buckling_safety_factor: float = 2.50
  min_fatigue_safety_factor: float = 1.50


@dataclass
class ConrodLoadCaseResult:
  """Metrics for a single connecting rod FEA load case."""

  load_case_id: str
  title: str
  applied_force_kn: float
  max_von_mises_mpa: float
  p95_von_mises_mpa: float
  max_displacement_mm: float
  yield_safety_factor: float
  is_pass: bool
  solve_time_s: float
  details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConrodFeaSummaryResult:
  """Complete structural, buckling, and fatigue evaluation results for the connecting rod."""

  config: ConrodFeaConfig
  mesh_stats: Dict[str, Any]
  lc3a_compression: ConrodLoadCaseResult
  lc3b_tension: ConrodLoadCaseResult
  critical_buckling_load_kn: float
  buckling_safety_factor: float
  fatigue_safety_factor: float
  is_buckling_safe: bool
  is_fatigue_safe: bool
  is_structural_safe: bool
  total_solve_time_s: float

  def to_dict(self) -> Dict[str, Any]:
    return {
        "config": asdict(self.config),
        "mesh_stats": self.mesh_stats,
        "lc3a_compression": asdict(self.lc3a_compression),
        "lc3b_tension": asdict(self.lc3b_tension),
        "critical_buckling_load_kn": self.critical_buckling_load_kn,
        "buckling_safety_factor": self.buckling_safety_factor,
        "fatigue_safety_factor": self.fatigue_safety_factor,
        "is_buckling_safe": self.is_buckling_safe,
        "is_fatigue_safe": self.is_fatigue_safe,
        "is_structural_safe": self.is_structural_safe,
        "total_solve_time_s": self.total_solve_time_s,
    }

  def to_json(self, filepath: Union[str, Path]) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
      json.dump(self.to_dict(), f, indent=2)


def generate_conrod_mesh(
    cad_step_path: Union[str, Path],
    output_inp_path: Union[str, Path],
    mesh_size_min: float = 3.0,
    mesh_size_max: float = 6.0,
    l_center_to_center_mm: float = 108.0,
) -> StructuralMeshResult:
  """Mesh the connecting rod STEP geometry and tag bearing surface node sets."""
  # Small pin bore: cylindrical surface at origin (y ~ 0, z ~ 0)
  small_filter = SurfaceFilter("NSMALL_PIN", lambda com, bbox: abs(com[1]) < 5.0 and abs(com[2]) < 5.0)
  # Big crank bore: cylindrical surface at y = l_mm (y ~ 108, z ~ 0)
  big_filter = SurfaceFilter("NBIG_CRANK", lambda com, bbox: abs(com[1] - l_center_to_center_mm) < 5.0 and abs(com[2]) < 5.0)

  return mesh_structural_part_c3d10(
      geometry=cad_step_path,
      output_inp_path=output_inp_path,
      mesh_size_min=mesh_size_min,
      mesh_size_max=mesh_size_max,
      surface_filters=[small_filter, big_filter],
      volume_name="EALL",
  )


def build_conrod_deck(
    mesh_inp_path: Union[str, Path],
    output_deck_path: Union[str, Path],
    job_title: str,
    axial_force_n: float,  # +Y pushes small end toward big end (compression)
    material: MaterialProperties = STEEL_4340,
) -> None:
  """Build CalculiX input deck for connecting rod axial load case."""
  with open(mesh_inp_path, "r") as f:
    mesh_content = f.read()

  deck_lines = [
      "*HEADING",
      job_title,
      mesh_content,
      f"*MATERIAL, NAME={material.name}",
      "*ELASTIC",
      f"{material.youngs_modulus_mpa:.1f}, {material.poissons_ratio:.2f}",
      "*DENSITY",
      f"{material.density_tonne_mm3:.3e}",
      f"*SOLID SECTION, ELSET=EALL, MATERIAL={material.name}",
      "*STEP",
      "*STATIC",
      "*BOUNDARY",
      # Fix big crank journal bore in X, Y, Z
      "NBIG_CRANK, 1, 3, 0.0",
      # Constrain small pin bore in X and Z
      "NSMALL_PIN, 1, 1, 0.0",
      "NSMALL_PIN, 3, 3, 0.0",
      "*CLOAD",
  ]

  # Distribute axial force along Y axis across NSMALL_PIN
  small_nodes = []
  in_small = False
  for line in mesh_content.splitlines():
    if line.startswith("*NSET, NSET=NSMALL_PIN"):
      in_small = True
      continue
    elif in_small and line.startswith("*"):
      in_small = False
      continue
    elif in_small:
      for token in line.split(","):
        t = token.strip()
        if t.isdigit():
          small_nodes.append(int(t))

  if not small_nodes:
    raise ValueError("No NSMALL_PIN nodes found in mesh deck.")

  fy_per_node = axial_force_n / len(small_nodes)
  for nid in small_nodes:
    deck_lines.append(f"{nid}, 2, {fy_per_node:.4f}")

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


def compute_column_buckling_load(
    youngs_modulus_mpa: float = 210000.0,
    flange_w_mm: float = 12.0,
    flange_t_mm: float = 3.5,
    web_t_mm: float = 3.0,
    height_mm: float = 16.0,
    effective_length_mm: float = 85.0,
) -> float:
  """Compute Euler critical buckling load for connecting rod I-beam shank.

  Evaluates the minor out-of-plane axis (Iyy) which governs connecting rod column stability.

  Returns:
    Critical buckling load in Newtons (N).
  """
  # Iyy = 2 * (1/12 * tf * b^3) + 1/12 * (h - 2*tf) * tw^3
  i_yy = 2.0 * ((1.0 / 12.0) * flange_t_mm * (flange_w_mm**3)) + (1.0 / 12.0) * (height_mm - 2.0 * flange_t_mm) * (web_t_mm**3)
  # P_cr = pi^2 * E * Iyy / L_eff^2 (pinned-pinned K=1.0)
  p_cr = (math.pi**2) * youngs_modulus_mpa * i_yy / (effective_length_mm**2)
  return p_cr


def run_conrod_fea(
    config: Optional[ConrodFeaConfig] = None,
    output_dir: Union[str, Path] = "outputs/fea/conrod",
    cad_step_path: Optional[Union[str, Path]] = None,
) -> ConrodFeaSummaryResult:
  """Execute complete structural, buckling, and fatigue FEA for the connecting rod.

  1. Meshes connecting rod CAD into quadratic 10-node tetrahedra (C3D10).
  2. Runs LC-3A: Peak Firing Compressive Load (+28.3 kN).
  3. Runs LC-3B: Redline Inertia Tensile Load (-2.54 kN @ 7000 RPM).
  4. Evaluates Euler column buckling safety factor (P_cr / F_comp >= 2.5).
  5. Evaluates modified Goodman fatigue safety factor under reversed loading.

  Args:
    config: Conrod FEA configuration parameters.
    output_dir: Output directory for mesh and CalculiX results.
    cad_step_path: Path to connecting_rod.step.

  Returns:
    ConrodFeaSummaryResult containing complete stress, buckling, and fatigue analysis.
  """
  cfg = config or ConrodFeaConfig()
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  t0_total = time.time()

  # 1. Resolve or generate CAD STEP
  step_file = Path(cad_step_path) if cad_step_path else out_dir / "connecting_rod.step"
  if not step_file.exists():
    from build123d import export_step

    rod_part = generate_connecting_rod(center_to_center_m=cfg.center_to_center_m)
    export_step(rod_part, str(step_file))

  # 2. Mesh connecting rod
  mesh_inp = out_dir / "conrod_mesh.inp"
  mesh_res = generate_conrod_mesh(
      cad_step_path=step_file,
      output_inp_path=mesh_inp,
      mesh_size_min=cfg.mesh_size_min_mm,
      mesh_size_max=cfg.mesh_size_max_mm,
      l_center_to_center_mm=cfg.center_to_center_m * 1000.0,
  )

  # 3. Calculate loads
  # Compressive gas load at firing TDC (0 deg CA)
  bore_mm = cfg.bore_m * 1000.0
  area_mm2 = (math.pi / 4.0) * (bore_mm**2)
  f_comp_n = cfg.peak_pressure_mpa * area_mm2  # +Y direction (+28.3 kN)

  # Tensile inertia load at overlap TDC (360 deg CA) @ redline
  r_m = cfg.stroke_m / 2.0
  omega_rad_s = 2.0 * math.pi * cfg.redline_rpm / 60.0
  lambda_ratio = r_m / cfg.center_to_center_m
  f_tensile_n = cfg.reciprocating_mass_kg * r_m * (omega_rad_s**2) * (1.0 + lambda_ratio)  # -Y direction (-2.54 kN)

  material = STEEL_4340

  # 4. Run LC-3A: Compression
  t0_3a = time.time()
  deck_3a = out_dir / "conrod_lc3a.inp"
  build_conrod_deck(
      mesh_inp_path=mesh_inp,
      output_deck_path=deck_3a,
      job_title=f"Conrod LC-3A: Firing Compression ({f_comp_n/1000.0:.1f} kN)",
      axial_force_n=f_comp_n,
      material=material,
  )
  results_3a = run_ccx_simulation("conrod_lc3a", out_dir)
  t_3a = time.time() - t0_3a

  sf_3a = material.yield_strength_mpa / max(0.1, results_3a.p95_von_mises_mpa)
  pass_3a = sf_3a >= cfg.min_yield_safety_factor

  lc3a_res = ConrodLoadCaseResult(
      load_case_id="LC-3A",
      title=f"Firing TDC Compression ({f_comp_n/1000.0:.1f} kN)",
      applied_force_kn=f_comp_n / 1000.0,
      max_von_mises_mpa=results_3a.max_von_mises_mpa,
      p95_von_mises_mpa=results_3a.p95_von_mises_mpa,
      max_displacement_mm=results_3a.max_disp_mm,
      yield_safety_factor=float(sf_3a),
      is_pass=pass_3a,
      solve_time_s=t_3a,
      details={
          "mean_von_mises_mpa": results_3a.mean_von_mises_mpa,
      },
  )

  # 5. Run LC-3B: Tension
  t0_3b = time.time()
  deck_3b = out_dir / "conrod_lc3b.inp"
  build_conrod_deck(
      mesh_inp_path=mesh_inp,
      output_deck_path=deck_3b,
      job_title=f"Conrod LC-3B: Redline Tension ({f_tensile_n/1000.0:.1f} kN)",
      axial_force_n=-f_tensile_n,  # -Y direction (tension)
      material=material,
  )
  results_3b = run_ccx_simulation("conrod_lc3b", out_dir)
  t_3b = time.time() - t0_3b

  sf_3b = material.yield_strength_mpa / max(0.1, results_3b.p95_von_mises_mpa)
  pass_3b = sf_3b >= cfg.min_yield_safety_factor

  lc3b_res = ConrodLoadCaseResult(
      load_case_id="LC-3B",
      title=f"Redline Overlap TDC Tension ({f_tensile_n/1000.0:.1f} kN @ {cfg.redline_rpm:.0f} RPM)",
      applied_force_kn=f_tensile_n / 1000.0,
      max_von_mises_mpa=results_3b.max_von_mises_mpa,
      p95_von_mises_mpa=results_3b.p95_von_mises_mpa,
      max_displacement_mm=results_3b.max_disp_mm,
      yield_safety_factor=float(sf_3b),
      is_pass=pass_3b,
      solve_time_s=t_3b,
      details={
          "mean_von_mises_mpa": results_3b.mean_von_mises_mpa,
      },
  )

  # 6. Column Buckling Assessment
  p_cr_n = compute_column_buckling_load(
      youngs_modulus_mpa=material.youngs_modulus_mpa,
      flange_w_mm=12.0,
      flange_t_mm=3.5,
      web_t_mm=3.0,
      height_mm=16.0,
      effective_length_mm=85.0,
  )
  sf_buckle = p_cr_n / max(1.0, f_comp_n)
  is_buckling_safe = sf_buckle >= cfg.min_buckling_safety_factor

  # 7. Modified Goodman Fatigue Assessment
  # Compressive stress: sigma_min (negative in mechanics convention)
  sigma_comp = results_3a.p95_von_mises_mpa
  sigma_tens = results_3b.p95_von_mises_mpa
  # Alternating stress: sigma_a = (sigma_tens - (-sigma_comp)) / 2 = (sigma_tens + sigma_comp) / 2
  sigma_a = (sigma_tens + sigma_comp) / 2.0
  # Mean stress: sigma_m = (sigma_tens - sigma_comp) / 2.0
  sigma_m = (sigma_tens - sigma_comp) / 2.0

  # Goodman relation: 1 / SF_fatigue = (sigma_a / Se) + (max(0, sigma_m) / Sut)
  goodman_lhs = (sigma_a / material.endurance_limit_mpa) + (max(0.0, sigma_m) / material.ultimate_strength_mpa)
  sf_fatigue = 1.0 / max(1e-4, goodman_lhs)
  is_fatigue_safe = sf_fatigue >= cfg.min_fatigue_safety_factor

  total_time = time.time() - t0_total
  is_structural_safe = pass_3a and pass_3b and is_buckling_safe and is_fatigue_safe

  summary = ConrodFeaSummaryResult(
      config=cfg,
      mesh_stats={
          "num_nodes": mesh_res.num_nodes,
          "num_elements": mesh_res.num_elements,
          "element_type": mesh_res.element_type,
          "small_pin_nodes": len(mesh_res.node_sets.get("NSMALL_PIN", [])),
          "big_crank_nodes": len(mesh_res.node_sets.get("NBIG_CRANK", [])),
          "meshing_time_s": mesh_res.wall_time_s,
      },
      lc3a_compression=lc3a_res,
      lc3b_tension=lc3b_res,
      critical_buckling_load_kn=float(p_cr_n / 1000.0),
      buckling_safety_factor=float(sf_buckle),
      fatigue_safety_factor=float(sf_fatigue),
      is_buckling_safe=is_buckling_safe,
      is_fatigue_safe=is_fatigue_safe,
      is_structural_safe=is_structural_safe,
      total_solve_time_s=total_time,
  )

  summary.to_json(out_dir / "conrod_fea_results.json")
  return summary
