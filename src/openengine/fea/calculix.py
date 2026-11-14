"""CalculiX ccx FEA input deck generator, runner, and .frd output parser."""

from collections import Counter
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from openengine.infra.podman import PodmanRunner

DEFAULT_CCX_PARALLEL: int = 8
DEFAULT_CCX_OMP_THREADS: int = 2
DEFAULT_BATCH_TIMEOUT_S: int = 600
ONE_WORKER: int = 1
BOUNDARY_FACE_COUNT: int = 1
MIN_NODE_TOKENS: int = 4
MIN_ELEM_TOKENS: int = 5
NODE_IDX_ONE: int = 1
NODE_IDX_TWO: int = 2
NODE_IDX_THREE: int = 3
NODE_IDX_FOUR: int = 4
X_COORD_IDX: int = 1
Y_COORD_IDX: int = 2
Z_COORD_IDX: int = 3
VON_MISES_SHEAR_COEFF: float = 6.0
VON_MISES_HALF_COEFF: float = 0.5
P95_PERCENTILE: int = 95
FRD_NODE_SLICE: slice = slice(3, 13)
FRD_FIELD_START: int = 13
FRD_FIELD_WIDTH: int = 12
FRD_DISP_COMPONENTS: int = 3
FRD_STRESS_COMPONENTS: int = 6
FRD_TEMP_COMPONENTS: int = 1
FRD_BLOCK_PREFIXES: Tuple[Tuple[str, str], ...] = (
    (" -4  DISP", "DISP"),
    (" -4  STRESS", "STRESS"),
    (" -4  TEMP", "TEMP"),
    (" -4  NDTEMP", "TEMP"),
)
INP_SECTION_HEADERS: Mapping[str, str] = {
    "*NODE": "node",
    "*ELEMENT": "elem",
}


@dataclass
class MaterialProperties:
  """Linear elastic and thermo-mechanical material definition for CalculiX."""

  name: str
  youngs_modulus_mpa: float  # E (MPa = N/mm^2)
  poissons_ratio: float  # nu
  density_tonne_mm3: float = 2.7e-9  # rho (tonne/mm^3, 2.7e-9 = 2700 kg/m^3)
  thermal_expansion_coeff: float = 20e-6  # alpha (1/K)
  thermal_conductivity_w_mm_k: float = 0.15  # k (W/(mm*K), 0.15 = 150 W/(m*K))
  yield_strength_mpa: float = 300.0  # Sy (MPa)
  ultimate_strength_mpa: float = 400.0  # Sut (MPa)
  endurance_limit_mpa: float = 150.0  # Se (MPa)


# Standard calibrated engine material specifications
ALUMINUM_4032_T6 = MaterialProperties(
    name="ALUMINUM_4032_T6",
    youngs_modulus_mpa=79000.0,
    poissons_ratio=0.33,
    density_tonne_mm3=2.68e-9,
    thermal_expansion_coeff=19.4e-6,
    thermal_conductivity_w_mm_k=0.154,
    yield_strength_mpa=315.0,  # 315 MPa at 20°C, ~210 MPa at 250°C
    ultimate_strength_mpa=380.0,
    endurance_limit_mpa=125.0,
)

STEEL_4340 = MaterialProperties(
    name="STEEL_4340",
    youngs_modulus_mpa=210000.0,
    poissons_ratio=0.29,
    density_tonne_mm3=7.85e-9,
    thermal_expansion_coeff=12.3e-6,
    thermal_conductivity_w_mm_k=0.044,
    yield_strength_mpa=850.0,
    ultimate_strength_mpa=1050.0,
    endurance_limit_mpa=480.0,
)

STEEL_GRADE_10_9 = MaterialProperties(
    name="STEEL_GRADE_10_9",
    youngs_modulus_mpa=210000.0,
    poissons_ratio=0.30,
    density_tonne_mm3=7.85e-9,
    thermal_expansion_coeff=12.0e-6,
    thermal_conductivity_w_mm_k=0.045,
    yield_strength_mpa=940.0,
    ultimate_strength_mpa=1040.0,
    endurance_limit_mpa=420.0,
)

ALUMINUM_7075_T6 = MaterialProperties(
    name="ALUMINUM_7075_T6",
    youngs_modulus_mpa=71700.0,
    poissons_ratio=0.33,
    density_tonne_mm3=2.81e-9,
    thermal_expansion_coeff=23.6e-6,
    thermal_conductivity_w_mm_k=0.130,
    yield_strength_mpa=503.0,
    ultimate_strength_mpa=572.0,
    endurance_limit_mpa=159.0,
)

ALUMINUM_6061_T6 = MaterialProperties(
    name="ALUMINUM_6061_T6",
    youngs_modulus_mpa=68900.0,
    poissons_ratio=0.33,
    density_tonne_mm3=2.70e-9,
    thermal_expansion_coeff=23.6e-6,
    thermal_conductivity_w_mm_k=0.167,
    yield_strength_mpa=276.0,
    ultimate_strength_mpa=310.0,
    endurance_limit_mpa=96.5,
)

STEEL_1144_STRESSPROOF = MaterialProperties(
    name="STEEL_1144_STRESSPROOF",
    youngs_modulus_mpa=200000.0,
    poissons_ratio=0.29,
    density_tonne_mm3=7.85e-9,
    thermal_expansion_coeff=11.5e-6,
    thermal_conductivity_w_mm_k=0.050,
    yield_strength_mpa=690.0,
    ultimate_strength_mpa=790.0,
    endurance_limit_mpa=345.0,
)

STEEL_4140_PREHARD = MaterialProperties(
    name="STEEL_4140_PREHARD",
    youngs_modulus_mpa=205000.0,
    poissons_ratio=0.29,
    density_tonne_mm3=7.85e-9,
    thermal_expansion_coeff=12.2e-6,
    thermal_conductivity_w_mm_k=0.0426,
    yield_strength_mpa=655.0,
    ultimate_strength_mpa=1020.0,
    endurance_limit_mpa=420.0,
)

BRONZE_SAE_660 = MaterialProperties(
    name="BRONZE_SAE_660",
    youngs_modulus_mpa=100000.0,
    poissons_ratio=0.34,
    density_tonne_mm3=8.91e-9,
    thermal_expansion_coeff=18.0e-6,
    thermal_conductivity_w_mm_k=0.059,
    yield_strength_mpa=140.0,
    ultimate_strength_mpa=240.0,
    endurance_limit_mpa=110.0,
)

DOM_STEEL_A513_T5 = MaterialProperties(
    name="DOM_STEEL_A513_T5",
    youngs_modulus_mpa=200000.0,
    poissons_ratio=0.29,
    density_tonne_mm3=7.85e-9,
    thermal_expansion_coeff=11.7e-6,
    thermal_conductivity_w_mm_k=0.052,
    yield_strength_mpa=485.0,
    ultimate_strength_mpa=550.0,
    endurance_limit_mpa=275.0,
)

DIECAST_CARB_ALLOY = MaterialProperties(
    name="DIECAST_CARB_ALLOY",
    youngs_modulus_mpa=85000.0,
    poissons_ratio=0.30,
    density_tonne_mm3=6.60e-9,
    thermal_expansion_coeff=27.0e-6,
    thermal_conductivity_w_mm_k=0.113,
    yield_strength_mpa=220.0,
    ultimate_strength_mpa=280.0,
    endurance_limit_mpa=85.0,
)

ALUMINA_STEEL_PLUG = MaterialProperties(
    name="ALUMINA_STEEL_PLUG",
    youngs_modulus_mpa=230000.0,
    poissons_ratio=0.26,
    density_tonne_mm3=5.40e-9,
    thermal_expansion_coeff=8.5e-6,
    thermal_conductivity_w_mm_k=0.028,
    yield_strength_mpa=380.0,
    ultimate_strength_mpa=480.0,
    endurance_limit_mpa=160.0,
)

PHENOLIC_ELECTRONICS = MaterialProperties(
    name="PHENOLIC_ELECTRONICS",
    youngs_modulus_mpa=9500.0,
    poissons_ratio=0.35,
    density_tonne_mm3=1.65e-9,
    thermal_expansion_coeff=32.0e-6,
    thermal_conductivity_w_mm_k=0.00045,
    yield_strength_mpa=75.0,
    ultimate_strength_mpa=95.0,
    endurance_limit_mpa=30.0,
)

VITON_FLUOROELASTOMER = MaterialProperties(
    name="VITON_FLUOROELASTOMER",
    youngs_modulus_mpa=18.0,
    poissons_ratio=0.45,
    density_tonne_mm3=1.85e-9,
    thermal_expansion_coeff=160.0e-6,
    thermal_conductivity_w_mm_k=0.00020,
    yield_strength_mpa=12.0,
    ultimate_strength_mpa=16.0,
    endurance_limit_mpa=6.0,
)

MATERIAL_CATALOG: Dict[str, MaterialProperties] = {
    mat.name: mat
    for mat in (
        ALUMINUM_4032_T6,
        STEEL_4340,
        STEEL_GRADE_10_9,
        ALUMINUM_7075_T6,
        ALUMINUM_6061_T6,
        STEEL_1144_STRESSPROOF,
        STEEL_4140_PREHARD,
        BRONZE_SAE_660,
        DOM_STEEL_A513_T5,
        DIECAST_CARB_ALLOY,
        ALUMINA_STEEL_PLUG,
        PHENOLIC_ELECTRONICS,
        VITON_FLUOROELASTOMER,
    )
}


def get_material_properties(name: str) -> MaterialProperties:
  """Return calibrated MaterialProperties for a standard catalog alloy name."""
  key = name.strip().upper()
  if key not in MATERIAL_CATALOG:
    raise KeyError(f"Unknown FEA material specification: {name}")
  return MATERIAL_CATALOG[key]


@dataclass
class FeaNodalResults:
  """Nodal field variables extracted from CalculiX .frd output."""

  num_nodes: int
  displacements: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
  stresses: Dict[int, Tuple[float, float, float, float, float, float]] = field(default_factory=dict)
  von_mises_stresses: Dict[int, float] = field(default_factory=dict)
  temperatures: Dict[int, float] = field(default_factory=dict)

  # Summary scalar metrics
  max_disp_mm: float = 0.0
  max_radial_growth_mm: float = 0.0
  max_von_mises_mpa: float = 0.0
  p95_von_mises_mpa: float = 0.0
  mean_von_mises_mpa: float = 0.0
  max_principal_stress_mpa: float = 0.0


def compute_von_mises(sxx: float, syy: float, szz: float, sxy: float, syz: float, szx: float) -> float:
  """Calculate von Mises equivalent stress from 6 Cauchy stress components."""
  term1 = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
  term2 = VON_MISES_SHEAR_COEFF * (sxy**2 + syz**2 + szx**2)
  return math.sqrt(VON_MISES_HALF_COEFF * (term1 + term2))


def compute_principal_stresses(sxx: float, syy: float, szz: float, sxy: float, syz: float, szx: float) -> Tuple[float, float, float]:
  """Calculate the three principal stresses (sigma1 >= sigma2 >= sigma3) from Cauchy tensor."""
  # Stress invariants
  i1 = sxx + syy + szz
  i2 = (sxx * syy + syy * szz + szz * sxx) - (sxy**2 + syz**2 + szx**2)
  i3 = sxx * (syy * szz - syz**2) - sxy * (sxy * szz - syz * szx) + szx * (sxy * syz - syy * szx)

  # Characteristic equation roots: lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0
  coeffs = [1.0, -i1, i2, -i3]
  roots = np.roots(coeffs)
  real_roots = sorted([float(r.real) for r in roots], reverse=True)
  return real_roots[0], real_roots[1], real_roots[2]


def _detect_frd_block(line: str, current_block: Optional[str]) -> Tuple[Optional[str], bool]:
  """Update active CalculiX FRD block state from header and terminator lines."""
  for prefix, block_name in FRD_BLOCK_PREFIXES:
    if line.startswith(prefix):
      return block_name, True
  if line.startswith(" -3"):
    return None, True
  return current_block, False


def _parse_frd_floats(line: str, count: int) -> Tuple[float, ...]:
  """Parse fixed-width CalculiX ASCII FRD float columns from a nodal result line."""
  values: List[float] = []
  for col in range(count):
    start = FRD_FIELD_START + col * FRD_FIELD_WIDTH
    end = start + FRD_FIELD_WIDTH
    values.append(float(line[start:end].strip()))
  return tuple(values)


def _record_frd_nodal_line(
    line: str,
    current_block: str,
    displacements: Dict[int, Tuple[float, float, float]],
    stresses: Dict[int, Tuple[float, float, float, float, float, float]],
    von_mises_map: Dict[int, float],
    temperatures: Dict[int, float],
) -> None:
  """Parse one FRD ' -1' nodal record line into the corresponding output maps."""
  try:
    node_id = int(line[FRD_NODE_SLICE].strip())
    if current_block == "DISP":
      ux, uy, uz = _parse_frd_floats(line, FRD_DISP_COMPONENTS)
      displacements[node_id] = (ux, uy, uz)
    elif current_block == "STRESS":
      sxx, syy, szz, sxy, syz, szx = _parse_frd_floats(line, FRD_STRESS_COMPONENTS)
      stresses[node_id] = (sxx, syy, szz, sxy, syz, szx)
      von_mises_map[node_id] = compute_von_mises(sxx, syy, szz, sxy, syz, szx)
    elif current_block == "TEMP":
      (temp_val,) = _parse_frd_floats(line, FRD_TEMP_COMPONENTS)
      temperatures[node_id] = temp_val
  except (ValueError, IndexError):
    return


def _summarize_frd_results(
    displacements: Dict[int, Tuple[float, float, float]],
    stresses: Dict[int, Tuple[float, float, float, float, float, float]],
    von_mises_map: Dict[int, float],
    temperatures: Dict[int, float],
) -> FeaNodalResults:
  """Aggregate displacement, von Mises, and principal stress metrics for parsed FRD maps."""
  max_disp = 0.0
  max_radial = 0.0
  for ux, uy, uz in displacements.values():
    max_disp = max(max_disp, math.sqrt(ux**2 + uy**2 + uz**2))
    max_radial = max(max_radial, math.sqrt(ux**2 + uy**2))

  vm_values = list(von_mises_map.values())
  max_vm = max(vm_values) if vm_values else 0.0
  mean_vm = float(np.mean(vm_values)) if vm_values else 0.0
  p95_vm = float(np.percentile(vm_values, P95_PERCENTILE)) if vm_values else 0.0
  p1_vals = [compute_principal_stresses(*s)[0] for s in stresses.values()]
  max_p1 = max(p1_vals) if p1_vals else 0.0

  return FeaNodalResults(
      num_nodes=len(von_mises_map) or len(displacements),
      displacements=displacements,
      stresses=stresses,
      von_mises_stresses=von_mises_map,
      temperatures=temperatures,
      max_disp_mm=max_disp,
      max_radial_growth_mm=max_radial,
      max_von_mises_mpa=max_vm,
      p95_von_mises_mpa=p95_vm,
      mean_von_mises_mpa=mean_vm,
      max_principal_stress_mpa=max_p1,
  )


def parse_frd_results(frd_filepath: Union[str, Path]) -> FeaNodalResults:
  """Parse CalculiX .frd ASCII results file for displacements, stresses, and temperatures.

  Args:
    frd_filepath: Path to the .frd file.

  Returns:
    FeaNodalResults containing node-wise field maps and maximum summary values.
  """
  frd_path = Path(frd_filepath)
  if not frd_path.exists():
    raise FileNotFoundError(f"CalculiX results file not found: {frd_path}")

  displacements: Dict[int, Tuple[float, float, float]] = {}
  stresses: Dict[int, Tuple[float, float, float, float, float, float]] = {}
  von_mises_map: Dict[int, float] = {}
  temperatures: Dict[int, float] = {}
  current_block: Optional[str] = None

  with open(frd_path, "r") as f:
    for line in f:
      current_block, header_matched = _detect_frd_block(line, current_block)
      if header_matched or current_block is None:
        continue
      if line.startswith(" -1"):
        _record_frd_nodal_line(line, current_block, displacements, stresses, von_mises_map, temperatures)

  return _summarize_frd_results(displacements, stresses, von_mises_map, temperatures)


def run_ccx_simulation(
    job_name: str,
    case_dir: Union[str, Path],
    runner: Optional[PodmanRunner] = None,
) -> FeaNodalResults:
  """Run CalculiX ccx solver inside the container and parse the .frd result.

  Args:
    job_name: Base name of the input deck (<job_name>.inp).
    case_dir: Working directory containing <job_name>.inp.
    runner: Optional PodmanRunner instance.

  Returns:
    FeaNodalResults with displacement and stress fields.
  """
  c_dir = Path(case_dir).resolve()
  inp_file = c_dir / f"{job_name}.inp"
  if not inp_file.exists():
    raise FileNotFoundError(f"CalculiX input file not found: {inp_file}")

  podman = runner or PodmanRunner()
  cmd = f"ccx -i {job_name}"
  res = podman.run_command(cmd, str(c_dir))

  frd_file = c_dir / f"{job_name}.frd"
  if not frd_file.exists():
    raise RuntimeError(
        f"CalculiX ccx failed to produce {frd_file.name}. Return code: {res.returncode}\n"
        f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    )

  return parse_frd_results(frd_file)


def _build_ccx_batch_command(unique_jobs: Sequence[str], max_parallel: int) -> str:
  """Construct one shell pipeline that runs CalculiX jobs in bounded parallel chunks."""
  stride = max(ONE_WORKER, int(max_parallel))
  chunks: List[str] = []
  for start in range(0, len(unique_jobs), stride):
    batch = unique_jobs[start : start + stride]
    launches = [
        f"OMP_NUM_THREADS={DEFAULT_CCX_OMP_THREADS} ccx -i {job} > {job}.ccx.log 2>&1 & p_{idx}=$!"
        for idx, job in enumerate(batch)
    ]
    waits = [f"wait $p_{idx} || exit 1" for idx in range(len(batch))]
    chunks.append("( " + "; ".join([*launches, *waits]) + " )")
  return " && ".join(chunks)


def run_ccx_batch(
    job_names: Sequence[str],
    case_dir: Union[str, Path],
    runner: Optional[PodmanRunner] = None,
    max_parallel: int = DEFAULT_CCX_PARALLEL,
    timeout_sec: int = DEFAULT_BATCH_TIMEOUT_S,
) -> Dict[str, FeaNodalResults]:
  """Run multiple CalculiX ccx jobs inside one Podman invocation and parse .frd results."""
  unique_jobs = list(dict.fromkeys(job_names))
  if not unique_jobs:
    return {}

  c_dir = Path(case_dir).resolve()
  for job in unique_jobs:
    inp_file = c_dir / f"{job}.inp"
    if not inp_file.exists():
      raise FileNotFoundError(f"CalculiX input file not found: {inp_file}")

  podman = runner or PodmanRunner()
  cmd = _build_ccx_batch_command(unique_jobs, max_parallel)
  podman.run_command(cmd, str(c_dir), timeout_sec=timeout_sec)

  results: Dict[str, FeaNodalResults] = {}
  for job in unique_jobs:
    frd_file = c_dir / f"{job}.frd"
    if not frd_file.exists() or frd_file.stat().st_size == 0:
      raise RuntimeError(f"CalculiX ccx batch failed to produce valid output: {frd_file}")
    results[job] = parse_frd_results(frd_file)
  return results


@dataclass(frozen=True, slots=True)
class BoundaryMeshData:
  """Parsed CalculiX node coordinates and outer triangular boundary mesh."""

  node_coords: Dict[int, Tuple[float, float, float]]
  boundary_triangles: Tuple[Tuple[int, int, int], ...]
  unique_nodes: Tuple[int, ...]


def boundary_triangles_from_tets(tets: List[Tuple[int, int, int, int]]) -> Tuple[Tuple[int, int, int], ...]:
  """Extract outward-facing surface triangles appearing only once across tetrahedra."""
  face_counts: Counter[Tuple[int, int, int]] = Counter()
  face_orientations: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
  for n0, n1, n2, n3 in tets:
    for face in ((n0, n1, n2), (n0, n2, n3), (n0, n3, n1), (n1, n3, n2)):
      sorted_vals = sorted(face)
      key = (sorted_vals[0], sorted_vals[1], sorted_vals[2])
      face_counts[key] += 1
      face_orientations[key] = face
  return tuple(face_orientations[k] for k, count in face_counts.items() if count == BOUNDARY_FACE_COUNT)


def _resolve_inp_section(line: str) -> Optional[str]:
  """Return updated parser section when encountering an INP keyword line, else None."""
  if not line.startswith("*"):
    return None
  for prefix, sec in INP_SECTION_HEADERS.items():
    if line.startswith(prefix):
      return sec
  return ""


def _append_inp_record(
    section: str,
    line: str,
    node_coords: Dict[int, Tuple[float, float, float]],
    tets: List[Tuple[int, int, int, int]],
) -> None:
  """Decode a numeric node or element record into the active INP mesh buffers."""
  if not line or not line[0].isdigit():
    return
  tokens = line.split(",")
  if section == "node" and len(tokens) >= MIN_NODE_TOKENS:
    node_coords[int(tokens[0])] = (float(tokens[X_COORD_IDX]), float(tokens[Y_COORD_IDX]), float(tokens[Z_COORD_IDX]))
  elif section == "elem" and len(tokens) >= MIN_ELEM_TOKENS:
    tets.append((int(tokens[NODE_IDX_ONE]), int(tokens[NODE_IDX_TWO]), int(tokens[NODE_IDX_THREE]), int(tokens[NODE_IDX_FOUR])))


def parse_calculix_inp_mesh(inp_path: Union[str, Path]) -> BoundaryMeshData:
  """Parse node coordinates and outer triangular boundary faces from a CalculiX .inp deck."""
  node_coords: Dict[int, Tuple[float, float, float]] = {}
  tets: List[Tuple[int, int, int, int]] = []
  section = ""

  for raw_line in Path(inp_path).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    next_section = _resolve_inp_section(line)
    if next_section is not None:
      section = next_section
      continue
    _append_inp_record(section, line, node_coords, tets)

  boundary_tris = boundary_triangles_from_tets(tets)
  unique_nodes = tuple(sorted({nid for tri in boundary_tris for nid in tri}))
  return BoundaryMeshData(
      node_coords=node_coords,
      boundary_triangles=boundary_tris,
      unique_nodes=unique_nodes,
  )
