"""CalculiX ccx FEA input deck generator, runner, and .frd output parser."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from openengine.infra.podman import PodmanRunner


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
  term2 = 6.0 * (sxy**2 + syz**2 + szx**2)
  return math.sqrt(0.5 * (term1 + term2))


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
  in_block = False

  with open(frd_path, "r") as f:
    for line in f:
      line_clean = line.strip()
      if line.startswith(" -4  DISP"):
        current_block = "DISP"
        in_block = True
        continue
      elif line.startswith(" -4  STRESS"):
        current_block = "STRESS"
        in_block = True
        continue
      elif line.startswith(" -4  TEMP") or line.startswith(" -4  NDTEMP"):
        current_block = "TEMP"
        in_block = True
        continue
      elif line.startswith(" -3"):
        in_block = False
        current_block = None
        continue

      if in_block and line.startswith(" -1"):
        try:
          node_id = int(line[3:13].strip())
          if current_block == "DISP":
            ux = float(line[13:25].strip())
            uy = float(line[25:37].strip())
            uz = float(line[37:49].strip())
            displacements[node_id] = (ux, uy, uz)
          elif current_block == "STRESS":
            sxx = float(line[13:25].strip())
            syy = float(line[25:37].strip())
            szz = float(line[37:49].strip())
            sxy = float(line[49:61].strip())
            syz = float(line[61:73].strip())
            szx = float(line[73:85].strip())
            stresses[node_id] = (sxx, syy, szz, sxy, syz, szx)
            vm = compute_von_mises(sxx, syy, szz, sxy, syz, szx)
            von_mises_map[node_id] = vm
          elif current_block == "TEMP":
            temp_val = float(line[13:25].strip())
            temperatures[node_id] = temp_val
        except (ValueError, IndexError):
          continue

  # Calculate statistics
  max_disp = 0.0
  max_radial = 0.0
  for nid, (ux, uy, uz) in displacements.items():
    disp_mag = math.sqrt(ux**2 + uy**2 + uz**2)
    radial_mag = math.sqrt(ux**2 + uy**2)
    if disp_mag > max_disp:
      max_disp = disp_mag
    if radial_mag > max_radial:
      max_radial = radial_mag

  vm_values = list(von_mises_map.values())
  max_vm = max(vm_values) if vm_values else 0.0
  mean_vm = float(np.mean(vm_values)) if vm_values else 0.0
  p95_vm = float(np.percentile(vm_values, 95)) if vm_values else 0.0

  # Max principal stress
  max_p1 = 0.0
  if stresses:
    p1_vals = [compute_principal_stresses(*s)[0] for s in stresses.values()]
    max_p1 = max(p1_vals)

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
