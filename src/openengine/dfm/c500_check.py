"""Automated NestWorks C500 B-Rep DFM validator and 1:1 full-twin extractor.

Audits live B-Rep solid parts, fixture envelopes, concave cylindrical radii,
MQL cast-iron bans, thread-mill specifications, and 4th-axis fallback rules,
and extracts true DFM twin parameters for downstream multi-physics solvers.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from OCP.BRepAdaptor import BRepAdaptor_Surface
from build123d import GeomType, Part

if TYPE_CHECKING:
  from openengine.cad.assemblies import C500FixtureReport, FixtureCheckItem
from openengine.cad.covers_and_aux import (
    FLYWHEEL_HUB_BORE_RADIUS_MM,
    FLYWHEEL_OUTER_RADIUS_MM,
    FLYWHEEL_RIM_INNER_RADIUS_MM,
    FLYWHEEL_RIM_WIDTH_Y_MM,
    FLYWHEEL_WEB_WIDTH_Y_MM,
)
from openengine.cad.twin_metrics import get_cad_valvetrain_masses_kg
from openengine.dfm.bom import DEFAULT_BOM_CONFIG_PATH, DEFAULT_ENGINE_SPECS_PATH, resolve_bom_from_files
from openengine.dfm.c500_spec import (
    C500Spec,
    DEFAULT_C500_CONFIG_PATH,
    MODE_THREE_AXIS_FOUR_SIDE_FLIP,
    PartMachiningSpec,
)
from openengine.fea.calculix import STEEL_1144_STRESSPROOF

MM_PER_M: float = 1000.0
TONNE_MM3_TO_KG_M3: float = 1.0e12
HALF_FACTOR: float = 0.5
QUARTIC_EXPONENT: int = 4
ROUND_DECIMALS: int = 4

RADIUS_TOL_MM: float = 1.0e-3
MIN_VALID_VOLUME_MM3: float = 1.0
DEFAULT_NO_CYLINDER_RADIUS_MM: float = 99.0
DEFAULT_SWEEP_STEP_DEG: float = 60.0
DEFAULT_HEAD_BOLT_PRELOAD_KN: float = 22.0

STEEL_DENSITY_KG_M3: float = STEEL_1144_STRESSPROOF.density_tonne_mm3 * TONNE_MM3_TO_KG_M3

AXIS_CHECK_COUNT: int = 3
ENVELOPE_TOL_MM: float = 0.5

PART_THREAD_MAP: Dict[str, Tuple[str, ...]] = {
    "poppet_valves": (),
    "one_piece_camshaft": ("M6x1.0",),
    "valve_guides_and_seats": (),
    "three_ring_piston": (),
    "trigger_wheel_and_retainers": ("M6x1.0",),
    "cylinder_head": ("M6x1.0", "M8x1.25", "M10x1.0"),
    "finned_cylinder_barrel": ("M8x1.25",),
    "split_crankcase_and_sump_pan": (
        "M6x1.0",
        "M8x1.25",
        "M12x1.5",
        "M16x1.5",
        "M20x1.5",
        "1/8-27 NPT",
    ),
    "split_connecting_rod": ("M6x1.0",),
    "three_piece_crankshaft_webs": ("M8x1.25",),
}


@dataclass(frozen=True, slots=True)
class BrepPartAuditItem:
  """Live B-Rep and DFM rule verification item for one ledger component."""

  part_id: str
  active_mode: str
  material: str
  uses_fallback: bool
  fallback_reason: str
  measured_dims_mm: Tuple[float, float, float]
  envelope_limit_mm: Tuple[float, float, float]
  stack_height_z_mm: float
  max_z_limit_mm: float
  pocket_depth_mm: float
  max_pocket_depth_mm: float
  min_cylindrical_radius_mm: float
  min_allowed_corner_radius_mm: float
  threads_checked: Tuple[str, ...]
  fixture_overlap_mm3: float
  watertight: bool
  violations: Tuple[str, ...]
  is_valid: bool


@dataclass(frozen=True, slots=True)
class FullTwinPhysicsParams:
  """Exact physics parameters extracted from the 1:1 DFM digital twin."""

  total_part_count: int
  dry_mass_kg: float
  wet_mass_kg: float
  reciprocating_mass_kg: float
  rotating_mass_kg: float
  effective_balance_factor: float
  crankshaft_mass_kg: float
  flywheel_mass_kg: float
  flywheel_outer_radius_m: float
  flywheel_hub_radius_m: float
  flywheel_cad_inertia_kg_m2: float
  dome_volume_cm3: float
  clearance_volume_cm3: float
  swept_volume_cm3: float
  compression_ratio: float
  oil_volume_liters: float
  bdc_dipper_immersion_mm: float
  valve_intake_mass_kg: float
  valve_exhaust_mass_kg: float
  retainer_mass_kg: float
  head_bolt_preload_kn: float


@dataclass(frozen=True, slots=True)
class C500DfmAuditReport:
  """Complete live B-Rep C500 DFM verification and twin extraction report."""

  machine_name: str
  has_fourth_axis: bool
  all_passed: bool
  fallback_count: int
  fallback_switches: Dict[str, str]
  part_audits: Tuple[BrepPartAuditItem, ...]
  bom_interface_valid: bool
  twin_params: Optional[FullTwinPhysicsParams]
  violations: Tuple[str, ...]

  def to_dict(self) -> Dict[str, object]:
    """Serialize report to a JSON-compatible dictionary."""
    return asdict(self)

  def to_json(self, filepath: Union[str, Path]) -> Path:
    """Write formatted JSON audit report to disk."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(self.to_dict(), indent=2))
    return out_path


def measure_min_cylindrical_radius_mm(part: Part) -> float:
  """Return the smallest cylindrical face radius in mm using OCC BRepAdaptor."""
  cyl_faces = part.faces().filter_by(GeomType.CYLINDER)
  radii = [float(BRepAdaptor_Surface(face.wrapped).Cylinder().Radius()) for face in cyl_faces]
  if not radii:
    return DEFAULT_NO_CYLINDER_RADIUS_MM
  return round(min(radii), ROUND_DECIMALS)


def _compute_dfm_flywheel_inertia_kg_m2() -> float:
  """Compute polar mass moment of inertia [kg*m^2] of the stepped 1144 flywheel."""
  r_outer = FLYWHEEL_OUTER_RADIUS_MM / MM_PER_M
  r_rim_in = FLYWHEEL_RIM_INNER_RADIUS_MM / MM_PER_M
  r_hub = FLYWHEEL_HUB_BORE_RADIUS_MM / MM_PER_M
  w_rim = FLYWHEEL_RIM_WIDTH_Y_MM / MM_PER_M
  w_web = FLYWHEEL_WEB_WIDTH_Y_MM / MM_PER_M

  web_term = w_web * ((r_rim_in**QUARTIC_EXPONENT) - (r_hub**QUARTIC_EXPONENT))
  rim_term = w_rim * ((r_outer**QUARTIC_EXPONENT) - (r_rim_in**QUARTIC_EXPONENT))
  inertia = HALF_FACTOR * math.pi * STEEL_DENSITY_KG_M3 * (web_term + rim_term)
  return round(float(inertia), ROUND_DECIMALS + AXIS_CHECK_COUNT)


def extract_full_twin_parameters(has_fourth_axis: bool = True) -> FullTwinPhysicsParams:
  """Extract exact mass, inertia, volume, and splash metrics from 1:1 CAD assemblies."""
  from openengine.cad.assemblies import (
      verify_crankcase_assembly,
      verify_dressed_engine,
      verify_head_assembly,
      verify_rotating_assembly,
  )

  rot = verify_rotating_assembly(0.0)
  head = verify_head_assembly(0.0, has_fourth_axis=has_fourth_axis)
  case = verify_crankcase_assembly()
  dressed = verify_dressed_engine(
      sweep_step_deg=DEFAULT_SWEEP_STEP_DEG,
      has_fourth_axis=has_fourth_axis,
  )

  in_mass, ex_mass, ret_mass = get_cad_valvetrain_masses_kg()

  return FullTwinPhysicsParams(
      total_part_count=dressed.total_part_count,
      dry_mass_kg=dressed.dry_mass_kg,
      wet_mass_kg=dressed.wet_mass_kg,
      reciprocating_mass_kg=rot.reciprocating_mass_kg,
      rotating_mass_kg=rot.rotating_mass_kg,
      effective_balance_factor=rot.effective_balance_factor,
      crankshaft_mass_kg=rot.crankshaft_mass_kg,
      flywheel_mass_kg=rot.flywheel_mass_kg,
      flywheel_outer_radius_m=FLYWHEEL_OUTER_RADIUS_MM / MM_PER_M,
      flywheel_hub_radius_m=FLYWHEEL_HUB_BORE_RADIUS_MM / MM_PER_M,
      flywheel_cad_inertia_kg_m2=_compute_dfm_flywheel_inertia_kg_m2(),
      dome_volume_cm3=head.dome_volume_cm3,
      clearance_volume_cm3=head.clearance_volume_cm3,
      swept_volume_cm3=head.swept_volume_cm3,
      compression_ratio=head.compression_ratio,
      oil_volume_liters=case.oil_volume_liters,
      bdc_dipper_immersion_mm=case.bdc_dipper_immersion_mm,
      valve_intake_mass_kg=in_mass,
      valve_exhaust_mass_kg=ex_mass,
      retainer_mass_kg=ret_mass,
      head_bolt_preload_kn=DEFAULT_HEAD_BOLT_PRELOAD_KN,
  )


def _check_barrel_brep_bounds(
    part_id: str,
    measured: Tuple[float, float, float],
    spec: C500Spec,
    violations: List[str],
) -> None:
  """Check 4-side indexed barrel max flip Z bound against live B-Rep height."""
  max_lateral = max(measured[0], measured[1])
  if max_lateral > spec.envelope.max_four_side_barrel_z_mm + ENVELOPE_TOL_MM:
    violations.append(
        f"{part_id}: live 4-side flip height {max_lateral:.2f} mm exceeds"
        f" {spec.envelope.max_four_side_barrel_z_mm:.1f} mm"
    )


def _audit_single_part(
    spec: C500Spec,
    ledger_item: PartMachiningSpec,
    fixture_item: "FixtureCheckItem",
    has_fourth_axis: bool,
) -> BrepPartAuditItem:
  """Validate one live B-Rep part against ledger, corner radius, threads, and fixture."""
  from openengine.cad.assemblies import get_c500_setup_part

  threads = PART_THREAD_MAP.get(ledger_item.part_id, ())
  env_check = spec.check_part(
      ledger_item,
      threads_used=threads,
      has_fourth_axis_override=has_fourth_axis,
  )
  violations = list(env_check.violations)

  posed = get_c500_setup_part(ledger_item.part_id, env_check.active_mode)
  watertight = bool(posed.is_valid and float(posed.volume) > MIN_VALID_VOLUME_MM3)
  if not watertight:
    violations.append(f"{ledger_item.part_id}: B-Rep solid is invalid or zero volume")

  min_cyl_r = measure_min_cylindrical_radius_mm(posed)
  min_allowed_r = spec.tooling.min_internal_corner_radius_mm
  if min_cyl_r + RADIUS_TOL_MM < min_allowed_r:
    violations.append(
        f"{ledger_item.part_id}: min cylindrical radius {min_cyl_r:.2f} mm below"
        f" {min_allowed_r:.2f} mm"
    )

  if ledger_item.mode == MODE_THREE_AXIS_FOUR_SIDE_FLIP:
    _check_barrel_brep_bounds(ledger_item.part_id, fixture_item.part_dims_mm, spec, violations)

  if not fixture_item.is_valid:
    violations.append(f"{ledger_item.part_id}: fixture check failed ({fixture_item.notes})")

  return BrepPartAuditItem(
      part_id=ledger_item.part_id,
      active_mode=env_check.active_mode,
      material=ledger_item.material,
      uses_fallback=env_check.uses_fallback,
      fallback_reason=env_check.fallback_reason,
      measured_dims_mm=fixture_item.part_dims_mm,
      envelope_limit_mm=fixture_item.envelope_limit_mm,
      stack_height_z_mm=fixture_item.stack_height_z_mm,
      max_z_limit_mm=fixture_item.max_z_limit_mm,
      pocket_depth_mm=ledger_item.pocket_depth_mm,
      max_pocket_depth_mm=spec.tooling.max_single_side_pocket_depth_mm,
      min_cylindrical_radius_mm=min_cyl_r,
      min_allowed_corner_radius_mm=min_allowed_r,
      threads_checked=threads,
      fixture_overlap_mm3=fixture_item.fixture_overlap_mm3,
      watertight=watertight,
      violations=tuple(violations),
      is_valid=len(violations) == 0,
  )


def run_c500_dfm_check(
    has_fourth_axis: bool = True,
    config_path: Path = DEFAULT_C500_CONFIG_PATH,
    include_twin_params: bool = True,
) -> C500DfmAuditReport:
  """Execute automated live B-Rep DFM audit and full-twin parameter extraction."""
  from openengine.cad.assemblies import verify_all_c500_fixtures

  spec = C500Spec.from_yaml(config_path)
  fixture_rep: "C500FixtureReport" = verify_all_c500_fixtures(has_fourth_axis=has_fourth_axis)
  fixture_by_id = {item.part_id: item for item in fixture_rep.items}

  bom_rep = resolve_bom_from_files(
      engine_path=DEFAULT_ENGINE_SPECS_PATH,
      c500_path=config_path,
      bom_path=DEFAULT_BOM_CONFIG_PATH,
      has_fourth_axis_override=has_fourth_axis,
  )

  audits = tuple(
      _audit_single_part(spec, part, fixture_by_id[part.part_id], has_fourth_axis)
      for part in spec.part_ledger
  )
  fallback_switches = {
      item.part_id: item.fallback_reason
      for item in audits
      if item.uses_fallback
  }
  top_violations: List[str] = []
  for item in audits:
    top_violations.extend(item.violations)
  if not bom_rep.validation.is_valid:
    top_violations.extend(bom_rep.validation.violations)

  twin_params = extract_full_twin_parameters(has_fourth_axis=has_fourth_axis) if include_twin_params else None
  return C500DfmAuditReport(
      machine_name=spec.machine_name,
      has_fourth_axis=has_fourth_axis,
      all_passed=(len(top_violations) == 0 and bom_rep.validation.is_valid),
      fallback_count=len(fallback_switches),
      fallback_switches=fallback_switches,
      part_audits=audits,
      bom_interface_valid=bom_rep.validation.is_valid,
      twin_params=twin_params,
      violations=tuple(top_violations),
  )
