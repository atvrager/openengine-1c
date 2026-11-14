"""NestWorks C500 3-Axis Fixture Plate & 4th-Axis Rotary Chuck Setup Assemblies."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from build123d import Axis, BuildPart, Compound, Cylinder, Box, Locations, Part

from openengine.cad.assemblies._common import overlap_volume_mm3
from openengine.cad.connecting_rod import generate_connecting_rod_parts
from openengine.cad.covers_and_aux import generate_trigger_wheel
from openengine.cad.crankcase import generate_crankcase_left, generate_sump_pan
from openengine.cad.crankshaft import generate_crankshaft_parts
from openengine.cad.cylinder_block import generate_cylinder_block
from openengine.cad.cylinder_head import generate_cylinder_head, generate_poppet_valve
from openengine.cad.piston import generate_piston
from openengine.cad.valvetrain_hardware import (
    generate_sohc_camshaft,
    generate_sohc_camshaft_parts,
    generate_valve_guide,
)
from openengine.dfm.c500_spec import C500Spec, MODE_ROTARY_4_AXIS, PartMachiningSpec

FIXTURE_PLATE_X_MM: float = 190.0
FIXTURE_PLATE_Y_MM: float = 210.0
FIXTURE_PLATE_THICK_MM: float = 12.0
ROTARY_CHUCK_OD_MM: float = 100.0
ROTARY_CHUCK_HEIGHT_MM: float = 45.0
SETUP_STANDOFF_Z_MM: float = 0.25
ENVELOPE_TOL_MM: float = 0.5
ROT_90_DEG: float = 90.0
VALVE_INTAKE_DIAM_M: float = 0.030
HALF_FACTOR: float = 0.5
ENV_AXES_COUNT: int = 3
OVERLAP_TOL_MM3: float = 1.0e-3
ROUND_OVERLAP_DECIMALS: int = 4


@dataclass(frozen=True, slots=True)
class FixtureCheckItem:
  """Machining fixture setup validation result for a single C500 ledger part."""

  part_id: str
  active_mode: str
  uses_fallback: bool
  part_dims_mm: Tuple[float, float, float]
  envelope_limit_mm: Tuple[float, float, float]
  stack_height_z_mm: float
  max_z_limit_mm: float
  fixture_overlap_mm3: float
  is_valid: bool
  notes: str


@dataclass(frozen=True, slots=True)
class C500FixtureReport:
  """Complete 10-part NestWorks C500 fixture and 4th-axis fallback summary."""

  has_fourth_axis: bool
  items: List[FixtureCheckItem]
  fallback_count: int
  all_passed: bool


def generate_c500_fixture_plate() -> Part:
  """Generate the NestWorks C500 190 x 210 x 12 mm fixture sub-plate in Z in [-12, 0]."""
  with BuildPart() as plate:
    with Locations((0.0, 0.0, -FIXTURE_PLATE_THICK_MM * HALF_FACTOR)):
      Box(FIXTURE_PLATE_X_MM, FIXTURE_PLATE_Y_MM, FIXTURE_PLATE_THICK_MM)
  return plate.part


def generate_c500_rotary_chuck() -> Part:
  """Generate the 100 mm OD x 45 mm 4th-axis rotary chuck sitting in Z in [-45, 0]."""
  with BuildPart() as chuck:
    with Locations((0.0, 0.0, -ROTARY_CHUCK_HEIGHT_MM * HALF_FACTOR)):
      Cylinder(radius=ROTARY_CHUCK_OD_MM * HALF_FACTOR, height=ROTARY_CHUCK_HEIGHT_MM)
  return chuck.part


def _seat_part_above_datum(part: Part) -> Part:
  """Center a setup-oriented B-Rep part in XY and place its bottom at Z = SETUP_STANDOFF_Z_MM."""
  bb = part.bounding_box()
  cx = (bb.min.X + bb.max.X) * HALF_FACTOR
  cy = (bb.min.Y + bb.max.Y) * HALF_FACTOR
  return part.translate((-cx, -cy, SETUP_STANDOFF_Z_MM - float(bb.min.Z)))


def get_c500_setup_part(part_id: str, active_mode: str) -> Part:
  """Return B-Rep solid oriented in its C500 fixture setup pose for a ledger ID."""
  is_rotary = active_mode == MODE_ROTARY_4_AXIS
  factories: Dict[str, Callable[[], Part]] = {
      "poppet_valves": lambda: generate_poppet_valve(head_diam_m=VALVE_INTAKE_DIAM_M),
      "one_piece_camshaft": lambda: (
          generate_sohc_camshaft(True).rotate(Axis.X, ROT_90_DEG)
          if is_rotary
          else generate_sohc_camshaft_parts(False)["intake_lobe_2_5d"].rotate(Axis.X, ROT_90_DEG)
      ),
      "valve_guides_and_seats": generate_valve_guide,
      "three_ring_piston": generate_piston,
      "trigger_wheel_and_retainers": lambda: generate_trigger_wheel().rotate(Axis.X, ROT_90_DEG),
      "cylinder_head": generate_cylinder_head,
      "finned_cylinder_barrel": generate_cylinder_block,
      "split_crankcase_and_sump_pan": lambda: generate_crankcase_left().rotate(Axis.X, ROT_90_DEG).rotate(Axis.Z, ROT_90_DEG),
      "split_connecting_rod": lambda: generate_connecting_rod_parts()[0].rotate(Axis.Y, ROT_90_DEG).rotate(Axis.Z, ROT_90_DEG),
      "three_piece_crankshaft_webs": lambda: generate_crankshaft_parts().left_web.rotate(Axis.X, ROT_90_DEG).rotate(Axis.Z, ROT_90_DEG),
  }
  if part_id not in factories:
    raise KeyError(f"No C500 fixture setup mapping for ledger part_id={part_id!r}")
  return _seat_part_above_datum(factories[part_id]())


def _measure_dims_mm(part: Part) -> Tuple[float, float, float]:
  """Return rounded (dx, dy, dz) bounding-box dimensions in mm."""
  bb = part.bounding_box()
  return (
      round(float(bb.max.X - bb.min.X), 2),
      round(float(bb.max.Y - bb.min.Y), 2),
      round(float(bb.max.Z - bb.min.Z), 2),
  )


def _evaluate_fixture_item(
    spec: C500Spec,
    ledger_item: PartMachiningSpec,
    has_fourth_axis: bool,
    fixture_plate: Part,
    rotary_chuck: Part,
) -> FixtureCheckItem:
  """Evaluate one C500 ledger part mounted on the plate or rotary chuck."""
  env_check = spec.check_part(ledger_item, has_fourth_axis_override=has_fourth_axis)
  posed = get_c500_setup_part(ledger_item.part_id, env_check.active_mode)
  dims = _measure_dims_mm(posed)

  uses_chuck = env_check.active_mode == MODE_ROTARY_4_AXIS
  fixture = rotary_chuck if uses_chuck else fixture_plate
  fixture_base_h = ROTARY_CHUCK_HEIGHT_MM if uses_chuck else FIXTURE_PLATE_THICK_MM
  stack_h = round(fixture_base_h + dims[2], 2)
  ov = overlap_volume_mm3(posed, fixture)

  lim = ledger_item.envelope_mm
  fits_env = all(dims[idx] <= (lim[idx] + ENVELOPE_TOL_MM) for idx in range(ENV_AXES_COUNT))
  fits_z = stack_h <= spec.envelope.max_z_clearance_mm
  valid = bool(env_check.is_valid and fits_env and fits_z and (ov <= OVERLAP_TOL_MM3))

  notes = env_check.fallback_reason if env_check.uses_fallback else (
      "OK" if valid else f"env={fits_env}, z={fits_z}, dfm={env_check.violations}"
  )
  return FixtureCheckItem(
      part_id=ledger_item.part_id,
      active_mode=env_check.active_mode,
      uses_fallback=env_check.uses_fallback,
      part_dims_mm=dims,
      envelope_limit_mm=lim,
      stack_height_z_mm=stack_h,
      max_z_limit_mm=spec.envelope.max_z_clearance_mm,
      fixture_overlap_mm3=round(ov, ROUND_OVERLAP_DECIMALS),
      is_valid=valid,
      notes=notes,
  )


def generate_c500_fixture_setup_compound(part_id: str, has_fourth_axis: bool = True) -> Compound:
  """Generate the part mounted on the C500 fixture plate or 4th-axis chuck for visual inspection."""
  spec = C500Spec.from_yaml()
  match = next((item for item in spec.part_ledger if item.part_id == part_id), None)
  if match is None:
    raise KeyError(f"Unknown C500 ledger part_id: {part_id}")
  env_check = spec.check_part(match, has_fourth_axis_override=has_fourth_axis)
  fixture = generate_c500_rotary_chuck() if env_check.active_mode == MODE_ROTARY_4_AXIS else generate_c500_fixture_plate()
  posed = get_c500_setup_part(part_id, env_check.active_mode)
  return Compound(children=[fixture, posed])


def verify_all_c500_fixtures(has_fourth_axis: bool = True) -> C500FixtureReport:
  """Verify all 10 C500 ledger parts against CNC work envelopes, stack Z, and fixture clearance."""
  spec = C500Spec.from_yaml()
  plate = generate_c500_fixture_plate()
  chuck = generate_c500_rotary_chuck()
  items = [
      _evaluate_fixture_item(spec, ledger_item, has_fourth_axis, plate, chuck)
      for ledger_item in spec.part_ledger
  ]
  fallbacks = sum(1 for item in items if item.uses_fallback)
  if has_fourth_axis:
    _ = generate_sump_pan()  # Ensure sump pan CAD remains importable alongside left case
  return C500FixtureReport(
      has_fourth_axis=has_fourth_axis,
      items=items,
      fallback_count=fallbacks,
      all_passed=all(item.is_valid for item in items),
  )
