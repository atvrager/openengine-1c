"""Full-Built 1:1 Billet + Offline COTS Dressed Digital Twin Assembly."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
from build123d import Axis, BuildPart, Compound, Cylinder, Locations, Part

from openengine.cad.assemblies._common import (
    find_cross_collisions,
    find_pairwise_collisions,
    part_mass_kg,
)
from openengine.cad.assemblies.crankcase_assembly import (
    BASE_CRANKCASE_MATERIALS,
    build_crankcase_assembly_parts,
    verify_crankcase_assembly,
)
from openengine.cad.assemblies.head_assembly import (
    BASE_HEAD_MATERIALS,
    build_head_assembly_parts,
    verify_head_assembly,
)
from openengine.cad.assemblies.rotating_assembly import (
    ROTATING_MATERIALS,
    build_rotating_assembly_parts,
    verify_rotating_assembly,
)
from openengine.cad.assembly import EngineAssembly
from openengine.cad.cots_parts import (
    generate_flat_washer,
    generate_hall_pickup_and_cdi,
    generate_roller_chain_25_loop,
    generate_shcs,
    generate_carburetor_vm26,
    generate_wrench_clearance_zones,
)
from openengine.cad.covers_and_aux import (
    generate_fan_shroud,
    generate_intake_manifold_adapter,
)
from openengine.cad.cylinder_block import (
    BASE_BOLT_XY_MM,
    HEAD_BOLT_XY_MM,
    generate_cylinder_block,
    generate_cylinder_liner,
)
from openengine.cad.cylinder_head import IN_FLANGE_Z_MM, MANIFOLD_BOLT_OFFSET_Y_MM
from openengine.cad.step_export import style_named_parts_as_compound

HEAD_WASHER_Z_MM: float = 44.02
HEAD_BOLT_Z_MM: float = 45.65
HEAD_BOLT_LEN_MM: float = 65.0
BASE_WASHER_Z_MM: float = -74.98
BASE_BOLT_Z_MM: float = -73.33
BASE_BOLT_LEN_MM: float = 32.0
SHCS_M8_DIAM_MM: float = 8.0
WASHER_M8_ID_MM: float = 8.4
WASHER_M8_OD_MM: float = 15.8
WASHER_M8_THICK_MM: float = 1.6

MANIFOLD_BOLT_X_MM: float = 68.25
MANIFOLD_BOLT_DIAM_MM: float = 6.0
MANIFOLD_BOLT_LEN_MM: float = 20.0
ROT_Y_90_DEG: float = 90.0

FUEL_CHARGE_RADIUS_MM: float = 18.0
FUEL_CHARGE_HEIGHT_MM: float = 2.5
FUEL_CHARGE_Z_MM: float = 2.2
FUEL_DENSITY_KG_MM3: float = 1.2e-9
TDC_CROWN_Z_TOL_MM: float = 0.1
FULL_FOUR_STROKE_DEG: float = 720.0
DEFAULT_SWEEP_STEP_DEG: float = 60.0
ROUND_MASS_DECIMALS: int = 4

SKU_HEAD_BOLT: str = "91290A452"
SKU_BASE_BOLT: str = "91290A442"
SKU_MANIFOLD_BOLT: str = "91290A326"

BASE_DRESSED_MATERIALS: Dict[str, str] = {
    **ROTATING_MATERIALS,
    **BASE_HEAD_MATERIALS,
    **BASE_CRANKCASE_MATERIALS,
    "cylinder_block": "ALUMINUM_6061_T6",
    "cylinder_liner": "DOM_STEEL_A513_T5",
    "roller_chain_25": "STEEL_4140_PREHARD",
    "intake_manifold_adapter": "ALUMINUM_6061_T6",
    "carburetor_vm26": "DIECAST_CARB_ALLOY",
    "fan_shroud": "ALUMINUM_6061_T6",
    "hall_pickup": "PHENOLIC_ELECTRONICS",
    "cdi_module": "PHENOLIC_ELECTRONICS",
    "fuel_charge": "FUEL_AIR_CHARGE",
}

STRUCTURAL_HOUSING_KEYS: Tuple[str, ...] = (
    "cylinder_head",
    "cylinder_block",
    "crankcase_left",
    "crankcase_right",
    "sump_pan",
    "cam_tower",
    "cam_cover",
    "right_timing_cover",
    "intake_manifold_adapter",
    "carburetor_vm26",
    "fan_shroud",
)

CORE_ANIMATED_KEYS: frozenset[str] = frozenset({
    "cylinder_head",
    "cylinder_block",
    "cylinder_liner",
    "crankshaft",
    "conrod_beam",
    "conrod_cap",
    "needle_bearing_nks28",
    "conrod_bolt_1",
    "conrod_bolt_2",
    "conrod_dowel_1",
    "conrod_dowel_2",
    "piston",
    "top_ring",
    "second_ring",
    "oil_ring",
    "wrist_pin",
    "circlip_left",
    "circlip_right",
    "valve_intake",
    "valve_exhaust",
    "retainer_intake",
    "retainer_exhaust",
    "camshaft",
    "ground_cam_core_shaft",
    "intake_lobe_2_5d",
    "exhaust_lobe_2_5d",
    "sprocket_cam_36t",
    "sprocket_crank_18t",
    "rocker_intake",
    "rocker_exhaust",
    "flywheel",
    "trigger_wheel",
})

CRANKCASE_LAYER_KEYS: Tuple[str, ...] = (
    "crankcase_left",
    "crankcase_right",
    "sump_pan",
    "cam_tower",
)

VALVETRAIN_LAYER_KEYS: Tuple[str, ...] = (
    "valve_seat_intake",
    "valve_seat_exhaust",
    "valve_guide_intake",
    "valve_guide_exhaust",
    "stem_seal_intake",
    "stem_seal_exhaust",
    "spring_intake",
    "spring_exhaust",
    "retainer_intake",
    "retainer_exhaust",
    "camshaft",
    "ground_cam_core_shaft",
    "intake_lobe_2_5d",
    "exhaust_lobe_2_5d",
    "sprocket_cam_36t",
    "sprocket_crank_18t",
    "roller_chain_25",
    "rocker_shaft_intake",
    "rocker_shaft_exhaust",
    "rocker_intake",
    "rocker_exhaust",
)

COVERS_AUX_LAYER_KEYS: Tuple[str, ...] = (
    "cam_cover",
    "right_timing_cover",
    "flywheel",
    "trigger_wheel",
    "intake_manifold_adapter",
    "carburetor_vm26",
    "fan_shroud",
    "hall_pickup",
    "cdi_module",
)

MOVING_KINEMATIC_KEYS: Tuple[str, ...] = (
    "crankshaft",
    "conrod_beam",
    "conrod_cap",
    "needle_bearing_nks28",
    "conrod_bolt_1",
    "conrod_bolt_2",
    "conrod_dowel_1",
    "conrod_dowel_2",
    "piston",
    "top_ring",
    "second_ring",
    "oil_ring",
    "wrist_pin",
    "circlip_left",
    "circlip_right",
    "valve_intake",
    "valve_exhaust",
    "spring_intake",
    "spring_exhaust",
    "retainer_intake",
    "retainer_exhaust",
    "camshaft",
    "ground_cam_core_shaft",
    "intake_lobe_2_5d",
    "exhaust_lobe_2_5d",
    "sprocket_crank_18t",
    "sprocket_cam_36t",
    "rocker_intake",
    "rocker_exhaust",
    "flywheel",
    "trigger_wheel",
)


@dataclass(frozen=True, slots=True)
class DressedEngineReport:
  """System-level verification report for the complete dressed 1:1 C500 billet engine."""

  total_part_count: int
  dry_mass_kg: float
  wet_mass_kg: float
  oil_volume_liters: float
  compression_ratio: float
  effective_balance_factor: float
  bdc_dipper_immersion_mm: float
  static_solid_collisions: List[str]
  kinematic_sweep_collisions: List[str]
  wrench_zone_collisions: List[str]
  is_valid: bool


def _add_barrel_and_aux_parts(parts: Dict[str, Part], crank_angle_deg: float, include_fluids: bool) -> None:
  """Add barrel, liner, #25 chain, intake/carb, fan shroud, ignition, and tie bolts."""
  parts["cylinder_block"] = generate_cylinder_block()
  parts["cylinder_liner"] = generate_cylinder_liner()
  parts["roller_chain_25"] = generate_roller_chain_25_loop()
  parts["intake_manifold_adapter"] = generate_intake_manifold_adapter()
  parts["carburetor_vm26"] = generate_carburetor_vm26()
  parts["fan_shroud"] = generate_fan_shroud()
  for key, sub_part in generate_hall_pickup_and_cdi().items():
    parts[key] = sub_part

  washer = generate_flat_washer(WASHER_M8_ID_MM, WASHER_M8_OD_MM, WASHER_M8_THICK_MM)
  head_bolt = generate_shcs(SHCS_M8_DIAM_MM, HEAD_BOLT_LEN_MM, SKU_HEAD_BOLT)
  for idx, (hx, hy) in enumerate(HEAD_BOLT_XY_MM, start=1):
    parts[f"head_washer_{idx}"] = washer.translate((hx, hy, HEAD_WASHER_Z_MM))
    parts[f"head_bolt_{idx}"] = head_bolt.translate((hx, hy, HEAD_BOLT_Z_MM))

  base_bolt = generate_shcs(SHCS_M8_DIAM_MM, BASE_BOLT_LEN_MM, SKU_BASE_BOLT)
  for idx, (bx, by) in enumerate(BASE_BOLT_XY_MM, start=1):
    parts[f"base_washer_{idx}"] = washer.translate((bx, by, BASE_WASHER_Z_MM))
    parts[f"base_bolt_{idx}"] = base_bolt.translate((bx, by, BASE_BOLT_Z_MM))

  manifold_bolt = generate_shcs(MANIFOLD_BOLT_DIAM_MM, MANIFOLD_BOLT_LEN_MM, SKU_MANIFOLD_BOLT).rotate(
      Axis.Y, ROT_Y_90_DEG
  )
  for idx, my in enumerate((-MANIFOLD_BOLT_OFFSET_Y_MM, MANIFOLD_BOLT_OFFSET_Y_MM), start=1):
    parts[f"manifold_bolt_{idx}"] = manifold_bolt.translate((MANIFOLD_BOLT_X_MM, my, IN_FLANGE_Z_MM))

  if include_fluids:
    st = EngineAssembly().solve_kinematics(crank_angle_deg)
    if st.piston_crown_z_mm <= TDC_CROWN_Z_TOL_MM:
      with BuildPart() as charge:
        with Locations((0.0, 0.0, FUEL_CHARGE_Z_MM)):
          Cylinder(radius=FUEL_CHARGE_RADIUS_MM, height=FUEL_CHARGE_HEIGHT_MM)
      parts["fuel_charge"] = charge.part


def build_dressed_engine_parts(
    crank_angle_deg: float = 0.0,
    has_fourth_axis: bool = True,
    include_fluids: bool = True,
) -> Dict[str, Part]:
  """Build every B-Rep component of the full 1:1 C500 + COTS engine digital twin."""
  parts: Dict[str, Part] = {}
  parts.update(build_rotating_assembly_parts(crank_angle_deg=crank_angle_deg))
  parts.update(build_head_assembly_parts(crank_angle_deg=crank_angle_deg, has_fourth_axis=has_fourth_axis))
  parts.update(build_crankcase_assembly_parts(include_oil_pool=include_fluids))
  _add_barrel_and_aux_parts(parts, crank_angle_deg=crank_angle_deg, include_fluids=include_fluids)
  return parts


def build_dressed_engine_layer_groups(
    crank_angle_deg: float = 0.0,
    has_fourth_axis: bool = True,
    parts: Dict[str, Part] | None = None,
) -> Dict[str, Compound]:
  """Partition dressed engine parts into styled toggleable dashboard/GLTF layers."""
  all_parts = (
      parts
      if parts is not None
      else build_dressed_engine_parts(
          crank_angle_deg=crank_angle_deg,
          has_fourth_axis=has_fourth_axis,
          include_fluids=True,
      )
  )
  excluded: Set[str] = set(CORE_ANIMATED_KEYS)

  def _collect(layer_name: str, keys: Tuple[str, ...]) -> Compound:
    layer_parts: Dict[str, Part] = {}
    for key in keys:
      if key in all_parts and key not in excluded:
        excluded.add(key)
        layer_parts[key] = all_parts[key]
    return style_named_parts_as_compound(
        layer_name,
        layer_parts,
        material_key_for_dressed_part,
    )

  crankcase_grp = _collect("crankcase_group", CRANKCASE_LAYER_KEYS)
  valvetrain_grp = _collect("valvetrain_group", VALVETRAIN_LAYER_KEYS)
  covers_grp = _collect("covers_and_aux_group", COVERS_AUX_LAYER_KEYS)
  excluded.add("sump_oil_pool")
  excluded.discard("fuel_charge")
  if "fuel_charge" in all_parts:
    excluded.add("fuel_charge")

  cots_parts = {
      name: part
      for name, part in all_parts.items()
      if name not in excluded
  }
  return {
      "crankcase_group": crankcase_grp,
      "valvetrain_group": valvetrain_grp,
      "covers_and_aux_group": covers_grp,
      "cots_hardware_group": style_named_parts_as_compound(
          "cots_hardware_group",
          cots_parts,
          material_key_for_dressed_part,
      ),
      "oil_pool_group": style_named_parts_as_compound(
          "oil_pool_group",
          {"sump_oil_pool": all_parts["sump_oil_pool"]},
          material_key_for_dressed_part,
      ),
  }


def generate_dressed_engine_assembly(
    crank_angle_deg: float = 0.0,
    has_fourth_axis: bool = True,
) -> Compound:
  """Assemble the complete oiled and dressed digital twin as a styled Compound."""
  parts = build_dressed_engine_parts(crank_angle_deg=crank_angle_deg, has_fourth_axis=has_fourth_axis, include_fluids=True)
  return style_named_parts_as_compound(
      "dressed_engine_assembly",
      parts,
      material_key_for_dressed_part,
  )


def material_key_for_dressed_part(name: str) -> str:
  """Resolve material density key for any component in the dressed engine twin."""
  if name in BASE_DRESSED_MATERIALS:
    return BASE_DRESSED_MATERIALS[name]
  if "dowel" in name:
    return "STEEL_4140_PREHARD"
  return "STEEL_GRADE_10_9"


_material_key_for_dressed_part = material_key_for_dressed_part


def _sweep_kinematic_collisions(sweep_step_deg: float, has_fourth_axis: bool) -> List[str]:
  """Check moving assembly parts against static housings and each other over 720 deg."""
  base_rigid = build_dressed_engine_parts(crank_angle_deg=0.0, has_fourth_axis=has_fourth_axis, include_fluids=False)
  moving_set = set(MOVING_KINEMATIC_KEYS)
  static_parts = {k: v for k, v in base_rigid.items() if k not in moving_set}
  hits: List[str] = []

  for angle_deg in np.arange(0.0, FULL_FOUR_STROKE_DEG, sweep_step_deg):
    rot_parts = build_rotating_assembly_parts(crank_angle_deg=float(angle_deg))
    head_parts = build_head_assembly_parts(crank_angle_deg=float(angle_deg), has_fourth_axis=has_fourth_axis)
    moving_now = {
        key: (rot_parts[key] if key in rot_parts else head_parts[key])
        for key in MOVING_KINEMATIC_KEYS
        if (key in rot_parts or key in head_parts)
    }
    _, inner_hits = find_pairwise_collisions(moving_now)
    _, outer_hits = find_cross_collisions(moving_now, static_parts)
    for label in inner_hits + outer_hits:
      hits.append(f"CA={angle_deg:.0f}: {label}")
  return hits


def verify_dressed_engine(
    sweep_step_deg: float = DEFAULT_SWEEP_STEP_DEG,
    has_fourth_axis: bool = True,
) -> DressedEngineReport:
  """Verify full dressed digital twin mass, CR, oil immersion, collisions, and wrench zones."""
  all_parts = build_dressed_engine_parts(crank_angle_deg=0.0, has_fourth_axis=has_fourth_axis, include_fluids=True)
  rigid_parts = {k: v for k, v in all_parts.items() if k not in ("sump_oil_pool", "fuel_charge")}
  _, static_hits = find_pairwise_collisions(rigid_parts)

  structural = {k: rigid_parts[k] for k in STRUCTURAL_HOUSING_KEYS if k in rigid_parts}
  _, wrench_hits = find_cross_collisions(generate_wrench_clearance_zones(), structural)
  kinematic_hits = _sweep_kinematic_collisions(sweep_step_deg=sweep_step_deg, has_fourth_axis=has_fourth_axis)

  dry_mass_kg = sum(part_mass_kg(part, _material_key_for_dressed_part(name)) for name, part in rigid_parts.items())
  oil_mass_kg = part_mass_kg(all_parts["sump_oil_pool"], "SAE_10W40_OIL")
  fuel_charge_kg = float(all_parts["fuel_charge"].volume) * FUEL_DENSITY_KG_MM3 if "fuel_charge" in all_parts else 0.0
  wet_mass_kg = dry_mass_kg + oil_mass_kg + fuel_charge_kg

  rot_rep = verify_rotating_assembly(0.0)
  head_rep = verify_head_assembly(0.0, has_fourth_axis=has_fourth_axis)
  case_rep = verify_crankcase_assembly()

  valid = (
      (len(static_hits) == 0)
      and (len(wrench_hits) == 0)
      and (len(kinematic_hits) == 0)
      and rot_rep.is_valid
      and head_rep.is_valid
      and case_rep.is_valid
  )
  return DressedEngineReport(
      total_part_count=len(all_parts),
      dry_mass_kg=round(dry_mass_kg, ROUND_MASS_DECIMALS),
      wet_mass_kg=round(wet_mass_kg, ROUND_MASS_DECIMALS),
      oil_volume_liters=case_rep.oil_volume_liters,
      compression_ratio=head_rep.compression_ratio,
      effective_balance_factor=rot_rep.effective_balance_factor,
      bdc_dipper_immersion_mm=case_rep.bdc_dipper_immersion_mm,
      static_solid_collisions=static_hits,
      kinematic_sweep_collisions=kinematic_hits,
      wrench_zone_collisions=wrench_hits,
      is_valid=valid,
  )
