"""Hierarchical SOHC Cylinder Head & Valvetrain CAD Sub-Assembly."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from build123d import Axis, Compound, Part

from openengine.cad.assemblies._common import (
    find_pairwise_collisions,
    part_mass_kg,
)
from openengine.cad.assembly import EngineAssembly, EngineAssemblyState
from openengine.cad.cots_parts import (
    generate_shcs,
    generate_spark_plug_cmr7h,
    generate_sprocket_25,
    generate_valve_spring_9573k35,
    generate_viton_oring,
)
from openengine.cad.covers_and_aux import generate_cam_cover
from openengine.cad.cylinder_head import (
    CAM_TOWER_BOLT_XY_MM,
    DEFAULT_VALVE_HEAD_EX_M,
    DEFAULT_VALVE_HEAD_IN_M,
    PLUG_ORIGIN_Y_MM,
    PLUG_ORIGIN_Z_MM,
    PLUG_TILT_DEG,
    VALVE_EX_X_MM,
    VALVE_IN_X_MM,
    VALVE_SEAT_Z_MM,
    generate_cylinder_head,
    generate_poppet_valve,
)
from openengine.cad.twin_metrics import get_cad_head_volumes_and_cr
from openengine.cad.valvetrain_hardware import (
    CAM_AXIS_Z_MM,
    COVER_BOLT_XY_MM,
    ROCKER_PIVOT_X_MM,
    ROCKER_PIVOT_Z_MM,
    generate_cam_tower,
    generate_rocker_arm,
    generate_rocker_shaft,
    generate_sohc_camshaft,
    generate_sohc_camshaft_parts,
    generate_spring_retainer,
    generate_valve_guide,
    generate_valve_seat,
)
from openengine.cad.step_export import style_named_parts_as_compound

VALVE_TILT_DEG: float = 15.0
VALVE_IN_DIAM_M: float = DEFAULT_VALVE_HEAD_IN_M
VALVE_EX_DIAM_M: float = DEFAULT_VALVE_HEAD_EX_M
SEAT_OFFSET_S_MM: float = 2.6
GUIDE_OFFSET_S_MM: float = 15.0
STEM_SEAL_OFFSET_S_MM: float = 44.0
STEM_SEAL_ID_MM: float = 5.2
STEM_SEAL_CS_MM: float = 1.5
SPRING_OFFSET_S_MM: float = 40.5
SPRING_INSTALLED_H_MM: float = 30.5
RETAINER_OFFSET_S_MM: float = 69.0

CAM_SPROCKET_TEETH: int = 36
CAM_SPROCKET_BORE_MM: float = 16.15
CAM_SPROCKET_Y_MM: float = 53.0
CAM_TO_CRANK_RATIO: float = 0.5
ROCKER_RATIO: float = 1.15
ROCKER_CONTACT_SCALE: float = 1.05

COVER_ORING_ID_MM: float = 29.8
COVER_ORING_CS_MM: float = 1.8
COVER_ORING_Y_MM: float = -2.0
COVER_ORING_Z_MM: float = 127.75
PLUG_S_OFFSET_MM: float = 1.0
PLUG_ROT_X_DEG: float = 66.0

TOWER_BOLT_Z_MM: float = 72.05
TOWER_BOLT_LEN_MM: float = 32.0
COVER_BOLT_Z_MM: float = 121.55
COVER_BOLT_LEN_MM: float = 26.0
SHCS_M6_DIAM_MM: float = 6.0

ROUND_MASS_DECIMALS: int = 4

SKU_STEM_SEAL: str = "9263K119"
SKU_COVER_ORING: str = "9263K305"
SKU_TOWER_BOLT: str = "91290A334"
SKU_COVER_BOLT: str = "91290A326"

BASE_HEAD_MATERIALS: Dict[str, str] = {
    "cylinder_head": "ALUMINUM_6061_T6",
    "valve_intake": "STEEL_4140_PREHARD",
    "valve_exhaust": "STEEL_4140_PREHARD",
    "valve_seat_intake": "STEEL_4140_PREHARD",
    "valve_seat_exhaust": "STEEL_4140_PREHARD",
    "valve_guide_intake": "BRONZE_SAE_660",
    "valve_guide_exhaust": "BRONZE_SAE_660",
    "stem_seal_intake": "VITON_FLUOROELASTOMER",
    "stem_seal_exhaust": "VITON_FLUOROELASTOMER",
    "spring_intake": "STEEL_4140_PREHARD",
    "spring_exhaust": "STEEL_4140_PREHARD",
    "retainer_intake": "ALUMINUM_7075_T6",
    "retainer_exhaust": "ALUMINUM_7075_T6",
    "cam_tower": "ALUMINUM_6061_T6",
    "camshaft": "STEEL_4140_PREHARD",
    "ground_cam_core_shaft": "STEEL_4140_PREHARD",
    "intake_lobe_2_5d": "STEEL_4140_PREHARD",
    "exhaust_lobe_2_5d": "STEEL_4140_PREHARD",
    "sprocket_cam_36t": "STEEL_4140_PREHARD",
    "rocker_shaft_intake": "STEEL_4140_PREHARD",
    "rocker_shaft_exhaust": "STEEL_4140_PREHARD",
    "rocker_intake": "ALUMINUM_7075_T6",
    "rocker_exhaust": "ALUMINUM_7075_T6",
    "cam_cover": "ALUMINUM_6061_T6",
    "cam_cover_oring": "VITON_FLUOROELASTOMER",
    "spark_plug_cmr7h": "ALUMINA_STEEL_PLUG",
}

HEAD_ONLY_KEYS: Tuple[str, ...] = ("cylinder_head",)
VALVETRAIN_KEYS: Tuple[str, ...] = (
    "valve_intake",
    "valve_exhaust",
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
    "rocker_shaft_intake",
    "rocker_shaft_exhaust",
    "rocker_intake",
    "rocker_exhaust",
)


@dataclass(frozen=True, slots=True)
class HeadAssemblyReport:
  """Chamber CR, component mass breakdown, and collision status for head sub-assembly."""

  crank_angle_deg: float
  has_fourth_axis: bool
  dome_volume_cm3: float
  clearance_volume_cm3: float
  swept_volume_cm3: float
  compression_ratio: float
  head_mass_kg: float
  valvetrain_mass_kg: float
  total_mass_kg: float
  max_overlap_mm3: float
  collision_pairs: List[str]
  is_valid: bool


def _place_valve_axis(
    part: Part,
    p_seat: Tuple[float, float, float],
    axis_dir: Tuple[float, float, float],
    rot_y_deg: float,
    s_mm: float,
) -> Part:
  """Orient a valve-axis part and translate it along the inclined stem axis."""
  pos = np.array(p_seat, dtype=float) + (np.array(axis_dir, dtype=float) * s_mm)
  return part.rotate(Axis.Y, rot_y_deg).translate(tuple(pos.tolist()))


def _add_valves_and_springs(parts: Dict[str, Part], st: EngineAssemblyState) -> None:
  """Add inclined intake/exhaust valves, seats, guides, stem seals, springs, and retainers."""
  sin_t = float(np.sin(np.radians(VALVE_TILT_DEG)))
  cos_t = float(np.cos(np.radians(VALVE_TILT_DEG)))
  configs = (
      ("intake", VALVE_IN_X_MM, (sin_t, 0.0, cos_t), VALVE_TILT_DEG, VALVE_IN_DIAM_M, st.intake_valve_lift_mm, True),
      ("exhaust", VALVE_EX_X_MM, (-sin_t, 0.0, cos_t), -VALVE_TILT_DEG, VALVE_EX_DIAM_M, st.exhaust_valve_lift_mm, False),
  )
  guide_base = generate_valve_guide()
  seal_base = generate_viton_oring(STEM_SEAL_ID_MM, STEM_SEAL_CS_MM, SKU_STEM_SEAL)
  retainer_base = generate_spring_retainer()

  for tag, vx, axis_dir, rot_y, head_d_m, lift_mm, is_in in configs:
    p_seat = (vx, 0.0, VALVE_SEAT_Z_MM)
    parts[f"valve_{tag}"] = _place_valve_axis(generate_poppet_valve(head_diam_m=head_d_m), p_seat, axis_dir, rot_y, -lift_mm)
    parts[f"valve_seat_{tag}"] = _place_valve_axis(generate_valve_seat(is_intake=is_in), p_seat, axis_dir, rot_y, SEAT_OFFSET_S_MM)
    parts[f"valve_guide_{tag}"] = _place_valve_axis(guide_base, p_seat, axis_dir, rot_y, GUIDE_OFFSET_S_MM)
    parts[f"stem_seal_{tag}"] = _place_valve_axis(seal_base, p_seat, axis_dir, rot_y, STEM_SEAL_OFFSET_S_MM)
    parts[f"spring_{tag}"] = _place_valve_axis(
        generate_valve_spring_9573k35(compressed_height_mm=SPRING_INSTALLED_H_MM - lift_mm),
        p_seat,
        axis_dir,
        rot_y,
        SPRING_OFFSET_S_MM,
    )
    parts[f"retainer_{tag}"] = _place_valve_axis(retainer_base, p_seat, axis_dir, rot_y, RETAINER_OFFSET_S_MM - lift_mm)


def _add_cam_rockers_and_fasteners(
    parts: Dict[str, Part],
    crank_angle_deg: float,
    has_fourth_axis: bool,
    st: EngineAssemblyState,
) -> None:
  """Add cam tower, 4-axis/3-axis camshaft, 36T sprocket, rockers, cover, plug, and SHCS."""
  cam_angle_deg = CAM_TO_CRANK_RATIO * crank_angle_deg
  parts["cam_tower"] = generate_cam_tower()

  if has_fourth_axis:
    parts["camshaft"] = generate_sohc_camshaft(True).rotate(Axis.Y, cam_angle_deg).translate((0.0, 0.0, CAM_AXIS_Z_MM))
  else:
    for cname, cpart in generate_sohc_camshaft_parts(False).items():
      parts[cname] = cpart.rotate(Axis.Y, cam_angle_deg).translate((0.0, 0.0, CAM_AXIS_Z_MM))

  parts["sprocket_cam_36t"] = (
      generate_sprocket_25(CAM_SPROCKET_TEETH, CAM_SPROCKET_BORE_MM)
      .rotate(Axis.Y, cam_angle_deg)
      .translate((0.0, CAM_SPROCKET_Y_MM, CAM_AXIS_Z_MM))
  )

  r_shaft = generate_rocker_shaft()
  parts["rocker_shaft_intake"] = r_shaft.translate((ROCKER_PIVOT_X_MM, 0.0, ROCKER_PIVOT_Z_MM))
  parts["rocker_shaft_exhaust"] = r_shaft.translate((-ROCKER_PIVOT_X_MM, 0.0, ROCKER_PIVOT_Z_MM))

  alpha_in = float(np.degrees(np.arcsin(np.clip((st.intake_valve_lift_mm / ROCKER_RATIO) / ROCKER_PIVOT_X_MM, -1.0, 1.0))))
  alpha_ex = float(np.degrees(np.arcsin(np.clip((st.exhaust_valve_lift_mm / ROCKER_RATIO) / ROCKER_PIVOT_X_MM, -1.0, 1.0))))
  parts["rocker_intake"] = generate_rocker_arm(True).rotate(
      Axis((ROCKER_PIVOT_X_MM, 0.0, ROCKER_PIVOT_Z_MM), (0.0, 1.0, 0.0)),
      alpha_in * ROCKER_CONTACT_SCALE,
  )
  parts["rocker_exhaust"] = generate_rocker_arm(False).rotate(
      Axis((-ROCKER_PIVOT_X_MM, 0.0, ROCKER_PIVOT_Z_MM), (0.0, 1.0, 0.0)),
      -alpha_ex * ROCKER_CONTACT_SCALE,
  )

  parts["cam_cover"] = generate_cam_cover()
  parts["cam_cover_oring"] = generate_viton_oring(COVER_ORING_ID_MM, COVER_ORING_CS_MM, SKU_COVER_ORING).translate(
      (0.0, COVER_ORING_Y_MM, COVER_ORING_Z_MM)
  )

  tilt_rad = np.radians(PLUG_TILT_DEG)
  plug_dir = (0.0, -float(np.cos(tilt_rad)), float(np.sin(tilt_rad)))
  parts["spark_plug_cmr7h"] = (
      generate_spark_plug_cmr7h()
      .rotate(Axis.X, PLUG_ROT_X_DEG)
      .translate((0.0, PLUG_ORIGIN_Y_MM + (plug_dir[1] * PLUG_S_OFFSET_MM), PLUG_ORIGIN_Z_MM + (plug_dir[2] * PLUG_S_OFFSET_MM)))
  )

  tower_bolt = generate_shcs(SHCS_M6_DIAM_MM, TOWER_BOLT_LEN_MM, SKU_TOWER_BOLT)
  for idx, (bx, by) in enumerate(CAM_TOWER_BOLT_XY_MM, start=1):
    parts[f"tower_bolt_{idx}"] = tower_bolt.translate((bx, by, TOWER_BOLT_Z_MM))

  cover_bolt = generate_shcs(SHCS_M6_DIAM_MM, COVER_BOLT_LEN_MM, SKU_COVER_BOLT)
  for idx, (cx, cy) in enumerate(COVER_BOLT_XY_MM, start=1):
    parts[f"cover_bolt_{idx}"] = cover_bolt.translate((cx, cy, COVER_BOLT_Z_MM))


def build_head_assembly_parts(
    crank_angle_deg: float = 0.0,
    has_fourth_axis: bool = True,
) -> Dict[str, Part]:
  """Build all SOHC head & valvetrain solids posed at crank_angle_deg."""
  st = EngineAssembly().solve_kinematics(crank_angle_deg)
  parts: Dict[str, Part] = {"cylinder_head": generate_cylinder_head()}
  _add_valves_and_springs(parts, st)
  _add_cam_rockers_and_fasteners(parts, crank_angle_deg, has_fourth_axis, st)
  return parts


def generate_head_assembly(
    crank_angle_deg: float = 0.0,
    has_fourth_axis: bool = True,
) -> Compound:
  """Assemble the SOHC head & valvetrain sub-assembly as a styled Compound."""
  return style_named_parts_as_compound(
      "head_assembly",
      build_head_assembly_parts(crank_angle_deg, has_fourth_axis),
      _material_key_for_head_part,
  )


def _material_key_for_head_part(name: str) -> str:
  """Resolve material catalog key for a head sub-assembly part name."""
  if name in BASE_HEAD_MATERIALS:
    return BASE_HEAD_MATERIALS[name]
  if name.startswith("tower_bolt_") or name.startswith("cover_bolt_"):
    return "STEEL_GRADE_10_9"
  return "STEEL_4140_PREHARD"


def verify_head_assembly(
    crank_angle_deg: float = 0.0,
    has_fourth_axis: bool = True,
) -> HeadAssemblyReport:
  """Verify head sub-assembly CR, mass rollup, and B-Rep solid clearances."""
  parts = build_head_assembly_parts(crank_angle_deg, has_fourth_axis=has_fourth_axis)
  max_ov, hits = find_pairwise_collisions(parts)

  dome_cm3, clearance_cm3, swept_cm3, cr = get_cad_head_volumes_and_cr()

  head_kg = sum(part_mass_kg(parts[k], _material_key_for_head_part(k)) for k in HEAD_ONLY_KEYS if k in parts)
  vt_kg = sum(part_mass_kg(parts[k], _material_key_for_head_part(k)) for k in VALVETRAIN_KEYS if k in parts)
  total_kg = sum(part_mass_kg(p, _material_key_for_head_part(k)) for k, p in parts.items())

  return HeadAssemblyReport(
      crank_angle_deg=crank_angle_deg,
      has_fourth_axis=has_fourth_axis,
      dome_volume_cm3=dome_cm3,
      clearance_volume_cm3=clearance_cm3,
      swept_volume_cm3=swept_cm3,
      compression_ratio=cr,
      head_mass_kg=round(head_kg, ROUND_MASS_DECIMALS),
      valvetrain_mass_kg=round(vt_kg, ROUND_MASS_DECIMALS),
      total_mass_kg=round(total_kg, ROUND_MASS_DECIMALS),
      max_overlap_mm3=round(max_ov, ROUND_MASS_DECIMALS),
      collision_pairs=hits,
      is_valid=(len(hits) == 0),
  )
