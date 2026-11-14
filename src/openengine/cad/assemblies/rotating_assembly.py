"""Hierarchical Rotating & Reciprocating CAD Sub-Assembly."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from build123d import Axis, Compound, Part

from openengine.cad.assemblies._common import (
    find_pairwise_collisions,
    part_mass_kg,
)
from openengine.cad.assembly import EngineAssembly, EngineAssemblyState
from openengine.cad.connecting_rod import generate_connecting_rod_parts
from openengine.cad.cots_parts import (
    generate_ball_bearing_6206,
    generate_dowel_pin,
    generate_external_circlip_16mm,
    generate_needle_bearing_nks28,
    generate_piston_ring_pack_68mm,
    generate_shcs,
    generate_sprocket_25,
    generate_wrist_pin_16x45,
)
from openengine.cad.covers_and_aux import generate_flywheel, generate_trigger_wheel
from openengine.cad.crankcase import CRANK_AXIS_Z_MM
from openengine.cad.crankshaft import (
    CRANK_SPROCKET_TEETH,
    calculate_crankshaft_balance,
    generate_crankshaft,
)
from openengine.cad.piston import generate_piston

KG_TO_G: float = 1000.0
TINY_MASS_EPS_G: float = 1.0e-9
ROUND_KG_DECIMALS: int = 4
ROT_X_NEG90_DEG: float = -90.0
ROT_X_POS90_DEG: float = 90.0
ROT_Y_NEG90_DEG: float = -90.0
ROT_Z_POS90_DEG: float = 90.0

CONROD_CENTER_TO_CENTER_MM: float = 108.0
CONROD_BOLT_OFFSET_X_MM: float = 4.2
CONROD_BOLT_HEAD_Y_MM: float = 121.0
CONROD_BOLT_OFFSET_Z_MM: float = 27.0
CONROD_BOLT_DIAM_MM: float = 6.0
CONROD_BOLT_LEN_MM: float = 26.0
CONROD_DOWEL_DIAM_MM: float = 5.0
CONROD_DOWEL_LEN_MM: float = 12.0

PISTON_PIN_LOCAL_OFFSET_Z_MM: float = 3.0
CIRCLIP_OFFSET_Y_MM: float = 23.5
MAIN_BEARING_OFFSET_Y_MM: float = 34.5
FLYWHEEL_OFFSET_Y_MM: float = -70.0
TRIGGER_WHEEL_OFFSET_Y_MM: float = -84.0
CRANK_SPROCKET_BORE_MM: float = 20.15
CRANK_SPROCKET_OFFSET_Y_MM: float = 53.0

SKU_CONROD_BOLT: str = "91290A330"
SKU_CONROD_DOWEL: str = "93600A149"

ROTATING_MATERIALS: Dict[str, str] = {
    "crankshaft": "STEEL_4140_PREHARD",
    "conrod_beam": "ALUMINUM_7075_T6",
    "conrod_cap": "ALUMINUM_7075_T6",
    "needle_bearing_nks28": "STEEL_4140_PREHARD",
    "conrod_bolt_1": "STEEL_GRADE_10_9",
    "conrod_bolt_2": "STEEL_GRADE_10_9",
    "conrod_dowel_1": "STEEL_4140_PREHARD",
    "conrod_dowel_2": "STEEL_4140_PREHARD",
    "piston": "ALUMINUM_4032_T6",
    "top_ring": "STEEL_4140_PREHARD",
    "second_ring": "STEEL_4140_PREHARD",
    "oil_ring": "STEEL_4140_PREHARD",
    "wrist_pin": "STEEL_4140_PREHARD",
    "circlip_left": "STEEL_4140_PREHARD",
    "circlip_right": "STEEL_4140_PREHARD",
    "main_bearing_left": "STEEL_4140_PREHARD",
    "main_bearing_right": "STEEL_4140_PREHARD",
    "flywheel": "STEEL_1144_STRESSPROOF",
    "trigger_wheel": "STEEL_1144_STRESSPROOF",
    "sprocket_crank_18t": "STEEL_4140_PREHARD",
}


@dataclass(frozen=True, slots=True)
class RotatingAssemblyReport:
  """Mass breakdown, crank balance factor, and collision status for rotating assembly."""

  crank_angle_deg: float
  piston_mass_g: float
  rings_mass_g: float
  wrist_pin_mass_g: float
  circlips_mass_g: float
  conrod_total_mass_g: float
  conrod_small_end_mass_g: float
  conrod_big_end_mass_g: float
  needle_bearing_mass_g: float
  reciprocating_mass_kg: float
  rotating_mass_kg: float
  crankshaft_mass_kg: float
  flywheel_mass_kg: float
  trigger_wheel_mass_g: float
  effective_balance_factor: float
  max_overlap_mm3: float
  collision_pairs: List[str]
  is_valid: bool


def _pose_rod_local(part: Part, st: EngineAssemblyState) -> Part:
  """Transform a connecting-rod local part (+Y toward crankpin) to master world pose."""
  return (
      part.rotate(Axis.Z, ROT_Z_POS90_DEG)
      .rotate(Axis.Y, ROT_Y_NEG90_DEG)
      .rotate(Axis.Y, -st.con_rod_angle_deg)
      .translate((0.0, 0.0, st.wrist_pin_z_mm))
  )


def _add_conrod_assembly_parts(parts: Dict[str, Part], st: EngineAssemblyState) -> None:
  """Add split 7075-T6 rod, needle bearing, SHCS bolts, and dowels in world pose."""
  beam, cap = generate_connecting_rod_parts()
  parts["conrod_beam"] = _pose_rod_local(beam, st)
  parts["conrod_cap"] = _pose_rod_local(cap, st)

  nks28 = generate_needle_bearing_nks28().translate((0.0, CONROD_CENTER_TO_CENTER_MM, 0.0))
  parts["needle_bearing_nks28"] = _pose_rod_local(nks28, st)

  bolt_base = generate_shcs(CONROD_BOLT_DIAM_MM, CONROD_BOLT_LEN_MM, SKU_CONROD_BOLT).rotate(Axis.X, ROT_X_NEG90_DEG)
  parts["conrod_bolt_1"] = _pose_rod_local(
      bolt_base.translate((CONROD_BOLT_OFFSET_X_MM, CONROD_BOLT_HEAD_Y_MM, CONROD_BOLT_OFFSET_Z_MM)),
      st,
  )
  parts["conrod_bolt_2"] = _pose_rod_local(
      bolt_base.translate((-CONROD_BOLT_OFFSET_X_MM, CONROD_BOLT_HEAD_Y_MM, -CONROD_BOLT_OFFSET_Z_MM)),
      st,
  )

  dowel_base = generate_dowel_pin(CONROD_DOWEL_DIAM_MM, CONROD_DOWEL_LEN_MM, SKU_CONROD_DOWEL).rotate(
      Axis.X, ROT_X_POS90_DEG
  )
  parts["conrod_dowel_1"] = _pose_rod_local(
      dowel_base.translate((-CONROD_BOLT_OFFSET_X_MM, CONROD_CENTER_TO_CENTER_MM, CONROD_BOLT_OFFSET_Z_MM)),
      st,
  )
  parts["conrod_dowel_2"] = _pose_rod_local(
      dowel_base.translate((CONROD_BOLT_OFFSET_X_MM, CONROD_CENTER_TO_CENTER_MM, -CONROD_BOLT_OFFSET_Z_MM)),
      st,
  )


def _add_piston_and_flywheel_parts(parts: Dict[str, Part], crank_angle_deg: float, st: EngineAssemblyState) -> None:
  """Add piston, rings, wrist pin, circlips, 6206 mains, flywheel, trigger wheel, and 18T sprocket."""
  parts["piston"] = generate_piston().translate((0.0, 0.0, st.wrist_pin_z_mm + PISTON_PIN_LOCAL_OFFSET_Z_MM))
  for ring_name, ring_part in generate_piston_ring_pack_68mm().items():
    parts[ring_name] = ring_part.translate((0.0, 0.0, st.wrist_pin_z_mm + PISTON_PIN_LOCAL_OFFSET_Z_MM))

  parts["wrist_pin"] = generate_wrist_pin_16x45().translate((0.0, 0.0, st.wrist_pin_z_mm))
  clip = generate_external_circlip_16mm()
  parts["circlip_left"] = clip.translate((0.0, -CIRCLIP_OFFSET_Y_MM, st.wrist_pin_z_mm))
  parts["circlip_right"] = clip.translate((0.0, CIRCLIP_OFFSET_Y_MM, st.wrist_pin_z_mm))

  brg = generate_ball_bearing_6206()
  parts["main_bearing_left"] = brg.translate((0.0, -MAIN_BEARING_OFFSET_Y_MM, CRANK_AXIS_Z_MM))
  parts["main_bearing_right"] = brg.translate((0.0, MAIN_BEARING_OFFSET_Y_MM, CRANK_AXIS_Z_MM))

  parts["flywheel"] = generate_flywheel().rotate(Axis.Y, crank_angle_deg).translate((0.0, FLYWHEEL_OFFSET_Y_MM, CRANK_AXIS_Z_MM))
  parts["trigger_wheel"] = (
      generate_trigger_wheel().rotate(Axis.Y, crank_angle_deg).translate((0.0, TRIGGER_WHEEL_OFFSET_Y_MM, CRANK_AXIS_Z_MM))
  )
  parts["sprocket_crank_18t"] = (
      generate_sprocket_25(CRANK_SPROCKET_TEETH, CRANK_SPROCKET_BORE_MM)
      .rotate(Axis.Y, crank_angle_deg)
      .translate((0.0, CRANK_SPROCKET_OFFSET_Y_MM, CRANK_AXIS_Z_MM))
  )


def build_rotating_assembly_parts(crank_angle_deg: float = 0.0) -> Dict[str, Part]:
  """Build all rotating & reciprocating B-Rep parts posed at crank_angle_deg."""
  st = EngineAssembly().solve_kinematics(crank_angle_deg)
  parts: Dict[str, Part] = {
      "crankshaft": generate_crankshaft().rotate(Axis.Y, crank_angle_deg).translate((0.0, 0.0, CRANK_AXIS_Z_MM)),
  }
  _add_conrod_assembly_parts(parts, st)
  _add_piston_and_flywheel_parts(parts, crank_angle_deg, st)
  return parts


def generate_rotating_assembly(crank_angle_deg: float = 0.0) -> Compound:
  """Assemble the rotating & reciprocating sub-assembly as a single Compound."""
  return Compound(children=list(build_rotating_assembly_parts(crank_angle_deg).values()))


def _conrod_cg_split_masses_g(parts: Dict[str, Part]) -> tuple[float, float, float]:
  """Split bare 2-piece conrod mass into small-end reciprocating and big-end rotating fractions."""
  beam_local, cap_local = generate_connecting_rod_parts()
  m_beam_g = part_mass_kg(parts["conrod_beam"], ROTATING_MATERIALS["conrod_beam"]) * KG_TO_G
  m_cap_g = part_mass_kg(parts["conrod_cap"], ROTATING_MATERIALS["conrod_cap"]) * KG_TO_G
  total_g = m_beam_g + m_cap_g
  y_cg = ((beam_local.center().Y * m_beam_g) + (cap_local.center().Y * m_cap_g)) / max(total_g, TINY_MASS_EPS_G)
  big_frac = float(np.clip(y_cg / CONROD_CENTER_TO_CENTER_MM, 0.0, 1.0))
  big_g = total_g * big_frac
  small_g = total_g - big_g
  return total_g, small_g, big_g


def verify_rotating_assembly(crank_angle_deg: float = 0.0) -> RotatingAssemblyReport:
  """Verify rotating sub-assembly solid clearances, reciprocating/rotating mass, and balance."""
  parts = build_rotating_assembly_parts(crank_angle_deg)
  max_ov, hits = find_pairwise_collisions(parts)

  piston_g = part_mass_kg(parts["piston"], ROTATING_MATERIALS["piston"]) * KG_TO_G
  rings_g = sum(part_mass_kg(parts[k], ROTATING_MATERIALS[k]) * KG_TO_G for k in ("top_ring", "second_ring", "oil_ring"))
  pin_g = part_mass_kg(parts["wrist_pin"], ROTATING_MATERIALS["wrist_pin"]) * KG_TO_G
  clips_g = (
      part_mass_kg(parts["circlip_left"], ROTATING_MATERIALS["circlip_left"])
      + part_mass_kg(parts["circlip_right"], ROTATING_MATERIALS["circlip_right"])
  ) * KG_TO_G

  conrod_total_g, small_end_g, big_end_g = _conrod_cg_split_masses_g(parts)
  nks_g = part_mass_kg(parts["needle_bearing_nks28"], ROTATING_MATERIALS["needle_bearing_nks28"]) * KG_TO_G
  rod_hw_g = sum(
      part_mass_kg(parts[k], ROTATING_MATERIALS[k]) * KG_TO_G
      for k in ("conrod_bolt_1", "conrod_bolt_2", "conrod_dowel_1", "conrod_dowel_2")
  )

  recip_kg = (piston_g + rings_g + pin_g + clips_g + small_end_g) / KG_TO_G
  rot_kg = (big_end_g + nks_g + rod_hw_g) / KG_TO_G
  bal = calculate_crankshaft_balance(reciprocating_mass_kg=recip_kg, rotating_mass_kg=rot_kg)
  bal_factor = float(bal["effective_balance_factor"])

  crank_kg = part_mass_kg(parts["crankshaft"], ROTATING_MATERIALS["crankshaft"])
  fw_kg = part_mass_kg(parts["flywheel"], ROTATING_MATERIALS["flywheel"])
  trig_g = part_mass_kg(parts["trigger_wheel"], ROTATING_MATERIALS["trigger_wheel"]) * KG_TO_G

  valid = (len(hits) == 0) and bool(bal["is_balanced"])
  return RotatingAssemblyReport(
      crank_angle_deg=crank_angle_deg,
      piston_mass_g=round(piston_g, 2),
      rings_mass_g=round(rings_g, 2),
      wrist_pin_mass_g=round(pin_g, 2),
      circlips_mass_g=round(clips_g, 2),
      conrod_total_mass_g=round(conrod_total_g, 2),
      conrod_small_end_mass_g=round(small_end_g, 2),
      conrod_big_end_mass_g=round(big_end_g, 2),
      needle_bearing_mass_g=round(nks_g, 2),
      reciprocating_mass_kg=round(recip_kg, ROUND_KG_DECIMALS),
      rotating_mass_kg=round(rot_kg, ROUND_KG_DECIMALS),
      crankshaft_mass_kg=round(crank_kg, ROUND_KG_DECIMALS),
      flywheel_mass_kg=round(fw_kg, ROUND_KG_DECIMALS),
      trigger_wheel_mass_g=round(trig_g, 2),
      effective_balance_factor=round(bal_factor, ROUND_KG_DECIMALS),
      max_overlap_mm3=round(max_ov, ROUND_KG_DECIMALS),
      collision_pairs=hits,
      is_valid=valid,
  )
