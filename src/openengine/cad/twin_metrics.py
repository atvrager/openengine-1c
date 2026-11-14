"""Cached collision-free B-Rep twin metric extractors for simulation and reporting."""

from functools import lru_cache
import math
from typing import Tuple

ROUND_MASS_DECIMALS: int = 4
ROUND_VOL_DECIMALS: int = 3
ROUND_SWEPT_DECIMALS: int = 2
ROUND_IMMERSION_DECIMALS: int = 2
CM_PER_M: float = 100.0
MM_PER_M: float = 1000.0
MM3_PER_LITER: float = 1.0e6
QUARTER_CIRCLE_FACTOR: float = 0.25
PAIR_COUNT: float = 2.0
TINY_MASS_EPS_KG: float = 1.0e-12
MIN_FRACTION: float = 0.0
MAX_FRACTION: float = 1.0
BDC_CRANK_ANGLE_DEG: float = 180.0
ROT_Y_NEG90_DEG: float = -90.0
ROT_Z_POS90_DEG: float = 90.0
RING_NAMES: Tuple[str, ...] = ("top_ring", "second_ring", "oil_ring")
MAT_VALVE_STEEL: str = "STEEL_4140_PREHARD"
MAT_RETAINER_ALUMINUM: str = "ALUMINUM_7075_T6"


@lru_cache(maxsize=1)
def get_cad_rotating_and_reciprocating_masses_kg() -> Tuple[float, float]:
  """Compute exact (rotating_kg, reciprocating_kg) from B-Rep CAD without collision checks."""
  from openengine.cad.assemblies._common import part_mass_kg
  from openengine.cad.assemblies.rotating_assembly import (
      CONROD_BOLT_DIAM_MM,
      CONROD_BOLT_LEN_MM,
      CONROD_DOWEL_DIAM_MM,
      CONROD_DOWEL_LEN_MM,
      ROTATING_MATERIALS,
      SKU_CONROD_BOLT,
      SKU_CONROD_DOWEL,
  )
  from openengine.cad.connecting_rod import (
      DEFAULT_CENTER_TO_CENTER_M,
      generate_connecting_rod_parts,
  )
  from openengine.cad.cots_parts import (
      generate_dowel_pin,
      generate_external_circlip_16mm,
      generate_needle_bearing_nks28,
      generate_piston_ring_pack_68mm,
      generate_shcs,
      generate_wrist_pin_16x45,
  )
  from openengine.cad.piston import generate_piston

  beam, cap = generate_connecting_rod_parts()
  m_beam_kg = part_mass_kg(beam, ROTATING_MATERIALS["conrod_beam"])
  m_cap_kg = part_mass_kg(cap, ROTATING_MATERIALS["conrod_cap"])
  rod_total_kg = m_beam_kg + m_cap_kg
  y_cg = ((float(beam.center().Y) * m_beam_kg) + (float(cap.center().Y) * m_cap_kg)) / max(
      rod_total_kg,
      TINY_MASS_EPS_KG,
  )
  big_frac = min(MAX_FRACTION, max(MIN_FRACTION, y_cg / (DEFAULT_CENTER_TO_CENTER_M * MM_PER_M)))
  big_end_kg = rod_total_kg * big_frac
  small_end_kg = rod_total_kg - big_end_kg

  piston_kg = part_mass_kg(generate_piston(), ROTATING_MATERIALS["piston"])
  ring_pack = generate_piston_ring_pack_68mm()
  rings_kg = sum(part_mass_kg(ring_pack[name], ROTATING_MATERIALS[name]) for name in RING_NAMES)
  pin_kg = part_mass_kg(generate_wrist_pin_16x45(), ROTATING_MATERIALS["wrist_pin"])
  clips_kg = PAIR_COUNT * part_mass_kg(generate_external_circlip_16mm(), ROTATING_MATERIALS["circlip_left"])

  nks_kg = part_mass_kg(generate_needle_bearing_nks28(), ROTATING_MATERIALS["needle_bearing_nks28"])
  bolts_kg = PAIR_COUNT * part_mass_kg(
      generate_shcs(CONROD_BOLT_DIAM_MM, CONROD_BOLT_LEN_MM, SKU_CONROD_BOLT),
      ROTATING_MATERIALS["conrod_bolt_1"],
  )
  dowels_kg = PAIR_COUNT * part_mass_kg(
      generate_dowel_pin(CONROD_DOWEL_DIAM_MM, CONROD_DOWEL_LEN_MM, SKU_CONROD_DOWEL),
      ROTATING_MATERIALS["conrod_dowel_1"],
  )

  recip_kg = piston_kg + rings_kg + pin_kg + clips_kg + small_end_kg
  rot_kg = big_end_kg + nks_kg + bolts_kg + dowels_kg
  return round(rot_kg, ROUND_MASS_DECIMALS), round(recip_kg, ROUND_MASS_DECIMALS)


def get_cad_rotating_mass_kg() -> float:
  """Return cached CAD big-end + needle bearing + fastener rotating mass in kg."""
  rot_kg, _ = get_cad_rotating_and_reciprocating_masses_kg()
  return rot_kg


def get_cad_reciprocating_mass_kg() -> float:
  """Return cached CAD piston + rings + pin + circlips + small-end reciprocating mass in kg."""
  _, recip_kg = get_cad_rotating_and_reciprocating_masses_kg()
  return recip_kg


@lru_cache(maxsize=1)
def get_cad_valvetrain_masses_kg() -> Tuple[float, float, float]:
  """Compute exact (intake_valve_kg, exhaust_valve_kg, retainer_kg) from B-Rep CAD."""
  from openengine.cad.assemblies._common import part_mass_kg
  from openengine.cad.cylinder_head import CylinderHeadParams, generate_poppet_valve
  from openengine.cad.valvetrain_hardware import generate_spring_retainer

  head_params = CylinderHeadParams()
  in_valve = generate_poppet_valve(head_diam_m=head_params.intake_valve_diam_m)
  ex_valve = generate_poppet_valve(head_diam_m=head_params.exhaust_valve_diam_m)
  retainer = generate_spring_retainer()
  return (
      round(part_mass_kg(in_valve, MAT_VALVE_STEEL), ROUND_MASS_DECIMALS),
      round(part_mass_kg(ex_valve, MAT_VALVE_STEEL), ROUND_MASS_DECIMALS),
      round(part_mass_kg(retainer, MAT_RETAINER_ALUMINUM), ROUND_MASS_DECIMALS),
  )


def get_cad_intake_valve_mass_kg() -> float:
  """Return cached CAD 4140 intake poppet valve mass in kg."""
  in_kg, _, _ = get_cad_valvetrain_masses_kg()
  return in_kg


def get_cad_exhaust_valve_mass_kg() -> float:
  """Return cached CAD 4140 exhaust poppet valve mass in kg."""
  _, ex_kg, _ = get_cad_valvetrain_masses_kg()
  return ex_kg


def get_cad_retainer_mass_kg() -> float:
  """Return cached CAD 7075-T6 valve spring retainer mass in kg."""
  _, _, ret_kg = get_cad_valvetrain_masses_kg()
  return ret_kg


@lru_cache(maxsize=1)
def get_cad_head_volumes_and_cr() -> Tuple[float, float, float, float]:
  """Compute exact (dome_cm3, clearance_cm3, swept_cm3, compression_ratio) from CAD."""
  from openengine.cad.crankshaft import DEFAULT_STROKE_M
  from openengine.cad.cylinder_block import DEFAULT_BORE_M
  from openengine.cad.cylinder_head import (
      GASKET_AND_CREVICE_CM3,
      calculate_combustion_chamber_volume,
  )

  _, dome_cm3_raw = calculate_combustion_chamber_volume()
  clearance_cm3_raw = float(dome_cm3_raw) + GASKET_AND_CREVICE_CM3
  swept_cm3 = round(
      math.pi * QUARTER_CIRCLE_FACTOR * ((DEFAULT_BORE_M * CM_PER_M) ** PAIR_COUNT) * (DEFAULT_STROKE_M * CM_PER_M),
      ROUND_SWEPT_DECIMALS,
  )
  cr = (swept_cm3 + clearance_cm3_raw) / clearance_cm3_raw
  return (
      round(float(dome_cm3_raw), ROUND_VOL_DECIMALS),
      round(clearance_cm3_raw, ROUND_VOL_DECIMALS),
      swept_cm3,
      round(cr, ROUND_VOL_DECIMALS),
  )


def get_cad_compression_ratio() -> float:
  """Return cached geometric compression ratio computed from live cylinder head B-Rep."""
  _, _, _, cr = get_cad_head_volumes_and_cr()
  return cr


@lru_cache(maxsize=1)
def get_cad_sump_splash_metrics() -> Tuple[float, float, float]:
  """Compute exact (oil_volume_liters, bdc_dipper_immersion_mm, dipper_tip_radius_mm)."""
  from build123d import Axis

  from openengine.cad.assembly import EngineAssembly
  from openengine.cad.connecting_rod import generate_connecting_rod_parts
  from openengine.cad.crankcase import (
      CRANK_AXIS_Z_MM,
      OIL_LEVEL_Z_MM,
      generate_sump_oil_pool,
  )

  oil_vol_liters = round(float(generate_sump_oil_pool().volume) / MM3_PER_LITER, ROUND_MASS_DECIMALS)
  st = EngineAssembly().solve_kinematics(BDC_CRANK_ANGLE_DEG)
  _, cap = generate_connecting_rod_parts()
  bdc_cap = (
      cap.rotate(Axis.Z, ROT_Z_POS90_DEG)
      .rotate(Axis.Y, ROT_Y_NEG90_DEG)
      .rotate(Axis.Y, -st.con_rod_angle_deg)
      .translate((MIN_FRACTION, MIN_FRACTION, st.wrist_pin_z_mm))
  )
  dipper_tip_z_mm = float(bdc_cap.bounding_box().min.Z)
  immersion_mm = round(float(OIL_LEVEL_Z_MM - dipper_tip_z_mm), ROUND_IMMERSION_DECIMALS)
  tip_radius_mm = round(abs(float(CRANK_AXIS_Z_MM - dipper_tip_z_mm)), ROUND_IMMERSION_DECIMALS)
  return oil_vol_liters, immersion_mm, tip_radius_mm


def get_cad_sump_oil_volume_l() -> float:
  """Return cached wet-sump static oil fill volume in liters from B-Rep sump pool."""
  vol_l, _, _ = get_cad_sump_splash_metrics()
  return vol_l


def get_cad_dipper_immersion_mm() -> float:
  """Return cached BDC conrod dipper immersion depth below static oil level in mm."""
  _, immersion_mm, _ = get_cad_sump_splash_metrics()
  return immersion_mm


def get_cad_dipper_tip_radius_mm() -> float:
  """Return cached BDC conrod dipper tip radius from crank axis in mm."""
  _, _, tip_radius_mm = get_cad_sump_splash_metrics()
  return tip_radius_mm
