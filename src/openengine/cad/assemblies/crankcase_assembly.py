"""Hierarchical Split Crankcase, Sloped Sump Pan, Seals, and 0.65 L Oil Pool Assembly."""

from dataclasses import dataclass
from typing import Dict, List
from build123d import Axis, Compound, Part

from openengine.cad.assemblies._common import (
    find_pairwise_collisions,
    overlap_volume_mm3,
    part_mass_kg,
)
from openengine.cad.assemblies.rotating_assembly import build_rotating_assembly_parts
from openengine.cad.cots_parts import (
    generate_breather_vent_npt,
    generate_dowel_pin,
    generate_magnetic_drain_plug_m12,
    generate_oil_fill_plug_m20,
    generate_shcs,
    generate_sight_glass_m16,
    generate_sump_perimeter_oring,
    generate_viton_shaft_seal_30x47x7,
)
from openengine.cad.covers_and_aux import generate_right_timing_cover
from openengine.cad.crankcase import (
    CRANK_AXIS_Z_MM,
    OIL_LEVEL_Z_MM,
    SIGHT_GLASS_Z_MM,
    SPLIT_BOLT_XZ_MM,
    SPLIT_DOWEL_XZ_MM,
    SUMP_BOLT_XY_MM,
    TIMING_COVER_BOLT_XZ_MM,
    generate_crankcase_left,
    generate_crankcase_right,
    generate_sump_oil_pool,
    generate_sump_pan,
)
from openengine.cad.step_export import style_named_parts_as_compound

MM3_PER_LITER: float = 1.0e6
BDC_CRANK_ANGLE_DEG: float = 180.0
MIN_IMMERSION_MM: float = 4.0
MAX_IMMERSION_MM: float = 6.0
MIN_SUBMERGED_VOL_MM3: float = 100.0
ROUND_REPORT_DECIMALS: int = 4

LEFT_SEAL_Y_MM: float = -48.0
RIGHT_SEAL_Y_MM: float = 61.5
SIGHT_GLASS_POS_MM: tuple[float, float, float] = (83.2, 30.0, -211.0)
FILL_PLUG_POS_MM: tuple[float, float, float] = (79.66, 30.0, -108.34)
FILL_PLUG_TILT_Y_DEG: float = 45.0
DRAIN_PLUG_POS_MM: tuple[float, float, float] = (-82.5, 0.0, -267.5)
BREATHER_POS_MM: tuple[float, float, float] = (-86.2, -34.0, -105.0)

SPLIT_BOLT_Y_MM: float = -21.05
SPLIT_BOLT_LEN_MM: float = 40.0
SUMP_BOLT_Z_MM: float = -250.05
SUMP_BOLT_LEN_MM: float = 22.0
TIMING_BOLT_Y_MM: float = 61.05
TIMING_BOLT_LEN_MM: float = 18.0
DOWEL_DIAM_MM: float = 6.0
DOWEL_LEN_MM: float = 16.0
SHCS_M6_DIAM_MM: float = 6.0
ROT_90_DEG: float = 90.0
ROT_180_DEG: float = 180.0

SKU_SPLIT_DOWEL: str = "93600A168"
SKU_SPLIT_BOLT: str = "91290A334"
SKU_SUMP_BOLT: str = "91290A326"
SKU_TIMING_BOLT: str = "91290A326"

BASE_CRANKCASE_MATERIALS: Dict[str, str] = {
    "crankcase_left": "ALUMINUM_6061_T6",
    "crankcase_right": "ALUMINUM_6061_T6",
    "sump_pan": "ALUMINUM_6061_T6",
    "right_timing_cover": "ALUMINUM_6061_T6",
    "sump_perimeter_oring": "VITON_FLUOROELASTOMER",
    "sump_oil_pool": "SAE_10W40_OIL",
    "left_shaft_seal": "VITON_FLUOROELASTOMER",
    "right_shaft_seal": "VITON_FLUOROELASTOMER",
    "sight_glass_m16": "STEEL_4140_PREHARD",
    "oil_fill_plug_m20": "STEEL_4140_PREHARD",
    "magnetic_drain_plug_m12": "STEEL_4140_PREHARD",
    "breather_vent_npt": "BRONZE_SAE_660",
}


@dataclass(frozen=True, slots=True)
class CrankcaseAssemblyReport:
  """Wet-sump oil volume, BDC dipper immersion, mass, and solid collision status."""

  oil_volume_liters: float
  oil_level_z_mm: float
  sight_glass_center_z_mm: float
  bdc_dipper_tip_z_mm: float
  bdc_dipper_immersion_mm: float
  bdc_dipper_submerged_vol_mm3: float
  case_mass_kg: float
  oil_mass_kg: float
  max_solid_overlap_mm3: float
  collision_pairs: List[str]
  is_valid: bool


def _add_seals_and_service_ports(parts: Dict[str, Part]) -> None:
  """Add rotary shaft seals, sight glass, fill plug, magnetic drain plug, and breather."""
  seal = generate_viton_shaft_seal_30x47x7()
  parts["left_shaft_seal"] = seal.translate((0.0, LEFT_SEAL_Y_MM, CRANK_AXIS_Z_MM))
  parts["right_shaft_seal"] = seal.translate((0.0, RIGHT_SEAL_Y_MM, CRANK_AXIS_Z_MM))

  parts["sight_glass_m16"] = generate_sight_glass_m16().translate(SIGHT_GLASS_POS_MM)
  parts["oil_fill_plug_m20"] = (
      generate_oil_fill_plug_m20().rotate(Axis.Y, FILL_PLUG_TILT_Y_DEG).translate(FILL_PLUG_POS_MM)
  )
  parts["magnetic_drain_plug_m12"] = (
      generate_magnetic_drain_plug_m12().rotate(Axis.Y, -ROT_90_DEG).translate(DRAIN_PLUG_POS_MM)
  )
  parts["breather_vent_npt"] = generate_breather_vent_npt().rotate(Axis.Y, -ROT_90_DEG).translate(BREATHER_POS_MM)


def _add_case_fasteners(parts: Dict[str, Part]) -> None:
  """Add split dowels, M6 split-case bolts, M6 sump bolts, and M6 timing-cover bolts."""
  dowel = generate_dowel_pin(DOWEL_DIAM_MM, DOWEL_LEN_MM, SKU_SPLIT_DOWEL).rotate(Axis.X, ROT_90_DEG)
  for idx, (dx, dz) in enumerate(SPLIT_DOWEL_XZ_MM, start=1):
    parts[f"split_dowel_{idx}"] = dowel.translate((dx, 0.0, dz))

  split_bolt = generate_shcs(SHCS_M6_DIAM_MM, SPLIT_BOLT_LEN_MM, SKU_SPLIT_BOLT).rotate(Axis.X, ROT_90_DEG)
  for idx, (sx, sz) in enumerate(SPLIT_BOLT_XZ_MM, start=1):
    parts[f"split_bolt_{idx}"] = split_bolt.translate((sx, SPLIT_BOLT_Y_MM, sz))

  sump_bolt = generate_shcs(SHCS_M6_DIAM_MM, SUMP_BOLT_LEN_MM, SKU_SUMP_BOLT).rotate(Axis.X, ROT_180_DEG)
  for idx, (sx, sy) in enumerate(SUMP_BOLT_XY_MM, start=1):
    parts[f"sump_bolt_{idx}"] = sump_bolt.translate((sx, sy, SUMP_BOLT_Z_MM))

  timing_bolt = generate_shcs(SHCS_M6_DIAM_MM, TIMING_BOLT_LEN_MM, SKU_TIMING_BOLT).rotate(Axis.X, -ROT_90_DEG)
  for idx, (tx, tz) in enumerate(TIMING_COVER_BOLT_XZ_MM, start=1):
    parts[f"timing_cover_bolt_{idx}"] = timing_bolt.translate((tx, TIMING_BOLT_Y_MM, tz))


def build_crankcase_assembly_parts(include_oil_pool: bool = True) -> Dict[str, Part]:
  """Build all crankcase, sump pan, seal, fastener, and optional oil-pool B-Rep parts."""
  parts: Dict[str, Part] = {
      "crankcase_left": generate_crankcase_left(),
      "crankcase_right": generate_crankcase_right(),
      "sump_pan": generate_sump_pan(),
      "right_timing_cover": generate_right_timing_cover(),
      "sump_perimeter_oring": generate_sump_perimeter_oring(),
  }
  if include_oil_pool:
    parts["sump_oil_pool"] = generate_sump_oil_pool()
  _add_seals_and_service_ports(parts)
  _add_case_fasteners(parts)
  return parts


def generate_crankcase_assembly(include_oil_pool: bool = True) -> Compound:
  """Assemble the split crankcase + sloped sump pan sub-assembly as a styled Compound."""
  return style_named_parts_as_compound(
      "crankcase_assembly",
      build_crankcase_assembly_parts(include_oil_pool=include_oil_pool),
      _material_key_for_case_part,
  )


def _material_key_for_case_part(name: str) -> str:
  """Resolve material catalog key for a crankcase sub-assembly part."""
  if name in BASE_CRANKCASE_MATERIALS:
    return BASE_CRANKCASE_MATERIALS[name]
  if name.startswith("split_dowel_"):
    return "STEEL_4140_PREHARD"
  return "STEEL_GRADE_10_9"


def verify_crankcase_assembly() -> CrankcaseAssemblyReport:
  """Verify crankcase/sump solid clearances, 0.65 L oil capacity, and BDC dipper immersion."""
  rigid_parts = build_crankcase_assembly_parts(include_oil_pool=False)
  max_ov, hits = find_pairwise_collisions(rigid_parts)

  oil_pool = generate_sump_oil_pool()
  oil_vol_liters = float(oil_pool.volume) / MM3_PER_LITER
  oil_kg = part_mass_kg(oil_pool, "SAE_10W40_OIL")
  case_kg = sum(part_mass_kg(p, _material_key_for_case_part(k)) for k, p in rigid_parts.items())

  bdc_cap = build_rotating_assembly_parts(BDC_CRANK_ANGLE_DEG)["conrod_cap"]
  dipper_tip_z_mm = float(bdc_cap.bounding_box().min.Z)
  immersion_mm = float(OIL_LEVEL_Z_MM - dipper_tip_z_mm)
  submerged_vol_mm3 = overlap_volume_mm3(bdc_cap, oil_pool)

  valid = (
      (len(hits) == 0)
      and (MIN_IMMERSION_MM <= immersion_mm <= MAX_IMMERSION_MM)
      and (submerged_vol_mm3 >= MIN_SUBMERGED_VOL_MM3)
  )
  return CrankcaseAssemblyReport(
      oil_volume_liters=round(oil_vol_liters, ROUND_REPORT_DECIMALS),
      oil_level_z_mm=round(OIL_LEVEL_Z_MM, 2),
      sight_glass_center_z_mm=round(SIGHT_GLASS_Z_MM, 2),
      bdc_dipper_tip_z_mm=round(dipper_tip_z_mm, 2),
      bdc_dipper_immersion_mm=round(immersion_mm, 2),
      bdc_dipper_submerged_vol_mm3=round(submerged_vol_mm3, 2),
      case_mass_kg=round(case_kg, ROUND_REPORT_DECIMALS),
      oil_mass_kg=round(oil_kg, ROUND_REPORT_DECIMALS),
      max_solid_overlap_mm3=round(max_ov, ROUND_REPORT_DECIMALS),
      collision_pairs=hits,
      is_valid=valid,
  )
