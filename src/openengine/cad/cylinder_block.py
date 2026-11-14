"""Parametric Billet Cylinder Block & DOM Liner Solid Model via build123d.

Generates watertight B-Rep CAD models of the NestWorks C500 4-side 3-axis
finned 6061-T6 cylinder barrel and DOM A513-T5 steel liner:
- 68.0 mm honed bore, 3.0 mm liner wall (74.0 mm OD), 105.0 mm length
- Shrink-fit 6061-T6 aluminum barrel with 11 cooling fins inside 120x120x85 mm
- Integrated +Y #25 cam-chain tunnel centered at Y = +53.0 mm
- 4x M8 head tie-bolt tapped bores at (+/-40.0, +/-40.0) mm
- 4x M8 base flange mounting bores at (+/-51.5, +/-26.0) mm with 19 mm wrench reliefs
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
from build123d import (
    Box,
    BuildPart,
    Cylinder,
    Locations,
    Mode,
    Part,
)

MM_PER_M: float = 1000.0
M_PER_MM: float = 0.001

DEFAULT_BORE_M: float = 0.068
DEFAULT_LINER_WALL_M: float = 0.003
DEFAULT_LINER_LENGTH_M: float = 0.105
DEFAULT_BLOCK_HEIGHT_M: float = 0.085
DEFAULT_BARREL_WALL_M: float = 0.007
DEFAULT_NUM_FINS: int = 11
DEFAULT_FIN_THICKNESS_M: float = 0.0022
DEFAULT_FIN_PITCH_M: float = 0.0055
DEFAULT_FIN_WIDTH_X_M: float = 0.118
DEFAULT_FIN_DEPTH_Y_M: float = 0.114

TOP_DECK_THICK_MM: float = 10.0
BASE_FLANGE_THICK_MM: float = 10.0
FLANGE_WIDTH_X_MM: float = 120.0
FLANGE_MIN_Y_MM: float = -56.0
FLANGE_MAX_Y_MM: float = 64.0
FIN_CENTER_Y_MM: float = 1.0
TOP_FIN_CENTER_OFFSET_MM: float = 15.5

LINER_BORE_CLEARANCE_MM: float = 0.02
HEAD_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (40.0, 40.0),
    (-40.0, 40.0),
    (40.0, -40.0),
    (-40.0, -40.0),
)
HEAD_BOLT_BOSS_RADIUS_MM: float = 8.0
HEAD_BOLT_BOSS_DEPTH_MM: float = 24.0
HEAD_BOLT_BORE_RADIUS_MM: float = 4.1
HEAD_BOLT_BORE_DEPTH_MM: float = 22.0

BASE_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (51.5, 26.0),
    (-51.5, 26.0),
    (51.5, -26.0),
    (-51.5, -26.0),
)
BASE_BOLT_BORE_RADIUS_MM: float = 4.25
BASE_WRENCH_RELIEF_RADIUS_MM: float = 9.5

CHAIN_BOSS_WIDTH_X_MM: float = 86.0
CHAIN_BOSS_MIN_Y_MM: float = 45.0
CHAIN_BOSS_MAX_Y_MM: float = 64.0
CHAIN_SLOT_WIDTH_X_MM: float = 78.0
CHAIN_SLOT_MIN_Y_MM: float = 48.8
CHAIN_SLOT_MAX_Y_MM: float = 57.2
THERMAL_BUDGET_W: float = 1300.0
CUT_MARGIN_MM: float = 4.0
HALF_FACTOR: float = 0.5
FIN_BASE_CLEARANCE_MM: float = 0.5
DEFAULT_AIR_HTC_W_M2K: float = 75.0
DEFAULT_AIR_TEMP_DIFF_K: float = 130.0
DEFAULT_FIN_EFFICIENCY: float = 0.85
MIN_THERMAL_MARGIN_PCT: float = 20.0


@dataclass
class CylinderBlockParams:
  """Geometric parameters for the finned 6061-T6 billet barrel and DOM liner."""

  bore_m: float = DEFAULT_BORE_M
  liner_wall_m: float = DEFAULT_LINER_WALL_M
  liner_length_m: float = DEFAULT_LINER_LENGTH_M
  block_height_m: float = DEFAULT_BLOCK_HEIGHT_M
  barrel_wall_m: float = DEFAULT_BARREL_WALL_M
  num_fins: int = DEFAULT_NUM_FINS
  fin_thickness_m: float = DEFAULT_FIN_THICKNESS_M
  fin_pitch_m: float = DEFAULT_FIN_PITCH_M
  fin_outer_width_m: float = DEFAULT_FIN_WIDTH_X_M
  fin_outer_depth_m: float = DEFAULT_FIN_DEPTH_Y_M


def generate_cylinder_liner(params: Optional[CylinderBlockParams] = None) -> Part:
  """Generate the shrunk-in honed DOM steel cylinder liner sleeve.

  Top deck face sits at Z = 0.0 mm and extends downward to Z = -105.0 mm.
  """
  p = params or CylinderBlockParams()
  r_inner = (p.bore_m * MM_PER_M) * HALF_FACTOR
  r_outer = r_inner + (p.liner_wall_m * MM_PER_M)
  h_liner = p.liner_length_m * MM_PER_M

  with BuildPart() as liner_builder:
    with Locations((0.0, 0.0, -h_liner * HALF_FACTOR)):
      Cylinder(radius=r_outer, height=h_liner)
      Cylinder(radius=r_inner, height=h_liner + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
  return liner_builder.part


def _add_block_structure(p: CylinderBlockParams) -> None:
  """Add barrel sleeve, flanges, cooling fins, chain boss, and bolt bosses."""
  r_liner_outer = ((p.bore_m * HALF_FACTOR) + p.liner_wall_m) * MM_PER_M
  r_barrel = r_liner_outer + (p.barrel_wall_m * MM_PER_M)
  h_block = p.block_height_m * MM_PER_M
  flange_depth_y = FLANGE_MAX_Y_MM - FLANGE_MIN_Y_MM
  flange_mid_y = (FLANGE_MAX_Y_MM + FLANGE_MIN_Y_MM) * HALF_FACTOR
  fin_w = p.fin_outer_width_m * MM_PER_M
  fin_d = p.fin_outer_depth_m * MM_PER_M
  fin_t = p.fin_thickness_m * MM_PER_M
  fin_pitch = p.fin_pitch_m * MM_PER_M

  with Locations((0.0, 0.0, -h_block * HALF_FACTOR)):
    Cylinder(radius=r_barrel, height=h_block)

  with Locations((0.0, flange_mid_y, -TOP_DECK_THICK_MM * HALF_FACTOR)):
    Box(FLANGE_WIDTH_X_MM, flange_depth_y, TOP_DECK_THICK_MM)
  with Locations((0.0, flange_mid_y, -h_block + (BASE_FLANGE_THICK_MM * HALF_FACTOR))):
    Box(FLANGE_WIDTH_X_MM, flange_depth_y, BASE_FLANGE_THICK_MM)

  for idx in range(p.num_fins):
    z_fin = -TOP_FIN_CENTER_OFFSET_MM - (idx * fin_pitch)
    if z_fin - (fin_t * HALF_FACTOR) > -h_block + BASE_FLANGE_THICK_MM + FIN_BASE_CLEARANCE_MM:
      with Locations((0.0, FIN_CENTER_Y_MM, z_fin)):
        Box(fin_w, fin_d, fin_t)

  chain_boss_d = CHAIN_BOSS_MAX_Y_MM - CHAIN_BOSS_MIN_Y_MM
  chain_boss_y = (CHAIN_BOSS_MAX_Y_MM + CHAIN_BOSS_MIN_Y_MM) * HALF_FACTOR
  with Locations((0.0, chain_boss_y, -h_block * HALF_FACTOR)):
    Box(CHAIN_BOSS_WIDTH_X_MM, chain_boss_d, h_block)

  for bx, by in HEAD_BOLT_XY_MM:
    with Locations((bx, by, -HEAD_BOLT_BOSS_DEPTH_MM * HALF_FACTOR)):
      Cylinder(radius=HEAD_BOLT_BOSS_RADIUS_MM, height=HEAD_BOLT_BOSS_DEPTH_MM)


def _subtract_block_bores(p: CylinderBlockParams) -> None:
  """Subtract liner bore, chain tunnel slot, head bolt holes, and base reliefs."""
  r_liner_bore = ((p.bore_m * HALF_FACTOR) + p.liner_wall_m) * MM_PER_M + LINER_BORE_CLEARANCE_MM
  h_block = p.block_height_m * MM_PER_M

  with Locations((0.0, 0.0, -h_block * HALF_FACTOR)):
    Cylinder(radius=r_liner_bore, height=h_block + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

  slot_d = CHAIN_SLOT_MAX_Y_MM - CHAIN_SLOT_MIN_Y_MM
  slot_y = (CHAIN_SLOT_MAX_Y_MM + CHAIN_SLOT_MIN_Y_MM) * HALF_FACTOR
  with Locations((0.0, slot_y, -h_block * HALF_FACTOR)):
    Box(CHAIN_SLOT_WIDTH_X_MM, slot_d, h_block + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

  for bx, by in HEAD_BOLT_XY_MM:
    with Locations((bx, by, -HEAD_BOLT_BORE_DEPTH_MM * HALF_FACTOR)):
      Cylinder(radius=HEAD_BOLT_BORE_RADIUS_MM, height=HEAD_BOLT_BORE_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

  relief_bot_z = -h_block + BASE_FLANGE_THICK_MM
  relief_top_z = CUT_MARGIN_MM * HALF_FACTOR
  relief_h = relief_top_z - relief_bot_z
  relief_mid_z = (relief_top_z + relief_bot_z) * HALF_FACTOR
  for bx, by in BASE_BOLT_XY_MM:
    with Locations((bx, by, -h_block * HALF_FACTOR)):
      Cylinder(radius=BASE_BOLT_BORE_RADIUS_MM, height=h_block + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
    with Locations((bx, by, relief_mid_z)):
      Cylinder(radius=BASE_WRENCH_RELIEF_RADIUS_MM, height=relief_h, mode=Mode.SUBTRACT)


def generate_cylinder_block(params: Optional[CylinderBlockParams] = None) -> Part:
  """Generate the finned 6061-T6 cylinder barrel B-Rep solid model.

  Top cylinder deck sits at Z = 0.0 mm and base deck flange at Z = -85.0 mm.
  """
  p = params or CylinderBlockParams()
  with BuildPart() as block_builder:
    _add_block_structure(p)
    _subtract_block_bores(p)
  return block_builder.part


def calculate_block_thermal_capacity(
    params: Optional[CylinderBlockParams] = None,
    air_heat_transfer_coeff: float = DEFAULT_AIR_HTC_W_M2K,
    temp_diff_k: float = DEFAULT_AIR_TEMP_DIFF_K,
    fin_efficiency: float = DEFAULT_FIN_EFFICIENCY,
) -> Dict[str, float]:
  """Calculate fin surface area and air-cooled heat rejection capacity."""
  p = params or CylinderBlockParams()
  r_barrel = (p.bore_m * HALF_FACTOR) + p.liner_wall_m + p.barrel_wall_m
  barrel_area = math.pi * (r_barrel**2)

  fin_face_area = max(0.0, (p.fin_outer_width_m * p.fin_outer_depth_m) - barrel_area)
  fin_perim = 2.0 * (p.fin_outer_width_m + p.fin_outer_depth_m)
  single_fin_area = (2.0 * fin_face_area) + (fin_perim * p.fin_thickness_m)
  fin_surface_area_m2 = p.num_fins * single_fin_area

  exposed_h = max(0.0, p.block_height_m - (p.num_fins * p.fin_thickness_m))
  exposed_barrel_area_m2 = 2.0 * math.pi * r_barrel * exposed_h

  flange_area = max(
      0.0,
      ((FLANGE_WIDTH_X_MM * M_PER_MM) * ((FLANGE_MAX_Y_MM - FLANGE_MIN_Y_MM) * M_PER_MM)) - barrel_area,
  )
  exterior_area_m2 = fin_surface_area_m2 + exposed_barrel_area_m2 + flange_area
  total_effective_area_m2 = (
      exposed_barrel_area_m2 + flange_area + (fin_efficiency * fin_surface_area_m2)
  )
  heat_rejection_w = air_heat_transfer_coeff * total_effective_area_m2 * temp_diff_k
  thermal_margin_percent = ((heat_rejection_w - THERMAL_BUDGET_W) / THERMAL_BUDGET_W) * 100.0

  return {
      "fin_surface_area_m2": fin_surface_area_m2,
      "exposed_barrel_area_m2": exposed_barrel_area_m2,
      "exterior_area_m2": exterior_area_m2,
      "total_effective_area_m2": total_effective_area_m2,
      "heat_rejection_w": heat_rejection_w,
      "thermal_dissipation_w": heat_rejection_w,
      "thermal_margin_percent": thermal_margin_percent,
      "is_adequate": (heat_rejection_w >= THERMAL_BUDGET_W) and (thermal_margin_percent >= MIN_THERMAL_MARGIN_PCT),
  }
