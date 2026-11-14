"""Parametric Split Crankcase, Sloped Sump Pan, and 0.65 L Oil Pool via build123d.

Generates watertight B-Rep CAD models for the NestWorks C500 3-axis machined
6061-T6 crankcase halves and bolt-on sloped wet-sump pan in master assembly
world coordinates (Z = 0 cylinder deck, crank axis at (0, 0, -159.0) mm):
- Left & right crankcase halves fit <= [52.0, 160.0, 178.0] mm C500 limits
- Spigot counterbore clears 105 mm honed DOM liner (Z in [-105, 0] mm)
- Dual 6206-2Z bearing pockets and 30x47x7 mm Viton rotary shaft seal pockets
- 45-deg M20x1.5 oil fill port, 1/8-27 NPT breather, M16x1.5 sight glass at Z = -211 mm
- Separate 37 mm sloped sump pan with 3.3-deg drain floor, Viton O-ring groove,
  and -X M12x1.5 magnetic drain port
- 0.65 L SAE 10W-40 static oil pool solid with top surface at Z = -211.0 mm
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from build123d import (
    Align,
    Box,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    extrude,
)

from openengine.cad.step_export import heal_part_edges

CRANK_AXIS_Z_MM: float = -159.0
BLOCK_BASE_Z_MM: float = -85.0
SUMP_SPLIT_Z_MM: float = -241.0
SUMP_BOTTOM_Z_MM: float = -278.0
OIL_LEVEL_Z_MM: float = -211.0

CASE_WIDTH_X_MM: float = 172.0
CASE_HALF_Y_MM: float = 52.0
LINER_SPIGOT_RADIUS_MM: float = 37.2
LINER_SPIGOT_BOTTOM_Z_MM: float = -106.5
UPPER_THROAT_WIDTH_X_MM: float = 90.0
UPPER_THROAT_HALF_Y_MM: float = 26.5

SWEEP_RADIUS_MM: float = 67.0
SWEEP_HALF_Y_MM: float = 26.5
OIL_BAY_WIDTH_X_MM: float = 140.0
OIL_BAY_HALF_Y_MM: float = 41.0

MAIN_THRU_RADIUS_MM: float = 17.0
BEARING_6206_RADIUS_MM: float = 31.1
BEARING_POCKET_MIN_Y_MM: float = 26.0
BEARING_POCKET_MAX_Y_MM: float = 43.0
LEFT_SEAL_RADIUS_MM: float = 23.6
LEFT_SEAL_MIN_Y_MM: float = 43.5
LEFT_SEAL_MAX_Y_MM: float = 52.5

TIMING_BAY_WIDTH_X_MM: float = 82.0
TIMING_BAY_MIN_Y_MM: float = 47.0
TIMING_BAY_MAX_Y_MM: float = 52.5
TIMING_BAY_BOTTOM_Z_MM: float = -192.0
TIMING_COVER_BOLT_XZ_MM: Tuple[Tuple[float, float], ...] = (
    (44.5, -94.0),
    (-44.5, -94.0),
    (44.5, -142.0),
    (-44.5, -142.0),
    (44.5, -188.0),
    (-44.5, -188.0),
)
TIMING_COVER_TAP_DEPTH_Y_MM: float = 10.0

BARREL_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (51.5, 26.0),
    (-51.5, 26.0),
    (51.5, -26.0),
    (-51.5, -26.0),
)
M8_TAP_DRILL_RADIUS_MM: float = 3.4
M8_TAP_DEPTH_MM: float = 24.0

SPLIT_BOLT_XZ_MM: Tuple[Tuple[float, float], ...] = (
    (78.0, -92.0),
    (-78.0, -92.0),
    (78.0, -135.0),
    (-78.0, -135.0),
    (78.0, -178.0),
    (-78.0, -178.0),
    (78.0, -228.0),
    (-78.0, -228.0),
)
SPLIT_DOWEL_XZ_MM: Tuple[Tuple[float, float], ...] = (
    (78.0, -159.0),
    (-78.0, -159.0),
)
SPLIT_CB_START_Y_MM: float = -21.0
SPLIT_TAP_DEPTH_Y_MM: float = 20.0
SPLIT_TAP_OVERSHOOT_Y_MM: float = 1.0
DOWEL_HALF_DEPTH_Y_MM: float = 9.0

SUMP_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (-65.0, -48.2),
    (-22.0, -48.2),
    (22.0, -48.2),
    (65.0, -48.2),
    (-65.0, 48.2),
    (-22.0, 48.2),
    (22.0, 48.2),
    (65.0, 48.2),
    (-79.0, 24.0),
    (79.0, 24.0),
)
SUMP_TAP_DEPTH_MM: float = 16.0
SUMP_BOLT_FLANGE_THICK_MM: float = 9.0
SUMP_ORING_CENTER_X_MM: float = 73.0
SUMP_ORING_CENTER_Y_MM: float = 43.5
SUMP_ORING_WIDTH_MM: float = 2.6
SUMP_ORING_DEPTH_MM: float = 1.9

SUMP_FLOOR_LEFT_Z_MM: float = -271.5
SUMP_FLOOR_RIGHT_Z_MM: float = -263.5
SUMP_FIN_THICK_MM: float = 2.5
SUMP_FIN_SPACING_X_MM: float = 16.0
SUMP_FIN_COUNT_HALF: int = 4

FILL_PORT_Y_MM: float = 30.0
FILL_PORT_Z_MM: float = -114.0
FILL_PORT_TILT_DEG: float = 45.0
FILL_PORT_BORE_RADIUS_MM: float = 9.3
FILL_PORT_SPOTFACE_RADIUS_MM: float = 13.0

SIGHT_GLASS_Y_MM: float = 30.0
SIGHT_GLASS_Z_MM: float = OIL_LEVEL_Z_MM
SIGHT_GLASS_BORE_RADIUS_MM: float = 7.3
SIGHT_GLASS_CB_RADIUS_MM: float = 11.0

BREATHER_Y_MM: float = -34.0
BREATHER_Z_MM: float = -105.0
BREATHER_BORE_RADIUS_MM: float = 4.3

DRAIN_PORT_Y_MM: float = 0.0
DRAIN_PORT_Z_MM: float = -267.5
DRAIN_BORE_RADIUS_MM: float = 5.3
DRAIN_SPOTFACE_RADIUS_MM: float = 9.0

M6_CLEARANCE_RADIUS_MM: float = 3.2
M6_TAP_DRILL_RADIUS_MM: float = 2.55
M6_CB_RADIUS_MM: float = 5.4
DOWEL_6MM_RADIUS_MM: float = 3.05

ROT_X_90_DEG: float = 90.0
ROT_Y_90_DEG: float = 90.0
CUT_MARGIN_MM: float = 4.0
OIL_POOL_WALL_CLEAR_MM: float = 0.2
HALF_FACTOR: float = 0.5
FILL_PORT_X_INSET_MM: float = 12.0
FILL_PORT_BORE_LEN_MM: float = 44.0
FILL_PORT_SF_X_OUTSET_MM: float = 3.0
FILL_PORT_SF_Z_OFFSET_MM: float = 15.0
FILL_PORT_SF_LEN_MM: float = 28.0
SPOTFACE_OFFSET_X_MM: float = 1.5
SPOTFACE_CUT_DEPTH_MM: float = 5.0
SUMP_FIN_EDGE_MARGIN_MM: float = 8.0
SUMP_FIN_CHANNEL_W_MM: float = 6.0


@dataclass
class CrankcaseParams:
  """Geometric parameters for the C500 split crankcase and wet-sump pan."""

  case_width_x_mm: float = CASE_WIDTH_X_MM
  case_half_y_mm: float = CASE_HALF_Y_MM
  sweep_radius_mm: float = SWEEP_RADIUS_MM
  oil_bay_width_x_mm: float = OIL_BAY_WIDTH_X_MM
  oil_bay_half_y_mm: float = OIL_BAY_HALF_Y_MM
  oil_level_z_mm: float = OIL_LEVEL_Z_MM


def _subtract_common_cavities(p: CrankcaseParams, y_sign: float) -> None:
  """Subtract shared crank sweep, throat, liner spigot, and lower oil bay."""
  bay_y_center = y_sign * (p.oil_bay_half_y_mm * HALF_FACTOR)
  sweep_y_center = y_sign * (SWEEP_HALF_Y_MM * HALF_FACTOR)
  throat_h = CRANK_AXIS_Z_MM - BLOCK_BASE_Z_MM

  # Lower wet-sump bay connecting sweep chamber to bolt-on sump pan
  oil_bay_z_h = CRANK_AXIS_Z_MM - SUMP_SPLIT_Z_MM
  with Locations((0.0, bay_y_center, SUMP_SPLIT_Z_MM + (oil_bay_z_h * HALF_FACTOR))):
    Box(p.oil_bay_width_x_mm, p.oil_bay_half_y_mm + CUT_MARGIN_MM, oil_bay_z_h + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

  # Upper crank/rod throat and DOM liner spigot bore
  throat_mid_z = (BLOCK_BASE_Z_MM + CRANK_AXIS_Z_MM) * HALF_FACTOR
  with Locations((0.0, y_sign * (UPPER_THROAT_HALF_Y_MM * HALF_FACTOR), throat_mid_z)):
    Box(UPPER_THROAT_WIDTH_X_MM, UPPER_THROAT_HALF_Y_MM + CUT_MARGIN_MM, abs(throat_h) + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

  spigot_h = BLOCK_BASE_Z_MM - LINER_SPIGOT_BOTTOM_Z_MM
  with Locations((0.0, 0.0, BLOCK_BASE_Z_MM - (spigot_h * HALF_FACTOR))):
    Cylinder(radius=LINER_SPIGOT_RADIUS_MM, height=spigot_h + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

  # Circular crank web + big-end sweep pocket
  with Locations((0.0, sweep_y_center, CRANK_AXIS_Z_MM)):
    Cylinder(
        radius=p.sweep_radius_mm,
        height=SWEEP_HALF_Y_MM + CUT_MARGIN_MM,
        rotation=(ROT_X_90_DEG, 0.0, 0.0),
        mode=Mode.SUBTRACT,
    )

  # Barrel base hold-down tapped holes and bottom M6 sump tapped holes
  for bx, by in BARREL_BOLT_XY_MM:
    if (by * y_sign) > 0.0:
      with Locations((bx, by, BLOCK_BASE_Z_MM - (M8_TAP_DEPTH_MM * HALF_FACTOR))):
        Cylinder(radius=M8_TAP_DRILL_RADIUS_MM, height=M8_TAP_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

  for sx, sy in SUMP_BOLT_XY_MM:
    if (sy * y_sign) > 0.0:
      with Locations((sx, sy, SUMP_SPLIT_Z_MM + (SUMP_TAP_DEPTH_MM * HALF_FACTOR))):
        Cylinder(radius=M6_TAP_DRILL_RADIUS_MM, height=SUMP_TAP_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

  # Alignment dowel bores straddling Y = 0 split line
  for dx, dz in SPLIT_DOWEL_XZ_MM:
    with Locations((dx, y_sign * (DOWEL_HALF_DEPTH_Y_MM * HALF_FACTOR), dz)):
      Cylinder(
          radius=DOWEL_6MM_RADIUS_MM,
          height=DOWEL_HALF_DEPTH_Y_MM + CUT_MARGIN_MM,
          rotation=(ROT_X_90_DEG, 0.0, 0.0),
          mode=Mode.SUBTRACT,
      )


def generate_crankcase_left(params: Optional[CrankcaseParams] = None) -> Part:
  """Generate the left (-Y flywheel side) 6061-T6 billet crankcase half."""
  p = params or CrankcaseParams()
  case_h = BLOCK_BASE_Z_MM - SUMP_SPLIT_Z_MM
  case_mid_z = (BLOCK_BASE_Z_MM + SUMP_SPLIT_Z_MM) * HALF_FACTOR
  x_wall_center = -(p.case_width_x_mm * HALF_FACTOR)

  with BuildPart() as left_case:
    with Locations((0.0, -p.case_half_y_mm * HALF_FACTOR, case_mid_z)):
      Box(p.case_width_x_mm, p.case_half_y_mm, case_h)

    _subtract_common_cavities(p, y_sign=-1.0)

    # Left main thru-bore, 6206-2Z bearing pocket, and outer Viton seal pocket
    with Locations((0.0, -p.case_half_y_mm * HALF_FACTOR, CRANK_AXIS_Z_MM)):
      Cylinder(radius=MAIN_THRU_RADIUS_MM, height=p.case_half_y_mm + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    brg_h = BEARING_POCKET_MAX_Y_MM - BEARING_POCKET_MIN_Y_MM
    brg_y = -(BEARING_POCKET_MIN_Y_MM + BEARING_POCKET_MAX_Y_MM) * HALF_FACTOR
    with Locations((0.0, brg_y, CRANK_AXIS_Z_MM)):
      Cylinder(radius=BEARING_6206_RADIUS_MM, height=brg_h, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    seal_h = LEFT_SEAL_MAX_Y_MM - LEFT_SEAL_MIN_Y_MM
    seal_y = -(LEFT_SEAL_MIN_Y_MM + LEFT_SEAL_MAX_Y_MM) * HALF_FACTOR
    with Locations((0.0, seal_y, CRANK_AXIS_Z_MM)):
      Cylinder(radius=LEFT_SEAL_RADIUS_MM, height=seal_h, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    # Split-case M6x40 clearance holes + outer M6 SHCS counterbores
    cb_len = p.case_half_y_mm + SPLIT_CB_START_Y_MM + 2.0
    cb_y = SPLIT_CB_START_Y_MM - (cb_len * HALF_FACTOR)
    for sx, sz in SPLIT_BOLT_XZ_MM:
      with Locations((sx, -p.case_half_y_mm * HALF_FACTOR, sz)):
        Cylinder(radius=M6_CLEARANCE_RADIUS_MM, height=p.case_half_y_mm + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)
      with Locations((sx, cb_y, sz)):
        Cylinder(radius=M6_CB_RADIUS_MM, height=cb_len, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    # 1/8-27 NPT crankcase breather vent on -X upper wall
    wall_thick = (p.case_width_x_mm - p.oil_bay_width_x_mm) * HALF_FACTOR
    with Locations((x_wall_center + (wall_thick * HALF_FACTOR), BREATHER_Y_MM, BREATHER_Z_MM)):
      Cylinder(radius=BREATHER_BORE_RADIUS_MM, height=wall_thick + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)

  return left_case.part


def generate_crankcase_right(params: Optional[CrankcaseParams] = None) -> Part:
  """Generate the right (+Y PTO/timing side) 6061-T6 billet crankcase half."""
  p = params or CrankcaseParams()
  case_h = BLOCK_BASE_Z_MM - SUMP_SPLIT_Z_MM
  case_mid_z = (BLOCK_BASE_Z_MM + SUMP_SPLIT_Z_MM) * HALF_FACTOR
  x_wall_outer = p.case_width_x_mm * HALF_FACTOR

  with BuildPart() as right_case:
    with Locations((0.0, p.case_half_y_mm * HALF_FACTOR, case_mid_z)):
      Box(p.case_width_x_mm, p.case_half_y_mm, case_h)

    _subtract_common_cavities(p, y_sign=1.0)

    # Right main thru-bore, 6206-2Z bearing pocket, and recessed #25 timing chain bay
    with Locations((0.0, p.case_half_y_mm * HALF_FACTOR, CRANK_AXIS_Z_MM)):
      Cylinder(radius=MAIN_THRU_RADIUS_MM, height=p.case_half_y_mm + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    brg_h = BEARING_POCKET_MAX_Y_MM - BEARING_POCKET_MIN_Y_MM
    brg_y = (BEARING_POCKET_MIN_Y_MM + BEARING_POCKET_MAX_Y_MM) * HALF_FACTOR
    with Locations((0.0, brg_y, CRANK_AXIS_Z_MM)):
      Cylinder(radius=BEARING_6206_RADIUS_MM, height=brg_h, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    timing_d = TIMING_BAY_MAX_Y_MM - TIMING_BAY_MIN_Y_MM
    timing_y = (TIMING_BAY_MIN_Y_MM + TIMING_BAY_MAX_Y_MM) * HALF_FACTOR
    timing_h = BLOCK_BASE_Z_MM - TIMING_BAY_BOTTOM_Z_MM
    timing_mid_z = (BLOCK_BASE_Z_MM + TIMING_BAY_BOTTOM_Z_MM) * HALF_FACTOR
    with Locations((0.0, timing_y, timing_mid_z)):
      Box(TIMING_BAY_WIDTH_X_MM, timing_d, timing_h + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

    # Tapped M6 split-line bores receiving M6x40 screws from left half
    split_tap_y_center = (SPLIT_TAP_DEPTH_Y_MM - SPLIT_TAP_OVERSHOOT_Y_MM) * HALF_FACTOR
    split_tap_len = SPLIT_TAP_DEPTH_Y_MM + SPLIT_TAP_OVERSHOOT_Y_MM
    for sx, sz in SPLIT_BOLT_XZ_MM:
      with Locations((sx, split_tap_y_center, sz)):
        Cylinder(radius=M6_TAP_DRILL_RADIUS_MM, height=split_tap_len, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    for tx, tz in TIMING_COVER_BOLT_XZ_MM:
      with Locations((tx, p.case_half_y_mm - (TIMING_COVER_TAP_DEPTH_Y_MM * HALF_FACTOR), tz)):
        Cylinder(radius=M6_TAP_DRILL_RADIUS_MM, height=TIMING_COVER_TAP_DEPTH_Y_MM + 1.0, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    # 45-deg M20x1.5 O-ring oil fill port recessed into +X upper wall
    with Locations((x_wall_outer - FILL_PORT_X_INSET_MM, FILL_PORT_Y_MM, FILL_PORT_Z_MM)):
      Cylinder(radius=FILL_PORT_BORE_RADIUS_MM, height=FILL_PORT_BORE_LEN_MM, rotation=(0.0, FILL_PORT_TILT_DEG, 0.0), mode=Mode.SUBTRACT)
    with Locations((x_wall_outer + FILL_PORT_SF_X_OUTSET_MM, FILL_PORT_Y_MM, FILL_PORT_Z_MM + FILL_PORT_SF_Z_OFFSET_MM)):
      Cylinder(radius=FILL_PORT_SPOTFACE_RADIUS_MM, height=FILL_PORT_SF_LEN_MM, rotation=(0.0, FILL_PORT_TILT_DEG, 0.0), mode=Mode.SUBTRACT)

    # M16x1.5 sight-glass port centered directly at static oil level Z = -211.0 mm
    wall_thick = (p.case_width_x_mm - p.oil_bay_width_x_mm) * HALF_FACTOR
    with Locations((x_wall_outer - (wall_thick * HALF_FACTOR), SIGHT_GLASS_Y_MM, SIGHT_GLASS_Z_MM)):
      Cylinder(radius=SIGHT_GLASS_BORE_RADIUS_MM, height=wall_thick + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)
    with Locations((x_wall_outer - SPOTFACE_OFFSET_X_MM, SIGHT_GLASS_Y_MM, SIGHT_GLASS_Z_MM)):
      Cylinder(radius=SIGHT_GLASS_CB_RADIUS_MM, height=SPOTFACE_CUT_DEPTH_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)

  return heal_part_edges(right_case.part)


def _build_sloped_sump_cavity(p: CrankcaseParams) -> Part:
  """Build sloped internal wet-sump volume draining down toward -X magnetic plug."""
  x_half = p.oil_bay_width_x_mm * HALF_FACTOR
  y_half = p.oil_bay_half_y_mm
  with BuildSketch(Plane.XZ) as profile:
    Polygon([
        (-x_half, SUMP_FLOOR_LEFT_Z_MM),
        (x_half, SUMP_FLOOR_RIGHT_Z_MM),
        (x_half, SUMP_SPLIT_Z_MM + CUT_MARGIN_MM),
        (-x_half, SUMP_SPLIT_Z_MM + CUT_MARGIN_MM),
    ])
  with BuildPart() as cavity:
    extrude(profile.sketch, amount=y_half, both=True)
  return cavity.part


def generate_sump_pan(params: Optional[CrankcaseParams] = None) -> Part:
  """Generate the 37 mm bolt-on 6061-T6 sloped wet-sump pan with Viton O-ring groove."""
  p = params or CrankcaseParams()
  sump_h = SUMP_SPLIT_Z_MM - SUMP_BOTTOM_Z_MM
  sump_mid_z = (SUMP_SPLIT_Z_MM + SUMP_BOTTOM_Z_MM) * HALF_FACTOR
  cavity = _build_sloped_sump_cavity(p)

  with BuildPart() as sump:
    with Locations((0.0, 0.0, sump_mid_z)):
      Box(p.case_width_x_mm, p.case_half_y_mm * 2.0, sump_h)

    with BuildPart(mode=Mode.SUBTRACT):
      from build123d import insert
      insert(cavity)

    # Perimeter 2.5 mm Viton O-ring groove along top sealing face (Z = -241.0)
    grooves = (
        (0.0, SUMP_ORING_CENTER_Y_MM, SUMP_ORING_CENTER_X_MM * 2.0 + SUMP_ORING_WIDTH_MM, SUMP_ORING_WIDTH_MM),
        (0.0, -SUMP_ORING_CENTER_Y_MM, SUMP_ORING_CENTER_X_MM * 2.0 + SUMP_ORING_WIDTH_MM, SUMP_ORING_WIDTH_MM),
        (SUMP_ORING_CENTER_X_MM, 0.0, SUMP_ORING_WIDTH_MM, SUMP_ORING_CENTER_Y_MM * 2.0 - SUMP_ORING_WIDTH_MM),
        (-SUMP_ORING_CENTER_X_MM, 0.0, SUMP_ORING_WIDTH_MM, SUMP_ORING_CENTER_Y_MM * 2.0 - SUMP_ORING_WIDTH_MM),
    )
    for gx, gy, gw, gd in grooves:
      with Locations((gx, gy, SUMP_SPLIT_Z_MM - (SUMP_ORING_DEPTH_MM * HALF_FACTOR))):
        Box(gw, gd, SUMP_ORING_DEPTH_MM + HALF_FACTOR, mode=Mode.SUBTRACT)

    # -X side M12x1.5 gravity magnetic drain port + crush-washer spotface
    wall_thick = (p.case_width_x_mm - p.oil_bay_width_x_mm) * HALF_FACTOR
    with Locations((-(p.case_width_x_mm * HALF_FACTOR) + (wall_thick * HALF_FACTOR), DRAIN_PORT_Y_MM, DRAIN_PORT_Z_MM)):
      Cylinder(radius=DRAIN_BORE_RADIUS_MM, height=wall_thick + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)
    with Locations((-(p.case_width_x_mm * HALF_FACTOR) + SPOTFACE_OFFSET_X_MM, DRAIN_PORT_Y_MM, DRAIN_PORT_Z_MM)):
      Cylinder(radius=DRAIN_SPOTFACE_RADIUS_MM, height=SPOTFACE_CUT_DEPTH_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)

    # Bottom cooling fin channels and 10x M6 mounting thru-holes + underside spotfaces
    fin_span_y = (p.oil_bay_half_y_mm * 2.0) - CUT_MARGIN_MM
    fin_span_x = p.oil_bay_width_x_mm - SUMP_FIN_EDGE_MARGIN_MM
    for idx in range(-SUMP_FIN_COUNT_HALF, SUMP_FIN_COUNT_HALF + 1):
      fx = idx * SUMP_FIN_SPACING_X_MM
      if abs(fx) < (fin_span_x * HALF_FACTOR):
        with Locations((fx, 0.0, SUMP_BOTTOM_Z_MM + (SUMP_FIN_THICK_MM * HALF_FACTOR))):
          Box(SUMP_FIN_CHANNEL_W_MM, fin_span_y, SUMP_FIN_THICK_MM + HALF_FACTOR, mode=Mode.SUBTRACT)

    cb_h = sump_h - SUMP_BOLT_FLANGE_THICK_MM
    for sx, sy in SUMP_BOLT_XY_MM:
      with Locations((sx, sy, sump_mid_z)):
        Cylinder(radius=M6_CLEARANCE_RADIUS_MM, height=sump_h + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
      with Locations((sx, sy, SUMP_BOTTOM_Z_MM + (cb_h * HALF_FACTOR))):
        Cylinder(radius=M6_CB_RADIUS_MM, height=cb_h, mode=Mode.SUBTRACT)

  return sump.part


def generate_sump_oil_pool(params: Optional[CrankcaseParams] = None) -> Part:
  """Generate the 0.65 L SAE 10W-40 static oil pool solid with top at Z = -211.0 mm."""
  p = params or CrankcaseParams()
  x_half = (p.oil_bay_width_x_mm * HALF_FACTOR) - OIL_POOL_WALL_CLEAR_MM
  y_half = p.oil_bay_half_y_mm - OIL_POOL_WALL_CLEAR_MM
  z_left = SUMP_FLOOR_LEFT_Z_MM + OIL_POOL_WALL_CLEAR_MM
  z_right = SUMP_FLOOR_RIGHT_Z_MM + OIL_POOL_WALL_CLEAR_MM

  with BuildSketch(Plane.XZ) as oil_profile:
    Polygon([
        (-x_half, z_left),
        (x_half, z_right),
        (x_half, p.oil_level_z_mm),
        (-x_half, p.oil_level_z_mm),
    ])
  with BuildPart() as pool:
    extrude(oil_profile.sketch, amount=y_half, both=True)
  return pool.part
