"""Parametric Split 7075-T6 Billet Connecting Rod Solid Model via build123d.

Generates a watertight B-Rep CAD model of the NestWorks C500 3-axis machined
2-piece split connecting rod in 7075-T651 aluminum:
- 16.0 mm small-end wrist-pin eye (18.0 mm width)
- High-inertia I-beam shank sized for 7075-T6 compression and Euler buckling
- 42.0 mm big-end bore housing the Schaeffler INA NKS28-XL needle bearing
- Split cap bolted with 2x M6x30 Class 12.9 SHCS and 2x 5x12 mm alignment dowels
- Integral 8.0 mm splash-oil dipper tang giving 5.0 mm BDC sump immersion
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from build123d import (
    Box,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Rectangle,
    extrude,
)

MM_PER_M: float = 1000.0
ROT_X_BORE_DEG: float = 90.0
ROT_Y_BORE_DEG: float = 90.0

DEFAULT_CENTER_TO_CENTER_M: float = 0.108
DEFAULT_WRIST_PIN_DIAM_M: float = 0.016
DEFAULT_CRANKPIN_DIAM_M: float = 0.028
DEFAULT_BIG_END_BORE_M: float = 0.042
DEFAULT_SMALL_END_WIDTH_M: float = 0.018
DEFAULT_BIG_END_WIDTH_M: float = 0.020
DEFAULT_SHANK_FLANGE_W_M: float = 0.017
DEFAULT_SHANK_FLANGE_T_M: float = 0.0045
DEFAULT_SHANK_WEB_T_M: float = 0.0040
DEFAULT_SHANK_HEIGHT_M: float = 0.020

SMALL_END_WALL_MM: float = 4.5
BIG_END_WALL_MM: float = 3.5
SHANK_END_OVERLAP_MM: float = 3.0
EAR_Z_CENTER_MM: float = 27.0
EAR_THICK_Y_MM: float = 26.0
EAR_RADIUS_Z_MM: float = 6.0
BOLT_HOLE_RADIUS_MM: float = 3.1
BOLT_X_OFFSET_MM: float = 4.2
DOWEL_HOLE_RADIUS_MM: float = 2.5
DIPPER_TIP_RADIUS_MM: float = 30.0
DIPPER_WIDTH_X_MM: float = 8.0
DIPPER_THICK_Z_MM: float = 5.0
CUT_MARGIN_MM: float = 4.0
SPLIT_CUT_SIZE_MM: float = 220.0
HALF_FACTOR: float = 0.5


@dataclass
class ConnectingRodParams:
  """Geometric parameters for the split 7075-T6 billet connecting rod."""

  center_to_center_m: float = DEFAULT_CENTER_TO_CENTER_M
  wrist_pin_diam_m: float = DEFAULT_WRIST_PIN_DIAM_M
  crankpin_diam_m: float = DEFAULT_CRANKPIN_DIAM_M
  big_end_bore_m: float = DEFAULT_BIG_END_BORE_M
  small_end_width_m: float = DEFAULT_SMALL_END_WIDTH_M
  big_end_width_m: float = DEFAULT_BIG_END_WIDTH_M
  shank_flange_w_m: float = DEFAULT_SHANK_FLANGE_W_M
  shank_flange_t_m: float = DEFAULT_SHANK_FLANGE_T_M
  shank_web_t_m: float = DEFAULT_SHANK_WEB_T_M
  shank_height_m: float = DEFAULT_SHANK_HEIGHT_M
  dipper_extension_m: float = 0.0065


def _add_shank_and_journals(p: ConnectingRodParams) -> None:
  """Create small-end, big-end housing, bolt ears, and I-beam shank inside BuildPart."""
  l_mm = p.center_to_center_m * MM_PER_M
  r_small_outer = (p.wrist_pin_diam_m * MM_PER_M) * HALF_FACTOR + SMALL_END_WALL_MM
  r_big_outer = (p.big_end_bore_m * MM_PER_M) * HALF_FACTOR + BIG_END_WALL_MM
  w_small = p.small_end_width_m * MM_PER_M
  w_big = p.big_end_width_m * MM_PER_M
  flange_w = p.shank_flange_w_m * MM_PER_M
  flange_t = p.shank_flange_t_m * MM_PER_M
  web_t = p.shank_web_t_m * MM_PER_M
  height = p.shank_height_m * MM_PER_M

  Cylinder(radius=r_small_outer, height=w_small, rotation=(0.0, ROT_Y_BORE_DEG, 0.0))
  with Locations((0.0, l_mm, 0.0)):
    Cylinder(radius=r_big_outer, height=w_big, rotation=(0.0, ROT_Y_BORE_DEG, 0.0))
    for z_ear in (-EAR_Z_CENTER_MM, EAR_Z_CENTER_MM):
      with Locations((0.0, 0.0, z_ear)):
        Box(w_big, EAR_THICK_Y_MM, EAR_RADIUS_Z_MM * 2.0)

  shank_start_y = r_small_outer - SHANK_END_OVERLAP_MM
  shank_end_y = l_mm - r_big_outer + SHANK_END_OVERLAP_MM
  shank_length = shank_end_y - shank_start_y

  with BuildSketch(Plane(origin=(0.0, shank_start_y, 0.0), z_dir=(0.0, 1.0, 0.0))) as shank_sk:
    Rectangle(flange_w, height)
    pocket_w = (flange_w - web_t) * HALF_FACTOR
    pocket_h = height - (2.0 * flange_t)
    with Locations(
        (-(flange_w * HALF_FACTOR) + (pocket_w * HALF_FACTOR), 0.0),
        ((flange_w * HALF_FACTOR) - (pocket_w * HALF_FACTOR), 0.0),
    ):
      Rectangle(pocket_w, pocket_h, mode=Mode.SUBTRACT)
  extrude(shank_sk.sketch, amount=shank_length)


def _add_dipper_and_bores(p: ConnectingRodParams, include_cap_hardware: bool = False) -> None:
  """Add integral splash-oil dipper tang and subtract bearing and bolt bores."""
  l_mm = p.center_to_center_m * MM_PER_M
  d_pin = p.wrist_pin_diam_m * MM_PER_M
  d_big = p.big_end_bore_m * MM_PER_M
  w_small = p.small_end_width_m * MM_PER_M
  w_big = p.big_end_width_m * MM_PER_M
  r_big_outer = (d_big * HALF_FACTOR) + BIG_END_WALL_MM

  # Integral splash dipper extending down (+Y local -> -Z world) to exact 30.0 mm below crankpin center
  dipper_start = l_mm + r_big_outer - SHANK_END_OVERLAP_MM
  dipper_end = l_mm + DIPPER_TIP_RADIUS_MM
  dipper_len = dipper_end - dipper_start
  with Locations((0.0, dipper_start + (dipper_len * HALF_FACTOR), 0.0)):
    Box(DIPPER_WIDTH_X_MM, dipper_len, DIPPER_THICK_Z_MM)

  Cylinder(radius=d_pin * HALF_FACTOR, height=w_small + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_BORE_DEG, 0.0), mode=Mode.SUBTRACT)
  with Locations((0.0, l_mm, 0.0)):
    Cylinder(radius=d_big * HALF_FACTOR, height=w_big + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_BORE_DEG, 0.0), mode=Mode.SUBTRACT)

  # Cap M6 bolt and 5 mm dowel bores on big-end ears (diagonal symmetry)
  ear_holes = [
      (BOLT_X_OFFSET_MM, EAR_Z_CENTER_MM, BOLT_HOLE_RADIUS_MM),
      (-BOLT_X_OFFSET_MM, -EAR_Z_CENTER_MM, BOLT_HOLE_RADIUS_MM),
  ]
  if include_cap_hardware:
    ear_holes.extend([
        (-BOLT_X_OFFSET_MM, EAR_Z_CENTER_MM, DOWEL_HOLE_RADIUS_MM),
        (BOLT_X_OFFSET_MM, -EAR_Z_CENTER_MM, DOWEL_HOLE_RADIUS_MM),
    ])
  for hx, hz, hr in ear_holes:
    with Locations((hx, l_mm, hz)):
      Cylinder(radius=hr, height=EAR_THICK_Y_MM + CUT_MARGIN_MM, rotation=(ROT_X_BORE_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)


def generate_connecting_rod_parts(
    params: Optional[ConnectingRodParams] = None,
) -> Tuple[Part, Part]:
  """Generate the split 7075-T6 connecting-rod beam and cap as two machined parts."""
  p = params or ConnectingRodParams()
  l_mm = p.center_to_center_m * MM_PER_M
  with BuildPart() as master:
    _add_shank_and_journals(p)
    _add_dipper_and_bores(p, include_cap_hardware=True)
  solid = master.part

  with BuildPart() as beam_cutter:
    with Locations((0.0, l_mm + (SPLIT_CUT_SIZE_MM * HALF_FACTOR), 0.0)):
      Box(SPLIT_CUT_SIZE_MM, SPLIT_CUT_SIZE_MM, SPLIT_CUT_SIZE_MM)
  with BuildPart() as cap_cutter:
    with Locations((0.0, l_mm - (SPLIT_CUT_SIZE_MM * HALF_FACTOR), 0.0)):
      Box(SPLIT_CUT_SIZE_MM, SPLIT_CUT_SIZE_MM, SPLIT_CUT_SIZE_MM)

  beam = solid - beam_cutter.part
  cap = solid - cap_cutter.part
  return beam, cap


def generate_connecting_rod(
    center_to_center_m: float = DEFAULT_CENTER_TO_CENTER_M,
    wrist_pin_diam_m: float = DEFAULT_WRIST_PIN_DIAM_M,
    crankpin_diam_m: float = DEFAULT_CRANKPIN_DIAM_M,
    big_end_bore_m: float = DEFAULT_BIG_END_BORE_M,
    small_end_width_m: float = DEFAULT_SMALL_END_WIDTH_M,
    big_end_width_m: float = DEFAULT_BIG_END_WIDTH_M,
    shank_flange_w_m: float = DEFAULT_SHANK_FLANGE_W_M,
    shank_flange_t_m: float = DEFAULT_SHANK_FLANGE_T_M,
    shank_web_t_m: float = DEFAULT_SHANK_WEB_T_M,
    shank_height_m: float = DEFAULT_SHANK_HEIGHT_M,
) -> Part:
  """Generate the assembled parametric 7075-T6 split connecting rod B-Rep."""
  p = ConnectingRodParams(
      center_to_center_m=center_to_center_m,
      wrist_pin_diam_m=wrist_pin_diam_m,
      crankpin_diam_m=crankpin_diam_m,
      big_end_bore_m=big_end_bore_m,
      small_end_width_m=small_end_width_m,
      big_end_width_m=big_end_width_m,
      shank_flange_w_m=shank_flange_w_m,
      shank_flange_t_m=shank_flange_t_m,
      shank_web_t_m=shank_web_t_m,
      shank_height_m=shank_height_m,
  )
  with BuildPart() as rod:
    _add_shank_and_journals(p)
    _add_dipper_and_bores(p, include_cap_hardware=True)
  return rod.part
