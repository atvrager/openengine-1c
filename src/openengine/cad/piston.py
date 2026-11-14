"""Parametric 4032-T6 Billet Three-Ring Piston Solid Model via build123d.

Generates a watertight B-Rep CAD model of the NestWorks C500 4-axis/3-axis
4032-T6 billet piston:
- 68.0 mm bore diameter, 24.0 mm compression height, 42.0 mm overall height
- Stepped internal cavity below ring belt for >= 2.5 mm ring-groove back-wall
- Three GX200 ring grooves (1.0 x 2.4 mm top, 1.0 x 2.4 mm second, 2.0 x 2.5 mm oil)
- 16.0 mm floating wrist-pin bore with dual DIN 471 circlip grooves (98541A143)
"""

from build123d import (
    Align,
    Box,
    BuildPart,
    Cylinder,
    Locations,
    Mode,
    Part,
    insert,
)

MM_PER_M: float = 1000.0
ROT_X_PIN_DEG: float = 90.0

DEFAULT_BORE_M: float = 0.068
DEFAULT_WRIST_PIN_DIAM_M: float = 0.016
DEFAULT_COMPRESSION_HEIGHT_M: float = 0.024
DEFAULT_TOTAL_HEIGHT_M: float = 0.042
DEFAULT_CROWN_THICKNESS_M: float = 0.0085
DEFAULT_SKIRT_WALL_THICKNESS_M: float = 0.0025

# GX200-RING-68MM groove schedule: (z_center_mm, axial_width_mm, radial_depth_mm)
RING_GROOVES_MM: tuple[tuple[float, float, float], ...] = (
    (17.5, 1.0, 2.4),
    (14.5, 1.0, 2.4),
    (11.2, 2.0, 2.5),
)
RING_BELT_MIN_WALL_MM: float = 5.0
RING_BELT_FLOOR_Z_MM: float = 9.5

PIN_BOSS_INNER_GAP_MM: float = 32.0
PIN_BOSS_OUTER_Y_MM: float = 26.5
PIN_BOSS_OUTER_RADIUS_MM: float = 13.0
PIN_BOSS_OUTER_CLEAR_MM: float = 0.2
CIRCLIP_OFFSET_Y_MM: float = 23.5
CIRCLIP_GROOVE_RADIUS_MM: float = 8.6
CIRCLIP_GROOVE_WIDTH_MM: float = 1.2
PIN_POCKET_WIDTH_X_MM: float = 34.0
PIN_POCKET_HEIGHT_Z_MM: float = 27.0
CUT_MARGIN_MM: float = 2.0
HALF_FACTOR: float = 0.5
MIN_BELT_CAVITY_H_MM: float = 0.5
PIN_BOSS_RADIAL_PAD_MM: float = 4.5
PIN_BOSS_STRUT_WIDTH_RATIO: float = 1.3
POCKET_BELT_CLEARANCE_MM: float = 0.5


def _build_ring_groove_cutter(r_outer_mm: float) -> Part:
  """Build annular ring groove cutter solids safely without nesting BuildPart contexts."""
  annuli: list[Part] = []
  for z_center, width_z, radial_depth in RING_GROOVES_MM:
    r_groove_inner = r_outer_mm - radial_depth
    with BuildPart() as outer_cyl:
      with Locations((0.0, 0.0, z_center)):
        Cylinder(radius=r_outer_mm + CUT_MARGIN_MM, height=width_z)
    with BuildPart() as inner_cyl:
      with Locations((0.0, 0.0, z_center)):
        Cylinder(radius=r_groove_inner, height=width_z + CUT_MARGIN_MM)
    annuli.append(outer_cyl.part - inner_cyl.part)
  cutter = annuli[0]
  for extra in annuli[1:]:
    cutter = cutter.fuse(extra)
  return cutter


def generate_piston(
    bore_m: float = DEFAULT_BORE_M,
    wrist_pin_diam_m: float = DEFAULT_WRIST_PIN_DIAM_M,
    compression_height_m: float = DEFAULT_COMPRESSION_HEIGHT_M,
    total_height_m: float = DEFAULT_TOTAL_HEIGHT_M,
    crown_thickness_m: float = DEFAULT_CROWN_THICKNESS_M,
    skirt_wall_thickness_m: float = DEFAULT_SKIRT_WALL_THICKNESS_M,
) -> Part:
  """Generate the 3D parametric billet piston solid model.

  Center of mass sits close to geometric mid-height Z = 0, placing the crown
  at Z = +21.0 mm, skirt base at Z = -21.0 mm, and wrist-pin axis along Y
  at Z = -3.0 mm.
  """
  r_outer = (bore_m * MM_PER_M) * HALF_FACTOR
  h_total = total_height_m * MM_PER_M
  h_half = h_total * HALF_FACTOR
  z_crown = h_half
  z_pin = z_crown - (compression_height_m * MM_PER_M)
  z_cavity_top = z_crown - (crown_thickness_m * MM_PER_M)

  r_inner_skirt = r_outer - (skirt_wall_thickness_m * MM_PER_M)
  r_inner_belt = min(r_inner_skirt, r_outer - RING_BELT_MIN_WALL_MM)
  belt_cav_h = max(MIN_BELT_CAVITY_H_MM, z_cavity_top - RING_BELT_FLOOR_Z_MM)
  skirt_cav_h = max(1.0, RING_BELT_FLOOR_Z_MM + h_half)

  d_pin = wrist_pin_diam_m * MM_PER_M
  boss_radius_z = max((d_pin * HALF_FACTOR) + PIN_BOSS_RADIAL_PAD_MM, PIN_BOSS_OUTER_RADIUS_MM)
  skirt_y_outer = r_outer - PIN_BOSS_OUTER_CLEAR_MM
  ring_groove_cutter = _build_ring_groove_cutter(r_outer)

  with BuildPart() as piston_builder:
    Cylinder(radius=r_outer, height=h_total)

    # Stepped internal cavity: narrow belt undercut + full skirt hollow
    if belt_cav_h > MIN_BELT_CAVITY_H_MM:
      with Locations((0.0, 0.0, RING_BELT_FLOOR_Z_MM + (belt_cav_h * HALF_FACTOR))):
        Cylinder(radius=r_inner_belt, height=belt_cav_h, mode=Mode.SUBTRACT)

    with Locations((0.0, 0.0, -h_half + (skirt_cav_h * HALF_FACTOR))):
      Cylinder(radius=r_inner_skirt, height=skirt_cav_h, mode=Mode.SUBTRACT)

    # Cylindrical internal wrist-pin bosses and crown-load struts
    with BuildPart(mode=Mode.ADD):
      boss_span_y = PIN_BOSS_OUTER_Y_MM - (PIN_BOSS_INNER_GAP_MM * HALF_FACTOR)
      boss_center_y = (PIN_BOSS_INNER_GAP_MM * HALF_FACTOR) + (boss_span_y * HALF_FACTOR)
      strut_h = (z_crown - z_pin) + 1.0
      strut_mid_z = z_pin + (strut_h * HALF_FACTOR)
      for sy in (-1.0, 1.0):
        with Locations((0.0, sy * boss_center_y, z_pin)):
          Cylinder(radius=boss_radius_z, height=boss_span_y, rotation=(ROT_X_PIN_DEG, 0.0, 0.0))
        with Locations((0.0, sy * boss_center_y, strut_mid_z)):
          Box(boss_radius_z * PIN_BOSS_STRUT_WIDTH_RATIO, boss_span_y, strut_h)
      Cylinder(radius=skirt_y_outer, height=h_total, mode=Mode.INTERSECT)

    # Three GX200-68MM piston ring grooves
    insert(ring_groove_cutter, mode=Mode.SUBTRACT)

    # Y-axis wrist-pin bore and dual DIN 471 circlip grooves
    with Locations((0.0, 0.0, z_pin)):
      Cylinder(
          radius=d_pin * HALF_FACTOR,
          height=r_outer * 2.0 + CUT_MARGIN_MM,
          rotation=(ROT_X_PIN_DEG, 0.0, 0.0),
          mode=Mode.SUBTRACT,
      )
      for sy in (-1.0, 1.0):
        with Locations((0.0, sy * CIRCLIP_OFFSET_Y_MM, 0.0)):
          Cylinder(
              radius=CIRCLIP_GROOVE_RADIUS_MM,
              height=CIRCLIP_GROOVE_WIDTH_MM,
              rotation=(ROT_X_PIN_DEG, 0.0, 0.0),
              mode=Mode.SUBTRACT,
          )

    # Side pin-boss weight-relief pockets outboard of circlip shoulders
    pocket_depth_y = (r_outer - PIN_BOSS_OUTER_Y_MM) + CUT_MARGIN_MM
    pocket_center_y = PIN_BOSS_OUTER_Y_MM + (pocket_depth_y * HALF_FACTOR)
    pocket_top_z = RING_BELT_FLOOR_Z_MM - POCKET_BELT_CLEARANCE_MM
    pocket_bot_z = -h_half - CUT_MARGIN_MM
    pocket_h = pocket_top_z - pocket_bot_z
    pocket_mid_z = (pocket_top_z + pocket_bot_z) * HALF_FACTOR
    for sy in (-1.0, 1.0):
      with Locations((0.0, sy * pocket_center_y, pocket_mid_z)):
        Box(
            PIN_POCKET_WIDTH_X_MM,
            pocket_depth_y,
            pocket_h,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
            mode=Mode.SUBTRACT,
        )

  return piston_builder.part
