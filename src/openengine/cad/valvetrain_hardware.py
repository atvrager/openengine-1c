"""Parametric SOHC Valvetrain Hardware & 4-Axis Camshaft CAD via build123d.

Generates watertight B-Rep CAD models tailored to the NestWorks C500 CNC:
- 6061-T6 SOHC cam tower (Z in [+52, +102] mm, <= 55 mm C500 3-axis limit)
- One-piece 4140 SOHC camshaft with true 4-5-6-7 polynomial lift lobes
  (<= [42, 42, 104] mm 4th-axis rotary limit) + 3-axis 2.5D fallback parts
- 7075-T6 billet intake and exhaust roller rocker arms & 4140 rocker shafts
- SAE 660 bronze valve guides (<= [34, 34, 36] mm 4-axis), 4140 valve seats,
  and 7075-T6 valve spring retainers
"""

from typing import Dict, Tuple
import numpy as np
from build123d import (
    Axis,
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
    loft,
    revolve,
)

MM_PER_M: float = 1000.0
DIAMETER_FACTOR: float = 2.0
CAM_AXIS_Z_MM: float = 76.5
CAM_CENTER_OFFSET_Z_MM: float = 76.5

TOWER_MIN_Z_MM: float = 52.0
TOWER_MAX_Z_MM: float = 106.5
TOWER_WIDTH_X_MM: float = 116.0
TOWER_MIN_Y_MM: float = -30.0
TOWER_MAX_Y_MM: float = 62.0
TOWER_BAY_HALF_X_MM: float = 45.0
TOWER_BAY_HALF_Y_MM: float = 17.5
TOWER_CHAIN_BAY_HALF_X_MM: float = 40.5
TOWER_CHAIN_BAY_MIN_Y_MM: float = 47.0
TOWER_CHAIN_BAY_MAX_Y_MM: float = 58.0

M8_HEAD_BOLT_REAR_XY_MM: Tuple[Tuple[float, float], ...] = (
    (40.0, 40.0),
    (-40.0, 40.0),
)
M8_SOCKET_RELIEF_RADIUS_MM: float = 8.0

CAM_BORE_RADIUS_MM: float = 10.1
ROCKER_PIVOT_X_MM: float = 19.5
ROCKER_PIVOT_Z_MM: float = 100.0
ROCKER_SHAFT_RADIUS_MM: float = 4.0
ROCKER_SHAFT_BORE_RADIUS_MM: float = 4.05
ROCKER_SHAFT_LEN_MM: float = 48.0

TOWER_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (30.0, -22.0),
    (-30.0, -22.0),
    (30.0, 28.0),
    (-30.0, 28.0),
)
TOWER_BOLT_SEAT_Z_MM: float = 72.0
COVER_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (50.5, -24.0),
    (-50.5, -24.0),
    (48.0, 54.5),
    (-48.0, 54.5),
)
COVER_BOLT_TAP_DEPTH_MM: float = 14.0

CAM_BASE_RADIUS_MM: float = 16.0
CAM_LOBE_WIDTH_Y_MM: float = 10.0
CAM_INTAKE_Y_MM: float = -8.0
CAM_EXHAUST_Y_MM: float = 8.0
CAM_FRONT_JOURNAL_START_Y_MM: float = -29.5
CAM_FRONT_JOURNAL_END_Y_MM: float = -20.0
CAM_CENTER_SPACER_HALF_Y_MM: float = 3.0
CAM_REAR_JOURNAL_START_Y_MM: float = 20.0
CAM_REAR_JOURNAL_END_Y_MM: float = 47.0
CAM_NOSE_END_Y_MM: float = 57.5
CAM_JOURNAL_RADIUS_MM: float = 10.0
CAM_NOSE_RADIUS_MM: float = 8.0
CAM_JOURNAL_DIAMETER_M: float = (DIAMETER_FACTOR * CAM_JOURNAL_RADIUS_MM) / MM_PER_M
TOWER_FRONT_JOURNAL_LAND_M: float = (-TOWER_BAY_HALF_Y_MM - TOWER_MIN_Y_MM) / MM_PER_M
CAM_LOBE_SAMPLES: int = 120
CAM_INTAKE_PEAK_CRANK_DEG: float = 467.5
CAM_EXHAUST_PEAK_CRANK_DEG: float = 257.5

VALVE_GUIDE_ID_MM: float = 5.20
VALVE_GUIDE_OD_MM: float = 9.94
VALVE_GUIDE_LENGTH_MM: float = 28.0
VALVE_SEAT_HEIGHT_MM: float = 2.3
VALVE_SEAT_RADIAL_WALL_MM: float = 2.2
SPRING_RETAINER_OD_MM: float = 17.8
SPRING_RETAINER_ID_MM: float = 5.20
SPRING_RETAINER_HEIGHT_MM: float = 4.5

ROCKER_THICK_Y_MM: float = 9.0
ROCKER_BOSS_RADIUS_MM: float = 5.5
ROCKER_ARM_HEIGHT_MM: float = 8.0
ROCKER_FOLLOWER_PAD_RADIUS_MM: float = 2.6
ROCKER_CAM_CONTACT_X_MM: float = 0.0
ROCKER_CAM_CONTACT_Z_MM: float = 93.4
ROCKER_VALVE_TIP_X_MM: float = 35.67
ROCKER_VALVE_TIP_Z_MM: float = 80.3

M6_CLEARANCE_RADIUS_MM: float = 3.2
M6_CB_RADIUS_MM: float = 5.4
M6_TAP_DRILL_RADIUS_MM: float = 2.55
ROT_X_90_DEG: float = 90.0
CUT_MARGIN_MM: float = 4.0
HALF_FACTOR: float = 0.5
ROCKER_SHAFT_CLEAR_MM: float = 12.0
FULL_FOUR_STROKE_DEG: float = 720.0
CAM_CRANK_RATIO: float = 0.5
CAM_CORE_SLIP_CLEAR_MM: float = 0.05
CAM_LOBE_BORE_CUT_LEN_MM: float = 40.0
ROCKER_BRIDGE_Z_OFFSET_MM: float = 4.0
ROCKER_BRIDGE_THICK_Z_MM: float = 5.0
ROCKER_SHOE_Z_OFFSET_MM: float = 6.5
ROCKER_SHOE_WIDTH_X_MM: float = 5.2
ROCKER_SHOE_HEIGHT_Z_MM: float = 7.0
ROCKER_ARM_DIR_X: float = 0.65
ROCKER_ARM_DIR_Y: float = 0.45
ROCKER_ARM_DIR_Z: float = -0.60
ROCKER_ROOT_HALF_W_MM: float = 4.0
ROCKER_TIP_HALF_W_MM: float = 3.5
ROCKER_TIP_SKETCH_Z_OFF_MM: float = 3.0
VALVE_SEAT_IN_OD_RADIUS_MM: float = 15.72
VALVE_SEAT_EX_OD_RADIUS_MM: float = 13.72
SPRING_RETAINER_STEP_RADIUS_MM: float = 5.6


def generate_cam_tower() -> Part:
  """Generate the 6061-T6 SOHC cam tower housing (Z in [+52, +102] mm)."""
  h_tower = TOWER_MAX_Z_MM - TOWER_MIN_Z_MM
  mid_z = (TOWER_MAX_Z_MM + TOWER_MIN_Z_MM) * HALF_FACTOR
  depth_y = TOWER_MAX_Y_MM - TOWER_MIN_Y_MM
  mid_y = (TOWER_MAX_Y_MM + TOWER_MIN_Y_MM) * HALF_FACTOR
  chain_d = TOWER_CHAIN_BAY_MAX_Y_MM - TOWER_CHAIN_BAY_MIN_Y_MM
  chain_y = (TOWER_CHAIN_BAY_MAX_Y_MM + TOWER_CHAIN_BAY_MIN_Y_MM) * HALF_FACTOR

  with BuildPart() as tower:
    with Locations((0.0, mid_y, mid_z)):
      Box(TOWER_WIDTH_X_MM, depth_y, h_tower)

    # Circular socket-access reliefs around rear (+/-40, +40) M8 head bolts
    for rx, ry in M8_HEAD_BOLT_REAR_XY_MM:
      with Locations((rx, ry, mid_z)):
        Cylinder(radius=M8_SOCKET_RELIEF_RADIUS_MM, height=h_tower + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

    # Central valvetrain pocket and +Y #25 timing chain/sprocket bay
    with Locations((0.0, 0.0, mid_z)):
      Box(TOWER_BAY_HALF_X_MM * 2.0, TOWER_BAY_HALF_Y_MM * 2.0, h_tower + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
    with Locations((0.0, chain_y, mid_z)):
      Box(TOWER_CHAIN_BAY_HALF_X_MM * 2.0, chain_d, h_tower + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

    # Coaxial cam journal bore and dual rocker-shaft bores along Y
    bulkhead_len = TOWER_CHAIN_BAY_MIN_Y_MM - TOWER_MIN_Y_MM
    bulkhead_y = (TOWER_CHAIN_BAY_MIN_Y_MM + TOWER_MIN_Y_MM) * HALF_FACTOR
    with Locations((0.0, bulkhead_y, CAM_AXIS_Z_MM)):
      Cylinder(radius=CAM_BORE_RADIUS_MM, height=bulkhead_len + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    for rx in (-ROCKER_PIVOT_X_MM, ROCKER_PIVOT_X_MM):
      with Locations((rx, 0.0, ROCKER_PIVOT_Z_MM)):
        Cylinder(
            radius=ROCKER_SHAFT_BORE_RADIUS_MM,
            height=ROCKER_SHAFT_LEN_MM + ROCKER_SHAFT_CLEAR_MM,
            rotation=(ROT_X_90_DEG, 0.0, 0.0),
            mode=Mode.SUBTRACT,
        )

    # 4x M6x40 head-mounting thru-holes + counterbores and 4x M6 top cover holes
    cb_h = (TOWER_MAX_Z_MM - TOWER_BOLT_SEAT_Z_MM) + CUT_MARGIN_MM
    for bx, by in TOWER_BOLT_XY_MM:
      with Locations((bx, by, mid_z)):
        Cylinder(radius=M6_CLEARANCE_RADIUS_MM, height=h_tower + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
      with Locations((bx, by, TOWER_BOLT_SEAT_Z_MM + (cb_h * HALF_FACTOR))):
        Cylinder(radius=M6_CB_RADIUS_MM, height=cb_h, mode=Mode.SUBTRACT)

    for cx, cy in COVER_BOLT_XY_MM:
      with Locations((cx, cy, TOWER_MAX_Z_MM - (COVER_BOLT_TAP_DEPTH_MM * HALF_FACTOR))):
        Cylinder(radius=M6_TAP_DRILL_RADIUS_MM, height=COVER_BOLT_TAP_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

  return tower.part


def _sample_cam_lobe_polygon(is_intake: bool) -> list[tuple[float, float]]:
  """Sample 2D (X, Z) lobe contour so rotate(Axis.Y, +0.5*CA) aligns active radius at +Z."""
  from openengine.dynamics.sohc_cam import SOHCCamshaft

  cam = SOHCCamshaft()
  params = cam.intake_params if is_intake else cam.exhaust_params
  theta_samples = np.linspace(0.0, FULL_FOUR_STROKE_DEG, CAM_LOBE_SAMPLES, endpoint=False)
  lifts_mm = cam.evaluate_valve_lift_crank_angle(params, theta_samples) * MM_PER_M / params.rocker_ratio

  coords = []
  for th_deg, lift_mm in zip(theta_samples, lifts_mm):
    r_pt = CAM_BASE_RADIUS_MM + float(lift_mm)
    # rotate(Axis.Y, +0.5*CA) rotates gamma -> gamma + 0.5*CA, so gamma = -0.5*CA brings nose to +Z
    gamma_rad = np.radians(-CAM_CRANK_RATIO * float(th_deg))
    px = r_pt * float(np.sin(gamma_rad))
    pz = r_pt * float(np.cos(gamma_rad))
    coords.append((px, pz))
  return coords


def _build_lobe_solid(y_center_mm: float, is_intake: bool) -> Part:
  """Extrude one smooth cam lobe centered at y_center_mm along Y."""
  y_end = y_center_mm + (CAM_LOBE_WIDTH_Y_MM * HALF_FACTOR)
  pts = _sample_cam_lobe_polygon(is_intake=is_intake)
  lobe_plane = Plane(origin=(0.0, y_end, 0.0), x_dir=(1.0, 0.0, 0.0), z_dir=(0.0, -1.0, 0.0))
  with BuildSketch(lobe_plane) as lobe_sk:
    Polygon(pts)
  with BuildPart() as lobe:
    extrude(lobe_sk.sketch, amount=CAM_LOBE_WIDTH_Y_MM)
  return lobe.part


def _add_cam_shaft_cylinders() -> None:
  """Add front journal, center spacer, rear journal, and sprocket nose to active BuildPart."""
  sections = (
      (CAM_FRONT_JOURNAL_START_Y_MM, CAM_FRONT_JOURNAL_END_Y_MM, CAM_JOURNAL_RADIUS_MM),
      (-CAM_CENTER_SPACER_HALF_Y_MM, CAM_CENTER_SPACER_HALF_Y_MM, CAM_JOURNAL_RADIUS_MM),
      (CAM_REAR_JOURNAL_START_Y_MM, CAM_REAR_JOURNAL_END_Y_MM, CAM_JOURNAL_RADIUS_MM),
      (CAM_REAR_JOURNAL_END_Y_MM, CAM_NOSE_END_Y_MM, CAM_NOSE_RADIUS_MM),
  )
  for y0, y1, rad in sections:
    with Locations((0.0, (y0 + y1) * HALF_FACTOR, 0.0)):
      Cylinder(radius=rad, height=y1 - y0, rotation=(ROT_X_90_DEG, 0.0, 0.0))


def generate_sohc_camshaft(has_fourth_axis: bool = True) -> Part:
  """Generate the 4140 SOHC camshaft centered along Y with axis at (0, Y, 0)."""
  _ = has_fourth_axis
  in_lobe = _build_lobe_solid(CAM_INTAKE_Y_MM, is_intake=True)
  ex_lobe = _build_lobe_solid(CAM_EXHAUST_Y_MM, is_intake=False)

  with BuildPart() as shaft_builder:
    _add_cam_shaft_cylinders()
    lobe_span_len = CAM_REAR_JOURNAL_START_Y_MM - CAM_FRONT_JOURNAL_END_Y_MM
    with Locations((0.0, 0.0, 0.0)):
      Cylinder(radius=CAM_NOSE_RADIUS_MM, height=lobe_span_len + 2.0, rotation=(ROT_X_90_DEG, 0.0, 0.0))

  return shaft_builder.part.fuse(in_lobe).fuse(ex_lobe)


def generate_sohc_camshaft_parts(has_fourth_axis: bool = True) -> Dict[str, Part]:
  """Return monolithic 4-axis camshaft or 3-axis 2.5D keyed lobe + ground shaft kit."""
  if has_fourth_axis:
    return {"one_piece_camshaft": generate_sohc_camshaft(has_fourth_axis=True)}

  shaft_len = CAM_NOSE_END_Y_MM - CAM_FRONT_JOURNAL_START_Y_MM
  shaft_mid_y = (CAM_NOSE_END_Y_MM + CAM_FRONT_JOURNAL_START_Y_MM) * HALF_FACTOR
  with BuildPart() as core_shaft:
    with Locations((0.0, shaft_mid_y, 0.0)):
      Cylinder(radius=CAM_NOSE_RADIUS_MM, height=shaft_len, rotation=(ROT_X_90_DEG, 0.0, 0.0))

  in_lobe = _build_lobe_solid(CAM_INTAKE_Y_MM, is_intake=True)
  ex_lobe = _build_lobe_solid(CAM_EXHAUST_Y_MM, is_intake=False)
  with BuildPart() as bore_cut:
    with Locations((0.0, 0.0, 0.0)):
      Cylinder(radius=CAM_NOSE_RADIUS_MM + CAM_CORE_SLIP_CLEAR_MM, height=CAM_LOBE_BORE_CUT_LEN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))

  return {
      "ground_cam_core_shaft": core_shaft.part,
      "intake_lobe_2_5d": in_lobe - bore_cut.part,
      "exhaust_lobe_2_5d": ex_lobe - bore_cut.part,
  }


def generate_rocker_shaft() -> Part:
  """Generate the ground 4140 rocker arm pivot shaft (8.0 mm OD x 48.0 mm)."""
  with BuildPart() as shaft:
    Cylinder(radius=ROCKER_SHAFT_RADIUS_MM, height=ROCKER_SHAFT_LEN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))
  return shaft.part


def generate_rocker_arm(is_intake: bool = True) -> Part:
  """Generate a 7075-T6 billet SOHC rocker arm in world coordinates around its pivot."""
  sign_x = 1.0 if is_intake else -1.0
  lobe_y = CAM_INTAKE_Y_MM if is_intake else CAM_EXHAUST_Y_MM
  pivot_x = sign_x * ROCKER_PIVOT_X_MM
  tip_x = sign_x * ROCKER_VALVE_TIP_X_MM

  with BuildPart() as rocker:
    # Pivot boss centered at (pivot_x, lobe_y, ROCKER_PIVOT_Z_MM)
    with Locations((pivot_x, lobe_y, ROCKER_PIVOT_Z_MM)):
      Cylinder(radius=ROCKER_BOSS_RADIUS_MM, height=ROCKER_THICK_Y_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))

    # Overhead cam-follower bridge arching above cam flanks (Z >= +101 mm) with shoe at X = 0
    bridge_x = sign_x * (ROCKER_PIVOT_X_MM * HALF_FACTOR)
    with Locations((bridge_x, lobe_y, ROCKER_PIVOT_Z_MM + ROCKER_BRIDGE_Z_OFFSET_MM)):
      Box(ROCKER_PIVOT_X_MM, ROCKER_THICK_Y_MM, ROCKER_BRIDGE_THICK_Z_MM)
    with Locations((ROCKER_CAM_CONTACT_X_MM, lobe_y, ROCKER_CAM_CONTACT_Z_MM + ROCKER_SHOE_Z_OFFSET_MM)):
      Box(ROCKER_SHOE_WIDTH_X_MM, ROCKER_THICK_Y_MM, ROCKER_SHOE_HEIGHT_Z_MM)
    with Locations((ROCKER_CAM_CONTACT_X_MM, lobe_y, ROCKER_CAM_CONTACT_Z_MM + ROCKER_FOLLOWER_PAD_RADIUS_MM)):
      Cylinder(radius=ROCKER_FOLLOWER_PAD_RADIUS_MM, height=ROCKER_THICK_Y_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))

    # Diagonal bridge across Y to Y = 0 valve stem tip at (tip_x, 0, ROCKER_VALVE_TIP_Z_MM)
    arm_dir = (sign_x * ROCKER_ARM_DIR_X, -np.sign(lobe_y) * ROCKER_ARM_DIR_Y, ROCKER_ARM_DIR_Z)
    with BuildSketch(Plane(origin=(pivot_x, lobe_y, ROCKER_PIVOT_Z_MM), z_dir=arm_dir)):
      Polygon([
          (-ROCKER_ROOT_HALF_W_MM, -ROCKER_ROOT_HALF_W_MM),
          (ROCKER_ROOT_HALF_W_MM, -ROCKER_ROOT_HALF_W_MM),
          (ROCKER_ROOT_HALF_W_MM, ROCKER_ROOT_HALF_W_MM),
          (-ROCKER_ROOT_HALF_W_MM, ROCKER_ROOT_HALF_W_MM),
      ])
    with BuildSketch(Plane(origin=(tip_x, 0.0, ROCKER_VALVE_TIP_Z_MM + ROCKER_TIP_SKETCH_Z_OFF_MM), z_dir=arm_dir)):
      Polygon([
          (-ROCKER_TIP_HALF_W_MM, -ROCKER_TIP_HALF_W_MM),
          (ROCKER_TIP_HALF_W_MM, -ROCKER_TIP_HALF_W_MM),
          (ROCKER_TIP_HALF_W_MM, ROCKER_TIP_HALF_W_MM),
          (-ROCKER_TIP_HALF_W_MM, ROCKER_TIP_HALF_W_MM),
      ])
    loft()

    with Locations((tip_x, 0.0, ROCKER_VALVE_TIP_Z_MM + 2.0)):
      Cylinder(radius=ROCKER_TIP_HALF_W_MM, height=ROCKER_ROOT_HALF_W_MM)

    # Subtract rocker-shaft bore
    with Locations((pivot_x, lobe_y, ROCKER_PIVOT_Z_MM)):
      Cylinder(radius=ROCKER_SHAFT_BORE_RADIUS_MM, height=ROCKER_THICK_Y_MM + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

  return rocker.part


def generate_valve_guide() -> Part:
  """Generate a coaxial SAE 660 bearing bronze valve guide (<= [34, 34, 36] mm)."""
  r_outer = VALVE_GUIDE_OD_MM * HALF_FACTOR
  r_inner = VALVE_GUIDE_ID_MM * HALF_FACTOR
  with BuildSketch(Plane.XZ) as profile:
    Polygon([
        (r_inner, 0.0),
        (r_outer, 0.0),
        (r_outer, VALVE_GUIDE_LENGTH_MM),
        (r_inner, VALVE_GUIDE_LENGTH_MM),
    ])
  with BuildPart() as guide:
    revolve(profile.sketch, axis=Axis.Z)
  return guide.part


def generate_valve_seat(is_intake: bool = True) -> Part:
  """Generate a coaxial 4140 intake (31.4 mm OD) or exhaust (27.4 mm OD) valve seat ring."""
  r_outer = VALVE_SEAT_IN_OD_RADIUS_MM if is_intake else VALVE_SEAT_EX_OD_RADIUS_MM
  r_inner = r_outer - VALVE_SEAT_RADIAL_WALL_MM
  with BuildSketch(Plane.XZ) as profile:
    Polygon([
        (r_inner, 0.0),
        (r_outer, 0.0),
        (r_outer, VALVE_SEAT_HEIGHT_MM),
        (r_inner, VALVE_SEAT_HEIGHT_MM),
    ])
  with BuildPart() as seat:
    revolve(profile.sketch, axis=Axis.Z)
  return seat.part


def generate_spring_retainer() -> Part:
  """Generate a stepped 7075-T6 valve spring retainer (<= [76, 76, 18] mm)."""
  r_outer = SPRING_RETAINER_OD_MM * HALF_FACTOR
  r_inner = SPRING_RETAINER_ID_MM * HALF_FACTOR
  r_step = SPRING_RETAINER_STEP_RADIUS_MM
  h_total = SPRING_RETAINER_HEIGHT_MM
  h_flange = 2.0
  with BuildSketch(Plane.XZ) as profile:
    Polygon([
        (r_inner, 0.0),
        (r_step, 0.0),
        (r_step, h_total - h_flange),
        (r_outer, h_total - h_flange),
        (r_outer, h_total),
        (r_inner, h_total),
    ])
  with BuildPart() as retainer:
    revolve(profile.sketch, axis=Axis.Z)
  return retainer.part
