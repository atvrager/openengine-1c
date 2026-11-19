"""3D-printable snap-fit rotating & reciprocating kit parts via build123d.

Generates all 10 unique printable STLs (11 physical pieces) in scaled world
assembly coordinates (origin = cylinder top deck Z = 0, crank axis at
(0, 0, -159.0 * scale)), posed kinematically at ``crank_angle_deg``.
"""

import math
from typing import Dict, List

from build123d import (
    Axis,
    Box,
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Polyline,
    Rectangle,
    Shape,
    extrude,
    insert,
    make_face,
    make_hull,
    revolve,
)

from openengine.cad.assembly import EngineAssembly
from openengine.cad.covers_and_aux import (
    FLYWHEEL_OUTER_RADIUS_MM,
    FLYWHEEL_RIM_INNER_RADIUS_MM,
    TRIGGER_ROOT_RADIUS_MM,
    TRIGGER_TIP_RADIUS_MM,
)
from openengine.cad.crankcase import CRANK_AXIS_Z_MM
from openengine.cad.crankshaft import (
    CRANKPIN_BOSS_MARGIN_MM,
    CW_CUT_HEIGHT_MM,
    CW_CUT_OFFSET_MM,
    CW_CUT_WIDTH_MM,
    MAIN_BOSS_MARGIN_MM,
)
from openengine.printing.primitives import (
    make_alignment_hole_cutter,
    make_alignment_pin,
    make_cantilever_snap_hook,
    make_keyed_bore_cutter,
    make_keyed_shaft_solid,
    make_snap_pocket_cutter,
    make_teardrop_bore_cutter,
)
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    SnapFitDesign,
    auto_size_cantilever_snap,
    min_printable_wall_mm,
    running_clearance_radial_mm,
    slip_clearance_radial_mm,
    snap_clearance_mm,
)

HALF_FACTOR: float = 0.5
ROT_90_DEG: float = 90.0
ROT_180_DEG: float = 180.0
CUT_MARGIN_MM: float = 2.0
FULL_CIRCLE_DEG: float = 360.0
MIN_GEOM_EPS_MM: float = 1.0e-4

# Crankshaft reference dimensions (1:1 mm)
CRANK_THROW_MM: float = 27.0
CRANKPIN_DIAM_MM: float = 28.0
CRANKPIN_GAP_Y_MM: float = 21.0
CRANKPIN_BORE_DIAM_MM: float = 15.0
WEB_THICKNESS_MM: float = 14.5
COUNTERWEIGHT_RADIUS_MM: float = 47.8
MAIN_JOURNAL_DIAM_MM: float = 30.0
LEFT_MAIN_LEN_MM: float = 28.0
RIGHT_MAIN_LEN_MM: float = 18.0
FLYWHEEL_HUB_DIAM_MM: float = 20.0
FLYWHEEL_HUB_LEN_MM: float = 34.0
NOSE_DIAM_MM: float = 12.0
NOSE_LEN_MM: float = 10.0
RIGHT_OUTER_SHAFT_END_Y_MM: float = 96.0
CRANKPIN_WEB_SOCKET_MM: float = 9.0
CRANKPIN_THRU_R_MM: float = 10.5
CRANK_KEY_FLAT_BASE_MM: float = 2.0
MIN_CRANK_KEY_FLAT_MM: float = 1.0
CRANK_GEAR_MAX_Y_RATIO: float = 61.6
CRANK_TIMING_FLAT_START_MM: float = 47.0
CRANK_TIMING_RAMP_START_MM: float = 62.0
CRANK_COVER_LAND_END_MM: float = 70.5
CRANK_PTO_FLAT_START_MM: float = 72.0
CRANK_FLYWHEEL_DETENT_Y_MM: float = -84.5
CRANK_PTO_DETENT_Y_MM: float = 92.0
DETENT_RIDGE_HALF_Y_MM: float = 1.25
DETENT_GROOVE_DEPTH_MM: float = 0.45

# Main bushing dimensions (1:1 mm)
BUSHING_OUTER_RADIUS_MM: float = 31.0
BUSHING_MIN_Y_MM: float = -42.5
BUSHING_MAX_Y_MM: float = -26.5
BUSHING_KEY_WIDTH_MM: float = 6.0
BUSHING_KEY_MIN_WIDTH_MM: float = 2.0
BUSHING_KEY_RADIAL_H_MM: float = 2.0

# Connecting rod dimensions (1:1 mm)
CONROD_LENGTH_MM: float = 108.0
WRIST_PIN_DIAM_MM: float = 16.0
CONROD_WIDTH_MM: float = 19.0
CONROD_SMALL_END_RAD_MM: float = 12.5
CONROD_BIG_END_RAD_MM: float = 20.0
CONROD_SHANK_HEIGHT_MM: float = 20.0
CONROD_EAR_Z_OUTER_MM: float = 33.0
CONROD_DIPPER_TIP_RADIUS_MM: float = 30.0
CONROD_DIPPER_WIDTH_X_MM: float = 8.0
CONROD_DIPPER_THICK_Z_MM: float = 5.0
CONROD_PIN_X_OFFSET_MM: float = 3.5

# Piston & wrist-pin dimensions (1:1 mm)
PISTON_BORE_MM: float = 68.0
PISTON_COMP_HEIGHT_MM: float = 24.0
PISTON_TOTAL_HEIGHT_MM: float = 42.0
PISTON_PIN_LOCAL_Z_MM: float = -3.0
PISTON_CROWN_THICK_MM: float = 8.5
PISTON_SKIRT_WALL_MM: float = 2.5
PISTON_PIN_BOSS_GAP_MM: float = 32.0
PISTON_PIN_BOSS_OUTER_Y_MM: float = 24.5
PISTON_PIN_BOSS_RAD_MM: float = 13.0
PISTON_POCKET_WIDTH_X_MM: float = 34.0
PISTON_MIN_GROOVE_W_MM: float = 1.4

# Flywheel & hand crank dimensions (1:1 mm)
FLYWHEEL_MIN_Y_MM: float = -83.0
FLYWHEEL_MAX_Y_MM: float = -57.0
FLYWHEEL_WEB_THICK_MM: float = 16.0
FLYWHEEL_TRIGGER_MIN_Y_MM: float = -86.75
FLYWHEEL_TRIGGER_MAX_Y_MM: float = -81.25
FLYWHEEL_TRIGGER_TEETH: int = 18
TIMING_COVER_OUTER_Y_BASE_MM: float = 68.0
HAND_CRANK_ARM_LENGTH_MM: float = 54.0
HAND_CRANK_KNOB_LENGTH_MM: float = 42.0


def _as_part(shape: Shape | Part, label: str) -> Part:
  """Normalize a build123d shape into a single-solid valid Part."""
  part = shape if isinstance(shape, Part) else Part(children=list(shape.solids()))
  solids = list(part.solids())
  if len(solids) != 1:
    raise ValueError(f"{label} expected 1 solid, got {len(solids)}")
  if not part.is_valid or float(part.volume) <= 0.0:
    raise ValueError(f"{label} produced invalid or zero-volume geometry")
  return part


def _y_cylinder(
    radius_mm: float,
    y_start_mm: float,
    y_end_mm: float,
    center_x_mm: float = 0.0,
    center_z_mm: float = 0.0,
) -> Part:
  """Create a cylinder aligned along the Y axis spanning [y_start_mm, y_end_mm]."""
  y_lo = min(y_start_mm, y_end_mm)
  y_hi = max(y_start_mm, y_end_mm)
  height = y_hi - y_lo
  y_mid = HALF_FACTOR * (y_lo + y_hi)
  with BuildPart() as builder:
    with Locations((center_x_mm, y_mid, center_z_mm)):
      Cylinder(radius=radius_mm, height=height, rotation=(ROT_90_DEG, 0.0, 0.0))
  return builder.part


def _y_keyed_shaft(
    radius_mm: float,
    y_start_mm: float,
    y_end_mm: float,
    flat_depth_mm: float,
    center_x_mm: float = 0.0,
    center_z_mm: float = 0.0,
    flat_dir: str = "+X",
) -> Part:
  """Build a D-flat shaft along Y from y_start_mm to y_end_mm."""
  length = abs(y_end_mm - y_start_mm)
  solid = make_keyed_shaft_solid(radius_mm=radius_mm, length_mm=length, flat_depth_mm=flat_depth_mm)
  if flat_dir == "+X":
    rot_x = -ROT_90_DEG if y_end_mm >= y_start_mm else ROT_90_DEG
    return solid.rotate(Axis.X, rot_x).translate((center_x_mm, y_start_mm, center_z_mm))
  if flat_dir == "+Z":
    if y_end_mm >= y_start_mm:
      return solid.rotate(Axis.Z, -ROT_90_DEG).rotate(Axis.X, -ROT_90_DEG).translate(
          (center_x_mm, y_start_mm, center_z_mm)
      )
    return solid.rotate(Axis.Z, ROT_90_DEG).rotate(Axis.X, ROT_90_DEG).translate(
        (center_x_mm, y_start_mm, center_z_mm)
    )
  raise ValueError(f"Unsupported flat_dir {flat_dir!r}")


def _y_keyed_bore(
    radius_mm: float,
    y_start_mm: float,
    y_end_mm: float,
    flat_depth_mm: float,
    clearance_mm: float,
    center_x_mm: float = 0.0,
    center_z_mm: float = 0.0,
    flat_dir: str = "+X",
) -> Part:
  """Build a mating D-flat bore cutter along Y from y_start_mm to y_end_mm."""
  length = abs(y_end_mm - y_start_mm)
  cutter = make_keyed_bore_cutter(
      radius_mm=radius_mm,
      length_mm=length,
      flat_depth_mm=flat_depth_mm,
      clearance_mm=clearance_mm,
  )
  if flat_dir == "+X":
    rot_x = -ROT_90_DEG if y_end_mm >= y_start_mm else ROT_90_DEG
    return cutter.rotate(Axis.X, rot_x).translate((center_x_mm, y_start_mm, center_z_mm))
  if flat_dir == "+Z":
    if y_end_mm >= y_start_mm:
      return cutter.rotate(Axis.Z, -ROT_90_DEG).rotate(Axis.X, -ROT_90_DEG).translate(
          (center_x_mm, y_start_mm, center_z_mm)
      )
    return cutter.rotate(Axis.Z, ROT_90_DEG).rotate(Axis.X, ROT_90_DEG).translate(
        (center_x_mm, y_start_mm, center_z_mm)
    )
  raise ValueError(f"Unsupported flat_dir {flat_dir!r}")


def _crank_key_flat_depth(s: float) -> float:
  """Return shared D-flat depth on D=20*s crank timing/flywheel/PTO journals."""
  return max(MIN_CRANK_KEY_FLAT_MM, CRANK_KEY_FLAT_BASE_MM * s)


def _crankpin_snap_params(cfg: PrintKitConfig) -> tuple[float, float, float, float, float]:
  """Compute filament-safe crankpin D-flat snap-tongue geometry.

  Returns:
    ``(flat_d, r_bore, tongue_w, tongue_len, detent_h)`` in mm.
  """
  s = cfg.scale
  min_wall_ns = min_printable_wall_mm(cfg, structural=False)
  r_pin = HALF_FACTOR * CRANKPIN_DIAM_MM * s
  flat_d = _crank_key_flat_depth(s)
  r_bore = min(
      HALF_FACTOR * CRANKPIN_BORE_DIAM_MM * s,
      max(1.2, r_pin - flat_d - max(min_wall_ns, 0.9)),
  )
  wall_under_flat = max(0.7, r_pin - flat_d - r_bore)
  beam_t = min(wall_under_flat, max(min_wall_ns, 0.85))
  web_socket = CRANKPIN_WEB_SOCKET_MM * s
  tongue_len = max(2.2, web_socket - (0.5 * s))
  tongue_w = max(2.6, 5.6 * s)

  eps_allow = cfg.filament_profile.allowable_snap_strain
  max_safe_detent = (eps_allow / (1.5 * 1.25)) * (tongue_len ** 2) / (1.5 * beam_t)
  detent_h = min(0.42, max(0.14, max_safe_detent))
  return flat_d, r_bore, tongue_w, tongue_len, detent_h


def _make_d_flat_detent_groove_cutter(
    radius_mm: float,
    flat_depth_mm: float,
    y_center_mm: float,
    half_width_y_mm: float,
    depth_mm: float,
) -> Part:
  """Cut a 45-degree chamfered detent groove into a +X D-flat shaft."""
  x_flat = radius_mm - flat_depth_mm
  x_in = x_flat - depth_mm
  x_out = x_flat + 1.0
  slope_y = depth_mm
  pts = [
      (x_out, y_center_mm - half_width_y_mm - slope_y),
      (x_flat, y_center_mm - half_width_y_mm - slope_y),
      (x_in, y_center_mm - half_width_y_mm),
      (x_in, y_center_mm + half_width_y_mm),
      (x_flat, y_center_mm + half_width_y_mm + slope_y),
      (x_out, y_center_mm + half_width_y_mm + slope_y),
  ]
  with BuildPart() as cutter:
    with BuildSketch(Plane.XY):
      with BuildLine():
        Polyline(*pts, close=True)
      make_face()
    extrude(amount=radius_mm + 2.0, both=True)
  return cutter.part


def _make_hub_detent_latch(
    cfg: PrintKitConfig,
    shaft_r_mm: float,
    flat_depth_mm: float,
    r_slip_mm: float,
    y_latch_center_mm: float,
    y_root_mm: float,
    ridge_depth_mm: float,
) -> tuple[Part, Part]:
  """Build an internal +X D-flat snap-detent ridge and side-slot reliefs for flywheel/hand crank.

  Returns:
    ``(relief_slots_cutter, detent_ridge_adder)`` in crank-local Y coordinates.
  """
  s = cfg.scale
  x_bore_flat = (shaft_r_mm - flat_depth_mm) + r_slip_mm
  arm_w = max(3.2, 6.0 * s)
  slot_w = max(0.7, 1.0 * s)
  arm_t = min_printable_wall_mm(cfg, structural=False)
  relief_r = max(0.8, 1.2 * s)
  # Back relief pocket opens all the way to the open end face (y_root_mm -> past y_latch_center_mm)
  # so the vertical slot prints completely support-free with zero ceiling overhang.
  y_open_end = y_latch_center_mm + (2.0 if y_latch_center_mm >= y_root_mm else -2.0)
  y_lo = min(y_open_end, y_root_mm)
  y_hi = max(y_open_end, y_root_mm)

  with BuildPart() as relief:
    # Back relief pocket behind D-flat wall + flanking Z slots creating a compliant cantilever tongue
    with Locations((x_bore_flat + arm_t + (HALF_FACTOR * relief_r), HALF_FACTOR * (y_lo + y_hi), 0.0)):
      Box(relief_r, y_hi - y_lo, arm_w + (2.0 * slot_w))
    for sz in (-1.0, 1.0):
      with Locations((
          x_bore_flat + HALF_FACTOR * (arm_t + relief_r) - 0.5,
          HALF_FACTOR * (y_lo + y_hi),
          sz * (HALF_FACTOR * arm_w + HALF_FACTOR * slot_w),
      )):
        Box(arm_t + relief_r + 1.0, y_hi - y_lo, slot_w)

  # 45-degree chamfered detent ridge protruding inward from x_bore_flat into shaft groove
  x_tip = x_bore_flat - ridge_depth_mm
  x_base = x_bore_flat + 0.4
  half_y = max(0.6, DETENT_RIDGE_HALF_Y_MM * s * 0.75)
  slope_y = ridge_depth_mm
  pts = [
      (x_base, y_latch_center_mm - half_y - slope_y),
      (x_tip, y_latch_center_mm - half_y),
      (x_tip, y_latch_center_mm + half_y),
      (x_base, y_latch_center_mm + half_y + slope_y),
  ]
  with BuildPart() as ridge:
    with BuildSketch(Plane.XY):
      with BuildLine():
        Polyline(*pts, close=True)
      make_face()
    extrude(amount=HALF_FACTOR * arm_w, both=True)

  return relief.part, ridge.part


def _build_scaled_web_sketch(s: float, min_wall_mm: float) -> BuildSketch:
  """Build scaled 1:1 counterweight web profile in the XZ plane."""
  r_throw = CRANK_THROW_MM * s
  r_pin = HALF_FACTOR * CRANKPIN_DIAM_MM * s
  r_main = HALF_FACTOR * MAIN_JOURNAL_DIAM_MM * s
  r_pin_boss = r_pin + max(min_wall_mm, CRANKPIN_BOSS_MARGIN_MM * s)
  r_main_boss = r_main + max(min_wall_mm, MAIN_BOSS_MARGIN_MM * s)
  r_max = COUNTERWEIGHT_RADIUS_MM * s
  cw_rect_z = (CW_CUT_OFFSET_MM * s) + (HALF_FACTOR * CW_CUT_HEIGHT_MM * s)

  with BuildSketch(Plane.XZ) as cheek_sketch:
    with BuildSketch(mode=Mode.PRIVATE) as arm_sk:
      with Locations((0.0, r_throw)):
        Circle(radius=r_pin_boss)
      with Locations((0.0, 0.0)):
        Circle(radius=r_main_boss)
      make_hull()
    insert(arm_sk.sketch)
    with BuildSketch(mode=Mode.PRIVATE) as cw_sk:
      Circle(radius=r_max)
      with Locations((0.0, cw_rect_z)):
        Rectangle(CW_CUT_WIDTH_MM * s, CW_CUT_HEIGHT_MM * s, mode=Mode.SUBTRACT)
    insert(cw_sk.sketch)
  return cheek_sketch


def _build_web_crankpin_socket_cutter(cfg: PrintKitConfig, side: int) -> Part:
  """Build the +Z D-keyed crankpin socket, 45-deg roof cone, and snap catch recess cutter."""
  s = cfg.scale
  r_slip = slip_clearance_radial_mm(cfg)
  c_snap = snap_clearance_mm(cfg)
  flat_d, _, tongue_w, _, detent_h = _crankpin_snap_params(cfg)

  r_throw = CRANK_THROW_MM * s
  r_pin = HALF_FACTOR * CRANKPIN_DIAM_MM * s
  w_pin = CRANKPIN_GAP_Y_MM * s
  t_web = WEB_THICKNESS_MM * s
  web_socket = CRANKPIN_WEB_SOCKET_MM * s
  r_thru = CRANKPIN_THRU_R_MM * s

  y_inner = side * HALF_FACTOR * w_pin
  y_sock_end = side * (HALF_FACTOR * w_pin + web_socket)
  r_sock = r_pin + r_slip
  cone_len = max(0.5, (r_sock - r_thru) * 1.12)

  # 2. 42-deg socket mouth bevel on +Z flat clearing crankpin D-flat transition
  mouth_len = (flat_d * 1.12) + r_slip
  y_mouth_end = y_inner + (side * mouth_len)
  z_flat_bore = r_throw + r_pin - flat_d + r_slip
  z_top_mouth = r_throw + r_sock + 0.5
  mouth_pts = [
      (y_inner - (side * 0.5), z_top_mouth),
      (y_inner - (side * 0.5), z_flat_bore - 0.1),
      (y_mouth_end, z_flat_bore - 0.1),
      (y_inner, r_throw + r_sock),
      (y_inner, z_top_mouth),
  ]
  if side < 0:
    mouth_pts = [
        (y_inner - (side * 0.5), z_top_mouth),
        (y_inner, z_top_mouth),
        (y_inner, r_throw + r_sock),
        (y_mouth_end, z_flat_bore - 0.1),
        (y_inner - (side * 0.5), z_flat_bore - 0.1),
    ]
  with BuildPart() as mouth_cut:
    with BuildSketch(Plane.YZ):
      with BuildLine():
        Polyline(*mouth_pts, close=True)
      make_face()
    extrude(amount=r_sock + 1.0, both=True)

  # 3. 45-deg conical ceiling tapering from R_sock to R_thru + thru-bore to outer web face
  with BuildPart() as cone_z:
    with BuildSketch(Plane.XZ):
      pts_z = [
          (0.0, 0.0),
          (r_sock, 0.0),
          (r_thru, cone_len),
          (r_thru, t_web - web_socket + 1.5),
          (0.0, t_web - web_socket + 1.5),
      ]
      with BuildLine():
        Polyline(*pts_z, close=True)
      make_face()
    revolve(axis=Axis.Z)
  rot_x = -ROT_90_DEG if side > 0 else ROT_90_DEG

  # 4. +Z catch pocket for crankpin D-flat snap-detent ridge (45-deg self-supporting roof)
  ridge_center_off = web_socket - (1.8 * s)
  half_y = max(0.6, 0.85 * s) + c_snap
  y_c = y_inner + (side * ridge_center_off)
  z_lo = r_throw + r_pin - flat_d - 0.1
  z_hi = r_throw + r_pin - flat_d + detent_h + c_snap
  slope = z_hi - z_lo
  catch_pts = [
      (y_c - half_y - slope, z_lo),
      (y_c - half_y, z_hi),
      (y_c + half_y, z_hi),
      (y_c + half_y + slope, z_lo),
  ]
  with BuildPart() as catch_cut:
    with BuildSketch(Plane.YZ):
      with BuildLine():
        Polyline(*catch_pts, close=True)
      make_face()
    extrude(amount=HALF_FACTOR * tongue_w + c_snap, both=True)

  with BuildPart() as cutter:
    # 1. D-keyed socket spanning [y_inner, y_sock_end]
    insert(_y_keyed_bore(r_pin, y_inner - (side * 0.5), y_sock_end, flat_d, r_slip, 0.0, r_throw, flat_dir="+Z"))
    insert(mouth_cut.part)
    insert(cone_z.part.rotate(Axis.X, rot_x).translate((0.0, y_sock_end, r_throw)))
    insert(catch_cut.part)

  return cutter.part


def _build_kit_crank_left_local(cfg: PrintKitConfig) -> Part:
  """Build kit_crank_left in crank-local coordinates (axis Y, pin center Z=+27*s)."""
  s = cfg.scale
  min_wall = min_printable_wall_mm(cfg, structural=True)
  cheek_sketch = _build_scaled_web_sketch(s, min_wall)

  w_pin = CRANKPIN_GAP_Y_MM * s
  t_web = WEB_THICKNESS_MM * s
  y_web_in = -HALF_FACTOR * w_pin
  y_web_out = y_web_in - t_web
  y_main_end = y_web_out - (LEFT_MAIN_LEN_MM * s)
  y_fw_end = y_main_end - (FLYWHEEL_HUB_LEN_MM * s)
  y_nose_end = y_fw_end - (NOSE_LEN_MM * s)

  r_main = HALF_FACTOR * MAIN_JOURNAL_DIAM_MM * s
  r_fw = HALF_FACTOR * FLYWHEEL_HUB_DIAM_MM * s
  r_nose = HALF_FACTOR * NOSE_DIAM_MM * s
  flat_d = _crank_key_flat_depth(s)

  with BuildPart() as left_crank:
    extrude(cheek_sketch.sketch.translate((0.0, y_web_in, 0.0)), amount=t_web, dir=(0.0, -1.0, 0.0))
    insert(_y_cylinder(r_main, y_main_end, y_web_out + 0.5))
    insert(_y_keyed_shaft(r_fw, y_main_end + 0.5, y_fw_end, flat_d, flat_dir="+X"))
    insert(_y_cylinder(r_nose, y_nose_end, y_fw_end + 0.5))
    insert(_build_web_crankpin_socket_cutter(cfg, side=-1), mode=Mode.SUBTRACT)
    insert(
        _make_d_flat_detent_groove_cutter(
            radius_mm=r_fw,
            flat_depth_mm=flat_d,
            y_center_mm=CRANK_FLYWHEEL_DETENT_Y_MM * s,
            half_width_y_mm=DETENT_RIDGE_HALF_Y_MM * s,
            depth_mm=DETENT_GROOVE_DEPTH_MM,
        ),
        mode=Mode.SUBTRACT,
    )
  return _as_part(left_crank.part, "kit_crank_left")


def _build_kit_crank_right_local(cfg: PrintKitConfig) -> Part:
  """Build kit_crank_right in crank-local coordinates with continuous D=20*s timing/PTO journal."""
  s = cfg.scale
  min_wall = min_printable_wall_mm(cfg, structural=True)
  cheek_sketch = _build_scaled_web_sketch(s, min_wall)

  w_pin = CRANKPIN_GAP_Y_MM * s
  t_web = WEB_THICKNESS_MM * s
  y_web_in = HALF_FACTOR * w_pin
  y_web_out = y_web_in + t_web
  y_main_end = y_web_out + (RIGHT_MAIN_LEN_MM * s)
  y_end = RIGHT_OUTER_SHAFT_END_Y_MM * s

  r_main = HALF_FACTOR * MAIN_JOURNAL_DIAM_MM * s
  r_shaft = HALF_FACTOR * FLYWHEEL_HUB_DIAM_MM * s
  flat_d = _crank_key_flat_depth(s)

  y_timing_start = CRANK_TIMING_FLAT_START_MM * s
  y_ramp_start = max(CRANK_TIMING_RAMP_START_MM * s, (CRANK_GEAR_MAX_Y_RATIO * s) + 0.15)
  y_ramp_end = y_ramp_start + (flat_d * 1.12)
  y_cover_end = max(CRANK_COVER_LAND_END_MM * s, y_ramp_end + 0.2)
  y_pto_start = max(CRANK_PTO_FLAT_START_MM * s, y_cover_end + 0.2)

  # Cut +X D-flats into continuous D=20*s cylinder with 45-deg transition ramp at timing exit
  x_flat = r_shaft - flat_d
  x_cut_out = r_shaft + 1.5
  timing_pts = [
      (x_flat, y_timing_start),
      (x_cut_out, y_timing_start),
      (x_cut_out, y_ramp_end),
      (x_flat, y_ramp_start),
  ]
  pto_pts = [
      (x_flat, y_pto_start),
      (x_cut_out, y_pto_start - flat_d),
      (x_cut_out, y_end + 1.0),
      (x_flat, y_end + 1.0),
  ]

  with BuildPart() as right_crank:
    extrude(cheek_sketch.sketch.translate((0.0, y_web_in, 0.0)), amount=t_web, dir=(0.0, 1.0, 0.0))
    insert(_y_cylinder(r_main, y_web_out - 0.5, y_main_end))
    insert(_y_cylinder(r_shaft, y_main_end - 0.5, y_end))

    for pts in (timing_pts, pto_pts):
      with BuildPart(mode=Mode.SUBTRACT):
        with BuildSketch(Plane.XY):
          with BuildLine():
            Polyline(*pts, close=True)
          make_face()
        extrude(amount=r_shaft + 2.0, both=True)

    insert(_build_web_crankpin_socket_cutter(cfg, side=1), mode=Mode.SUBTRACT)
    insert(
        _make_d_flat_detent_groove_cutter(
            radius_mm=r_shaft,
            flat_depth_mm=flat_d,
            y_center_mm=CRANK_PTO_DETENT_Y_MM * s,
            half_width_y_mm=DETENT_RIDGE_HALF_Y_MM * s,
            depth_mm=DETENT_GROOVE_DEPTH_MM,
        ),
        mode=Mode.SUBTRACT,
    )
  return _as_part(right_crank.part, "kit_crank_right")


def _build_kit_crankpin_local(cfg: PrintKitConfig) -> Part:
  """Build hollow kit_crankpin with +Z D-keyed snap-detent ends in crank-local coordinates."""
  s = cfg.scale
  flat_d, r_bore, tongue_w, tongue_len, detent_h = _crankpin_snap_params(cfg)
  r_throw = CRANK_THROW_MM * s
  r_pin = HALF_FACTOR * CRANKPIN_DIAM_MM * s
  w_pin = CRANKPIN_GAP_Y_MM * s
  web_socket = CRANKPIN_WEB_SOCKET_MM * s
  y_half_total = (HALF_FACTOR * w_pin) + web_socket
  y_half_journal = HALF_FACTOR * w_pin

  slot_w = max(0.7, 1.0 * s)
  ridge_off = web_socket - (1.8 * s)
  ridge_half_y = max(0.55, 0.75 * s)
  z_flat = r_throw + r_pin - flat_d
  z_top_cut = r_throw + r_pin + 1.5

  with BuildPart() as pin:
    insert(_y_cylinder(r_pin, -y_half_total, y_half_total, center_z_mm=r_throw))
    insert(_y_cylinder(r_bore, -y_half_total - 1.0, y_half_total + 1.0, center_z_mm=r_throw), mode=Mode.SUBTRACT)

    for sy in (-1.0, 1.0):
      # +Z D-flat with 42-deg self-supporting bevel transition starting at |Y| = 10.5*s
      bevel_y = flat_d * 1.12
      flat_pts = [
          (sy * y_half_journal, z_top_cut),
          (sy * y_half_journal, r_throw + r_pin + 0.05),
          (sy * (y_half_journal + bevel_y), z_flat),
          (sy * (y_half_total + 1.0), z_flat),
          (sy * (y_half_total + 1.0), z_top_cut),
      ]
      if sy < 0.0:
        flat_pts = list(reversed(flat_pts))
      with BuildPart(mode=Mode.SUBTRACT):
        with BuildSketch(Plane.YZ):
          with BuildLine():
            Polyline(*flat_pts, close=True)
          make_face()
        extrude(amount=r_pin + 1.5, both=True)

      # Flanking axial slots along +Z flat creating a compliant snap-detent tongue
      y_slot_in = y_half_total - tongue_len
      y_slot_mid = sy * (y_slot_in + HALF_FACTOR * (tongue_len + 0.5))
      for sx in (-1.0, 1.0):
        with Locations((
            sx * (HALF_FACTOR * tongue_w + HALF_FACTOR * slot_w),
            y_slot_mid,
            r_throw + HALF_FACTOR * (r_bore + r_pin - flat_d),
        )):
          Box(slot_w, tongue_len + 0.5, (r_pin - flat_d - r_bore) + 2.0, mode=Mode.SUBTRACT)

      # 45-degree ramped snap-detent ridge on the +Z D-flat tongue
      y_c = sy * (y_half_journal + ridge_off)
      z_peak = z_flat + detent_h
      z_base = z_flat - 0.3
      slope = detent_h
      ridge_pts = [
          (y_c - ridge_half_y - slope, z_base),
          (y_c - ridge_half_y, z_peak),
          (y_c + ridge_half_y, z_peak),
          (y_c + ridge_half_y + slope, z_base),
      ]
      with BuildPart(mode=Mode.ADD):
        with BuildSketch(Plane.YZ):
          with BuildLine():
            Polyline(*ridge_pts, close=True)
          make_face()
        extrude(amount=HALF_FACTOR * tongue_w, both=True)

  return _as_part(pin.part, "kit_crankpin")


def _conrod_geometry_params(cfg: PrintKitConfig) -> dict[str, float | SnapFitDesign]:
  """Compute shared scaled connecting-rod dimensions and cantilever snap design."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg, structural=True)

  l_mm = CONROD_LENGTH_MM * s
  r_small_bore = (HALF_FACTOR * WRIST_PIN_DIAM_MM * s) + r_run
  r_big_bore = (HALF_FACTOR * CRANKPIN_DIAM_MM * s) + r_run
  r_small_outer = max(CONROD_SMALL_END_RAD_MM * s, r_small_bore + max(min_wall, 3.0 * s))
  r_big_outer = max(CONROD_BIG_END_RAD_MM * s, r_big_bore + max(min_wall * 1.4, 4.0 * s))

  w_rod = CONROD_WIDTH_MM * s
  shank_h = CONROD_SHANK_HEIGHT_MM * s
  ear_z_out = max(CONROD_EAR_Z_OUTER_MM * s, r_big_outer + max(4.8, 10.5 * s))

  snap_len = max(7.0, 14.0 * s)
  snap_width = min(w_rod - (2.0 * min_wall), max(4.5, 10.0 * s))
  snap_design = auto_size_cantilever_snap(cfg, available_length_mm=snap_len, width_mm=snap_width)

  pin_r = max(0.95, 2.0 * s)
  pin_len = max(2.6, 5.0 * s)
  pin_z = r_big_outer + HALF_FACTOR * (ear_z_out - r_big_outer - (snap_design.tip_thickness_mm + snap_design.undercut_mm))
  pin_x = min(CONROD_PIN_X_OFFSET_MM * s, (HALF_FACTOR * w_rod) - pin_r - min_wall)

  return {
      "l_mm": l_mm,
      "r_small_bore": r_small_bore,
      "r_big_bore": r_big_bore,
      "r_small_outer": r_small_outer,
      "r_big_outer": r_big_outer,
      "w_rod": w_rod,
      "shank_h": shank_h,
      "ear_z_out": ear_z_out,
      "snap_design": snap_design,
      "pin_r": pin_r,
      "pin_len": pin_len,
      "pin_z": pin_z,
      "pin_x": pin_x,
  }


def _place_conrod_snap_shape(
    shape: Part,
    side_z: float,
    l_mm: float,
    ear_z_out: float,
    y_root_offset: float,
) -> Part:
  """Orient a hook or pocket cutter from primitives frame onto +/-Z conrod ear.

  Local hook frame maps: ``+Z_prim -> -Y_rod``, ``+X_prim -> -side_z * Z_rod`` (inward),
  and ``Y_prim -> side_z * X_rod`` (width centered symmetrically at ``x = 0``).
  """
  return (
      shape.rotate(Axis.Y, side_z * ROT_90_DEG)
      .rotate(Axis.Z, -side_z * ROT_90_DEG)
      .translate((0.0, l_mm + y_root_offset, side_z * ear_z_out))
  )


def _build_kit_conrod_parts_local(cfg: PrintKitConfig) -> tuple[Part, Part]:
  """Build (kit_conrod_beam, kit_conrod_cap) in local rod coords (X=bore, Y=rod, Z=ears)."""
  s = cfg.scale
  r_slip = slip_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg, structural=True)
  g = _conrod_geometry_params(cfg)

  l_mm = float(g["l_mm"])
  r_small_bore = float(g["r_small_bore"])
  r_big_bore = float(g["r_big_bore"])
  r_small_outer = float(g["r_small_outer"])
  r_big_outer = float(g["r_big_outer"])
  w_rod = float(g["w_rod"])
  shank_h = float(g["shank_h"])
  ear_z_out = float(g["ear_z_out"])
  snap_design: SnapFitDesign = g["snap_design"]  # type: ignore[assignment]
  pin_r = float(g["pin_r"])
  pin_len = float(g["pin_len"])
  pin_z = float(g["pin_z"])
  pin_x = float(g["pin_x"])

  # --- BEAM (y <= l_mm) ---
  y_root_off = max(2.2, 4.5 * s)
  y_beam_ear_bot = l_mm - (snap_design.beam_length_mm - y_root_off + snap_design.hook_tip_flat_mm + 6.5)
  y_beam_ear_top = l_mm
  beam_ear_h = y_beam_ear_top - y_beam_ear_bot
  beam_ear_mid_y = HALF_FACTOR * (y_beam_ear_top + y_beam_ear_bot)

  shank_y0 = r_small_outer - (2.0 * s)
  shank_y1 = l_mm - r_big_outer + (2.0 * s)
  shank_len = shank_y1 - shank_y0
  flange_t = max(min_wall, 4.0 * s)
  web_t = max(min_wall, 4.0 * s)
  pocket_depth = max(1.0, w_rod - web_t) * 0.45

  hook_solid = make_cantilever_snap_hook(snap_design)
  pocket_cutter = make_snap_pocket_cutter(snap_design, snap_design.clearance_mm)
  pin_solid = make_alignment_pin(pin_r, pin_len, chamfer_mm=0.35).rotate(Axis.X, ROT_90_DEG)
  hole_cutter = make_alignment_hole_cutter(pin_r, pin_len + 0.6, r_slip).rotate(Axis.X, ROT_90_DEG)

  # Build beam silhouette in Plane.YZ and extrude symmetrically so there are zero coplanar seams
  with BuildPart() as beam:
    with BuildSketch(Plane.YZ):
      with Locations((0.0, 0.0)):
        Circle(radius=r_small_outer)
      with Locations((l_mm, 0.0)):
        Circle(radius=r_big_outer)
      with Locations((shank_y0 + HALF_FACTOR * shank_len, 0.0)):
        Rectangle(shank_len, shank_h)
      with Locations((beam_ear_mid_y, 0.0)):
        Rectangle(beam_ear_h, 2.0 * ear_z_out)
      with Locations((l_mm + 20.0, 0.0)):
        Rectangle(40.0, 2.5 * ear_z_out, mode=Mode.SUBTRACT)
    extrude(amount=HALF_FACTOR * w_rod, both=True)

    # Top (+X) weight-relief pocket leaving a flat -X face on Z=0 when oriented
    with Locations((HALF_FACTOR * w_rod - HALF_FACTOR * pocket_depth + 0.5, shank_y0 + HALF_FACTOR * shank_len, 0.0)):
      Box(pocket_depth + 1.0, max(2.0, shank_len - (4.0 * s)), max(2.0, shank_h - (2.0 * flange_t)), mode=Mode.SUBTRACT)

    # Subtract small end, big end bores, diagonal alignment holes, and snap pockets
    Cylinder(radius=r_small_bore, height=w_rod + CUT_MARGIN_MM, rotation=(0.0, ROT_90_DEG, 0.0), mode=Mode.SUBTRACT)
    with Locations((0.0, l_mm, 0.0)):
      Cylinder(radius=r_big_bore, height=w_rod + CUT_MARGIN_MM, rotation=(0.0, ROT_90_DEG, 0.0), mode=Mode.SUBTRACT)

    insert(hole_cutter.translate((-pin_x, l_mm + 0.05, pin_z)), mode=Mode.SUBTRACT)
    insert(hole_cutter.translate((pin_x, l_mm + 0.05, -pin_z)), mode=Mode.SUBTRACT)

    for sz in (-1.0, 1.0):
      insert(_place_conrod_snap_shape(pocket_cutter, sz, l_mm, ear_z_out, y_root_off), mode=Mode.SUBTRACT)

  # --- CAP (y >= l_mm) ---
  cap_ear_h = max(3.5, 9.0 * s)
  cap_ear_mid_y = l_mm + HALF_FACTOR * cap_ear_h
  dipper_w = max(min_wall, CONROD_DIPPER_WIDTH_X_MM * s)
  dipper_t = max(min_wall, CONROD_DIPPER_THICK_Z_MM * s)
  dipper_tip_y = l_mm + min(CONROD_DIPPER_TIP_RADIUS_MM * s, 30.0 * (s / 0.5))
  dipper_start_y = l_mm + r_big_outer - (2.0 * s)
  dipper_len = max(2.0, dipper_tip_y - dipper_start_y)

  with BuildPart() as cap:
    with Locations((0.0, l_mm, 0.0)):
      Cylinder(radius=r_big_outer, height=w_rod, rotation=(0.0, ROT_90_DEG, 0.0))

    for sz in (-1.0, 1.0):
      with Locations((0.0, cap_ear_mid_y, sz * HALF_FACTOR * (r_big_outer * 0.7 + ear_z_out))):
        Box(w_rod, cap_ear_h, ear_z_out - (r_big_outer * 0.7))

    with Locations((0.0, dipper_start_y + HALF_FACTOR * dipper_len, 0.0)):
      Box(dipper_w, dipper_len, dipper_t)

    # Cut away beam half (y < l_mm) cleanly at the split plane
    with Locations((0.0, l_mm - 40.0, 0.0)):
      Box(80.0, 80.0, 120.0, mode=Mode.SUBTRACT)

    with Locations((0.0, l_mm, 0.0)):
      Cylinder(radius=r_big_bore, height=w_rod + CUT_MARGIN_MM, rotation=(0.0, ROT_90_DEG, 0.0), mode=Mode.SUBTRACT)

    # Add diagonal alignment pins projecting -Y from y = l_mm
    insert(pin_solid.translate((-pin_x, l_mm + 0.2, pin_z)))
    insert(pin_solid.translate((pin_x, l_mm + 0.2, -pin_z)))

    # Add cantilever snap hooks on cap ears + solid root buttresses connecting to cap body
    for sz in (-1.0, 1.0):
      insert(_place_conrod_snap_shape(hook_solid, sz, l_mm, ear_z_out, y_root_off))
      buttress_h = max(2.2, y_root_off + 0.5)
      with Locations((0.0, l_mm + HALF_FACTOR * buttress_h, sz * (ear_z_out - HALF_FACTOR * snap_design.root_thickness_mm))):
        Box(snap_design.beam_width_mm, buttress_h, snap_design.root_thickness_mm + 1.5)

  return _as_part(beam.part, "kit_conrod_beam"), _as_part(cap.part, "kit_conrod_cap")


def _pose_rod_local(part: Part, con_rod_angle_deg: float, wrist_pin_z_mm: float) -> Part:
  """Transform a connecting-rod local part (+Y toward crankpin) into world coordinates."""
  return (
      part.rotate(Axis.Z, ROT_90_DEG)
      .rotate(Axis.Y, -ROT_90_DEG)
      .rotate(Axis.Y, -con_rod_angle_deg)
      .translate((0.0, 0.0, wrist_pin_z_mm))
  )


def _piston_ring_groove_schedule(
    s: float,
    z_crown: float,
    z_pin: float,
    r_pin_bore: float,
    min_wall: float,
) -> tuple[float, List[tuple[float, float, float]]]:
  """Compute self-supporting decorative piston ring grooves clear of pin bore.

  Returns:
    ``(belt_floor_z, [(z_center, outer_width_z, radial_depth), ...])``
  """
  groove_w = max(PISTON_MIN_GROOVE_W_MM, 2.0 * s)
  land_w = max(min_wall, 1.2)
  d_groove = min(0.85 * HALF_FACTOR * groove_w, 0.85, 1.8 * s)

  # Since teardrop bore roof points toward -Z (skirt), +Z pin bore ceiling is purely circular
  min_belt_floor = z_pin + r_pin_bore + min_wall
  target_floor = max(min_belt_floor, 9.5 * s)
  available_span = z_crown - target_floor
  if available_span < (groove_w + (2.0 * land_w)):
    target_floor = min_belt_floor
    available_span = max(0.0, z_crown - target_floor)

  max_grooves = 3 if s >= 0.7 else 2
  n_grooves = 0
  for count in range(max_grooves, 0, -1):
    needed = (count * groove_w) + ((count + 1) * land_w)
    if available_span + 1.0e-6 >= needed:
      n_grooves = count
      break

  if n_grooves == 0:
    return target_floor, []

  used_grooves_h = n_grooves * groove_w
  equal_land = (available_span - used_grooves_h) / (n_grooves + 1)
  grooves: List[tuple[float, float, float]] = []
  cursor = z_crown - equal_land
  for _ in range(n_grooves):
    z_c = cursor - (HALF_FACTOR * groove_w)
    grooves.append((z_c, groove_w, d_groove))
    cursor -= groove_w + equal_land
  return target_floor, grooves


def _build_kit_piston_local(cfg: PrintKitConfig) -> Part:
  """Build kit_piston in piston-local coords (crown at +21*s, pin axis at Z=-3*s)."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg, structural=True)

  r_outer = (HALF_FACTOR * PISTON_BORE_MM * s) - r_run
  h_total = PISTON_TOTAL_HEIGHT_MM * s
  h_half = HALF_FACTOR * h_total
  z_crown = h_half
  z_pin = z_crown - (PISTON_COMP_HEIGHT_MM * s)
  crown_t = max(min_wall * 1.25, PISTON_CROWN_THICK_MM * s)
  skirt_wall = max(min_wall * 1.2, PISTON_SKIRT_WALL_MM * s)
  r_inner_skirt = r_outer - skirt_wall
  r_pin_bore = (HALF_FACTOR * WRIST_PIN_DIAM_MM * s) + r_run

  belt_floor_z, grooves = _piston_ring_groove_schedule(s, z_crown, z_pin, r_pin_bore, min_wall)
  groove_depth_max = max((g[2] for g in grooves), default=0.0)
  r_inner_belt = min(r_inner_skirt, r_outer - groove_depth_max - max(min_wall, 2.0 * s))
  z_cavity_top = z_crown - crown_t

  # Build stepped + 45-deg chamfered internal cavity cutter (self-supporting when crown-down!)
  cav_pts: List[tuple[float, float]] = [
      (0.0, -h_half - 1.0),
      (r_inner_skirt, -h_half - 1.0),
      (r_inner_skirt, belt_floor_z),
  ]
  dr_cav = max(0.0, r_inner_skirt - r_inner_belt)
  z_cone_top = min(z_cavity_top, belt_floor_z + dr_cav)
  if dr_cav > MIN_GEOM_EPS_MM and z_cone_top > belt_floor_z + MIN_GEOM_EPS_MM:
    cav_pts.append((r_inner_belt, z_cone_top))
  if z_cavity_top > z_cone_top + MIN_GEOM_EPS_MM:
    cav_pts.append((r_inner_belt, z_cavity_top))
  cav_pts.append((0.0, z_cavity_top))

  with BuildPart() as cav_solid:
    with BuildSketch(Plane.XZ):
      with BuildLine():
        Polyline(*cav_pts, close=True)
      make_face()
    revolve(axis=Axis.Z)

  # Build trapezoidal 42-deg-sloped ring-groove annulus cutters (zero overhang when crown-down)
  groove_cutters: List[Part] = []
  for z_c, width_z, depth_r in grooves:
    half_w = HALF_FACTOR * width_z
    bevel = min(half_w * 0.95, depth_r * 1.12)
    g_pts = [
        (r_outer + 1.0, z_c - half_w),
        (r_outer - depth_r, z_c - half_w + bevel),
        (r_outer - depth_r, z_c + half_w - bevel),
        (r_outer + 1.0, z_c + half_w),
    ]
    with BuildPart() as gc:
      with BuildSketch(Plane.XZ):
        with BuildLine():
          Polyline(*g_pts, close=True)
        make_face()
      revolve(axis=Axis.Z)
    groove_cutters.append(gc.part)

  boss_inner_gap = PISTON_PIN_BOSS_GAP_MM * s
  boss_outer_y = PISTON_PIN_BOSS_OUTER_Y_MM * s
  boss_span_y = boss_outer_y - (HALF_FACTOR * boss_inner_gap)
  boss_center_y = (HALF_FACTOR * boss_inner_gap) + (HALF_FACTOR * boss_span_y)
  boss_rad = max(r_pin_bore + max(min_wall, 1.5), PISTON_PIN_BOSS_RAD_MM * s)
  skirt_y_outer = r_outer - (0.2 * s)

  # Pin bore teardrop rotated 180 deg around Y so 45-deg roof points toward skirt (-Z -> +Z_print!)
  td_bore = (
      make_teardrop_bore_cutter(
          radius_mm=r_pin_bore,
          length_mm=(2.0 * r_outer) + 4.0,
          axis="Y",
      )
      .rotate(Axis.Y, ROT_180_DEG)
      .translate((0.0, 0.0, z_pin))
  )

  with BuildPart() as piston:
    Cylinder(radius=r_outer, height=h_total)
    insert(cav_solid.part, mode=Mode.SUBTRACT)

    # Internal pin bosses and crown-load struts
    with BuildPart(mode=Mode.ADD):
      strut_h = (z_crown - z_pin) + (1.0 * s)
      strut_mid_z = z_pin + (HALF_FACTOR * strut_h)
      for sy in (-1.0, 1.0):
        with Locations((0.0, sy * boss_center_y, z_pin)):
          Cylinder(radius=boss_rad, height=boss_span_y, rotation=(ROT_90_DEG, 0.0, 0.0))
        with Locations((0.0, sy * boss_center_y, strut_mid_z)):
          Box(boss_rad * 2.0, boss_span_y, strut_h)
      Cylinder(radius=skirt_y_outer, height=h_total, mode=Mode.INTERSECT)

    for gc_part in groove_cutters:
      insert(gc_part, mode=Mode.SUBTRACT)

    insert(td_bore, mode=Mode.SUBTRACT)

    # Side pin-boss weight-relief pockets outboard of boss_outer_y
    pocket_depth_y = (r_outer - boss_outer_y) + 2.0
    pocket_center_y = boss_outer_y + (HALF_FACTOR * pocket_depth_y)
    pocket_top_z = belt_floor_z - max(0.5, 0.5 * s)
    pocket_bot_z = -h_half - 2.0
    pocket_h = pocket_top_z - pocket_bot_z
    pocket_mid_z = HALF_FACTOR * (pocket_top_z + pocket_bot_z)
    pocket_w_x = min(PISTON_POCKET_WIDTH_X_MM * s, 1.8 * r_outer)
    for sy in (-1.0, 1.0):
      with Locations((0.0, sy * pocket_center_y, pocket_mid_z)):
        Box(pocket_w_x, pocket_depth_y, pocket_h, mode=Mode.SUBTRACT)

  return _as_part(piston.part, "kit_piston")


def _build_kit_wrist_pin_local(cfg: PrintKitConfig) -> Part:
  """Build hollow kit_wrist_pin with -Y stop flange and +Y split cantilever snap-barb."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg, structural=True)
  min_wall_ns = min_printable_wall_mm(cfg, structural=False)

  r_pin = HALF_FACTOR * WRIST_PIN_DIAM_MM * s
  pin_wall = max(min_wall, 2.8 * s)
  r_bore = max(0.7, r_pin - pin_wall)
  actual_wall = r_pin - r_bore

  boss_outer_y = PISTON_PIN_BOSS_OUTER_Y_MM * s
  flange_t = max(min_wall, 2.5 * s)
  r_outer_piston = (HALF_FACTOR * PISTON_BORE_MM * s) - r_run
  max_safe_flange_r = math.sqrt(max(1.0, (r_outer_piston - 0.6) ** 2 - (boss_outer_y + flange_t) ** 2))
  flange_r = min(max_safe_flange_r, r_pin + max(0.8, 2.0 * s))

  barb_start_y = boss_outer_y + 0.25
  barb_len = max(1.6, 3.0 * s)
  y_tip = barb_start_y + barb_len

  slot_len = max(6.0, 14.0 * s)
  eps_allow = cfg.filament_profile.allowable_snap_strain
  max_safe_barb = (eps_allow / (1.5 * 1.25)) * (slot_len ** 2) / (1.5 * actual_wall)
  barb_undercut = min(0.45, max(0.18, max_safe_barb))
  barb_r_max = min(
      r_pin + barb_undercut,
      math.sqrt(max(1.0, (r_outer_piston - 0.6) ** 2 - y_tip ** 2)),
  )
  barb_h = max(0.15, barb_r_max - r_pin)

  # Build pin profile along +Z in XZ plane then revolve and rotate (-90, 0, 0) to +Y
  z_base = -boss_outer_y - flange_t
  z_barb_lo = barb_start_y - z_base
  z_barb_hi = y_tip - z_base
  z_total = y_tip - z_base

  # Split barb with 42-deg self-supporting retention shoulder and 42-deg lead-in nose
  dz_ret = barb_h * 1.12
  z_peak = z_barb_lo + dz_ret
  z_lead_start = min(z_barb_hi - 0.3, z_peak + max(0.4, 0.5 * s))

  pts = [
      (r_bore, 0.0),
      (flange_r, 0.0),
      (flange_r, flange_t),
      (r_pin, flange_t),
      (r_pin, z_barb_lo),
      (r_pin + barb_h, z_peak),
      (r_pin + barb_h, z_lead_start),
      (r_pin, z_barb_hi),
      (r_bore, z_barb_hi),
  ]

  slot_w = max(min_wall_ns * 0.85, 1.4 * s)
  with BuildPart() as pin_z:
    with BuildSketch(Plane.XZ):
      with BuildLine():
        Polyline(*pts, close=True)
      make_face()
    revolve(axis=Axis.Z)

    # Split +Y end in YZ plane (slots cut along X in [-slot_w/2, +slot_w/2]) to leave +X/-X prongs
    with Locations((0.0, 0.0, z_total - (HALF_FACTOR * slot_len) + 0.5)):
      Box(slot_w, (2.0 * (r_pin + barb_h)) + 2.0, slot_len + 1.0, mode=Mode.SUBTRACT)

  return _as_part(pin_z.part.rotate(Axis.X, -ROT_90_DEG).translate((0.0, z_base, 0.0)), "kit_wrist_pin")


def _build_kit_main_bushing_local(cfg: PrintKitConfig, y_min: float, y_max: float) -> Part:
  """Build symmetric main bearing sleeve bushing with +Z anti-rotation tab and ID lube flutes."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)

  r_outer = BUSHING_OUTER_RADIUS_MM * s
  r_bore = (HALF_FACTOR * MAIN_JOURNAL_DIAM_MM * s) + r_run
  key_w = max(BUSHING_KEY_MIN_WIDTH_MM, BUSHING_KEY_WIDTH_MM * s)
  key_top_z = r_outer + (BUSHING_KEY_RADIAL_H_MM * s)
  length = y_max - y_min
  y_mid = HALF_FACTOR * (y_min + y_max)

  flute_depth = min(0.6, max(0.25, 0.4 * s))
  flute_w = max(0.8, 1.8 * s)

  with BuildPart() as bushing:
    insert(_y_cylinder(r_outer, y_min, y_max))
    # +Z anti-rotation key tab centered at X=0 extending radially to +33*s
    with Locations((0.0, y_mid, HALF_FACTOR * (r_outer * 0.85 + key_top_z))):
      Box(key_w, length, key_top_z - (r_outer * 0.85))
    insert(_y_cylinder(r_bore, y_min - 1.0, y_max + 1.0), mode=Mode.SUBTRACT)

    # 4 shallow axial oil/grease flutes at 45-deg diagonals
    for deg in (45.0, 135.0, 225.0, 315.0):
      rad = math.radians(deg)
      fx = r_bore * math.cos(rad)
      fz = r_bore * math.sin(rad)
      with Locations((fx, y_mid, fz)):
        Box(flute_w, length + 2.0, flute_depth * 2.0, rotation=(0.0, -deg, 0.0), mode=Mode.SUBTRACT)

  return _as_part(bushing.part, "kit_main_bushing")


def _build_kit_flywheel_local(cfg: PrintKitConfig) -> Part:
  """Build kit_flywheel with integrated 18-1 trigger ring and D-keyed snap-detent bore."""
  s = cfg.scale
  r_slip = slip_clearance_radial_mm(cfg)
  min_wall_ns = min_printable_wall_mm(cfg, structural=False)

  r_outer = FLYWHEEL_OUTER_RADIUS_MM * s
  r_rim_in = FLYWHEEL_RIM_INNER_RADIUS_MM * s
  r_shaft = HALF_FACTOR * FLYWHEEL_HUB_DIAM_MM * s
  flat_d = _crank_key_flat_depth(s)

  y_max = FLYWHEEL_MAX_Y_MM * s  # -57*s (flat bed contact face when printed at (90,0,0))
  y_min = FLYWHEEL_MIN_Y_MM * s  # -83*s
  y_web_min = y_max - (FLYWHEEL_WEB_THICK_MM * s)  # -73*s
  y_trig_max = FLYWHEEL_TRIGGER_MAX_Y_MM * s  # -81.25*s
  y_trig_min = FLYWHEEL_TRIGGER_MIN_Y_MM * s  # -86.75*s

  r_trig_root = TRIGGER_ROOT_RADIUS_MM * s
  r_trig_tip = TRIGGER_TIP_RADIUS_MM * s
  r_pedestal_base = max(r_shaft + 5.0, r_trig_tip - ((y_web_min - y_trig_max) / 1.12))

  # Cross-section in XY (X = radius, Y = crank-local Y), revolved around Axis.Y
  profile_pts = [
      (r_shaft, y_max),
      (r_outer, y_max),
      (r_outer, y_min),
      (r_rim_in, y_min),
      (r_rim_in, y_web_min),
      (r_pedestal_base, y_web_min),
      (r_trig_tip, y_trig_max),
      (r_shaft, y_trig_max),
  ]

  # 18-1 visual trigger ring (tooth #0 omitted for index gap)
  trig_pts: List[tuple[float, float]] = []
  tooth_step = FULL_CIRCLE_DEG / FLYWHEEL_TRIGGER_TEETH
  for idx in range(FLYWHEEL_TRIGGER_TEETH):
    base_deg = idx * tooth_step
    if idx == 0:
      for deg in (base_deg - 4.0, base_deg + 4.0):
        rad = math.radians(deg)
        trig_pts.append((r_trig_root * math.cos(rad), r_trig_root * math.sin(rad)))
    else:
      for deg, rad_val in (
          (base_deg - 5.2, r_trig_root),
          (base_deg - 3.6, r_trig_tip),
          (base_deg + 3.6, r_trig_tip),
          (base_deg + 5.2, r_trig_root),
      ):
        rad = math.radians(deg)
        trig_pts.append((rad_val * math.cos(rad), rad_val * math.sin(rad)))

  relief_cut, detent_ridge = _make_hub_detent_latch(
      cfg=cfg,
      shaft_r_mm=r_shaft,
      flat_depth_mm=flat_d,
      r_slip_mm=r_slip,
      y_latch_center_mm=CRANK_FLYWHEEL_DETENT_Y_MM * s,
      y_root_mm=y_web_min + 2.0,
      ridge_depth_mm=max(0.25, DETENT_GROOVE_DEPTH_MM - 0.12),
  )

  with BuildPart() as trig_ring:
    with BuildSketch(Plane.XZ):
      Polygon(trig_pts)
    extrude(amount=(y_trig_max - y_trig_min) + 0.1, dir=(0.0, -1.0, 0.0))

  # Combine revolved body + shifted trigger ring + keyed bore + snap detent
  with BuildPart() as flywheel:
    with BuildSketch(Plane.XY):
      with BuildLine():
        Polyline(*profile_pts, close=True)
      make_face()
    revolve(axis=Axis.Y)

    insert(trig_ring.part.translate((0.0, y_trig_max + 0.05, 0.0)))

    # Cut 6 radial decorative/finger windows through flat web if space allows
    win_r_mid = HALF_FACTOR * (r_pedestal_base + r_rim_in)
    win_r_span = (r_rim_in - r_pedestal_base) - (4.0 * max(min_wall_ns, 1.5))
    if win_r_span >= 4.0 and r_pedestal_base > 0.0:
      for k in range(6):
        ang = k * 60.0
        wx = win_r_mid * math.cos(math.radians(ang))
        wz = win_r_mid * math.sin(math.radians(ang))
        with Locations((wx, HALF_FACTOR * (y_web_min + y_max), wz)):
          Cylinder(radius=0.35 * win_r_span, height=(y_max - y_web_min) + 2.0, rotation=(ROT_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    insert(
        _y_keyed_bore(r_shaft, y_max + 1.0, y_trig_min - 1.0, flat_d, r_slip, flat_dir="+X"),
        mode=Mode.SUBTRACT,
    )
    insert(relief_cut, mode=Mode.SUBTRACT)
    insert(detent_ridge)

  return _as_part(flywheel.part, "kit_flywheel")


def _build_kit_hand_crank_local(cfg: PrintKitConfig) -> Part:
  """Build ergonomic D-keyed kit_hand_crank clearing right_timing_cover at +Y PTO."""
  s = cfg.scale
  r_slip = slip_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg, structural=True)

  y_cover_clear = (TIMING_COVER_OUTER_Y_BASE_MM * s) + min_wall + 0.8
  y_start = max(CRANK_PTO_FLAT_START_MM * s, y_cover_clear)
  y_end = RIGHT_OUTER_SHAFT_END_Y_MM * s
  if y_end - y_start < 4.0:
    y_start = y_end - 4.0

  r_shaft = HALF_FACTOR * FLYWHEEL_HUB_DIAM_MM * s
  flat_d = _crank_key_flat_depth(s)
  r_hub = max(r_shaft + r_slip + max(min_wall * 1.5, 2.2), 16.0 * s)

  arm_thick = min(y_end - y_start, max(4.0, 12.0 * s))
  arm_len = HAND_CRANK_ARM_LENGTH_MM * s
  arm_z_center = -arm_len
  r_tip = max(5.0, 9.0 * s)

  knob_len = max(14.0, HAND_CRANK_KNOB_LENGTH_MM * s)
  knob_y0 = y_start + arm_thick
  knob_r_base = max(3.5, min(r_tip - 0.5, 7.0 * s))
  knob_r_mid = max(4.5, 9.5 * s)
  knob_r_top = max(3.8, 7.8 * s)

  # Revolve ergonomic knob profile around Axis.Y in XY plane
  knob_pts = [
      (0.0, knob_y0 - 0.5),
      (knob_r_base, knob_y0 - 0.5),
      (knob_r_mid, knob_y0 + (0.65 * knob_len)),
      (knob_r_top, knob_y0 + knob_len - 1.5),
      (knob_r_top - 1.5, knob_y0 + knob_len),
      (0.0, knob_y0 + knob_len),
  ]

  relief_cut, detent_ridge = _make_hub_detent_latch(
      cfg=cfg,
      shaft_r_mm=r_shaft,
      flat_depth_mm=flat_d,
      r_slip_mm=r_slip,
      y_latch_center_mm=CRANK_PTO_DETENT_Y_MM * s,
      y_root_mm=y_start + 1.5,
      ridge_depth_mm=max(0.25, DETENT_GROOVE_DEPTH_MM - 0.12),
  )

  # Tapered crank arm hull from hub (0, 0) to tip (0, -arm_len) in XZ
  with BuildPart() as arm_part:
    with BuildSketch(Plane.XZ):
      with Locations((0.0, 0.0)):
        Circle(radius=r_hub)
      with Locations((0.0, arm_z_center)):
        Circle(radius=r_tip)
      make_hull()
    extrude(amount=arm_thick, dir=(0.0, -1.0, 0.0))

  # Ergonomic textured grip knob
  with BuildPart() as knob:
    with BuildSketch(Plane.XY):
      with BuildLine():
        Polyline(*knob_pts, close=True)
      make_face()
    revolve(axis=Axis.Y)
    # 8 shallow longitudinal grip flutes along +Y
    for k in range(8):
      ang = k * 45.0
      fx = (knob_r_mid - 0.1) * math.cos(math.radians(ang))
      fz = (knob_r_mid - 0.1) * math.sin(math.radians(ang))
      with Locations((fx, knob_y0 + HALF_FACTOR * knob_len, fz)):
        Cylinder(radius=max(0.5, 1.0 * s), height=knob_len - 2.0, rotation=(ROT_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

  with BuildPart() as crank:
    insert(_y_cylinder(r_hub, y_start, y_end))
    insert(arm_part.part.translate((0.0, y_start + arm_thick, 0.0)))
    insert(knob.part.translate((0.0, 0.0, arm_z_center)))

    # D-keyed bore + snap detent
    insert(
        _y_keyed_bore(r_shaft, y_start - 1.0, y_end + 1.0, flat_d, r_slip, flat_dir="+X"),
        mode=Mode.SUBTRACT,
    )
    insert(relief_cut, mode=Mode.SUBTRACT)
    insert(detent_ridge)

  return _as_part(crank.part, "kit_hand_crank")


def build_kit_rotating_parts(
    cfg: PrintKitConfig,
    crank_angle_deg: float = 0.0,
) -> Dict[str, KitPart]:
  """Build canonical rotating & reciprocating kit STLs in scaled world coordinates.

  Returns 10 unique printable ``KitPart`` entries (11 physical pieces including
  ``quantity=2`` for ``kit_main_bushing``) posed kinematically at ``crank_angle_deg``.
  """
  s = cfg.scale
  z_crank_axis = CRANK_AXIS_Z_MM * s
  st = EngineAssembly().solve_kinematics(crank_angle_deg)
  z_wrist = st.wrist_pin_z_mm * s
  beta_deg = st.con_rod_angle_deg

  def _pose_crank(part: Part) -> Part:
    return part.rotate(Axis.Y, crank_angle_deg).translate((0.0, 0.0, z_crank_axis))

  crank_left = _pose_crank(_build_kit_crank_left_local(cfg))
  crank_right = _pose_crank(_build_kit_crank_right_local(cfg))
  crankpin = _pose_crank(_build_kit_crankpin_local(cfg))
  flywheel = _pose_crank(_build_kit_flywheel_local(cfg))
  hand_crank = _pose_crank(_build_kit_hand_crank_local(cfg))

  beam_local, cap_local = _build_kit_conrod_parts_local(cfg)
  conrod_beam = _pose_rod_local(beam_local, beta_deg, z_wrist)
  conrod_cap = _pose_rod_local(cap_local, beta_deg, z_wrist)

  piston = _build_kit_piston_local(cfg).translate((0.0, 0.0, z_wrist - (PISTON_PIN_LOCAL_Z_MM * s)))
  wrist_pin = _build_kit_wrist_pin_local(cfg).translate((0.0, 0.0, z_wrist))
  main_bushing = _build_kit_main_bushing_local(
      cfg, BUSHING_MIN_Y_MM * s, BUSHING_MAX_Y_MM * s
  ).translate((0.0, 0.0, z_crank_axis))

  return {
      "kit_crank_left": KitPart(
          name="kit_crank_left",
          part=crank_left,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(-ROT_90_DEG, 0.0, 0.0),
          notes="Left counterweight web, 30*s main journal, D-keyed flywheel hub w/ detent, -Y nose.",
      ),
      "kit_crank_right": KitPart(
          name="kit_crank_right",
          part=crank_right,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Right counterweight web, 30*s main journal, continuous 20*s D-keyed timing/PTO shaft.",
      ),
      "kit_crankpin": KitPart(
          name="kit_crankpin",
          part=crankpin,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(-ROT_90_DEG, 0.0, 0.0),
          notes="Hollow 28*s crankpin with +Z D-keyed ends and snap-detent indexing webs at 0 deg.",
      ),
      "kit_conrod_beam": KitPart(
          name="kit_conrod_beam",
          part=conrod_beam,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Split conrod beam with 16*s+2*r_run small end, 28*s+2*r_run big end, pin holes, snap pockets.",
      ),
      "kit_conrod_cap": KitPart(
          name="kit_conrod_cap",
          part=conrod_cap,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Split conrod cap with diagonal alignment pins, dual cantilever snap arms, splash dipper.",
      ),
      "kit_piston": KitPart(
          name="kit_piston",
          part=piston,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes="Crown-down printable piston with 45-deg teardrop pin bore and clamped decorative grooves.",
      ),
      "kit_wrist_pin": KitPart(
          name="kit_wrist_pin",
          part=wrist_pin,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Hollow 16*s wrist pin with -Y stop flange and +Y split cantilever snap-barb.",
      ),
      "kit_main_bushing": KitPart(
          name="kit_main_bushing",
          part=main_bushing,
          quantity=2,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Symmetric main bearing sleeve (OD=62*s, ID=30*s+2*r_run) w/ +Z key tab & lube flutes.",
      ),
      "kit_flywheel": KitPart(
          name="kit_flywheel",
          part=flywheel,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(-ROT_90_DEG, 0.0, 0.0),
          notes="160*s OD flywheel w/ integrated 18-1 trigger ring, D-keyed bore, and snap detent.",
      ),
      "kit_hand_crank": KitPart(
          name="kit_hand_crank",
          part=hand_crank,
          quantity=1,
          category="rotating",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Ergonomic fluted hand crank D-keyed onto +Y PTO clearing right timing cover.",
      ),
  }


def build_kit_rotating_assembly_parts(
    cfg: PrintKitConfig,
    crank_angle_deg: float = 0.0,
) -> Dict[str, KitPart]:
  """Build all 11 posed physical rotating-assembly instances for full collision checks."""
  s = cfg.scale
  z_crank_axis = CRANK_AXIS_Z_MM * s
  canonical = build_kit_rotating_parts(cfg, crank_angle_deg=crank_angle_deg)

  assembled: Dict[str, KitPart] = {}
  for name, kp in canonical.items():
    if name == "kit_main_bushing":
      assembled["kit_main_bushing_left"] = KitPart(
          name="kit_main_bushing_left",
          part=kp.part,
          quantity=1,
          category=kp.category,
          preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
          notes=kp.notes,
      )
      right_bushing = _build_kit_main_bushing_local(
          cfg, -BUSHING_MAX_Y_MM * s, -BUSHING_MIN_Y_MM * s
      ).translate((0.0, 0.0, z_crank_axis))
      assembled["kit_main_bushing_right"] = KitPart(
          name="kit_main_bushing_right",
          part=right_bushing,
          quantity=1,
          category=kp.category,
          preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
          notes=kp.notes,
      )
    else:
      assembled[name] = kp
  return assembled
