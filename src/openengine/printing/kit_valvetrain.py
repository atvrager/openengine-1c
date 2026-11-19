"""3D-printable SOHC valvetrain and 5-gear 2:1 timing drive kit builder via build123d.

Generates watertight, support-friendly B-Rep solids in scaled master assembly
world coordinates (``s = cfg.scale``, cylinder deck ``Z = 0``, crank axis
``(0, 0, -159.0 * s)``, cam axis ``(0, 0, +76.5 * s)``), posed kinematically
at ``crank_angle_deg`` for direct hand-crankable assembly with ``kit_rotating``
and ``kit_housings``.
"""

import functools
import math
from typing import Dict

from build123d import (
    Align,
    Axis,
    Box,
    BuildLine,
    BuildPart,
    BuildSketch,
    CenterArc,
    Circle,
    Cone,
    Cylinder,
    Line,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Rectangle,
    extrude,
    loft,
    make_face,
    revolve,
)
import numpy as np

from openengine.dynamics.sohc_cam import SOHCCamshaft, ValvetrainMassSpring
from openengine.printing.primitives import (
    make_compliant_wave_spring,
    make_keyed_bore_cutter,
    make_keyed_shaft_solid,
)
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    min_printable_wall_mm,
    running_clearance_radial_mm,
    slip_clearance_radial_mm,
)

# Master world datum constants (1:1 mm) matching OpenEngine CAD twin & kit blueprint.
CRANK_AXIS_Z_MM: float = -159.0
CAM_AXIS_Z_MM: float = 76.5
TIMING_CENTER_DISTANCE_MM: float = CAM_AXIS_Z_MM - CRANK_AXIS_Z_MM  # 235.5 mm
TIMING_PLANE_Y_MM: float = 56.0
TIMING_TUNNEL_WALL_Y_MM: float = 48.0
MIN_GEAR_FACE_WIDTH_MM: float = 4.0
GEAR_FACE_WIDTH_1TO1_MM: float = 11.2

# 5-gear 2:1 timing train parameters: 14T crank + 3x 24T idlers + 28T cam (sum = 93*m).
CRANK_GEAR_TEETH: int = 14
IDLER_GEAR_TEETH: int = 24
CAM_GEAR_TEETH: int = 28
TIMING_MODULE_DIVISOR: float = 93.0
IDLER_K_STEPS: tuple[int, int, int] = (19, 43, 67)
GEAR_PRESSURE_ANGLE_DEG: float = 20.0
GEAR_ADDENDUM_COEFF: float = 1.0
GEAR_DEDENDUM_COEFF: float = 1.25
GEAR_BACKLASH_MM: float = 0.20
GEAR_INVOLUTE_SAMPLES: int = 7

# Timing shaft / pin / socket mating dimensions.
CRANK_TIMING_JOURNAL_R_1TO1_MM: float = 10.0
CRANK_TIMING_FLAT_MIN_MM: float = 1.0
CRANK_TIMING_FLAT_1TO1_MM: float = 2.0
IDLER_PIN_MIN_R_MM: float = 2.0
IDLER_PIN_R_1TO1_MM: float = 5.0
IDLER_PIN_MIN_Y_1TO1_MM: float = 44.0
IDLER_PIN_MAX_Y_1TO1_MM: float = 67.5
IDLER_SHOULDER_MAX_DR_MM: float = 1.0
IDLER_SHOULDER_MIN_DR_MM: float = 0.5
IDLER_GEAR_AXIAL_CLEAR_MM: float = 0.15
IDLER_PIN_CHAMFER_MM: float = 0.30
IDLER_PIN_BED_CHAMFER_MM: float = 0.15

# Split keyed SOHC camshaft dimensions.
CAM_CORE_MIN_R_MM: float = 3.0
CAM_CORE_R_1TO1_MM: float = 10.0
CAM_NOSE_MIN_R_MM: float = 2.5
CAM_NOSE_R_1TO1_MM: float = 8.0
CAM_FLAT_MIN_MM: float = 0.8
CAM_FLAT_1TO1_MM: float = 1.6
CAM_CORE_MIN_Y_1TO1_MM: float = -32.0
CAM_CORE_FLANGE_END_Y_1TO1_MM: float = -30.05
CAM_CORE_KEY_START_Y_1TO1_MM: float = -15.5
CAM_CORE_KEY_END_Y_1TO1_MM: float = 14.0
CAM_CORE_NOSE_START_Y_1TO1_MM: float = 40.0
CAM_CORE_NOSE_END_Y_1TO1_MM: float = 62.0
CAM_BASE_R_1TO1_MM: float = 16.0
CAM_LOBE_WIDTH_1TO1_MM: float = 10.0
CAM_INTAKE_Y_1TO1_MM: float = -8.0
CAM_EXHAUST_Y_1TO1_MM: float = 8.0
CAM_SLEEVE_HALF_Y_1TO1_MM: float = 3.0
CAM_SLEEVE_R_1TO1_MM: float = 13.5

# Rocker arms & shafts.
ROCKER_PIVOT_X_1TO1_MM: float = 19.5
ROCKER_PIVOT_Z_1TO1_MM: float = 100.0
ROCKER_SHAFT_MIN_R_MM: float = 1.3
ROCKER_SHAFT_R_1TO1_MM: float = 3.2
ROCKER_SHAFT_MIN_FLANGE_T_MM: float = 0.8
ROCKER_SHAFT_FLANGE_T_1TO1_MM: float = 1.5
ROCKER_SHAFT_MAX_Y_1TO1_MM: float = 17.45
ROCKER_TOWER_FRONT_Y_1TO1_MM: float = -30.0
ROCKER_THICK_Y_1TO1_MM: float = 9.0
ROCKER_BOSS_R_1TO1_MM: float = 5.5
ROCKER_RATIO: float = 1.15
ROCKER_CONTACT_SCALE: float = 1.05

# Poppet valves, compliant wave springs, and snap retainers.
VALVE_IN_X_1TO1_MM: float = 16.0
VALVE_EX_X_1TO1_MM: float = -16.0
VALVE_SEAT_Z_1TO1_MM: float = 6.0
VALVE_TILT_DEG: float = 15.0
VALVE_HEAD_IN_1TO1_MM: float = 30.0
VALVE_HEAD_EX_1TO1_MM: float = 26.0
VALVE_STEM_MIN_D_MM: float = 3.5
VALVE_STEM_D_1TO1_MM: float = 5.0
VALVE_MARGIN_MIN_MM: float = 1.2
VALVE_MARGIN_1TO1_MM: float = 0.8
VALVE_SEAT_H_MIN_MM: float = 0.8
VALVE_SEAT_H_1TO1_MM: float = 1.8
VALVE_NECK_BLEND_1TO1_MM: float = 8.0
VALVE_STEM_LEN_1TO1_MM: float = 76.0
VALVE_RETAINER_TOP_S_1TO1_MM: float = 73.5

SPRING_POCKET_INBOARD_X_1TO1_MM: float = 17.326
SPRING_OFFSET_S_1TO1_MM: float = 40.5
SPRING_MIN_RADIAL_CLEAR_MM: float = 0.90
SPRING_INNER_R_1TO1_MM: float = 5.8
SPRING_OUTER_R_1TO1_MM: float = 8.9
SPRING_OUTER_MAX_SCALE_R_MM: float = 10.4
SPRING_TOP_GAP_MM: float = 0.05
SPRING_MIN_ACTIVE_HEIGHT_MM: float = 5.0

RETAINER_FLANGE_MIN_H_MM: float = 1.4
RETAINER_FLANGE_H_1TO1_MM: float = 2.0
RETAINER_STEP_MIN_H_MM: float = 1.0
RETAINER_STEP_H_1TO1_MM: float = 2.5
RETAINER_PINCH_MM: float = 0.06
VALVE_GROOVE_AXIAL_CLEAR_MM: float = 0.10
VALVE_NECK_MIN_R_MM: float = 1.15
VALVE_NECK_MIN_STEP_MM: float = 0.55

HALF_FACTOR: float = 0.5
ROT_90_DEG: float = 90.0
ROT_180_DEG: float = 180.0
CUT_MARGIN_MM: float = 2.0


def _ensure_single_part(shape: Part, label: str) -> Part:
  """Normalize a build123d shape into a valid single-solid Part."""
  solids = list(shape.solids())
  if len(solids) != 1:
    raise RuntimeError(f"Expected 1 solid for {label}, got {len(solids)}")
  out = Part(children=solids)
  out.label = label
  return out


def _timing_params(cfg: PrintKitConfig) -> tuple[float, float, float, float, float]:
  """Return (scale, module_mm, face_width_mm, gear_y_mm, pin_r_mm)."""
  s = cfg.scale
  cd = TIMING_CENTER_DISTANCE_MM * s
  module_mm = cd / TIMING_MODULE_DIVISOR
  face_w = max(MIN_GEAR_FACE_WIDTH_MM, GEAR_FACE_WIDTH_1TO1_MM * s)
  gear_y = TIMING_PLANE_Y_MM * s
  pin_r = max(IDLER_PIN_MIN_R_MM, IDLER_PIN_R_1TO1_MM * s)
  return s, module_mm, face_w, gear_y, pin_r


def _involute_gear_profile_xy(
    n_teeth: int,
    module_mm: float,
    backlash_mm: float = GEAR_BACKLASH_MM,
) -> list[tuple[float, float]]:
  """Generate closed 2D involute spur gear polygon with tooth 0 centered on +Y."""
  alpha = math.radians(GEAR_PRESSURE_ANGLE_DEG)
  rp = HALF_FACTOR * n_teeth * module_mm
  rb = rp * math.cos(alpha)
  ra = rp + (GEAR_ADDENDUM_COEFF * module_mm)
  rr = max(HALF_FACTOR * module_mm, rp - (GEAR_DEDENDUM_COEFF * module_mm))

  inv_alpha = math.tan(alpha) - alpha
  tooth_thick_p = (HALF_FACTOR * math.pi * module_mm) - (HALF_FACTOR * backlash_mm)
  half_ang_b = (HALF_FACTOR * tooth_thick_p / rp) + inv_alpha
  pitch_ang = (2.0 * math.pi) / n_teeth

  r_start = max(rb, rr)
  t_start = math.sqrt(max(0.0, ((r_start / rb) ** 2) - 1.0))
  t_tip = math.sqrt(max(0.0, ((ra / rb) ** 2) - 1.0))

  pts: list[tuple[float, float]] = []
  n_inv = GEAR_INVOLUTE_SAMPLES
  for k in range(n_teeth):
    c_ang = k * pitch_ang
    ang_v = c_ang - (HALF_FACTOR * pitch_ang)
    pts.append((rr * math.sin(ang_v), rr * math.cos(ang_v)))
    if rr < rb:
      pts.append((rr * math.sin(c_ang - half_ang_b), rr * math.cos(c_ang - half_ang_b)))

    for j in range(n_inv):
      t = t_start + ((t_tip - t_start) * (j / (n_inv - 1)))
      r = rb * math.sqrt(1.0 + (t * t))
      ang = c_ang - (half_ang_b - (t - math.atan(t)))
      pts.append((r * math.sin(ang), r * math.cos(ang)))

    for j in range(n_inv - 1, -1, -1):
      t = t_start + ((t_tip - t_start) * (j / (n_inv - 1)))
      r = rb * math.sqrt(1.0 + (t * t))
      ang = c_ang + (half_ang_b - (t - math.atan(t)))
      pts.append((r * math.sin(ang), r * math.cos(ang)))

    if rr < rb:
      pts.append((rr * math.sin(c_ang + half_ang_b), rr * math.cos(c_ang + half_ang_b)))

  return pts


@functools.lru_cache(maxsize=1)
def _cam_kinematics_engine() -> SOHCCamshaft:
  """Lightweight cached SOHC camshaft kinematics instance avoiding B-Rep mass lookup."""
  return SOHCCamshaft(spring_params=ValvetrainMassSpring(valve_mass_kg=0.04, retainer_mass_kg=0.01))


@functools.lru_cache(maxsize=2)
def _sample_cam_lobe_1to1(is_intake: bool) -> tuple[tuple[float, float], ...]:
  """Cached 1:1 (px, pz) cam lobe contour points matching valvetrain_hardware.py."""
  cam = _cam_kinematics_engine()
  params = cam.intake_params if is_intake else cam.exhaust_params
  theta_samples = np.linspace(0.0, 720.0, 120, endpoint=False)
  lifts_mm = cam.evaluate_valve_lift_crank_angle(params, theta_samples) * 1000.0 / params.rocker_ratio
  coords: list[tuple[float, float]] = []
  for th_deg, lift_mm in zip(theta_samples, lifts_mm):
    r_pt = CAM_BASE_R_1TO1_MM + float(lift_mm)
    gamma_rad = math.radians(-0.5 * float(th_deg))
    coords.append((r_pt * math.sin(gamma_rad), r_pt * math.cos(gamma_rad)))
  return tuple(coords)


@functools.lru_cache(maxsize=32)
def _cached_wave_spring(r_out_mm: float, r_in_mm: float, height_mm: float) -> Part:
  """Cached compliant wave spring solid for given rounded dimensions."""
  return make_compliant_wave_spring(
      outer_radius_mm=r_out_mm,
      inner_radius_mm=r_in_mm,
      free_height_mm=height_mm,
      wire_radial_mm=r_out_mm - r_in_mm,
      waves=3,
      turns=2,
  )


@functools.lru_cache(maxsize=16)
def _cached_gear_solid_xy(
    n_teeth: int,
    module_mm: float,
    face_w: float,
    bore_radius_mm: float,
    flat_depth_mm: float | None,
    clearance_mm: float,
) -> Part:
  """Cached spur gear centered at origin in XY with bore subtracted along Z."""
  if flat_depth_mm is not None:
    bore_cutter_xy = make_keyed_bore_cutter(
        radius_mm=bore_radius_mm,
        length_mm=face_w + CUT_MARGIN_MM,
        flat_depth_mm=flat_depth_mm,
        clearance_mm=clearance_mm,
    ).translate((0.0, 0.0, -(HALF_FACTOR * face_w) - (HALF_FACTOR * CUT_MARGIN_MM)))
  else:
    with BuildPart() as round_bp:
      Cylinder(radius=bore_radius_mm + clearance_mm, height=face_w + CUT_MARGIN_MM)
    bore_cutter_xy = round_bp.part

  pts = _involute_gear_profile_xy(n_teeth=n_teeth, module_mm=module_mm)
  with BuildPart() as gear_bp:
    with BuildSketch(Plane.XY):
      Polygon(pts)
    extrude(amount=HALF_FACTOR * face_w, both=True)
  return Part(gear_bp.part.cut(bore_cutter_xy).wrapped)


def _pose_gear_xy(
    name: str,
    gear_xy: Part,
    y_center: float,
    z_center: float,
    phase_rot_y_deg: float,
) -> Part:
  """Map XY gear (+X->+X, +Y->+Z, Z->-Y) and pose at (0, y_center, z_center)."""
  gear_world = gear_xy.rotate(Axis.X, ROT_90_DEG).rotate(Axis.Y, phase_rot_y_deg).translate(
      (0.0, y_center, z_center)
  )
  return _ensure_single_part(gear_world, name)


@functools.lru_cache(maxsize=8)
def _cached_idler_pin_local(s: float, face_w: float, gear_y: float, pin_r: float) -> Part:
  """Cached stepped, support-free idler pin along +Y with -Y end at Y=y_min."""
  y_min = IDLER_PIN_MIN_Y_1TO1_MM * s
  y_max = IDLER_PIN_MAX_Y_1TO1_MM * s
  y_wall = TIMING_TUNNEL_WALL_Y_MM * s
  y_shoulder_end = gear_y - (HALF_FACTOR * face_w) - IDLER_GEAR_AXIAL_CLEAR_MM

  gap = y_shoulder_end - y_wall
  # <= 41-deg cone keeps discrete STL facets safely above the 45-deg overhang threshold.
  shoulder_dr = min(
      IDLER_SHOULDER_MAX_DR_MM,
      max(IDLER_SHOULDER_MIN_DR_MM, gap * 0.78),
  )
  r_shoulder = pin_r + shoulder_dr
  flare_len = min(gap * 0.96, shoulder_dr * 1.18)
  y_flare_end = y_wall + flare_len

  chamf_bed = min(IDLER_PIN_BED_CHAMFER_MM, 0.12 * pin_r)
  chamf_tip = min(IDLER_PIN_CHAMFER_MM, 0.25 * pin_r)

  # Build along +Z (z = Y - y_min) so -Y end sits on z=0, then rotate -90 deg X -> +Y.
  total_len = y_max - y_min
  z_wall = y_wall - y_min
  z_flare = y_flare_end - y_min
  z_sh_end = y_shoulder_end - y_min

  profile_pts: list[tuple[float, float]] = [
      (0.0, 0.0),
      (pin_r - chamf_bed, 0.0),
      (pin_r, chamf_bed),
      (pin_r, z_wall),
      (r_shoulder, z_flare),
  ]
  if z_sh_end > z_flare + 1.0e-4:
    profile_pts.append((r_shoulder, z_sh_end))
  profile_pts.extend([
      (pin_r, z_sh_end),
      (pin_r, total_len - chamf_tip),
      (pin_r - chamf_tip, total_len),
      (0.0, total_len),
  ])

  with BuildPart() as pin_bp:
    with BuildSketch(Plane.XZ):
      Polygon(profile_pts)
    revolve(axis=Axis.Z)

  return pin_bp.part.rotate(Axis.X, -ROT_90_DEG).translate((0.0, y_min, 0.0))


def _cam_specs(cfg: PrintKitConfig) -> tuple[float, float, float, float]:
  """Return (scale, cam_core_r, cam_nose_r, flat_depth) matching Board #14."""
  s = cfg.scale
  core_r = max(CAM_CORE_MIN_R_MM, CAM_CORE_R_1TO1_MM * s)
  nose_r = max(CAM_NOSE_MIN_R_MM, CAM_NOSE_R_1TO1_MM * s)
  flat_d = max(CAM_FLAT_MIN_MM, CAM_FLAT_1TO1_MM * s)
  return s, core_r, nose_r, flat_d


@functools.lru_cache(maxsize=8)
def _cached_camshaft_core_local(
    s: float,
    core_r: float,
    nose_r: float,
    flat_d: float,
    face_w: float,
    gear_y: float,
) -> Part:
  """Cached unposed split D-keyed camshaft core in world orientation at cam_angle=0 (centered Z=0)."""
  flange_r = core_r + max(1.2, 2.0 * s)

  y_flange_0 = CAM_CORE_MIN_Y_1TO1_MM * s
  y_flange_1 = CAM_CORE_FLANGE_END_Y_1TO1_MM * s
  y_key_0 = CAM_CORE_KEY_START_Y_1TO1_MM * s
  y_key_1 = CAM_CORE_KEY_END_Y_1TO1_MM * s
  y_trans_end = y_key_1 + max(1.5 * s, flat_d)
  y_nose_0 = CAM_CORE_NOSE_START_Y_1TO1_MM * s
  y_nose_1 = max(CAM_CORE_NOSE_END_Y_1TO1_MM * s, gear_y + HALF_FACTOR * face_w)

  with BuildPart() as flange_bp:
    Cylinder(radius=flange_r, height=y_flange_1 - y_flange_0, align=(Align.CENTER, Align.CENTER, Align.MIN))
  flange_part = flange_bp.part.translate((0.0, 0.0, y_flange_0))

  with BuildPart() as front_bp:
    Cylinder(radius=core_r, height=y_key_0 - y_flange_1, align=(Align.CENTER, Align.CENTER, Align.MIN))
  front_part = front_bp.part.translate((0.0, 0.0, y_flange_1))

  key_part = make_keyed_shaft_solid(
      radius_mm=core_r,
      length_mm=y_key_1 - y_key_0,
      flat_depth_mm=flat_d,
  ).translate((0.0, 0.0, y_key_0))

  x_flat = core_r - flat_d
  y_flat = math.sqrt(max(1.0e-12, (core_r * core_r) - (x_flat * x_flat)))
  ang_deg = math.degrees(math.atan2(y_flat, x_flat))
  with BuildPart() as trans_bp:
    with BuildSketch(Plane.XY.offset(y_key_1)) as sk_a:
      with BuildLine():
        CenterArc((0.0, 0.0), core_r, start_angle=ang_deg, arc_size=360.0 - (2.0 * ang_deg))
        Line((x_flat, -y_flat), (x_flat, y_flat))
      make_face()
    with BuildSketch(Plane.XY.offset(y_trans_end)) as sk_b:
      Circle(radius=core_r)
    loft([sk_a.sketch.faces()[0], sk_b.sketch.faces()[0]], ruled=True)

  with BuildPart() as rear_bp:
    Cylinder(radius=core_r, height=y_nose_0 - y_trans_end, align=(Align.CENTER, Align.CENTER, Align.MIN))
  rear_part = rear_bp.part.translate((0.0, 0.0, y_trans_end))

  nose_part = make_keyed_shaft_solid(
      radius_mm=nose_r,
      length_mm=y_nose_1 - y_nose_0,
      flat_depth_mm=flat_d,
  ).translate((0.0, 0.0, y_nose_0))

  core_z = flange_part.fuse(front_part, key_part, trans_bp.part, rear_part, nose_part)
  return core_z.rotate(Axis.X, -ROT_90_DEG)


def _build_camshaft_core(cfg: PrintKitConfig, cam_angle_deg: float) -> Part:
  """Pose split D-keyed camshaft core at cam_angle_deg."""
  s, core_r, nose_r, flat_d = _cam_specs(cfg)
  _, _, face_w, gear_y, _ = _timing_params(cfg)
  core_local = _cached_camshaft_core_local(s, core_r, nose_r, flat_d, face_w, gear_y)
  core_world = core_local.rotate(Axis.Y, cam_angle_deg).translate((0.0, 0.0, CAM_AXIS_Z_MM * s))
  return _ensure_single_part(core_world, "kit_camshaft_core")


def _build_lobe_solid_xy(is_intake: bool, s: float, z_start: float, width_mm: float) -> Part:
  """Extrude one smooth 4-5-6-7 polynomial cam lobe along +Z (mapped to +Y later)."""
  pts_xy = [(px * s, -pz * s) for px, pz in _sample_cam_lobe_1to1(is_intake)]
  with BuildPart() as lobe_bp:
    with BuildSketch(Plane.XY.offset(z_start)):
      Polygon(pts_xy)
    extrude(amount=width_mm)
  return lobe_bp.part


@functools.lru_cache(maxsize=8)
def _cached_cam_lobes_local(
    s: float,
    core_r: float,
    flat_d: float,
    r_slip: float,
    min_wall: float,
) -> Part:
  """Cached unposed dual-lobe D-keyed sleeve in world orientation at cam_angle=0 (centered Z=0)."""
  lobe_w = CAM_LOBE_WIDTH_1TO1_MM * s
  y_in = CAM_INTAKE_Y_1TO1_MM * s
  y_ex = CAM_EXHAUST_Y_1TO1_MM * s
  half_spacer = CAM_SLEEVE_HALF_Y_1TO1_MM * s
  base_r = CAM_BASE_R_1TO1_MM * s
  sleeve_r = min(base_r, max(core_r + r_slip + min_wall, CAM_SLEEVE_R_1TO1_MM * s))

  in_lobe = _build_lobe_solid_xy(True, s, z_start=y_in - (HALF_FACTOR * lobe_w), width_mm=lobe_w)
  ex_lobe = _build_lobe_solid_xy(False, s, z_start=y_ex - (HALF_FACTOR * lobe_w), width_mm=lobe_w)

  with BuildPart() as spacer_bp:
    with Locations((0.0, 0.0, -half_spacer)):
      Cylinder(radius=sleeve_r, height=2.0 * half_spacer, align=(Align.CENTER, Align.CENTER, Align.MIN))
    flare_h = max(0.3, min(half_spacer, base_r - sleeve_r))
    if base_r > sleeve_r + 1.0e-4:
      with Locations((0.0, 0.0, half_spacer - flare_h)):
        Cone(
            bottom_radius=sleeve_r,
            top_radius=base_r,
            height=flare_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

  total_span = (2.0 * half_spacer) + (2.0 * lobe_w)
  bore_cut = make_keyed_bore_cutter(
      radius_mm=core_r,
      length_mm=total_span + CUT_MARGIN_MM,
      flat_depth_mm=flat_d,
      clearance_mm=r_slip,
  ).translate((0.0, 0.0, -(HALF_FACTOR * total_span) - (HALF_FACTOR * CUT_MARGIN_MM)))

  sleeve_xy = Part(in_lobe.fuse(ex_lobe, spacer_bp.part).cut(bore_cut).wrapped)
  return sleeve_xy.rotate(Axis.X, -ROT_90_DEG)


def _build_cam_lobes(cfg: PrintKitConfig, cam_angle_deg: float) -> Part:
  """Pose dual-lobe D-keyed cam sleeve at cam_angle_deg."""
  s, core_r, _, flat_d = _cam_specs(cfg)
  r_slip = slip_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg)
  sleeve_local = _cached_cam_lobes_local(s, core_r, flat_d, r_slip, min_wall)
  sleeve_world = sleeve_local.rotate(Axis.Y, cam_angle_deg).translate((0.0, 0.0, CAM_AXIS_Z_MM * s))
  return _ensure_single_part(sleeve_world, "kit_cam_lobes")


@functools.lru_cache(maxsize=8)
def _cached_rocker_shaft_local(s: float) -> tuple[Part, float]:
  """Cached stepped rocker shaft oriented along +Y and (solid, y_flange_0)."""
  shaft_r = max(ROCKER_SHAFT_MIN_R_MM, ROCKER_SHAFT_R_1TO1_MM * s)
  flange_r = shaft_r + min(1.5, 2.0 * s)
  flange_t = max(ROCKER_SHAFT_MIN_FLANGE_T_MM, ROCKER_SHAFT_FLANGE_T_1TO1_MM * s)

  y_front_face = (ROCKER_TOWER_FRONT_Y_1TO1_MM * s) - (0.05 * s)
  y_flange_0 = min(-31.0 * s, y_front_face - flange_t)
  y_rear_end = ROCKER_SHAFT_MAX_Y_1TO1_MM * s
  chamf = min(0.3, 0.25 * shaft_r)

  total_len = y_rear_end - y_flange_0
  z_step = y_front_face - y_flange_0
  pts = [
      (0.0, 0.0),
      (flange_r, 0.0),
      (flange_r, z_step),
      (shaft_r, z_step),
      (shaft_r, total_len - chamf),
      (shaft_r - chamf, total_len),
      (0.0, total_len),
  ]
  with BuildPart() as shaft_bp:
    with BuildSketch(Plane.XZ):
      Polygon(pts)
    revolve(axis=Axis.Z)

  return shaft_bp.part.rotate(Axis.X, -ROT_90_DEG), y_flange_0


@functools.lru_cache(maxsize=8)
def _cached_rocker_arm_intake(s: float, r_run: float, min_wall: float) -> Part:
  """Cached intake-side rocker arm built cleanly with unified boss/bridge sketch (zero seams)."""
  shaft_r = max(ROCKER_SHAFT_MIN_R_MM, ROCKER_SHAFT_R_1TO1_MM * s)
  bore_r = shaft_r + r_run
  boss_r = max(ROCKER_BOSS_R_1TO1_MM * s, bore_r + min_wall + 0.15)

  pivot_x = ROCKER_PIVOT_X_1TO1_MM * s
  pivot_y = CAM_INTAKE_Y_1TO1_MM * s
  pivot_z = ROCKER_PIVOT_Z_1TO1_MM * s
  thick_y = ROCKER_THICK_Y_1TO1_MM * s

  body_plane = Plane(origin=(0.0, pivot_y, 0.0), x_dir=(1.0, 0.0, 0.0), z_dir=(0.0, -1.0, 0.0))
  with BuildPart() as body_bp:
    with BuildSketch(body_plane):
      with Locations((pivot_x, pivot_z)):
        Circle(radius=boss_r)
      with Locations((HALF_FACTOR * pivot_x, pivot_z + (4.0 * s))):
        Rectangle(pivot_x, 5.0 * s)
      with Locations((0.0, (93.4 + 6.5) * s)):
        Rectangle(5.2 * s, 7.0 * s)
      with Locations((0.0, (93.4 + 2.6) * s)):
        Circle(radius=2.6 * s)
    extrude(amount=HALF_FACTOR * thick_y, both=True)

  tip_x = 35.67 * s
  tip_z = 80.3 * s
  arm_dir = (0.65, 0.45, -0.60)
  with BuildPart() as arm_bp:
    with BuildSketch(Plane(origin=(pivot_x, pivot_y, pivot_z), z_dir=arm_dir)):
      Rectangle(8.0 * s, 8.0 * s)
    with BuildSketch(Plane(origin=(tip_x, 0.0, tip_z + (3.0 * s)), z_dir=arm_dir)):
      Rectangle(7.0 * s, 7.0 * s)
    loft()
    with Locations((tip_x, 0.0, tip_z + (2.0 * s))):
      Cylinder(radius=3.5 * s, height=4.0 * s)

  with BuildPart() as bore_bp:
    with Locations((pivot_x, pivot_y, pivot_z)):
      Cylinder(radius=bore_r, height=thick_y + CUT_MARGIN_MM, rotation=(ROT_90_DEG, 0.0, 0.0))

  arm = Part(body_bp.part.fuse(arm_bp.part).cut(bore_bp.part).wrapped)
  return _ensure_single_part(arm, "kit_rocker_arm")


def _build_base_rocker_arm_intake(cfg: PrintKitConfig) -> Part:
  """Return cached intake-side rocker arm at rest with printable thickened pivot boss."""
  return _cached_rocker_arm_intake(
      cfg.scale,
      running_clearance_radial_mm(cfg),
      min_printable_wall_mm(cfg),
  )


def _pose_rocker_arm(
    base_intake_arm: Part,
    s: float,
    is_intake: bool,
    valve_lift_1to1_mm: float,
    label: str,
) -> Part:
  """Pose the canonical rocker arm to intake or exhaust side at current valve lift."""
  alpha_deg = math.degrees(math.asin(min(1.0, max(-1.0, (valve_lift_1to1_mm / ROCKER_RATIO) / ROCKER_PIVOT_X_1TO1_MM))))
  rot_deg = alpha_deg * ROCKER_CONTACT_SCALE
  if is_intake:
    posed = base_intake_arm.rotate(Axis((ROCKER_PIVOT_X_1TO1_MM * s, 0.0, ROCKER_PIVOT_Z_1TO1_MM * s), (0.0, 1.0, 0.0)), rot_deg)
  else:
    posed = base_intake_arm.rotate(Axis.Z, ROT_180_DEG).rotate(
        Axis((-ROCKER_PIVOT_X_1TO1_MM * s, 0.0, ROCKER_PIVOT_Z_1TO1_MM * s), (0.0, 1.0, 0.0)),
        -rot_deg,
    )
  return _ensure_single_part(posed, label)


def _valve_and_spring_specs(cfg: PrintKitConfig) -> dict[str, float]:
  """Compute consistent clamped valve stem, neck groove, wave spring, and retainer dimensions."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg)

  r_stem = HALF_FACTOR * max(VALVE_STEM_MIN_D_MM, VALVE_STEM_D_1TO1_MM * s)
  margin = max(VALVE_MARGIN_MIN_MM, VALVE_MARGIN_1TO1_MM * s)
  seat_h = max(VALVE_SEAT_H_MIN_MM, VALVE_SEAT_H_1TO1_MM * s)
  stem_len = VALVE_STEM_LEN_1TO1_MM * s

  ret_flange_h = max(RETAINER_FLANGE_MIN_H_MM, RETAINER_FLANGE_H_1TO1_MM * s)
  ret_step_h = max(RETAINER_STEP_MIN_H_MM, RETAINER_STEP_H_1TO1_MM * s)
  s_ret_top = VALVE_RETAINER_TOP_S_1TO1_MM * s
  s_flange_bot = s_ret_top - ret_flange_h
  s_ret_bot = s_flange_bot - ret_step_h

  r_neck = max(VALVE_NECK_MIN_R_MM, r_stem - max(VALVE_NECK_MIN_STEP_MM, 0.30 * r_stem))

  r_spring_in = max(r_stem + r_run + SPRING_MIN_RADIAL_CLEAR_MM, SPRING_INNER_R_1TO1_MM * s)
  target_r_out = (10.0 * s) if s < 0.8 else (SPRING_OUTER_R_1TO1_MM * s)
  r_spring_out = min(SPRING_OUTER_MAX_SCALE_R_MM * s, max(r_spring_in + max(min_wall, 1.2), target_r_out))

  # Ensure spring bottom clears cylinder_head's vertical spring pocket inboard wall at small scales.
  sin_t = math.sin(math.radians(VALVE_TILT_DEG))
  cos_t = math.cos(math.radians(VALVE_TILT_DEG))
  s_min_inboard = ((SPRING_POCKET_INBOARD_X_1TO1_MM * s) + (r_spring_out * cos_t) - (VALVE_IN_X_1TO1_MM * s)) / sin_t
  s_spring_bot = max(SPRING_OFFSET_S_1TO1_MM * s, s_min_inboard + 0.2)
  spring_installed_h = (s_flange_bot - SPRING_TOP_GAP_MM) - s_spring_bot

  return {
      "s": s,
      "r_run": r_run,
      "r_slip": r_slip,
      "r_stem": r_stem,
      "margin": margin,
      "seat_h": seat_h,
      "stem_len": stem_len,
      "ret_flange_h": ret_flange_h,
      "ret_step_h": ret_step_h,
      "s_ret_top": s_ret_top,
      "s_flange_bot": s_flange_bot,
      "s_ret_bot": s_ret_bot,
      "r_neck": r_neck,
      "r_spring_in": r_spring_in,
      "r_spring_out": r_spring_out,
      "s_spring_bot": s_spring_bot,
      "spring_installed_h": spring_installed_h,
  }


def _place_on_valve_axis(part: Part, is_intake: bool, s: float, s_offset_mm: float) -> Part:
  """Orient a local +Z valve-axis part and place it along intake/exhaust inclined axis."""
  sin_t = math.sin(math.radians(VALVE_TILT_DEG))
  cos_t = math.cos(math.radians(VALVE_TILT_DEG))
  if is_intake:
    vx = VALVE_IN_X_1TO1_MM * s
    axis_dir = (sin_t, 0.0, cos_t)
    rot_y = VALVE_TILT_DEG
  else:
    vx = VALVE_EX_X_1TO1_MM * s
    axis_dir = (-sin_t, 0.0, cos_t)
    rot_y = -VALVE_TILT_DEG

  px = vx + (axis_dir[0] * s_offset_mm)
  pz = (VALVE_SEAT_Z_1TO1_MM * s) + (axis_dir[2] * s_offset_mm)
  return part.rotate(Axis.Y, rot_y).translate((px, 0.0, pz))


@functools.lru_cache(maxsize=8)
def _cached_poppet_valve_local(
    s: float,
    is_intake: bool,
    r_stem: float,
    margin: float,
    seat_h: float,
    stem_len: float,
    r_neck: float,
    groove_bot: float,
    groove_top: float,
) -> Part:
  """Cached unposed poppet valve revolved along +Z with <=40-deg neck groove flare."""
  head_d = (VALVE_HEAD_IN_1TO1_MM if is_intake else VALVE_HEAD_EX_1TO1_MM) * s
  r_head = HALF_FACTOR * head_d
  r_seat_top = r_head - seat_h
  neck_blend = VALVE_NECK_BLEND_1TO1_MM * s
  flare_top = min(stem_len - 0.15, groove_top + (1.25 * (r_stem - r_neck)))

  pts = [
      (0.0, 0.0),
      (r_head, 0.0),
      (r_head, margin),
      (r_seat_top, margin + seat_h),
      (r_stem, margin + seat_h + neck_blend),
      (r_stem, groove_bot),
      (r_neck, groove_bot),
      (r_neck, groove_top),
      (r_stem, flare_top),
      (r_stem, stem_len),
      (0.0, stem_len),
  ]
  with BuildPart() as valve_bp:
    with BuildSketch(Plane.XZ):
      Polygon(pts)
    revolve(axis=Axis.Z)
  return valve_bp.part


def _build_poppet_valve(
    specs: dict[str, float],
    is_intake: bool,
    lift_mm: float,
    label: str,
) -> Part:
  """Pose printable intake/exhaust poppet valve along valve axis at lift_mm."""
  s = specs["s"]
  valve_local = _cached_poppet_valve_local(
      s=s,
      is_intake=is_intake,
      r_stem=specs["r_stem"],
      margin=specs["margin"],
      seat_h=specs["seat_h"],
      stem_len=specs["stem_len"],
      r_neck=specs["r_neck"],
      groove_bot=specs["s_flange_bot"] - VALVE_GROOVE_AXIAL_CLEAR_MM,
      groove_top=specs["s_ret_top"] + VALVE_GROOVE_AXIAL_CLEAR_MM,
  )
  placed = _place_on_valve_axis(valve_local, is_intake=is_intake, s=s, s_offset_mm=-lift_mm)
  return _ensure_single_part(placed, label)


def _build_valve_spring(
    specs: dict[str, float],
    is_intake: bool,
    lift_mm: float,
    label: str,
) -> Part:
  """Build a compliant wave spring sized inside cylinder_head pocket and compressed by lift."""
  spring_h = round(max(SPRING_MIN_ACTIVE_HEIGHT_MM, specs["spring_installed_h"] - lift_mm), 4)
  r_in = round(specs["r_spring_in"], 4)
  r_out = round(specs["r_spring_out"], 4)
  spring_local = _cached_wave_spring(r_out, r_in, spring_h)
  placed = _place_on_valve_axis(spring_local, is_intake=is_intake, s=specs["s"], s_offset_mm=specs["s_spring_bot"])
  return _ensure_single_part(placed, label)


@functools.lru_cache(maxsize=8)
def _cached_spring_retainer_local(
    r_out: float,
    r_step: float,
    r_stem: float,
    r_neck: float,
    r_slip: float,
    step_h: float,
    flange_h: float,
) -> Part:
  """Cached canonical side-snap C-slot retainer collar along local +Z."""
  r_bore_step = r_stem + r_slip
  r_bore_flange = r_neck + min(0.15, r_slip)
  slot_half_flange = max(0.5, r_neck - RETAINER_PINCH_MM)
  slot_len = r_out + 1.0

  with BuildPart() as ret_bp:
    with Locations((0.0, 0.0, HALF_FACTOR * step_h)):
      Cylinder(radius=r_step, height=step_h)
    with Locations((0.0, 0.0, step_h + (HALF_FACTOR * flange_h))):
      Cylinder(radius=r_out, height=flange_h)
    with Locations((0.0, 0.0, HALF_FACTOR * step_h)):
      Cylinder(radius=r_bore_step, height=step_h + 0.2, mode=Mode.SUBTRACT)
    with Locations((0.0, 0.0, step_h + (HALF_FACTOR * flange_h))):
      Cylinder(radius=r_bore_flange, height=flange_h + 0.2, mode=Mode.SUBTRACT)
    with Locations((0.0, HALF_FACTOR * slot_len, HALF_FACTOR * step_h)):
      Box(2.0 * r_bore_step, slot_len, step_h + 0.2, mode=Mode.SUBTRACT)
    with Locations((0.0, HALF_FACTOR * slot_len, step_h + (HALF_FACTOR * flange_h))):
      Box(2.0 * slot_half_flange, slot_len, flange_h + 0.2, mode=Mode.SUBTRACT)

  return ret_bp.part


def _solve_scaled_lifts(crank_angle_deg: float, s: float) -> tuple[float, float, float, float]:
  """Evaluate 4-stroke valve lifts without CAD twin rebuilds.

  Returns:
    ``(in_lift_1to1_mm, ex_lift_1to1_mm, in_lift_scaled_mm, ex_lift_scaled_mm)``.
  """
  cam = _cam_kinematics_engine()
  th_arr = np.array([crank_angle_deg], dtype=float)
  in_1to1 = float(cam.evaluate_valve_lift_crank_angle(cam.intake_params, th_arr)[0] * 1000.0)
  ex_1to1 = float(cam.evaluate_valve_lift_crank_angle(cam.exhaust_params, th_arr)[0] * 1000.0)
  return in_1to1, ex_1to1, in_1to1 * s, ex_1to1 * s


def _build_all_posed_components(
    cfg: PrintKitConfig,
    crank_angle_deg: float,
) -> dict[str, KitPart]:
  """Construct all 20 physical valvetrain instances posed at crank_angle_deg."""
  s, module_mm, face_w, gear_y, pin_r = _timing_params(cfg)
  _, _, cam_nose_r, cam_flat_d = _cam_specs(cfg)
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  lift_in_1to1, lift_ex_1to1, lift_in_mm, lift_ex_mm = _solve_scaled_lifts(crank_angle_deg, s)
  cam_angle_deg = HALF_FACTOR * crank_angle_deg

  # 1. 5-Gear 2:1 timing drive + 3 stepped idler pins.
  crank_gear_xy = _cached_gear_solid_xy(
      CRANK_GEAR_TEETH,
      module_mm,
      face_w,
      CRANK_TIMING_JOURNAL_R_1TO1_MM * s,
      max(CRANK_TIMING_FLAT_MIN_MM, CRANK_TIMING_FLAT_1TO1_MM * s),
      r_slip,
  )
  crank_gear = _pose_gear_xy(
      "kit_timing_gear_crank_14t",
      crank_gear_xy,
      y_center=gear_y,
      z_center=CRANK_AXIS_Z_MM * s,
      phase_rot_y_deg=crank_angle_deg,
  )

  idler_gear_xy = _cached_gear_solid_xy(IDLER_GEAR_TEETH, module_mm, face_w, pin_r, None, r_run)
  idler_pin_local = _cached_idler_pin_local(s, face_w, gear_y, pin_r)

  idler_ratio = CRANK_GEAR_TEETH / IDLER_GEAR_TEETH
  idler_phases_deg = (
      7.5 - (crank_angle_deg * idler_ratio),
      crank_angle_deg * idler_ratio,
      7.5 - (crank_angle_deg * idler_ratio),
  )
  idler_z_centers = tuple((CRANK_AXIS_Z_MM * s) + (k * module_mm) for k in IDLER_K_STEPS)

  cam_gear_xy = _cached_gear_solid_xy(
      CAM_GEAR_TEETH,
      module_mm,
      face_w,
      cam_nose_r,
      cam_flat_d,
      r_slip,
  )
  cam_gear = _pose_gear_xy(
      "kit_timing_gear_cam_28t",
      cam_gear_xy,
      y_center=gear_y,
      z_center=CAM_AXIS_Z_MM * s,
      phase_rot_y_deg=cam_angle_deg,
  )

  # 2. Split keyed camshaft + dual-lobe sleeve.
  cam_core = _build_camshaft_core(cfg, cam_angle_deg=cam_angle_deg)
  cam_lobes = _build_cam_lobes(cfg, cam_angle_deg=cam_angle_deg)

  # 3. Rocker shafts & rocker arms.
  r_shaft_local, r_shaft_y0 = _cached_rocker_shaft_local(s)
  rocker_shaft_in = _ensure_single_part(
      r_shaft_local.translate((ROCKER_PIVOT_X_1TO1_MM * s, r_shaft_y0, ROCKER_PIVOT_Z_1TO1_MM * s)),
      "kit_rocker_shaft_intake",
  )
  rocker_shaft_ex = _ensure_single_part(
      r_shaft_local.translate((-ROCKER_PIVOT_X_1TO1_MM * s, r_shaft_y0, ROCKER_PIVOT_Z_1TO1_MM * s)),
      "kit_rocker_shaft_exhaust",
  )
  base_arm = _build_base_rocker_arm_intake(cfg)
  rocker_in = _pose_rocker_arm(base_arm, s, True, lift_in_1to1, "kit_rocker_arm_intake")
  rocker_ex = _pose_rocker_arm(base_arm, s, False, lift_ex_1to1, "kit_rocker_arm_exhaust")

  # 4. Valves, compliant wave springs, and side-snap retainers.
  specs = _valve_and_spring_specs(cfg)
  valve_in = _build_poppet_valve(specs, True, lift_in_mm, "kit_valve_intake")
  valve_ex = _build_poppet_valve(specs, False, lift_ex_mm, "kit_valve_exhaust")
  spring_in = _build_valve_spring(specs, True, lift_in_mm, "kit_valve_spring_intake")
  spring_ex = _build_valve_spring(specs, False, lift_ex_mm, "kit_valve_spring_exhaust")
  ret_local = _cached_spring_retainer_local(
      specs["r_spring_out"],
      specs["r_spring_in"] - 0.15,
      specs["r_stem"],
      specs["r_neck"],
      specs["r_slip"],
      specs["ret_step_h"],
      specs["ret_flange_h"],
  )
  retainer_in = _ensure_single_part(
      _place_on_valve_axis(ret_local, True, s, specs["s_ret_bot"] - lift_in_mm),
      "kit_spring_retainer_intake",
  )
  retainer_ex = _ensure_single_part(
      _place_on_valve_axis(ret_local, False, s, specs["s_ret_bot"] - lift_ex_mm),
      "kit_spring_retainer_exhaust",
  )

  parts: Dict[str, KitPart] = {
      "kit_timing_gear_crank_14t": KitPart(
          name="kit_timing_gear_crank_14t",
          part=crank_gear,
          quantity=1,
          category="timing_drive",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="14T involute crank timing gear with D-keyed bore.",
      ),
  }
  for idx, (z_i, ph_i) in enumerate(zip(idler_z_centers, idler_phases_deg), start=1):
    g_name = f"kit_timing_gear_idler_24t_{idx}"
    p_name = f"kit_timing_idler_pin_{idx}"
    parts[g_name] = KitPart(
        name=g_name,
        part=_pose_gear_xy(g_name, idler_gear_xy, y_center=gear_y, z_center=z_i, phase_rot_y_deg=ph_i),
        quantity=1,
        category="timing_drive",
        preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
        notes=f"24T involute timing idler gear #{idx}.",
    )
    parts[p_name] = KitPart(
        name=p_name,
        part=_ensure_single_part(idler_pin_local.translate((0.0, 0.0, z_i)), p_name),
        quantity=1,
        category="timing_drive",
        preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
        notes=f"Stepped timing idler pin #{idx} with 45-deg flared shoulder.",
    )

  parts.update({
      "kit_timing_gear_cam_28t": KitPart(
          name="kit_timing_gear_cam_28t",
          part=cam_gear,
          quantity=1,
          category="timing_drive",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="28T involute cam timing gear with D-keyed bore.",
      ),
      "kit_camshaft_core": KitPart(
          name="kit_camshaft_core",
          part=cam_core,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Split D-keyed SOHC camshaft core sliding axially from -Y.",
      ),
      "kit_cam_lobes": KitPart(
          name="kit_cam_lobes",
          part=cam_lobes,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Dual 4-5-6-7 polynomial intake/exhaust lobe sleeve with D-keyed bore.",
      ),
      "kit_rocker_shaft_intake": KitPart(
          name="kit_rocker_shaft_intake",
          part=rocker_shaft_in,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Intake rocker pivot shaft with front retaining flange.",
      ),
      "kit_rocker_shaft_exhaust": KitPart(
          name="kit_rocker_shaft_exhaust",
          part=rocker_shaft_ex,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Exhaust rocker pivot shaft with front retaining flange.",
      ),
      "kit_rocker_arm_intake": KitPart(
          name="kit_rocker_arm_intake",
          part=rocker_in,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=None,
          notes="Intake rocker arm with thickened printable pivot boss.",
      ),
      "kit_rocker_arm_exhaust": KitPart(
          name="kit_rocker_arm_exhaust",
          part=rocker_ex,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=None,
          notes="Exhaust rocker arm with thickened printable pivot boss.",
      ),
      "kit_valve_intake": KitPart(
          name="kit_valve_intake",
          part=valve_in,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(0.0, -VALVE_TILT_DEG, 0.0),
          notes="Intake poppet valve with clamped stem/margin and retainer neck groove.",
      ),
      "kit_valve_exhaust": KitPart(
          name="kit_valve_exhaust",
          part=valve_ex,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(0.0, VALVE_TILT_DEG, 0.0),
          notes="Exhaust poppet valve with clamped stem/margin and retainer neck groove.",
      ),
      "kit_valve_spring_intake": KitPart(
          name="kit_valve_spring_intake",
          part=spring_in,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(0.0, -VALVE_TILT_DEG, 0.0),
          notes="Compliant wave flexure intake valve spring.",
      ),
      "kit_valve_spring_exhaust": KitPart(
          name="kit_valve_spring_exhaust",
          part=spring_ex,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(0.0, VALVE_TILT_DEG, 0.0),
          notes="Compliant wave flexure exhaust valve spring.",
      ),
      "kit_spring_retainer_intake": KitPart(
          name="kit_spring_retainer_intake",
          part=retainer_in,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(ROT_180_DEG, VALVE_TILT_DEG, 0.0),
          notes="Side-snap C-slot intake valve spring retainer collar.",
      ),
      "kit_spring_retainer_exhaust": KitPart(
          name="kit_spring_retainer_exhaust",
          part=retainer_ex,
          quantity=1,
          category="valvetrain",
          preferred_rotation_xyz_deg=(ROT_180_DEG, -VALVE_TILT_DEG, 0.0),
          notes="Side-snap C-slot exhaust valve spring retainer collar.",
      ),
  })
  return parts


def build_kit_valvetrain_parts(
    cfg: PrintKitConfig,
    crank_angle_deg: float = 0.0,
) -> dict[str, KitPart]:
  """Return the 12 canonical STL BOM KitParts for the valvetrain and timing drive."""
  all_parts = _build_all_posed_components(cfg, crank_angle_deg=crank_angle_deg)

  def _retag(key: str, canonical_name: str, quantity: int, notes: str) -> KitPart:
    kp = all_parts[key]
    kp.part.label = canonical_name
    return KitPart(
        name=canonical_name,
        part=kp.part,
        quantity=quantity,
        category=kp.category,
        preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
        notes=notes,
    )

  return {
      "kit_timing_gear_crank_14t": all_parts["kit_timing_gear_crank_14t"],
      "kit_timing_gear_idler_24t": _retag(
          "kit_timing_gear_idler_24t_1",
          "kit_timing_gear_idler_24t",
          quantity=3,
          notes="24T involute timing idler gear (print 3x for k=19,43,67).",
      ),
      "kit_timing_gear_cam_28t": all_parts["kit_timing_gear_cam_28t"],
      "kit_timing_idler_pin": _retag(
          "kit_timing_idler_pin_1",
          "kit_timing_idler_pin",
          quantity=3,
          notes="Stepped timing idler pin with 45-deg flared shoulder (print 3x).",
      ),
      "kit_camshaft_core": all_parts["kit_camshaft_core"],
      "kit_cam_lobes": all_parts["kit_cam_lobes"],
      "kit_rocker_shaft": _retag(
          "kit_rocker_shaft_intake",
          "kit_rocker_shaft",
          quantity=2,
          notes="Rocker pivot shaft with front retaining flange (print 2x).",
      ),
      "kit_rocker_arm": _retag(
          "kit_rocker_arm_intake",
          "kit_rocker_arm",
          quantity=2,
          notes="SOHC rocker arm with thickened printable pivot boss (print 2x).",
      ),
      "kit_valve_intake": all_parts["kit_valve_intake"],
      "kit_valve_exhaust": all_parts["kit_valve_exhaust"],
      "kit_valve_spring": _retag(
          "kit_valve_spring_intake",
          "kit_valve_spring",
          quantity=2,
          notes="Compliant wave flexure valve spring (print 2x).",
      ),
      "kit_spring_retainer": _retag(
          "kit_spring_retainer_intake",
          "kit_spring_retainer",
          quantity=2,
          notes="Side-snap C-slot valve spring retainer collar (print 2x).",
      ),
  }


def build_kit_valvetrain_assembly_parts(
    cfg: PrintKitConfig,
    crank_angle_deg: float = 0.0,
) -> dict[str, KitPart]:
  """Return all 20 individually posed valvetrain KitPart instances for assembly checks."""
  return _build_all_posed_components(cfg, crank_angle_deg=crank_angle_deg)
