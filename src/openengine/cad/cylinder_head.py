"""Parametric SOHC Pent-Roof Cylinder Head & Poppet Valve CAD via build123d.

Generates watertight B-Rep CAD models for the NestWorks C500 3-axis machined
6061-T6 cylinder head and 4140 poppet valves:
- Envelope 120.0 x 124.0 x 52.0 mm (<= 130 x 130 x 52 mm C500 3-axis limit)
- 30-degree pent-roof chamber tuned to ~20.5 cm3 (9.0:1 CR with gasket/crevice)
- Lowered intake (Z = +34 mm) and exhaust (Z = +24 mm) ports with M6 flange holes
- Counterbored M8 tie-bolt holes, bronze guide / seat / spring pockets, #25 chain tunnel
- Angled (-Y, 24 deg) NGK CMR7H M10x1.0 spark plug bore + 21 mm deep-socket well
- Watertight negative intake port fluid domain for OpenFOAM CFD flow bench
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from build123d import (
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Vector,
    extrude,
    insert,
    loft,
    revolve,
)

from openengine.cad.cylinder_block import DEFAULT_BORE_M

MM_PER_M: float = 1000.0
VALVE_IN_X_MM: float = 16.0
VALVE_EX_X_MM: float = -16.0
VALVE_SEAT_Z_MM: float = 6.0

IN_FLANGE_Z_MM: float = 34.0
IN_MID_X_MM: float = 36.0
IN_MID_Z_MM: float = 22.0
IN_MID_DIR_X: float = 0.85
IN_MID_DIR_Z: float = 0.52
EX_FLANGE_Z_MM: float = 24.0
EX_MID_X_MM: float = -36.0
EX_MID_Z_MM: float = 19.5

HEAD_CENTER_Y_MM: float = 2.0
CHAMBER_BASE_Z_MM: float = -3.0
PORT_MID_TAPER_MM: float = 1.5
SEAT_RECESS_HEIGHT_MM: float = 8.0
VALVE_GUIDE_BORE_DIAM_MM: float = 10.05
VALVE_GUIDE_START_S_MM: float = 15.0
VALVE_GUIDE_LEN_S_MM: float = 36.0
SEAT_POCKET_EXTRA_DIAM_MM: float = 1.6
SEAT_POCKET_HEIGHT_MM: float = 6.0
SPRING_POCKET_RADIUS_MM: float = 11.0
SPRING_POCKET_FLOOR_Z_MM: float = 42.0

HEAD_BOLT_COUNTERBORE_RADIUS_MM: float = 8.25
HEAD_BOLT_SEAT_Z_MM: float = 44.0
CAM_TOWER_BOLT_XY_MM: Tuple[Tuple[float, float], ...] = (
    (30.0, -22.0),
    (-30.0, -22.0),
    (30.0, 28.0),
    (-30.0, 28.0),
)
CAM_TOWER_BOLT_RADIUS_MM: float = 2.6
CAM_TOWER_BOLT_DEPTH_MM: float = 14.0
MANIFOLD_BOLT_OFFSET_Y_MM: float = 23.0
MANIFOLD_BOLT_RADIUS_MM: float = 2.6
MANIFOLD_BOLT_DEPTH_MM: float = 16.0

CHAIN_SLOT_WIDTH_X_MM: float = 78.0
CHAIN_SLOT_MIN_Y_MM: float = 48.8
CHAIN_SLOT_MAX_Y_MM: float = 57.2

PLUG_ORIGIN_Y_MM: float = -13.5
PLUG_ORIGIN_Z_MM: float = 8.5
PLUG_TILT_DEG: float = 24.0
PLUG_THREAD_RADIUS_MM: float = 5.1
PLUG_THREAD_END_S_MM: float = 17.0
PLUG_SOCKET_WELL_RADIUS_MM: float = 10.5
PLUG_SOCKET_START_S_MM: float = 15.5
PLUG_SOCKET_END_S_MM: float = 75.0
FIN_GROOVE_X_OFFSETS_MM: Tuple[float, ...] = (-22.0, 22.0)
FIN_GROOVE_Z_MM: Tuple[float, float, float] = (16.0, 26.0, 36.0)
CUT_MARGIN_MM: float = 4.0
HALF_FACTOR: float = 0.5
HALF_MM_FROM_DIAM_M: float = 500.0
DEFAULT_VALVE_HEAD_IN_M: float = 0.030
DEFAULT_VALVE_HEAD_EX_M: float = 0.026
DEFAULT_VALVE_STEM_M: float = 0.005
GASKET_AND_CREVICE_CM3: float = 4.05
DEFAULT_VALVE_STEM_LEN_M: float = 0.076
DEFAULT_VALVE_SEAT_ANGLE_DEG: float = 45.0
DEFAULT_VALVE_MARGIN_M: float = 0.0008
VALVE_SEAT_FACE_H_MM: float = 1.8
VALVE_NECK_BLEND_MM: float = 8.0
CHAMBER_SKETCH_HALF_X_MM: float = 40.0
VALVE_STEM_CUT_MID_S_MM: float = 30.0
VALVE_STEM_CLEAR_MM: float = 0.5
VALVE_STEM_CUT_LEN_MM: float = 70.0
ROT_90_DEG: float = 90.0
EX_MID_DIR_X: float = -0.8
EX_MID_DIR_Z: float = 0.4
FIN_GROOVE_SIZE_XY_MM: float = 12.0
FIN_GROOVE_HEIGHT_Z_MM: float = 3.0
DEFAULT_CFD_CYL_DEPTH_M: float = 0.050
DOME_CUT_Z_MM: float = -5.0
DOME_CUT_XY_MM: float = 100.0
DOME_CUT_HEIGHT_Z_MM: float = 10.0


@dataclass
class CylinderHeadParams:
  """Geometric specifications for the C500 billet SOHC pent-roof cylinder head."""

  bore_m: float = DEFAULT_BORE_M
  intake_valve_diam_m: float = DEFAULT_VALVE_HEAD_IN_M
  exhaust_valve_diam_m: float = DEFAULT_VALVE_HEAD_EX_M
  valve_angle_deg: float = 15.0
  intake_downdraft_angle_deg: float = 24.0
  intake_flange_diam_m: float = 0.032
  exhaust_flange_diam_m: float = 0.028
  spark_plug_diam_m: float = 0.010
  valve_stem_diam_m: float = DEFAULT_VALVE_STEM_M
  head_width_m: float = 0.120
  head_length_m: float = 0.124
  head_height_m: float = 0.052
  head_bolt_pitch_m: float = 0.080
  head_bolt_hole_diam_m: float = 0.0085
  chamber_apex_height_m: float = 0.0105
  squish_half_width_m: float = 0.0265


def generate_poppet_valve(
    head_diam_m: float = DEFAULT_VALVE_HEAD_IN_M,
    stem_diam_m: float = DEFAULT_VALVE_STEM_M,
    stem_length_m: float = DEFAULT_VALVE_STEM_LEN_M,
    seat_angle_deg: float = DEFAULT_VALVE_SEAT_ANGLE_DEG,
    margin_thickness_m: float = DEFAULT_VALVE_MARGIN_M,
) -> Part:
  """Generate a parametric revolved solid model of a poppet valve."""
  r_head = head_diam_m * HALF_MM_FROM_DIAM_M
  r_stem = stem_diam_m * HALF_MM_FROM_DIAM_M
  length = stem_length_m * MM_PER_M
  margin = margin_thickness_m * MM_PER_M
  seat_h = VALVE_SEAT_FACE_H_MM
  r_seat_top = r_head - seat_h * np.tan(np.radians(seat_angle_deg))

  with BuildSketch(Plane.XZ) as profile:
    Polygon([
        (0.0, 0.0),
        (r_head, 0.0),
        (r_head, margin),
        (r_seat_top, margin + seat_h),
        (r_stem, margin + seat_h + VALVE_NECK_BLEND_MM),
        (r_stem, length),
        (0.0, length),
    ])

  with BuildPart() as valve:
    revolve(profile.sketch, axis=Axis.Z)
  return valve.part


def _build_chamber_cutter(p: CylinderHeadParams, z_base: float = CHAMBER_BASE_Z_MM) -> Part:
  """Build pent-roof combustion chamber + optional below-deck bore volume."""
  bore_r = (p.bore_m * MM_PER_M) * HALF_FACTOR
  apex_h = p.chamber_apex_height_m * MM_PER_M
  squish_y = p.squish_half_width_m * MM_PER_M
  slope_deg = p.valve_angle_deg

  z_roof_edge = apex_h - CHAMBER_SKETCH_HALF_X_MM * np.tan(np.radians(slope_deg))
  with BuildSketch(Plane.XZ) as roof_sketch:
    Polygon([
        (-CHAMBER_SKETCH_HALF_X_MM, z_base),
        (-CHAMBER_SKETCH_HALF_X_MM, z_roof_edge),
        (0.0, apex_h),
        (CHAMBER_SKETCH_HALF_X_MM, z_roof_edge),
        (CHAMBER_SKETCH_HALF_X_MM, z_base),
    ])
  with BuildPart() as p1:
    extrude(roof_sketch.sketch, amount=squish_y, both=True)
  with BuildPart() as p2:
    with Locations((0.0, 0.0, ((apex_h - z_base) * HALF_FACTOR) + z_base)):
      Cylinder(radius=bore_r, height=apex_h - z_base)
  return p1.part & p2.part


def _build_port_cutter(
    p_flange: Vector,
    p_mid: Vector,
    p_seat: Vector,
    flange_dir: Vector,
    mid_dir: Vector,
    seat_dir: Vector,
    r_flange: float,
    r_seat: float,
    seat_rot_y_deg: float,
) -> Part:
  """Loft a watertight intake or exhaust runner and unite its valve seat pocket."""
  plane_f = Plane(origin=p_flange, z_dir=flange_dir)
  plane_m = Plane(origin=p_mid, z_dir=mid_dir)
  plane_s = Plane(origin=p_seat, z_dir=seat_dir)
  r_throat = r_seat - 1.0

  with BuildPart() as port:
    with BuildSketch(plane_f):
      Circle(radius=r_flange)
    with BuildSketch(plane_m):
      Circle(radius=r_flange - PORT_MID_TAPER_MM)
    with BuildSketch(plane_s):
      Circle(radius=r_throat)
    loft()
    with Locations(p_seat):
      Cylinder(radius=r_seat, height=SEAT_RECESS_HEIGHT_MM, rotation=(0.0, seat_rot_y_deg, 0.0))
  return port.part


def _subtract_valvetrain_pockets(p: CylinderHeadParams, h_mm: float) -> None:
  """Subtract guide counterbores, valve seat pockets, and upper spring pockets."""
  slope_deg = p.valve_angle_deg
  in_dir = Vector(np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg)))
  ex_dir = Vector(-np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg)))
  p_in = Vector(VALVE_IN_X_MM, 0.0, VALVE_SEAT_Z_MM)
  p_ex = Vector(VALVE_EX_X_MM, 0.0, VALVE_SEAT_Z_MM)
  guide_mid_s = VALVE_GUIDE_START_S_MM + (VALVE_GUIDE_LEN_S_MM * HALF_FACTOR)
  spring_h = (h_mm - SPRING_POCKET_FLOOR_Z_MM) + CUT_MARGIN_MM
  spring_mid_z = SPRING_POCKET_FLOOR_Z_MM + (spring_h * HALF_FACTOR)

  for p_seat, axis_dir, rot_deg, valve_d_m in (
      (p_in, in_dir, slope_deg, p.intake_valve_diam_m),
      (p_ex, ex_dir, -slope_deg, p.exhaust_valve_diam_m),
  ):
    with Locations(p_seat + axis_dir * VALVE_STEM_CUT_MID_S_MM):
      Cylinder(
          radius=(p.valve_stem_diam_m * MM_PER_M + VALVE_STEM_CLEAR_MM) * HALF_FACTOR,
          height=VALVE_STEM_CUT_LEN_MM,
          rotation=(0.0, rot_deg, 0.0),
          mode=Mode.SUBTRACT,
      )
    with Locations(p_seat + axis_dir * guide_mid_s):
      Cylinder(radius=VALVE_GUIDE_BORE_DIAM_MM * HALF_FACTOR, height=VALVE_GUIDE_LEN_S_MM, rotation=(0.0, rot_deg, 0.0), mode=Mode.SUBTRACT)
    r_seat_pkt = ((valve_d_m * MM_PER_M) + SEAT_POCKET_EXTRA_DIAM_MM) * HALF_FACTOR
    with Locations(p_seat + axis_dir * 2.0):
      Cylinder(radius=r_seat_pkt, height=SEAT_POCKET_HEIGHT_MM, rotation=(0.0, rot_deg, 0.0), mode=Mode.SUBTRACT)

  spring_x_offset = (h_mm - VALVE_SEAT_Z_MM) * float(np.tan(np.radians(slope_deg)))
  for sx in (VALVE_IN_X_MM + spring_x_offset, VALVE_EX_X_MM - spring_x_offset):
    with Locations((sx, 0.0, spring_mid_z)):
      Cylinder(radius=SPRING_POCKET_RADIUS_MM, height=spring_h, mode=Mode.SUBTRACT)


def _subtract_spark_plug_and_mounting(p: CylinderHeadParams, w_mm: float, h_mm: float) -> None:
  """Subtract -Y angled spark plug bore/socket well, head bolts, cam bolts, and chain slot."""
  tilt_rad = np.radians(PLUG_TILT_DEG)
  plug_origin = Vector(0.0, PLUG_ORIGIN_Y_MM, PLUG_ORIGIN_Z_MM)
  plug_dir = Vector(0.0, -float(np.cos(tilt_rad)), float(np.sin(tilt_rad)))
  plug_rot_x_deg = ROT_90_DEG - PLUG_TILT_DEG

  thread_len = PLUG_THREAD_END_S_MM + CUT_MARGIN_MM
  with Locations(plug_origin + plug_dir * ((PLUG_THREAD_END_S_MM - CUT_MARGIN_MM) * HALF_FACTOR)):
    Cylinder(radius=PLUG_THREAD_RADIUS_MM, height=thread_len, rotation=(plug_rot_x_deg, 0.0, 0.0), mode=Mode.SUBTRACT)

  well_len = PLUG_SOCKET_END_S_MM - PLUG_SOCKET_START_S_MM
  well_mid_s = (PLUG_SOCKET_END_S_MM + PLUG_SOCKET_START_S_MM) * HALF_FACTOR
  with Locations(plug_origin + plug_dir * well_mid_s):
    Cylinder(radius=PLUG_SOCKET_WELL_RADIUS_MM, height=well_len, rotation=(plug_rot_x_deg, 0.0, 0.0), mode=Mode.SUBTRACT)

  stud_pitch = (p.head_bolt_pitch_m * MM_PER_M) * HALF_FACTOR
  r_stud = (p.head_bolt_hole_diam_m * MM_PER_M) * HALF_FACTOR
  cb_h = (h_mm - HEAD_BOLT_SEAT_Z_MM) + CUT_MARGIN_MM
  for sx in (-stud_pitch, stud_pitch):
    for sy in (-stud_pitch, stud_pitch):
      with Locations((sx, sy, h_mm * HALF_FACTOR)):
        Cylinder(radius=r_stud, height=h_mm + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
      with Locations((sx, sy, HEAD_BOLT_SEAT_Z_MM + (cb_h * HALF_FACTOR))):
        Cylinder(radius=HEAD_BOLT_COUNTERBORE_RADIUS_MM, height=cb_h, mode=Mode.SUBTRACT)

  for bx, by in CAM_TOWER_BOLT_XY_MM:
    with Locations((bx, by, h_mm - (CAM_TOWER_BOLT_DEPTH_MM * HALF_FACTOR))):
      Cylinder(radius=CAM_TOWER_BOLT_RADIUS_MM, height=CAM_TOWER_BOLT_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

  for flange_x, flange_z in ((w_mm * HALF_FACTOR, IN_FLANGE_Z_MM), (-w_mm * HALF_FACTOR, EX_FLANGE_Z_MM)):
    for my in (-MANIFOLD_BOLT_OFFSET_Y_MM, MANIFOLD_BOLT_OFFSET_Y_MM):
      with Locations((flange_x - np.sign(flange_x) * (MANIFOLD_BOLT_DEPTH_MM * HALF_FACTOR), my, flange_z)):
        Cylinder(radius=MANIFOLD_BOLT_RADIUS_MM, height=MANIFOLD_BOLT_DEPTH_MM + 1.0, rotation=(0.0, ROT_90_DEG, 0.0), mode=Mode.SUBTRACT)

  slot_d = CHAIN_SLOT_MAX_Y_MM - CHAIN_SLOT_MIN_Y_MM
  slot_y = (CHAIN_SLOT_MAX_Y_MM + CHAIN_SLOT_MIN_Y_MM) * HALF_FACTOR
  with Locations((0.0, slot_y, h_mm * HALF_FACTOR)):
    Box(CHAIN_SLOT_WIDTH_X_MM, slot_d, h_mm + CUT_MARGIN_MM, mode=Mode.SUBTRACT)


def generate_cylinder_head(params: Optional[CylinderHeadParams] = None) -> Part:
  """Generate the watertight 3-axis billet SOHC pent-roof cylinder head B-Rep."""
  p = params or CylinderHeadParams()
  w = p.head_width_m * MM_PER_M
  length = p.head_length_m * MM_PER_M
  h = p.head_height_m * MM_PER_M
  slope_deg = p.valve_angle_deg
  front_y = HEAD_CENTER_Y_MM - (length * HALF_FACTOR)

  chamber_cutter = _build_chamber_cutter(p, z_base=CHAMBER_BASE_Z_MM)
  in_port = _build_port_cutter(
      Vector(w * HALF_FACTOR, 0.0, IN_FLANGE_Z_MM),
      Vector(IN_MID_X_MM, 0.0, IN_MID_Z_MM),
      Vector(VALVE_IN_X_MM, 0.0, VALVE_SEAT_Z_MM),
      Vector(1.0, 0.0, 0.0),
      Vector(IN_MID_DIR_X, 0.0, IN_MID_DIR_Z),
      Vector(np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg))),
      (p.intake_flange_diam_m * MM_PER_M) * HALF_FACTOR,
      (p.intake_valve_diam_m * MM_PER_M) * HALF_FACTOR,
      slope_deg,
  )
  ex_port = _build_port_cutter(
      Vector(-w * HALF_FACTOR, 0.0, EX_FLANGE_Z_MM),
      Vector(EX_MID_X_MM, 0.0, EX_MID_Z_MM),
      Vector(VALVE_EX_X_MM, 0.0, VALVE_SEAT_Z_MM),
      Vector(-1.0, 0.0, 0.0),
      Vector(EX_MID_DIR_X, 0.0, EX_MID_DIR_Z),
      Vector(-np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg))),
      (p.exhaust_flange_diam_m * MM_PER_M) * HALF_FACTOR,
      (p.exhaust_valve_diam_m * MM_PER_M) * HALF_FACTOR,
      -slope_deg,
  )

  with BuildPart() as head:
    with Locations((0.0, HEAD_CENTER_Y_MM, h * HALF_FACTOR)):
      Box(w, length, h)
    insert(chamber_cutter, mode=Mode.SUBTRACT)
    insert(in_port, mode=Mode.SUBTRACT)
    insert(ex_port, mode=Mode.SUBTRACT)
    _subtract_valvetrain_pockets(p, h)
    _subtract_spark_plug_and_mounting(p, w, h)
    for gx in FIN_GROOVE_X_OFFSETS_MM:
      for gz in FIN_GROOVE_Z_MM:
        with Locations((gx, front_y, gz)):
          Box(FIN_GROOVE_SIZE_XY_MM, FIN_GROOVE_SIZE_XY_MM, FIN_GROOVE_HEIGHT_Z_MM, mode=Mode.SUBTRACT)

  return head.part


def extract_intake_port_fluid_domain(
    params: Optional[CylinderHeadParams] = None,
    cylinder_depth_m: float = DEFAULT_CFD_CYL_DEPTH_M,
) -> Part:
  """Extract the negative fluid volume of the intake port, valve seat, and cylinder bore."""
  p = params or CylinderHeadParams()
  w = p.head_width_m * MM_PER_M
  bore_r = (p.bore_m * MM_PER_M) * HALF_FACTOR
  depth_mm = cylinder_depth_m * MM_PER_M
  slope_deg = p.valve_angle_deg

  port = _build_port_cutter(
      Vector(w * HALF_FACTOR, 0.0, IN_FLANGE_Z_MM),
      Vector(IN_MID_X_MM, 0.0, IN_MID_Z_MM),
      Vector(VALVE_IN_X_MM, 0.0, VALVE_SEAT_Z_MM),
      Vector(1.0, 0.0, 0.0),
      Vector(IN_MID_DIR_X, 0.0, IN_MID_DIR_Z),
      Vector(np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg))),
      (p.intake_flange_diam_m * MM_PER_M) * HALF_FACTOR,
      (p.intake_valve_diam_m * MM_PER_M) * HALF_FACTOR,
      slope_deg,
  )

  chamber_and_cyl = _build_chamber_cutter(p, z_base=-depth_mm)
  with BuildPart() as cyl_filler:
    with Locations((0.0, 0.0, -depth_mm * HALF_FACTOR)):
      Cylinder(radius=bore_r, height=depth_mm)

  chamber_full = chamber_and_cyl.fuse(cyl_filler.part)
  return port.fuse(chamber_full)


def calculate_combustion_chamber_volume(
    params: Optional[CylinderHeadParams] = None,
) -> Tuple[float, float]:
  """Calculate the geometric combustion chamber dome volume above deck face (Z = 0)."""
  p = params or CylinderHeadParams()
  chamber = _build_chamber_cutter(p, z_base=CHAMBER_BASE_Z_MM)
  with BuildPart() as cutter:
    with Locations((0.0, 0.0, DOME_CUT_Z_MM)):
      Box(DOME_CUT_XY_MM, DOME_CUT_XY_MM, DOME_CUT_HEIGHT_Z_MM)
  dome = chamber - cutter.part
  vol_mm3 = float(dome.volume)
  return vol_mm3, vol_mm3 / MM_PER_M
