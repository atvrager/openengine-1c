"""Parametric Billet Covers, Flywheel, Trigger Wheel, Manifold, and Shroud CAD.

Generates watertight B-Rep CAD models machined on the NestWorks C500:
- 6061-T6 SOHC cam cover (Z in [+102, +124] mm) with Viton O-ring groove
- 6061-T6 right timing cover capping the #25 chain bay & 30x47x7 Viton seal
- 1144 Stressproof flywheel (160 mm OD <= 165 mm C500 3-axis limit)
- 1144 36-1 ignition trigger wheel (74 mm OD <= 76 mm C500 ledger limit)
- 6061-T6 intake manifold adapter (cylinder head to Mikuni VM26 carburetor)
- 6061-T6 flywheel cooling fan shroud on -Y side
"""

from typing import List, Tuple
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
    revolve,
)

from openengine.cad.crankcase import (
    BLOCK_BASE_Z_MM,
    CRANK_AXIS_Z_MM,
    TIMING_COVER_BOLT_XZ_MM,
)
from openengine.cad.cylinder_head import IN_FLANGE_Z_MM, MANIFOLD_BOLT_OFFSET_Y_MM
from openengine.cad.valvetrain_hardware import (
    COVER_BOLT_XY_MM,
    M8_HEAD_BOLT_REAR_XY_MM,
    M8_SOCKET_RELIEF_RADIUS_MM,
    TOWER_BAY_HALF_X_MM,
    TOWER_BAY_HALF_Y_MM,
    TOWER_CHAIN_BAY_HALF_X_MM,
    TOWER_CHAIN_BAY_MAX_Y_MM,
    TOWER_CHAIN_BAY_MIN_Y_MM,
    TOWER_MAX_Y_MM,
    TOWER_MAX_Z_MM,
    TOWER_MIN_Y_MM,
    TOWER_WIDTH_X_MM,
)

CAM_COVER_HEIGHT_Z_MM: float = 22.0
CAM_COVER_CAVITY_TOP_Z_MM: float = 123.0
CAM_COVER_ORING_Y_MM: float = -2.0
CAM_COVER_ORING_RAD_MID_MM: float = 16.0
CAM_COVER_ORING_RAD_HALF_W_MM: float = 1.25
CAM_COVER_ORING_DEPTH_MM: float = 1.6

TIMING_COVER_WIDTH_X_MM: float = 96.0
TIMING_COVER_MIN_Y_MM: float = 52.0
TIMING_COVER_MAX_Y_MM: float = 68.0
TIMING_COVER_TOP_Z_MM: float = BLOCK_BASE_Z_MM - 0.2
TIMING_COVER_BOT_Z_MM: float = -196.0
TIMING_POCKET_WIDTH_X_MM: float = 82.0
TIMING_POCKET_MAX_Y_MM: float = 57.5
TIMING_POCKET_BOT_Z_MM: float = -192.0
RIGHT_SEAL_RADIUS_MM: float = 23.6
RIGHT_SEAL_MIN_Y_MM: float = 56.8
RIGHT_SEAL_MAX_Y_MM: float = 66.2
PTO_CLEAR_BORE_RADIUS_MM: float = 12.0

FLYWHEEL_OUTER_RADIUS_MM: float = 80.0
FLYWHEEL_RIM_INNER_RADIUS_MM: float = 58.0
FLYWHEEL_HUB_BORE_RADIUS_MM: float = 10.1
FLYWHEEL_RIM_WIDTH_Y_MM: float = 26.0
FLYWHEEL_WEB_WIDTH_Y_MM: float = 12.0

TRIGGER_TEETH_COUNT: int = 36
TRIGGER_ROOT_RADIUS_MM: float = 31.5
TRIGGER_TIP_RADIUS_MM: float = 37.0
TRIGGER_THICK_Y_MM: float = 5.5
TRIGGER_BORE_RADIUS_MM: float = 10.1
TRIGGER_TOOTH_HALF_DEG: float = 2.4

ADAPTER_START_X_MM: float = 60.2
ADAPTER_END_X_MM: float = 90.0
ADAPTER_FLANGE_THICK_X_MM: float = 8.0
ADAPTER_FLANGE_Y_MM: float = 64.0
ADAPTER_FLANGE_Z_MM: float = 38.0
ADAPTER_NECK_OUTER_R_MM: float = 17.5
ADAPTER_INLET_R_MM: float = 16.0
ADAPTER_CARB_R_MM: float = 13.0

SHROUD_WIDTH_X_MM: float = 128.0
SHROUD_MIN_Y_MM: float = -106.0
SHROUD_MAX_Y_MM: float = -98.5
SHROUD_MIN_Z_MM: float = -210.0
SHROUD_MAX_Z_MM: float = -76.0
SHROUD_CENTER_HOLE_R_MM: float = 15.0

M6_CLEARANCE_RADIUS_MM: float = 3.2
M6_CB_RADIUS_MM: float = 5.4
M6_CB_DEPTH_MM: float = 7.0
ROT_X_90_DEG: float = 90.0
ROT_Y_90_DEG: float = 90.0
CUT_MARGIN_MM: float = 4.0
HALF_FACTOR: float = 0.5
ORING_GROOVE_CLEAR_MM: float = 0.4
TRIGGER_TOOTH_ROOT_HALF_DEG: float = 4.0
FULL_CIRCLE_DEG: float = 360.0


def generate_cam_cover() -> Part:
  """Generate the 6061-T6 SOHC cam cover (Z in [+102, +124] mm) with 36T sprocket dome."""
  base_z = TOWER_MAX_Z_MM
  top_z = base_z + CAM_COVER_HEIGHT_Z_MM
  mid_z = (base_z + top_z) * HALF_FACTOR
  depth_y = TOWER_MAX_Y_MM - TOWER_MIN_Y_MM
  mid_y = (TOWER_MAX_Y_MM + TOWER_MIN_Y_MM) * HALF_FACTOR
  cav_h = CAM_COVER_CAVITY_TOP_Z_MM - base_z
  cav_mid_z = base_z + (cav_h * HALF_FACTOR)
  chain_d = TOWER_CHAIN_BAY_MAX_Y_MM - TOWER_CHAIN_BAY_MIN_Y_MM
  chain_y = (TOWER_CHAIN_BAY_MAX_Y_MM + TOWER_CHAIN_BAY_MIN_Y_MM) * HALF_FACTOR

  with BuildPart() as cover:
    with Locations((0.0, mid_y, mid_z)):
      Box(TOWER_WIDTH_X_MM, depth_y, CAM_COVER_HEIGHT_Z_MM)

    for rx, ry in M8_HEAD_BOLT_REAR_XY_MM:
      with Locations((rx, ry, mid_z)):
        Cylinder(radius=M8_SOCKET_RELIEF_RADIUS_MM, height=CAM_COVER_HEIGHT_Z_MM + CUT_MARGIN_MM, mode=Mode.SUBTRACT)

    # Hollow underside pockets clearing rockers and 36T cam sprocket
    with Locations((0.0, 0.0, cav_mid_z - 1.0)):
      Box(TOWER_BAY_HALF_X_MM * 2.0, TOWER_BAY_HALF_Y_MM * 2.0, cav_h + 2.0, mode=Mode.SUBTRACT)
    with Locations((0.0, chain_y, cav_mid_z - 1.0)):
      Box(TOWER_CHAIN_BAY_HALF_X_MM * 2.0, chain_d, cav_h + 2.0, mode=Mode.SUBTRACT)

    # Top Viton 30x2.0 mm O-ring inspection groove
    with BuildPart(mode=Mode.SUBTRACT):
      with Locations((0.0, CAM_COVER_ORING_Y_MM, top_z - (CAM_COVER_ORING_DEPTH_MM * HALF_FACTOR))):
        Cylinder(radius=CAM_COVER_ORING_RAD_MID_MM + CAM_COVER_ORING_RAD_HALF_W_MM, height=CAM_COVER_ORING_DEPTH_MM + ORING_GROOVE_CLEAR_MM)
        Cylinder(radius=CAM_COVER_ORING_RAD_MID_MM - CAM_COVER_ORING_RAD_HALF_W_MM, height=CAM_COVER_ORING_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

    # 4x M6x20 mounting clearance holes + top spotfaces
    for cx, cy in COVER_BOLT_XY_MM:
      with Locations((cx, cy, mid_z)):
        Cylinder(radius=M6_CLEARANCE_RADIUS_MM, height=CAM_COVER_HEIGHT_Z_MM + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
      with Locations((cx, cy, top_z - (M6_CB_DEPTH_MM * HALF_FACTOR))):
        Cylinder(radius=M6_CB_RADIUS_MM, height=M6_CB_DEPTH_MM + 1.0, mode=Mode.SUBTRACT)

  return cover.part


def generate_right_timing_cover() -> Part:
  """Generate the 6061-T6 right timing cover enclosing the #25 chain and 30x47x7 seal."""
  cover_d = TIMING_COVER_MAX_Y_MM - TIMING_COVER_MIN_Y_MM
  cover_y = (TIMING_COVER_MAX_Y_MM + TIMING_COVER_MIN_Y_MM) * HALF_FACTOR
  cover_h = TIMING_COVER_TOP_Z_MM - TIMING_COVER_BOT_Z_MM
  cover_z = (TIMING_COVER_TOP_Z_MM + TIMING_COVER_BOT_Z_MM) * HALF_FACTOR
  pkt_d = TIMING_POCKET_MAX_Y_MM - TIMING_COVER_MIN_Y_MM
  pkt_y = TIMING_COVER_MIN_Y_MM + (pkt_d * HALF_FACTOR)
  pkt_h = TIMING_COVER_TOP_Z_MM - TIMING_POCKET_BOT_Z_MM
  pkt_z = TIMING_COVER_TOP_Z_MM - (pkt_h * HALF_FACTOR)
  seal_d = RIGHT_SEAL_MAX_Y_MM - RIGHT_SEAL_MIN_Y_MM
  seal_y = (RIGHT_SEAL_MAX_Y_MM + RIGHT_SEAL_MIN_Y_MM) * HALF_FACTOR

  with BuildPart() as cover:
    with Locations((0.0, cover_y, cover_z)):
      Box(TIMING_COVER_WIDTH_X_MM, cover_d, cover_h)

    with Locations((0.0, pkt_y - 1.0, pkt_z + 1.0)):
      Box(TIMING_POCKET_WIDTH_X_MM, pkt_d + 2.0, pkt_h + 2.0, mode=Mode.SUBTRACT)

    with Locations((0.0, cover_y, CRANK_AXIS_Z_MM)):
      Cylinder(radius=PTO_CLEAR_BORE_RADIUS_MM, height=cover_d + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)
    with Locations((0.0, seal_y, CRANK_AXIS_Z_MM)):
      Cylinder(radius=RIGHT_SEAL_RADIUS_MM, height=seal_d, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

    for bx, bz in TIMING_COVER_BOLT_XZ_MM:
      with Locations((bx, cover_y, bz)):
        Cylinder(radius=M6_CLEARANCE_RADIUS_MM, height=cover_d + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)
      with Locations((bx, TIMING_COVER_MAX_Y_MM - (M6_CB_DEPTH_MM * HALF_FACTOR), bz)):
        Cylinder(radius=M6_CB_RADIUS_MM, height=M6_CB_DEPTH_MM + 1.0, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

  return cover.part


def generate_flywheel() -> Part:
  """Generate the 1144 steel inertia flywheel centered on Y axis (OD 160 mm <= 165 mm)."""
  rim_half = FLYWHEEL_RIM_WIDTH_Y_MM * HALF_FACTOR
  web_half = FLYWHEEL_WEB_WIDTH_Y_MM * HALF_FACTOR
  with BuildSketch(Plane.XY) as profile:
    Polygon([
        (FLYWHEEL_HUB_BORE_RADIUS_MM, -web_half),
        (FLYWHEEL_RIM_INNER_RADIUS_MM, -web_half),
        (FLYWHEEL_RIM_INNER_RADIUS_MM, -rim_half),
        (FLYWHEEL_OUTER_RADIUS_MM, -rim_half),
        (FLYWHEEL_OUTER_RADIUS_MM, rim_half),
        (FLYWHEEL_RIM_INNER_RADIUS_MM, rim_half),
        (FLYWHEEL_RIM_INNER_RADIUS_MM, web_half),
        (FLYWHEEL_HUB_BORE_RADIUS_MM, web_half),
    ])
  with BuildPart() as fw:
    revolve(profile.sketch, axis=Axis.Y)
  return fw.part


def generate_trigger_wheel() -> Part:
  """Generate the 1144 36-1 toothed ignition trigger wheel centered on Y axis (OD 74 mm <= 76 mm)."""
  pts: List[Tuple[float, float]] = []
  for idx in range(TRIGGER_TEETH_COUNT):
    base_deg = idx * (FULL_CIRCLE_DEG / TRIGGER_TEETH_COUNT)
    if idx == 0:
      # Missing tooth #0 index gap
      for deg in (base_deg - TRIGGER_TOOTH_HALF_DEG, base_deg + TRIGGER_TOOTH_HALF_DEG):
        rad = np.radians(deg)
        pts.append((TRIGGER_ROOT_RADIUS_MM * float(np.cos(rad)), TRIGGER_ROOT_RADIUS_MM * float(np.sin(rad))))
    else:
      for deg, rad_val in (
          (base_deg - TRIGGER_TOOTH_ROOT_HALF_DEG, TRIGGER_ROOT_RADIUS_MM),
          (base_deg - TRIGGER_TOOTH_HALF_DEG, TRIGGER_TIP_RADIUS_MM),
          (base_deg + TRIGGER_TOOTH_HALF_DEG, TRIGGER_TIP_RADIUS_MM),
          (base_deg + TRIGGER_TOOTH_ROOT_HALF_DEG, TRIGGER_ROOT_RADIUS_MM),
      ):
        rad = np.radians(deg)
        pts.append((rad_val * float(np.cos(rad)), rad_val * float(np.sin(rad))))

  with BuildSketch(Plane.XZ) as profile:
    Polygon(pts)
  with BuildPart() as wheel:
    extrude(profile.sketch, amount=TRIGGER_THICK_Y_MM * HALF_FACTOR, both=True)
    Cylinder(radius=TRIGGER_BORE_RADIUS_MM, height=TRIGGER_THICK_Y_MM + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)
  return wheel.part


def generate_intake_manifold_adapter() -> Part:
  """Generate the 6061-T6 intake manifold flange-to-VM26 adapter in world coordinates."""
  total_len = ADAPTER_END_X_MM - ADAPTER_START_X_MM
  flange_mid_x = ADAPTER_START_X_MM + (ADAPTER_FLANGE_THICK_X_MM * HALF_FACTOR)
  neck_len = total_len - ADAPTER_FLANGE_THICK_X_MM
  neck_mid_x = ADAPTER_START_X_MM + ADAPTER_FLANGE_THICK_X_MM + (neck_len * HALF_FACTOR)

  with BuildPart() as adapter:
    with Locations((flange_mid_x, 0.0, IN_FLANGE_Z_MM)):
      Box(ADAPTER_FLANGE_THICK_X_MM, ADAPTER_FLANGE_Y_MM, ADAPTER_FLANGE_Z_MM)
    with Locations((neck_mid_x, 0.0, IN_FLANGE_Z_MM)):
      Cylinder(radius=ADAPTER_NECK_OUTER_R_MM, height=neck_len, rotation=(0.0, ROT_Y_90_DEG, 0.0))

    # Bore matched to 32 mm head runner -> 26 mm VM26 throat
    with BuildPart(mode=Mode.SUBTRACT):
      with Locations((ADAPTER_START_X_MM + (total_len * HALF_FACTOR), 0.0, IN_FLANGE_Z_MM)):
        Cylinder(radius=ADAPTER_CARB_R_MM, height=total_len + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
      with Locations((flange_mid_x, 0.0, IN_FLANGE_Z_MM)):
        Cylinder(radius=ADAPTER_INLET_R_MM, height=ADAPTER_FLANGE_THICK_X_MM + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))

    for my in (-MANIFOLD_BOLT_OFFSET_Y_MM, MANIFOLD_BOLT_OFFSET_Y_MM):
      with Locations((flange_mid_x, my, IN_FLANGE_Z_MM)):
        Cylinder(radius=M6_CLEARANCE_RADIUS_MM, height=ADAPTER_FLANGE_THICK_X_MM + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)

  return adapter.part


def generate_fan_shroud() -> Part:
  """Generate the -Y flywheel cooling fan shroud plate in world coordinates."""
  shroud_d = SHROUD_MAX_Y_MM - SHROUD_MIN_Y_MM
  shroud_y = (SHROUD_MAX_Y_MM + SHROUD_MIN_Y_MM) * HALF_FACTOR
  shroud_h = SHROUD_MAX_Z_MM - SHROUD_MIN_Z_MM
  shroud_z = (SHROUD_MAX_Z_MM + SHROUD_MIN_Z_MM) * HALF_FACTOR

  with BuildPart() as shroud:
    with Locations((0.0, shroud_y, shroud_z)):
      Box(SHROUD_WIDTH_X_MM, shroud_d, shroud_h)
    with Locations((0.0, shroud_y, CRANK_AXIS_Z_MM)):
      Cylinder(radius=SHROUD_CENTER_HOLE_R_MM, height=shroud_d + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)

  return shroud.part
