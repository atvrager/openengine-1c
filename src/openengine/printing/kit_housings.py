"""3D-printable snap-fit engine housing, cover, manifold, and shroud parts.

Generates watertight ``build123d.Part`` solids in scaled master-assembly world
coordinates (master origin ``(0, 0, 0)`` at cylinder deck ``Z = 0``, crank axis at
``(0, 0, -159.0 * scale)``, cam axis at ``(0, 0, +76.5 * scale)``). All walls,
cooling fins, horizontal teardrop bores, alignment pins, and cantilever snap hooks
are parameterized from ``PrintKitConfig`` process tolerances so parts remain
printable and snap-assemblable across filaments and scales ``0.4 .. 1.0``.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cone,
    Cylinder,
    Location,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Shape,
    extrude,
    loft,
)

from openengine.printing.primitives import (
    make_alignment_hole_cutter,
    make_alignment_pin,
    make_cantilever_snap_hook,
    make_snap_pocket_cutter,
    make_teardrop_bore_cutter,
)
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    SnapFitDesign,
    auto_size_cantilever_snap,
    min_fin_thickness_mm,
    min_printable_wall_mm,
    running_clearance_radial_mm,
    slip_clearance_radial_mm,
)

HALF_FACTOR: float = 0.5
ROT_90_DEG: float = 90.0
ROT_180_DEG: float = 180.0

# Master engine datums (1:1 nominal mm before cfg.scale)
CRANK_AXIS_Z_MM: float = -159.0
CAM_AXIS_Z_MM: float = 76.5
TIMING_CD_MM: float = CAM_AXIS_Z_MM - CRANK_AXIS_Z_MM  # 235.5 mm
TIMING_MODULE_DIVISOR: float = 93.0
IDLER_MODULE_STEPS: Tuple[int, int, int] = (19, 43, 67)
IDLER_PIN_BASE_RADIUS_MM: float = 5.0
MIN_IDLER_PIN_RADIUS_MM: float = 2.0

# Shared coplanar +Y timing tunnel datums (1:1 nominal mm)
TIMING_TUNNEL_HALF_X_MM: float = 40.0
TIMING_TUNNEL_FLOOR_Y_MM: float = 48.0
TIMING_TUNNEL_BACK_Y_MM: float = 52.0
TIMING_POCKET_HALF_X_MM: float = 41.0
TIMING_POCKET_MAX_Y_MM: float = 64.0
IDLER_SOCKET_START_Y_MM: float = 43.5
TIMING_COVER_WIDTH_X_MM: float = 96.0
TIMING_COVER_BOT_Z_MM: float = -196.0
TIMING_COVER_POCKET_BOT_Z_MM: float = -192.0
TIMING_COVER_OUTER_Y_BASE_MM: float = 70.0
TIMING_COVER_SOCKET_END_Y_MM: float = 68.0
TIMING_COVER_SNAP_X_MM: float = 44.5
TIMING_COVER_SNAP_Z_MM: Tuple[float, float, float] = (-135.0, 26.0, 84.0)

# Crankcase and sump datums (1:1 nominal mm)
BLOCK_BASE_Z_MM: float = -85.0
SUMP_SPLIT_Z_MM: float = -241.0
SUMP_BOTTOM_Z_MM: float = -278.0
CASE_WIDTH_X_MM: float = 172.0
CASE_HALF_Y_MM: float = 52.0
OIL_BAY_WIDTH_X_MM: float = 140.0
OIL_BAY_HALF_Y_MM: float = 41.0
UPPER_THROAT_WIDTH_X_MM: float = 90.0
UPPER_THROAT_HALF_Y_MM: float = 26.5
SWEEP_RADIUS_MM: float = 67.0
LINER_SPIGOT_BOTTOM_Z_MM: float = -106.5
LINER_SPIGOT_BLOCK_BOT_Z_MM: float = -105.0
MAIN_POCKET_RADIUS_MM: float = 31.0
MAIN_POCKET_MIN_Y_MM: float = 26.0
MAIN_POCKET_MAX_Y_MM: float = 43.0
MAIN_THRU_RADIUS_MM: float = 17.0
BUSHING_KEY_WIDTH_BASE_MM: float = 6.0
BUSHING_KEY_MIN_WIDTH_MM: float = 2.4
BUSHING_KEY_HEIGHT_BASE_MM: float = 3.5
BUSHING_KEY_MIN_HEIGHT_MM: float = 2.0

SUMP_FLOOR_LEFT_Z_MM: float = -271.5
SUMP_FLOOR_RIGHT_Z_MM: float = -263.5
SUMP_LIP_WALL_BASE_MM: float = 3.0
SUMP_LIP_HEIGHT_BASE_MM: float = 4.0

# Cylinder block datums (1:1 nominal mm)
BORE_RADIUS_MM: float = 34.0
LINER_WALL_BASE_MM: float = 3.0
BARREL_WALL_BASE_MM: float = 10.0
FLANGE_WIDTH_X_MM: float = 120.0
FLANGE_MIN_Y_MM: float = -56.0
TOP_DECK_THICK_MM: float = 10.0
BASE_FLANGE_THICK_MM: float = 10.0
NUM_KIT_FINS: int = 5
BLOCK_TIMING_BOSS_HALF_X_MM: float = 48.0

# Cylinder head, cam tower, cam cover datums (1:1 nominal mm)
HEAD_WIDTH_X_MM: float = 120.0
HEAD_MIN_Y_MM: float = -60.0
HEAD_HEIGHT_Z_MM: float = 52.0
CHAMBER_APEX_Z_MM: float = 10.5
CHAMBER_SQUISH_HALF_Y_MM: float = 26.5
VALVE_TILT_DEG: float = 15.0
VALVE_SEAT_X_MM: float = 16.0
VALVE_SEAT_Z_MM: float = 6.0
VALVE_IN_RADIUS_MM: float = 15.0
VALVE_EX_RADIUS_MM: float = 13.0
VALVE_STEM_BASE_R_MM: float = 2.5
VALVE_STEM_MIN_R_MM: float = 1.75
SPRING_POCKET_RADIUS_MM: float = 11.0
SPRING_POCKET_FLOOR_Z_MM: float = 42.0
IN_FLANGE_Z_MM: float = 34.0
EX_FLANGE_Z_MM: float = 24.0
IN_FLANGE_R_MM: float = 16.0
EX_FLANGE_R_MM: float = 14.0
PLUG_ORIGIN_Y_MM: float = -13.5
PLUG_ORIGIN_Z_MM: float = 8.5
PLUG_TILT_DEG: float = 24.0
PLUG_SHANK_BASE_R_MM: float = 4.9
PLUG_SHANK_MIN_R_MM: float = 2.0
PLUG_WELL_BASE_R_MM: float = 10.5
PLUG_WELL_MIN_R_MM: float = 4.5

TOWER_MIN_Z_MM: float = 52.0
TOWER_MAX_Z_MM: float = 106.5
TOWER_WIDTH_X_MM: float = 116.0
TOWER_MIN_Y_MM: float = -30.0
TOWER_BAY_HALF_X_MM: float = 45.0
TOWER_BAY_HALF_Y_MM: float = 17.5
CAM_CORE_BASE_R_MM: float = 10.0
CAM_CORE_MIN_R_MM: float = 3.0
ROCKER_PIVOT_X_MM: float = 19.5
ROCKER_PIVOT_Z_MM: float = 100.0
ROCKER_SHAFT_BASE_R_MM: float = 3.2
ROCKER_SHAFT_MIN_R_MM: float = 1.3
ROCKER_BORE_END_Y_MM: float = 26.0

CAM_COVER_TOP_Z_MM: float = 128.5
CAM_COVER_CAVITY_TOP_Z_MM: float = 123.0

# Shroud, intake/carb, and spark plug datums (1:1 nominal mm)
SHROUD_WIDTH_X_MM: float = 128.0
SHROUD_MIN_Y_MM: float = -106.0
SHROUD_MAX_Y_MM: float = -98.5
SHROUD_MIN_Z_MM: float = -210.0
SHROUD_MAX_Z_MM: float = -76.0
SHROUD_CENTER_HOLE_R_MM: float = 15.0
SHROUD_ARM_XZ_MM: Tuple[Tuple[float, float], ...] = (
    (-74.0, -96.0),
    (74.0, -96.0),
    (-74.0, -226.0),
    (74.0, -226.0),
)

INTAKE_START_X_MM: float = 60.0
INTAKE_FLANGE_THICK_X_MM: float = 8.0
INTAKE_FLANGE_Y_MM: float = 64.0
INTAKE_FLANGE_MIN_Z_MM: float = 15.0
INTAKE_FLANGE_MAX_Z_MM: float = 51.5
CARB_START_X_MM: float = 90.0
CARB_END_X_MM: float = 144.3
CARB_VENTURI_R_MM: float = 13.0
CARB_SLIDE_TOP_Z_MM: float = 88.0
CARB_BOWL_BOT_Z_MM: float = -4.0
CARB_BOWL_WIDTH_Y_MM: float = 38.0
CARB_TOWER_R_MM: float = 14.0

PLUG_REACH_LEN_MM: float = 15.0
PLUG_HEX_LEN_MM: float = 11.0
PLUG_CERAMIC_LEN_MM: float = 24.0
PLUG_TERMINAL_LEN_MM: float = 8.0


@dataclass(frozen=True, slots=True)
class _SnapJoint:
  """Placed cantilever snap hook solid, male relief cutter, and female pocket cutter."""

  hook: Part
  pocket_cutter: Part
  male_relief_cutter: Optional[Part] = None


def _as_part(shape: Shape) -> Part:
  """Normalize a build123d Solid/Compound/Part into a single-wrapper Part."""
  if isinstance(shape, Part):
    return shape
  solids = list(shape.solids())
  if not solids:
    raise ValueError("Expected non-empty solid geometry in generated housing")
  return Part(children=solids)


def _idler_z_mm(idx: int, scale: float) -> float:
  """Return scaled Z center of idler gear ``idx`` in {0, 1, 2}."""
  step = IDLER_MODULE_STEPS[idx]
  return (CRANK_AXIS_Z_MM + (step / TIMING_MODULE_DIVISOR) * TIMING_CD_MM) * scale


def _idler_pin_radius_mm(scale: float) -> float:
  """Return scaled printable idler pin radius matching kit_valvetrain."""
  return max(MIN_IDLER_PIN_RADIUS_MM, IDLER_PIN_BASE_RADIUS_MM * scale)


def _cover_outer_y_mm(cfg: PrintKitConfig) -> float:
  """Return +Y outer face of timing cover and cam cover preserving rear socket wall."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)
  return max(TIMING_COVER_OUTER_Y_BASE_MM * s, TIMING_COVER_SOCKET_END_Y_MM * s + w_min)


def _timing_cover_pocket_max_y_mm(cfg: PrintKitConfig) -> float:
  """Return +Y inner pocket face of timing cover."""
  return TIMING_POCKET_MAX_Y_MM * cfg.scale


def _teardrop_bore(
    radius_mm: float,
    length_mm: float,
    axis: str = "Y",
    crest_up_world: bool = False,
) -> Part:
  """Create horizontal teardrop bore cutter with 45-deg roof pointing toward print +Z."""
  cutter = make_teardrop_bore_cutter(radius_mm=radius_mm, length_mm=length_mm, axis=axis)
  if crest_up_world:
    return cutter
  axis_key = axis.strip().upper()
  rot_axis = Axis.Y if axis_key == "Y" else Axis.X
  return cutter.rotate(rot_axis, ROT_180_DEG)


def _place_in_frame(
    part: Part,
    origin: Tuple[float, float, float],
    x_dir: Tuple[float, float, float],
    z_dir: Tuple[float, float, float],
) -> Part:
  """Transform a local (+Z beam, +X barb) part into the requested world frame."""
  loc = Location(Plane(origin=origin, x_dir=x_dir, z_dir=z_dir))
  return part.moved(loc)


def _build_snap_joint(
    cfg: PrintKitConfig,
    beam_len_mm: float,
    width_mm: float,
    root_origin: Tuple[float, float, float],
    z_dir: Tuple[float, float, float],
    x_dir: Tuple[float, float, float],
    recess_depth_mm: float = 0.0,
    anchor_depth_mm: float = 1.4,
    open_back_relief: bool = False,
) -> _SnapJoint:
  """Build a scaled, filament-adaptive cantilever snap hook + mating pocket cutter."""
  design: SnapFitDesign = auto_size_cantilever_snap(
      cfg,
      available_length_mm=beam_len_mm,
      width_mm=width_mm,
      retention_angle_deg=45.0,
  )
  hook_local = make_cantilever_snap_hook(design)
  with BuildPart() as anchor_bp:
    with Locations((HALF_FACTOR * design.root_thickness_mm, 0.0, -HALF_FACTOR * anchor_depth_mm)):
      Box(design.root_thickness_mm, design.beam_width_mm, anchor_depth_mm + 0.1)
  hook_anchored = _as_part(hook_local.fuse(anchor_bp.part))

  pocket_local = make_snap_pocket_cutter(design, design.clearance_mm)
  male_relief_world: Optional[Part] = None
  if recess_depth_mm > 0.0:
    c = design.clearance_mm
    x_min = -(design.undercut_mm + c + (1.5 if open_back_relief else 0.0))
    x_max = design.tip_thickness_mm + design.undercut_mm + c + (0.5 if open_back_relief else 0.0)
    z_start = 0.0
    z_span = recess_depth_mm + 0.2
    with BuildPart() as relief_bp:
      with Locations((HALF_FACTOR * (x_min + x_max), 0.0, z_start + HALF_FACTOR * z_span)):
        Box(x_max - x_min, design.beam_width_mm + 2.0 * c, z_span)
    male_relief_world = _place_in_frame(relief_bp.part, root_origin, x_dir, z_dir)

  return _SnapJoint(
      hook=_place_in_frame(hook_anchored, root_origin, x_dir, z_dir),
      pocket_cutter=_place_in_frame(pocket_local, root_origin, x_dir, z_dir),
      male_relief_cutter=male_relief_world,
  )


def _build_pin_pair(
    cfg: PrintKitConfig,
    radius_mm: float,
    length_mm: float,
    origin: Tuple[float, float, float],
    z_dir: Tuple[float, float, float],
    x_dir: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    embed_mm: float = 1.2,
) -> Tuple[Part, Part]:
  """Build an embedded alignment pin and mating slip-clearance hole cutter."""
  r_slip = slip_clearance_radial_mm(cfg)
  chamfer_mm = min(0.5, 0.3 * radius_mm)
  pin_local = make_alignment_pin(radius_mm=radius_mm, length_mm=length_mm + embed_mm, chamfer_mm=chamfer_mm)
  pin_shifted = pin_local.translate((0.0, 0.0, -embed_mm))

  hole_local = make_alignment_hole_cutter(
      radius_mm=radius_mm,
      depth_mm=length_mm + 0.8,
      clearance_mm=r_slip,
  ).translate((0.0, 0.0, -0.3))

  return (
      _place_in_frame(pin_shifted, origin, x_dir, z_dir),
      _place_in_frame(hole_local, origin, x_dir, z_dir),
  )


def _apply_cut_list(body: Part, cutters: Sequence[Part]) -> Part:
  """Subtract non-empty cutters cleanly from a Part."""
  if not cutters:
    return body
  return _as_part(body.cut(*cutters))


def _apply_fuse_list(body: Part, solids: Sequence[Part]) -> Part:
  """Fuse solids cleanly into a Part."""
  out = body
  for item in solids:
    out = _as_part(out.fuse(item))
  return out


def _apply_male_snaps(
    body: Part,
    joints: Sequence[_SnapJoint],
    other_cutters: Sequence[Part],
    other_solids: Sequence[Part] = (),
) -> Part:
  """Cut male relief pockets BEFORE fusing snap hooks so beams never get severed."""
  reliefs = [j.male_relief_cutter for j in joints if j.male_relief_cutter is not None]
  part = _apply_fuse_list(body, other_solids)
  part = _apply_cut_list(part, list(other_cutters) + reliefs)
  return _apply_fuse_list(part, [j.hook for j in joints])


# ---------------------------------------------------------------------------
# Interface joint factories (ensuring identical coords across mating parts)
# ---------------------------------------------------------------------------

def _crankcase_split_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Part], list[Part], list[_SnapJoint]]:
  """Build Y=0 split alignment pins and 4 recessed perimeter snaps (clear of -X cutaway)."""
  s = cfg.scale
  r_pin = max(1.5, 3.0 * s)
  l_pin = max(2.5, 5.0 * s)

  pin_coords = ((-78.0 * s, -96.0 * s), (78.0 * s, -96.0 * s), (78.0 * s, -190.0 * s))
  pins: list[Part] = []
  holes: list[Part] = []
  for px, pz in pin_coords:
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=(px, 0.0, pz),
        z_dir=(0.0, 1.0, 0.0),
        x_dir=(1.0, 0.0, 0.0),
    )
    pins.append(pin)
    holes.append(hole)

  l_snap = min(10.0, max(7.0, 16.0 * s))
  snap_specs = (
      ((-62.0 * s, 0.0, -98.0 * s), (-1.0, 0.0, 0.0), min(6.0, 12.0 * s), False),
      ((62.0 * s, 0.0, -98.0 * s), (1.0, 0.0, 0.0), min(6.0, 12.0 * s), False),
      ((-78.0 * s, 0.0, -230.5 * s), (-1.0, 0.0, 0.0), min(5.0, 11.0 * s), False),
      ((78.0 * s, 0.0, -230.5 * s), (1.0, 0.0, 0.0), min(5.0, 11.0 * s), False),
  )
  joints = [
      _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=origin,
          z_dir=(0.0, 1.0, 0.0),
          x_dir=x_dir,
      )
      for origin, x_dir, w_snap, _ in snap_specs
  ]
  return pins, holes, joints


def _sump_snap_joints(cfg: PrintKitConfig) -> list[_SnapJoint]:
  """Build the 6 perimeter snap hooks attaching kit_sump_pan up onto crankcase base."""
  s = cfg.scale
  l_snap = min(10.0, max(7.0, 16.0 * s))
  w_y = min(6.0, 14.0 * s)
  w_x = min(6.0, 12.0 * s)

  # Hook back sits flush at outer perimeter and barbs latch inward so outer faces stay clean.
  specs = (
      ((-48.0 * s, -CASE_HALF_Y_MM * s, SUMP_SPLIT_Z_MM * s), (0.0, 1.0, 0.0), w_y),
      ((48.0 * s, -CASE_HALF_Y_MM * s, SUMP_SPLIT_Z_MM * s), (0.0, 1.0, 0.0), w_y),
      ((-48.0 * s, CASE_HALF_Y_MM * s, SUMP_SPLIT_Z_MM * s), (0.0, -1.0, 0.0), w_y),
      ((48.0 * s, CASE_HALF_Y_MM * s, SUMP_SPLIT_Z_MM * s), (0.0, -1.0, 0.0), w_y),
      ((-HALF_FACTOR * CASE_WIDTH_X_MM * s, -38.0 * s, SUMP_SPLIT_Z_MM * s), (1.0, 0.0, 0.0), w_x),
      ((HALF_FACTOR * CASE_WIDTH_X_MM * s, 38.0 * s, SUMP_SPLIT_Z_MM * s), (-1.0, 0.0, 0.0), w_x),
  )
  return [
      _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=origin,
          z_dir=(0.0, 0.0, 1.0),
          x_dir=x_dir,
      )
      for origin, x_dir, w_snap in specs
  ]


def _block_to_case_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Part], list[Part], list[_SnapJoint]]:
  """Build Z=-85*s alignment pins and 4 corner snaps between block and crankcase."""
  s = cfg.scale
  r_pin = max(1.5, 3.0 * s)
  l_pin = max(2.5, 5.0 * s)
  pins: list[Part] = []
  holes: list[Part] = []
  for px, py in ((51.5 * s, -26.0 * s), (-51.5 * s, 26.0 * s)):
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=(px, py, BLOCK_BASE_Z_MM * s),
        z_dir=(0.0, 0.0, -1.0),
        x_dir=(1.0, 0.0, 0.0),
    )
    pins.append(pin)
    holes.append(hole)

  l_snap = min(11.0, max(8.0, 20.0 * s))
  w_snap = min(6.0, 12.0 * s)
  joints = [
      _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=(sx, sy, BLOCK_BASE_Z_MM * s),
          z_dir=(0.0, 0.0, -1.0),
          x_dir=(1.0 if sx > 0 else -1.0, 0.0, 0.0),
      )
      for sx in (-52.0 * s, 52.0 * s)
      for sy in (-38.0 * s, 38.0 * s)
  ]
  return pins, holes, joints


def _head_to_block_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Part], list[Part], list[_SnapJoint], list[Part]]:
  """Build Z=0 diagonal alignment pins and 4 exterior corner latch tabs (head -> block)."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)
  r_pin = max(1.5, 3.0 * s)
  l_pin = max(2.2, min(5.0 * s, TOP_DECK_THICK_MM * s - w_min - 0.3))
  pins: list[Part] = []
  holes: list[Part] = []
  for px, py in ((40.0 * s, -40.0 * s), (-40.0 * s, 40.0 * s)):
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=(px, py, 0.0),
        z_dir=(0.0, 0.0, -1.0),
        x_dir=(1.0, 0.0, 0.0),
    )
    pins.append(pin)
    holes.append(hole)

  l_snap = min(13.0, max(8.5, 22.0 * s))
  w_snap = min(6.0, 12.0 * s)
  z_catch_target = -TOP_DECK_THICK_MM * s
  z_root = z_catch_target + l_snap
  joints: list[_SnapJoint] = []
  block_side_notches: list[Part] = []
  half_flange_x = HALF_FACTOR * FLANGE_WIDTH_X_MM * s

  for sx_sign in (-1.0, 1.0):
    x_outer = sx_sign * half_flange_x
    for sy in (-46.0 * s, 42.0 * s):
      # Hook back stays flush at outer X face, barb latches inward under top deck flange.
      joint = _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=(x_outer, sy, z_root),
          z_dir=(0.0, 0.0, -1.0),
          x_dir=(-sx_sign, 0.0, 0.0),
          recess_depth_mm=z_root,
          open_back_relief=True,
      )
      joints.append(joint)

      # Shallow side guide notch across block top deck Z in [-10*s, 0] for flush tab seating.
      d_design = auto_size_cantilever_snap(
          cfg, available_length_mm=l_snap, width_mm=w_snap, retention_angle_deg=45.0
      )
      c = d_design.clearance_mm
      notch_depth = d_design.tip_thickness_mm + d_design.undercut_mm + c
      notch_h = TOP_DECK_THICK_MM * s + 1.0
      x_mid = x_outer - sx_sign * HALF_FACTOR * notch_depth
      with BuildPart() as notch_bp:
        with Locations((x_mid, sy, -HALF_FACTOR * TOP_DECK_THICK_MM * s)):
          Box(notch_depth + 0.5, d_design.beam_width_mm + 2.0 * c, notch_h)
      block_side_notches.append(notch_bp.part)

  return pins, holes, joints, block_side_notches


def _tower_to_head_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Part], list[Part], list[_SnapJoint]]:
  """Build Z=+52*s alignment pins and 4 snap-posts from cam tower down into head."""
  s = cfg.scale
  r_pin = max(1.2, 2.5 * s)
  l_pin = max(2.5, 5.0 * s)
  pins: list[Part] = []
  holes: list[Part] = []
  for px in (-51.0 * s, 51.0 * s):
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=(px, 24.0 * s, TOWER_MIN_Z_MM * s),
        z_dir=(0.0, 0.0, -1.0),
        x_dir=(1.0, 0.0, 0.0),
    )
    pins.append(pin)
    holes.append(hole)

  l_snap = min(12.0, max(7.5, 18.0 * s))
  w_snap = min(5.5, 11.0 * s)
  snap_specs = (
      ((-49.0 * s, -22.0 * s, TOWER_MIN_Z_MM * s), (-1.0, 0.0, 0.0)),
      ((49.0 * s, -22.0 * s, TOWER_MIN_Z_MM * s), (1.0, 0.0, 0.0)),
      ((-30.0 * s, 28.0 * s, TOWER_MIN_Z_MM * s), (0.0, 1.0, 0.0)),
      ((30.0 * s, 28.0 * s, TOWER_MIN_Z_MM * s), (0.0, 1.0, 0.0)),
  )
  joints = [
      _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=origin,
          z_dir=(0.0, 0.0, -1.0),
          x_dir=x_dir,
      )
      for origin, x_dir in snap_specs
  ]
  return pins, holes, joints


def _cover_to_tower_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Part], list[Part], list[_SnapJoint]]:
  """Build Z=+106.5*s alignment pins and 4 exterior side snap tabs (cam cover -> tower)."""
  s = cfg.scale
  r_pin = max(1.2, 2.5 * s)
  l_pin = max(2.2, 4.5 * s)
  pins: list[Part] = []
  holes: list[Part] = []
  for px in (-51.0 * s, 51.0 * s):
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=(px, 36.0 * s, TOWER_MAX_Z_MM * s),
        z_dir=(0.0, 0.0, -1.0),
        x_dir=(1.0, 0.0, 0.0),
    )
    pins.append(pin)
    holes.append(hole)

  l_snap = min(13.0, max(8.5, 22.0 * s))
  w_snap = min(6.0, 12.0 * s)
  z_root = TOWER_MAX_Z_MM * s + 0.52 * l_snap
  recess = z_root - TOWER_MAX_Z_MM * s
  half_w = HALF_FACTOR * TOWER_WIDTH_X_MM * s
  joints: list[_SnapJoint] = []
  for sx_sign in (-1.0, 1.0):
    x_outer = sx_sign * half_w
    for sy in (-8.0 * s, 40.0 * s):
      joints.append(
          _build_snap_joint(
              cfg,
              beam_len_mm=l_snap,
              width_mm=w_snap,
              root_origin=(x_outer, sy, z_root),
              z_dir=(0.0, 0.0, -1.0),
              x_dir=(-sx_sign, 0.0, 0.0),
              recess_depth_mm=recess,
              open_back_relief=True,
          )
      )
  return pins, holes, joints


def _timing_cover_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Tuple[str, Part, Part]], list[Tuple[str, _SnapJoint]]]:
  """Build 2 alignment pins and 6 -Y perimeter snap hooks on timing cover -> housing stack."""
  s = cfg.scale
  r_pin = max(1.2, 2.5 * s)
  l_pin = max(2.2, 4.5 * s)
  y_iface = TIMING_TUNNEL_BACK_Y_MM * s

  pin_targets = (
      ("crankcase_right", (TIMING_COVER_SNAP_X_MM * s, y_iface, -100.0 * s)),
      ("cam_tower", (-TIMING_COVER_SNAP_X_MM * s, y_iface, 68.0 * s)),
  )
  pin_list: list[Tuple[str, Part, Part]] = []
  for host, origin in pin_targets:
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=origin,
        z_dir=(0.0, -1.0, 0.0),
        x_dir=(1.0, 0.0, 0.0),
    )
    pin_list.append((host, pin, hole))

  l_snap = min(12.0, max(8.0, 20.0 * s))
  w_snap = min(6.0, 12.0 * s)
  recess = 0.5 * l_snap
  y_root = y_iface + recess
  half_cov_x = HALF_FACTOR * TIMING_COVER_WIDTH_X_MM * s

  snap_hosts = (
      ("crankcase_right", TIMING_COVER_SNAP_Z_MM[0] * s),
      ("cylinder_head", TIMING_COVER_SNAP_Z_MM[1] * s),
      ("cam_tower", TIMING_COVER_SNAP_Z_MM[2] * s),
  )
  snap_list: list[Tuple[str, _SnapJoint]] = []
  for host, sz in snap_hosts:
    for sx_sign in (-1.0, 1.0):
      x_outer = sx_sign * half_cov_x
      joint = _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=(x_outer, y_root, sz),
          z_dir=(0.0, -1.0, 0.0),
          x_dir=(-sx_sign, 0.0, 0.0),
          recess_depth_mm=recess,
          open_back_relief=True,
      )
      snap_list.append((host, joint))
  return pin_list, snap_list


def _shroud_snap_joints(cfg: PrintKitConfig) -> list[_SnapJoint]:
  """Build 4 +Y snap hooks on fan shroud standoff arms latching onto crankcase_left."""
  s = cfg.scale
  l_snap = min(10.0, max(7.0, 16.0 * s))
  w_snap = min(5.5, 11.0 * s)
  y_iface = -CASE_HALF_Y_MM * s
  return [
      _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=(ax * s, y_iface, az * s),
          z_dir=(0.0, 1.0, 0.0),
          x_dir=(1.0 if ax > 0 else -1.0, 0.0, 0.0),
      )
      for ax, az in SHROUD_ARM_XZ_MM
  ]


def _intake_pins_and_snaps(
    cfg: PrintKitConfig,
) -> Tuple[list[Part], list[Part], list[_SnapJoint]]:
  """Build 2 alignment pins and 2 snap clips on +X head flange -> intake/carb unit."""
  s = cfg.scale
  r_pin = max(1.2, 2.5 * s)
  l_pin = max(2.0, min(4.0, 5.0 * s))
  pins: list[Part] = []
  holes: list[Part] = []
  for py in (-22.0 * s, 22.0 * s):
    pin, hole = _build_pin_pair(
        cfg,
        radius_mm=r_pin,
        length_mm=l_pin,
        origin=(INTAKE_START_X_MM * s, py, 43.0 * s),
        z_dir=(1.0, 0.0, 0.0),
        x_dir=(0.0, 0.0, 1.0),
    )
    pins.append(pin)
    holes.append(hole)

  l_snap = min(11.0, max(8.0, 16.0 * s))
  w_snap = min(4.0, 9.0 * s)
  recess = max(0.0, l_snap - (INTAKE_FLANGE_THICK_X_MM * s - 0.5))
  x_root = INTAKE_START_X_MM * s - recess
  joints = [
      _build_snap_joint(
          cfg,
          beam_len_mm=l_snap,
          width_mm=w_snap,
          root_origin=(x_root, sy, 19.5 * s),
          z_dir=(1.0, 0.0, 0.0),
          x_dir=(0.0, 1.0 if sy > 0 else -1.0, 0.0),
          recess_depth_mm=recess,
      )
      for sy in (-28.0 * s, 28.0 * s)
  ]
  return pins, holes, joints


# ---------------------------------------------------------------------------
# Part Builders
# ---------------------------------------------------------------------------

def _build_crankcase_half(
    cfg: PrintKitConfig,
    is_right: bool,
    split_pins: Sequence[Part],
    split_holes: Sequence[Part],
    split_snaps: Sequence[_SnapJoint],
    sump_snaps: Sequence[_SnapJoint],
    block_holes: Sequence[Part],
    block_snaps: Sequence[_SnapJoint],
    timing_pins: Sequence[Tuple[str, Part, Part]],
    timing_snaps: Sequence[Tuple[str, _SnapJoint]],
    shroud_snaps: Sequence[_SnapJoint],
) -> Part:
  """Generate kit_crankcase_left (Y <= 0) or kit_crankcase_right (Y >= 0)."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  y_sign = 1.0 if is_right else -1.0
  case_w = CASE_WIDTH_X_MM * s
  case_half_y = CASE_HALF_Y_MM * s
  case_h = (BLOCK_BASE_Z_MM - SUMP_SPLIT_Z_MM) * s
  case_mid_z = HALF_FACTOR * (BLOCK_BASE_Z_MM + SUMP_SPLIT_Z_MM) * s

  spigot_wall = max(w_min, LINER_WALL_BASE_MM * s)
  r_spigot_bore = BORE_RADIUS_MM * s + spigot_wall + r_slip

  with BuildPart() as half_bp:
    with Locations((0.0, y_sign * HALF_FACTOR * case_half_y, case_mid_z)):
      Box(case_w, case_half_y, case_h)

  cutters: list[Part] = []

  # Lower wet-sump bay (Z in [-241*s, -159*s])
  oil_h = (CRANK_AXIS_Z_MM - SUMP_SPLIT_Z_MM) * s
  oil_d = OIL_BAY_HALF_Y_MM * s + 1.0
  with BuildPart() as oil_bp:
    with Locations((0.0, y_sign * (HALF_FACTOR * oil_d - 0.5), SUMP_SPLIT_Z_MM * s + HALF_FACTOR * oil_h - 1.0)):
      Box(OIL_BAY_WIDTH_X_MM * s, oil_d, oil_h + 2.0)
  cutters.append(oil_bp.part)

  # Upper crank/rod throat (Z in [-159*s, -85*s])
  throat_h = (BLOCK_BASE_Z_MM - CRANK_AXIS_Z_MM) * s
  throat_d = UPPER_THROAT_HALF_Y_MM * s + 1.0
  with BuildPart() as throat_bp:
    with Locations((0.0, y_sign * (HALF_FACTOR * throat_d - 0.5), CRANK_AXIS_Z_MM * s + HALF_FACTOR * throat_h + 1.0)):
      Box(UPPER_THROAT_WIDTH_X_MM * s, throat_d, throat_h + 2.0)
  cutters.append(throat_bp.part)

  # Crank sweep circular pocket
  sweep_len = UPPER_THROAT_HALF_Y_MM * s + 1.0
  with BuildPart() as sweep_bp:
    with Locations((0.0, y_sign * (HALF_FACTOR * sweep_len - 0.5), CRANK_AXIS_Z_MM * s)):
      Cylinder(radius=SWEEP_RADIUS_MM * s, height=sweep_len, rotation=(ROT_90_DEG, 0.0, 0.0))
  cutters.append(sweep_bp.part)

  # Liner spigot bore + 45-deg self-supporting taper below -106.5*s for (180, 0, 0) printability
  spigot_h = (BLOCK_BASE_Z_MM - LINER_SPIGOT_BOTTOM_Z_MM) * s
  with BuildPart() as spigot_bp:
    with Locations((0.0, 0.0, BLOCK_BASE_Z_MM * s - HALF_FACTOR * spigot_h + 1.0)):
      Cylinder(radius=r_spigot_bore, height=spigot_h + 2.0)
    r_throat_y = UPPER_THROAT_HALF_Y_MM * s
    if r_spigot_bore > r_throat_y + 0.2:
      taper_h = r_spigot_bore - r_throat_y
      with Locations((0.0, 0.0, LINER_SPIGOT_BOTTOM_Z_MM * s - taper_h)):
        Cone(
            bottom_radius=r_throat_y,
            top_radius=r_spigot_bore,
            height=taper_h,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
  cutters.append(spigot_bp.part)

  # Teardrop main bushing pocket (|Y| in [25.5*s, 43*s]) + +Z anti-rotation key slot
  r_pkt = MAIN_POCKET_RADIUS_MM * s + r_slip
  pkt_min_y = (MAIN_POCKET_MIN_Y_MM - 0.5) * s
  pkt_max_y = MAIN_POCKET_MAX_Y_MM * s
  pkt_len = pkt_max_y - pkt_min_y
  pkt_mid_y = y_sign * HALF_FACTOR * (pkt_min_y + pkt_max_y)
  cutters.append(
      _teardrop_bore(r_pkt, pkt_len, axis="Y", crest_up_world=False).translate(
          (0.0, pkt_mid_y, CRANK_AXIS_Z_MM * s)
      )
  )

  key_w = max(BUSHING_KEY_MIN_WIDTH_MM, BUSHING_KEY_WIDTH_BASE_MM * s) + 2.0 * r_slip
  key_top_z = (CRANK_AXIS_Z_MM + MAIN_POCKET_RADIUS_MM) * s + max(BUSHING_KEY_MIN_HEIGHT_MM, BUSHING_KEY_HEIGHT_BASE_MM * s) + r_slip
  key_bot_z = (CRANK_AXIS_Z_MM + MAIN_POCKET_RADIUS_MM - 2.0) * s
  with BuildPart() as key_bp:
    with Locations((0.0, pkt_mid_y, HALF_FACTOR * (key_top_z + key_bot_z))):
      Box(key_w, pkt_len, key_top_z - key_bot_z)
  cutters.append(key_bp.part)

  # Teardrop main thru-bore (|Y| in [42.5*s, 52.5*s])
  r_thru = MAIN_THRU_RADIUS_MM * s + r_run
  thru_min_y = (MAIN_POCKET_MAX_Y_MM - 0.5) * s
  thru_max_y = (CASE_HALF_Y_MM + 1.0) * s
  thru_len = thru_max_y - thru_min_y
  thru_mid_y = y_sign * HALF_FACTOR * (thru_min_y + thru_max_y)
  cutters.append(
      _teardrop_bore(r_thru, thru_len, axis="Y", crest_up_world=False).translate(
          (0.0, thru_mid_y, CRANK_AXIS_Z_MM * s)
      )
  )

  # Half-specific timing bay / service bosses / shroud / cover interfaces
  add_solids: list[Part] = []
  if is_right:
    bay_h = (BLOCK_BASE_Z_MM - TIMING_COVER_POCKET_BOT_Z_MM) * s
    bay_d = (TIMING_TUNNEL_BACK_Y_MM - TIMING_TUNNEL_FLOOR_Y_MM) * s + 1.0
    bay_mid_y = TIMING_TUNNEL_FLOOR_Y_MM * s + HALF_FACTOR * bay_d
    with BuildPart() as tbay_bp:
      with Locations((0.0, bay_mid_y, TIMING_COVER_POCKET_BOT_Z_MM * s + HALF_FACTOR * bay_h + 1.0)):
        Box(2.0 * TIMING_TUNNEL_HALF_X_MM * s, bay_d, bay_h + 2.0)
    cutters.append(tbay_bp.part)

    # 45-deg chamfered floor at Z = -192*s eliminating bottom timing bay overhang
    chamf_z = min(6.0 * s, (TIMING_TUNNEL_BACK_Y_MM - TIMING_TUNNEL_FLOOR_Y_MM) * s + 1.0)
    with BuildSketch(Plane.YZ) as bay_chamf_sk:
      Polygon([
          (TIMING_TUNNEL_FLOOR_Y_MM * s, TIMING_COVER_POCKET_BOT_Z_MM * s + chamf_z),
          (TIMING_TUNNEL_BACK_Y_MM * s + 1.0, TIMING_COVER_POCKET_BOT_Z_MM * s + chamf_z),
          (TIMING_TUNNEL_BACK_Y_MM * s + 1.0, TIMING_COVER_POCKET_BOT_Z_MM * s),
      ])
    with BuildPart() as bay_chamf_bp:
      extrude(bay_chamf_sk.sketch, amount=TIMING_TUNNEL_HALF_X_MM * s, both=True)
    cutters.append(bay_chamf_bp.part)
    sock_len = (TIMING_TUNNEL_FLOOR_Y_MM - IDLER_SOCKET_START_Y_MM) * s + 0.5
    sock_mid_y = IDLER_SOCKET_START_Y_MM * s + HALF_FACTOR * sock_len
    cutters.append(
        _teardrop_bore(_idler_pin_radius_mm(s) + r_slip, sock_len, axis="Y", crest_up_world=False).translate(
            (0.0, sock_mid_y, _idler_z_mm(0, s))
        )
    )

    # Solid cosmetic sight-glass & oil-fill bosses on +X wall (plugged/non-hollow)
    for bz, br in ((-211.0 * s, 9.0 * s), (-114.0 * s, 10.0 * s)):
      boss_h = max(1.0, 2.0 * s)
      with BuildPart() as boss_bp:
        with Locations((HALF_FACTOR * case_w - 0.5, 30.0 * s, bz)):
          Cone(
              bottom_radius=br,
              top_radius=max(1.0, br - boss_h),
              height=boss_h + 0.5,
              rotation=(0.0, ROT_90_DEG, 0.0),
              align=(Align.CENTER, Align.CENTER, Align.MIN),
          )
      add_solids.append(boss_bp.part)

    for host, _, hole in timing_pins:
      if host == "crankcase_right":
        cutters.append(hole)
    for host, joint in timing_snaps:
      if host == "crankcase_right":
        cutters.append(joint.pocket_cutter)
  else:
    for j in shroud_snaps:
      cutters.append(j.pocket_cutter)

  # Split-line pins/snaps, block base pins/snaps, and sump snaps
  if is_right:
    cutters.extend(split_holes)
    cutters.extend(j.pocket_cutter for j in split_snaps)
    for hole in block_holes:
      if hole.bounding_box().center().Y > 0.0:
        cutters.append(hole)
    for j in block_snaps:
      if j.pocket_cutter.bounding_box().center().Y > 0.0:
        cutters.append(j.pocket_cutter)
    for j in sump_snaps:
      if j.pocket_cutter.bounding_box().center().Y > -1.0:
        cutters.append(j.pocket_cutter)
  else:
    add_solids.extend(split_pins)
    add_solids.extend(j.hook for j in split_snaps)
    for j in split_snaps:
      if j.male_relief_cutter is not None:
        cutters.append(j.male_relief_cutter)
    for hole in block_holes:
      if hole.bounding_box().center().Y < 0.0:
        cutters.append(hole)
    for j in block_snaps:
      if j.pocket_cutter.bounding_box().center().Y < 0.0:
        cutters.append(j.pocket_cutter)
    for j in sump_snaps:
      if j.pocket_cutter.bounding_box().center().Y < 1.0:
        cutters.append(j.pocket_cutter)

  # Optional -X cutaway viewing window with 45-deg self-supporting upper corners
  if cfg.cutaway:
    y_outer = 32.0 * s
    y_inner = 25.5 * s
    z_bot = -220.0 * s
    z_top = -108.0 * s
    arch_drop = min(20.0 * s, 0.35 * (z_top - z_bot))
    z_shoulder = z_bot + arch_drop
    y_ridge = max(0.0, y_outer - arch_drop)
    with BuildSketch(Plane.YZ) as win_sk:
      Polygon([
          (-y_outer, z_top),
          (-y_outer, z_shoulder),
          (-y_ridge, z_bot),
          (y_ridge, z_bot),
          (y_outer, z_shoulder),
          (y_outer, z_top),
      ])
    with BuildPart() as win_outer:
      extrude(win_sk.sketch, amount=24.0 * s)
    cutters.append(win_outer.part.translate((-92.0 * s, 0.0, 0.0)))

    with BuildSketch(Plane.YZ) as win_in_sk:
      Polygon([
          (-y_inner, z_top),
          (-y_inner, z_shoulder),
          (-max(0.0, y_inner - arch_drop), z_bot),
          (max(0.0, y_inner - arch_drop), z_bot),
          (y_inner, z_shoulder),
          (y_inner, z_top),
      ])
    with BuildPart() as win_inner:
      extrude(win_in_sk.sketch, amount=25.0 * s)
    cutters.append(win_inner.part.translate((-68.5 * s, 0.0, 0.0)))

  part = _apply_fuse_list(half_bp.part, add_solids)
  return _apply_cut_list(part, cutters)


def _build_sump_pan(cfg: PrintKitConfig, sump_snaps: Sequence[_SnapJoint]) -> Part:
  """Generate kit_sump_pan with raised inner locating lip, sloped drain floor, and 6 snaps."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)
  r_slip = slip_clearance_radial_mm(cfg)

  case_w = CASE_WIDTH_X_MM * s
  case_d = 2.0 * CASE_HALF_Y_MM * s
  sump_h = (SUMP_SPLIT_Z_MM - SUMP_BOTTOM_Z_MM) * s
  sump_mid_z = HALF_FACTOR * (SUMP_SPLIT_Z_MM + SUMP_BOTTOM_Z_MM) * s

  floor_l_z = max(SUMP_FLOOR_LEFT_Z_MM * s, SUMP_BOTTOM_Z_MM * s + w_min)
  floor_r_z = max(SUMP_FLOOR_RIGHT_Z_MM * s, floor_l_z + 2.0 * s)
  bay_half_x = HALF_FACTOR * OIL_BAY_WIDTH_X_MM * s
  bay_half_y = OIL_BAY_HALF_Y_MM * s

  with BuildSketch(Plane.XZ) as cav_sk:
    Polygon([
        (-bay_half_x, floor_l_z),
        (bay_half_x, floor_r_z),
        (bay_half_x, SUMP_SPLIT_Z_MM * s + 0.5),
        (-bay_half_x, SUMP_SPLIT_Z_MM * s + 0.5),
    ])
  with BuildPart() as cav_bp:
    extrude(cav_sk.sketch, amount=bay_half_y, both=True)

  lip_wall = max(w_min, SUMP_LIP_WALL_BASE_MM * s)
  lip_h = max(2.0, SUMP_LIP_HEIGHT_BASE_MM * s)
  lip_out_x = bay_half_x - r_slip
  lip_out_y = bay_half_y - r_slip
  lip_in_x = max(w_min, lip_out_x - lip_wall)
  lip_in_y = max(w_min, lip_out_y - lip_wall)

  # 45-deg self-supporting inner taper from locating lip ID down to sump oil bay wall
  taper_h = lip_wall + 1.0
  with BuildPart() as lip_bp:
    with Locations((0.0, 0.0, SUMP_SPLIT_Z_MM * s - taper_h)):
      Box(2.0 * (bay_half_x + 2.0), 2.0 * (bay_half_y + 2.0), taper_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations((0.0, 0.0, SUMP_SPLIT_Z_MM * s - 0.5)):
      Box(2.0 * lip_out_x, 2.0 * lip_out_y, lip_h + 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with BuildSketch(Plane.XY.offset(SUMP_SPLIT_Z_MM * s - taper_h - 0.1)) as bot_sk:
      Polygon([(-bay_half_x, -bay_half_y), (bay_half_x, -bay_half_y), (bay_half_x, bay_half_y), (-bay_half_x, bay_half_y)])
    with BuildSketch(Plane.XY.offset(SUMP_SPLIT_Z_MM * s)) as top_sk:
      Polygon([(-lip_in_x, -lip_in_y), (lip_in_x, -lip_in_y), (lip_in_x, lip_in_y), (-lip_in_x, lip_in_y)])
    loft(mode=Mode.SUBTRACT)
    with Locations((0.0, 0.0, SUMP_SPLIT_Z_MM * s - 0.1)):
      Box(2.0 * lip_in_x, 2.0 * lip_in_y, lip_h + 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

  # Solid drain boss on -X wall (45-deg self-supporting underside)
  boss_r = 9.0 * s
  boss_h = max(1.0, 2.0 * s)
  with BuildPart() as drain_bp:
    with Locations((-HALF_FACTOR * case_w + 0.5, 0.0, -267.5 * s)):
      Cone(
          bottom_radius=boss_r,
          top_radius=max(1.0, boss_r - boss_h),
          height=boss_h + 0.5,
          rotation=(0.0, -ROT_90_DEG, 0.0),
          align=(Align.CENTER, Align.CENTER, Align.MIN),
      )

  with BuildPart() as sump_bp:
    with Locations((0.0, 0.0, sump_mid_z)):
      Box(case_w, case_d, sump_h)

  # Subtract oil cavity BEFORE fusing locating lip so cavity never trims the lip base
  pan_hollow = _apply_cut_list(sump_bp.part, [cav_bp.part])
  return _apply_male_snaps(pan_hollow, sump_snaps, (), [lip_bp.part, drain_bp.part])


def _build_cylinder_block(
    cfg: PrintKitConfig,
    block_pins: Sequence[Part],
    block_snaps: Sequence[_SnapJoint],
    head_holes: Sequence[Part],
    block_side_notches: Sequence[Part],
) -> Part:
  """Generate kit_cylinder_block with merged spigot, 45-deg beveled fins, and timing tunnel."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)
  fin_min = min_fin_thickness_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  bore_r = BORE_RADIUS_MM * s
  spigot_wall = max(w_min, LINER_WALL_BASE_MM * s)
  r_spigot = bore_r + spigot_wall
  r_barrel = bore_r + max(1.5 * w_min, BARREL_WALL_BASE_MM * s)

  flange_w = FLANGE_WIDTH_X_MM * s
  flange_min_y = FLANGE_MIN_Y_MM * s
  flange_max_y = TIMING_TUNNEL_BACK_Y_MM * s
  flange_d = flange_max_y - flange_min_y
  flange_mid_y = HALF_FACTOR * (flange_min_y + flange_max_y)
  top_t = max(w_min, TOP_DECK_THICK_MM * s)
  base_t = max(w_min, BASE_FLANGE_THICK_MM * s)
  block_h = -BLOCK_BASE_Z_MM * s
  spigot_h = (BLOCK_BASE_Z_MM - LINER_SPIGOT_BLOCK_BOT_Z_MM) * s

  add_solids: list[Part] = []
  with BuildPart() as barrel_bp:
    with Locations((0.0, 0.0, -HALF_FACTOR * block_h)):
      Cylinder(radius=r_barrel, height=block_h)
    with Locations((0.0, 0.0, BLOCK_BASE_Z_MM * s - HALF_FACTOR * spigot_h + 0.5)):
      Cylinder(radius=r_spigot, height=spigot_h + 0.5)
    with Locations((0.0, flange_mid_y, -HALF_FACTOR * top_t)):
      Box(flange_w, flange_d, top_t)
    with Locations((0.0, flange_mid_y, -block_h + HALF_FACTOR * base_t)):
      Box(flange_w, flange_d, base_t)

  # <= 40-deg self-supporting skirt from base flange top up to circular barrel
  base_top_z = -block_h + base_t
  skirt_h = max(4.0 * s, 1.25 * (HALF_FACTOR * flange_w - r_barrel))
  with BuildPart() as skirt_bp:
    with BuildSketch(Plane.XY.offset(base_top_z - 0.2)):
      Polygon([
          (-HALF_FACTOR * flange_w, flange_min_y),
          (HALF_FACTOR * flange_w, flange_min_y),
          (HALF_FACTOR * flange_w, flange_max_y),
          (-HALF_FACTOR * flange_w, flange_max_y),
      ])
    with BuildSketch(Plane.XY.offset(base_top_z + skirt_h)):
      Circle(radius=0.98 * r_barrel)
    loft()
  add_solids.append(skirt_bp.part)

  # +Y timing tunnel boss coplanar with right timing cover X in [-48*s, +48*s]
  boss_half_x = max(BLOCK_TIMING_BOSS_HALF_X_MM * s, TIMING_TUNNEL_HALF_X_MM * s + w_min)
  boss_min_y = 38.0 * s
  boss_d = TIMING_TUNNEL_BACK_Y_MM * s - boss_min_y
  with BuildPart() as boss_bp:
    with Locations((0.0, boss_min_y + HALF_FACTOR * boss_d, -HALF_FACTOR * block_h)):
      Box(2.0 * boss_half_x, boss_d, block_h)
  add_solids.append(boss_bp.part)

  # 45-deg self-supporting beveled cooling fins (bevel on +Z_world -> +Z_build when printed (180,0,0))
  fin_zone_top = -top_t
  fin_zone_bot = -block_h + base_t
  fin_span_z = fin_zone_top - fin_zone_bot
  fin_t = max(fin_min, 2.4 * s)
  fin_pitch = max(2.0 * fin_min, fin_span_z / (NUM_KIT_FINS + 0.5))
  n_fins = max(2, int(fin_span_z // fin_pitch))
  fin_proj = min(9.0 * s, max(2.0, fin_pitch - fin_t - 0.35))
  r_fin = r_barrel + fin_proj

  # Clip fins cleanly before +Y timing tunnel wall
  with BuildPart() as fin_clip_bp:
    with Locations((0.0, -HALF_FACTOR * (120.0 * s) + TIMING_TUNNEL_FLOOR_Y_MM * s, -HALF_FACTOR * block_h)):
      Box(2.0 * r_fin + 4.0, 120.0 * s, block_h + 2.0)

  for idx in range(n_fins):
    z_center = fin_zone_top - (idx + 0.85) * fin_pitch
    z_tip_bot = z_center - HALF_FACTOR * fin_t
    z_tip_top = z_center + HALF_FACTOR * fin_t
    if z_tip_bot <= fin_zone_bot + 0.3 or z_tip_top + fin_proj >= fin_zone_top - 0.1:
      continue
    with BuildPart() as fin_bp:
      with Locations((0.0, 0.0, z_center)):
        Cylinder(radius=r_fin, height=fin_t)
      with Locations((0.0, 0.0, z_tip_top)):
        Cone(
            bottom_radius=r_fin,
            top_radius=r_barrel,
            height=fin_proj,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    add_solids.append(_as_part(fin_bp.part.intersect(fin_clip_bp.part)))

  add_solids.extend(block_pins)
  add_solids.extend(j.hook for j in block_snaps)

  cutters: list[Part] = []
  # Seamless cylinder bore
  with BuildPart() as bore_bp:
    with Locations((0.0, 0.0, HALF_FACTOR * (1.0 + LINER_SPIGOT_BLOCK_BOT_Z_MM * s - 1.0))):
      Cylinder(radius=bore_r, height=block_h + spigot_h + 2.0)
  cutters.append(bore_bp.part)

  # Open +Y timing tunnel
  tun_d = (TIMING_TUNNEL_BACK_Y_MM - TIMING_TUNNEL_FLOOR_Y_MM) * s + 1.0
  with BuildPart() as tun_bp:
    with Locations((0.0, TIMING_TUNNEL_FLOOR_Y_MM * s + HALF_FACTOR * tun_d, -HALF_FACTOR * block_h)):
      Box(2.0 * TIMING_TUNNEL_HALF_X_MM * s, tun_d, block_h + 2.0)
  cutters.append(tun_bp.part)

  # Idler-2 blind socket at Y in [43.5*s, 48.0*s]
  sock_len = (TIMING_TUNNEL_FLOOR_Y_MM - IDLER_SOCKET_START_Y_MM) * s + 0.5
  sock_mid_y = IDLER_SOCKET_START_Y_MM * s + HALF_FACTOR * sock_len
  cutters.append(
      _teardrop_bore(_idler_pin_radius_mm(s) + r_slip, sock_len, axis="Y", crest_up_world=False).translate(
          (0.0, sock_mid_y, _idler_z_mm(1, s))
      )
  )

  # Conrod big-end clearance arch scallops at spigot bottom (Z in [-105*s, -95*s], |Y| <= 13*s)
  arch_y_half = max(5.5, 13.0 * s)
  arch_z_bot = LINER_SPIGOT_BLOCK_BOT_Z_MM * s - 0.5
  arch_z_top = (LINER_SPIGOT_BLOCK_BOT_Z_MM + 10.0) * s
  with BuildSketch(Plane.YZ) as arch_sk:
    Polygon([
        (-arch_y_half, arch_z_bot),
        (-arch_y_half, arch_z_top - 0.5 * arch_y_half),
        (0.0, arch_z_top),
        (arch_y_half, arch_z_top - 0.5 * arch_y_half),
        (arch_y_half, arch_z_bot),
    ])
  with BuildPart() as arch_bp:
    extrude(arch_sk.sketch, amount=r_spigot + 2.0, both=True)
  cutters.append(arch_bp.part)

  cutters.extend(head_holes)
  cutters.extend(block_side_notches)
  cutters.extend(j.male_relief_cutter for j in block_snaps if j.male_relief_cutter is not None)

  # Optional -Y cutaway viewing window (45-deg arch at -72*s for (180,0,0) printability)
  if cfg.cutaway:
    x_win = 26.0 * s
    z_bot = -72.0 * s
    z_top = -10.0 * s
    arch_drop = min(x_win, 0.45 * (z_top - z_bot))
    with BuildSketch(Plane.XZ) as win_sk:
      Polygon([
          (-x_win, z_top),
          (-x_win, z_bot + arch_drop),
          (0.0, z_bot),
          (x_win, z_bot + arch_drop),
          (x_win, z_top),
      ])
    with BuildPart() as win_bp:
      extrude(win_sk.sketch, amount=30.0 * s)
    cutters.append(win_bp.part.translate((0.0, -28.0 * s, 0.0)))

  return _apply_cut_list(_apply_fuse_list(barrel_bp.part, add_solids), cutters)


def _teardrop_sketch_profile(radius_mm: float, arc_samples: int = 24) -> None:
  """Populate active XY sketch with a 42-deg self-supporting teardrop crest along +Y."""
  pts = []
  start_deg = 138.0
  span_deg = 264.0
  for idx in range(arc_samples + 1):
    ang = math.radians(start_deg + span_deg * idx / arc_samples)
    pts.append((radius_mm * math.cos(ang), radius_mm * math.sin(ang)))
  pts.append((0.0, radius_mm / math.sin(math.radians(42.0))))
  Polygon(pts)


def _build_teardrop_port_cutter(
    p_flange: Tuple[float, float, float],
    p_mid: Tuple[float, float, float],
    p_seat: Tuple[float, float, float],
    flange_dir: Tuple[float, float, float],
    mid_dir: Tuple[float, float, float],
    seat_dir: Tuple[float, float, float],
    r_flange: float,
    r_valve_head: float,
    r_slip: float,
    seat_rot_y_deg: float,
    scale: float,
) -> Part:
  """Loft a self-supporting teardrop intake/exhaust port and valve seat pocket."""
  # Orient sketch +Y toward -Z_world so 45-deg teardrop crest faces +Z_build in (180,0,0) print.
  x_sign = -1.0 if flange_dir[0] >= 0.0 else 1.0

  def _plane(orig: Tuple[float, float, float], z_d: Tuple[float, float, float]) -> Plane:
    return Plane(origin=orig, x_dir=(0.0, x_sign, 0.0), z_dir=z_d)

  r_throat = max(2.0, r_valve_head - max(0.8, 1.2 * scale))
  r_mid = max(r_throat, r_flange - 1.5 * scale)

  with BuildPart() as port_bp:
    with BuildSketch(_plane(p_flange, flange_dir)):
      _teardrop_sketch_profile(r_flange)
    with BuildSketch(_plane(p_mid, mid_dir)):
      _teardrop_sketch_profile(r_mid)
    with BuildSketch(_plane(p_seat, seat_dir)):
      _teardrop_sketch_profile(r_throat)
    loft()

    # Valve seat pocket clearing clamped valve margin + seat cone + neck transition
    seat_h = max(6.5, 8.0 * scale)
    s_center = HALF_FACTOR * seat_h - 1.5 * scale
    with Locations((
        p_seat[0] + seat_dir[0] * s_center,
        p_seat[1],
        p_seat[2] + seat_dir[2] * s_center,
    )):
      Cylinder(radius=r_valve_head + r_slip, height=seat_h, rotation=(0.0, seat_rot_y_deg, 0.0))

  return port_bp.part


def _build_cylinder_head(
    cfg: PrintKitConfig,
    head_pins: Sequence[Part],
    head_snaps: Sequence[_SnapJoint],
    tower_holes: Sequence[Part],
    tower_snaps: Sequence[_SnapJoint],
    timing_snaps: Sequence[Tuple[str, _SnapJoint]],
    intake_pins: Sequence[Part],
    intake_snaps: Sequence[_SnapJoint],
) -> Part:
  """Generate kit_cylinder_head with merged guides/seats, teardrop ports, and snap interfaces."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  head_w = HEAD_WIDTH_X_MM * s
  min_y = HEAD_MIN_Y_MM * s
  max_y = TIMING_TUNNEL_BACK_Y_MM * s
  head_d = max_y - min_y
  mid_y = HALF_FACTOR * (min_y + max_y)
  head_h = HEAD_HEIGHT_Z_MM * s

  with BuildPart() as head_bp:
    with Locations((0.0, mid_y, HALF_FACTOR * head_h)):
      Box(head_w, head_d, head_h)

  cutters: list[Part] = []

  # Pent-roof combustion dome
  bore_r = BORE_RADIUS_MM * s
  apex_z = CHAMBER_APEX_Z_MM * s
  squish_y = CHAMBER_SQUISH_HALF_Y_MM * s
  roof_edge_z = apex_z - (40.0 * s) * math.tan(math.radians(VALVE_TILT_DEG))
  with BuildSketch(Plane.XZ) as roof_sk:
    Polygon([
        (-40.0 * s, -2.0),
        (-40.0 * s, roof_edge_z),
        (0.0, apex_z),
        (40.0 * s, roof_edge_z),
        (40.0 * s, -2.0),
    ])
  with BuildPart() as roof_bp:
    extrude(roof_sk.sketch, amount=squish_y, both=True)
  with BuildPart() as bore_dome_bp:
    with Locations((0.0, 0.0, HALF_FACTOR * (apex_z - 2.0))):
      Cylinder(radius=bore_r, height=apex_z + 2.0)
  cutters.append(_as_part(roof_bp.part.intersect(bore_dome_bp.part)))

  # Teardrop intake (+X) and exhaust (-X) ports
  sin_t = math.sin(math.radians(VALVE_TILT_DEG))
  cos_t = math.cos(math.radians(VALVE_TILT_DEG))
  in_port = _build_teardrop_port_cutter(
      p_flange=(HALF_FACTOR * head_w + 0.5, 0.0, IN_FLANGE_Z_MM * s),
      p_mid=(36.0 * s, 0.0, 22.0 * s),
      p_seat=(VALVE_SEAT_X_MM * s, 0.0, VALVE_SEAT_Z_MM * s),
      flange_dir=(1.0, 0.0, 0.0),
      mid_dir=(0.85, 0.0, 0.52),
      seat_dir=(sin_t, 0.0, cos_t),
      r_flange=IN_FLANGE_R_MM * s,
      r_valve_head=VALVE_IN_RADIUS_MM * s,
      r_slip=r_slip,
      seat_rot_y_deg=VALVE_TILT_DEG,
      scale=s,
  )
  ex_port = _build_teardrop_port_cutter(
      p_flange=(-HALF_FACTOR * head_w - 0.5, 0.0, EX_FLANGE_Z_MM * s),
      p_mid=(-36.0 * s, 0.0, 19.5 * s),
      p_seat=(-VALVE_SEAT_X_MM * s, 0.0, VALVE_SEAT_Z_MM * s),
      flange_dir=(-1.0, 0.0, 0.0),
      mid_dir=(-0.80, 0.0, 0.40),
      seat_dir=(-sin_t, 0.0, cos_t),
      r_flange=EX_FLANGE_R_MM * s,
      r_valve_head=VALVE_EX_RADIUS_MM * s,
      r_slip=r_slip,
      seat_rot_y_deg=-VALVE_TILT_DEG,
      scale=s,
  )
  cutters.extend([in_port, ex_port])

  # Merged valve stem guide bores + spring pockets (with 45-deg under-floor bevel for (180,0,0))
  stem_r = max(VALVE_STEM_MIN_R_MM, VALVE_STEM_BASE_R_MM * s)
  guide_r = stem_r + r_run
  spring_r = SPRING_POCKET_RADIUS_MM * s
  spring_floor_z = SPRING_POCKET_FLOOR_Z_MM * s
  spring_h = head_h - spring_floor_z + 2.0
  spring_x_off = (HEAD_HEIGHT_Z_MM - VALVE_SEAT_Z_MM) * s * math.tan(math.radians(VALVE_TILT_DEG))

  for vx_sign, rot_deg in ((1.0, VALVE_TILT_DEG), (-1.0, -VALVE_TILT_DEG)):
    axis_dir = (vx_sign * sin_t, 0.0, cos_t)
    stem_len = 66.0 * s + 10.0
    stem_mid = 30.0 * s
    with BuildPart() as guide_bp:
      with Locations((vx_sign * VALVE_SEAT_X_MM * s + axis_dir[0] * stem_mid, 0.0, VALVE_SEAT_Z_MM * s + axis_dir[2] * stem_mid)):
        Cylinder(radius=guide_r, height=stem_len, rotation=(0.0, rot_deg, 0.0))
    cutters.append(guide_bp.part)

    sx = vx_sign * (VALVE_SEAT_X_MM * s + spring_x_off)
    taper_h = max(0.5, spring_r - guide_r)
    with BuildPart() as sp_bp:
      with Locations((sx, 0.0, spring_floor_z + HALF_FACTOR * spring_h)):
        Cylinder(radius=spring_r, height=spring_h)
      with Locations((sx, 0.0, spring_floor_z - taper_h)):
        Cone(
            bottom_radius=guide_r,
            top_radius=spring_r,
            height=taper_h + 0.2,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    cutters.append(sp_bp.part)

  # Angled spark plug bore (24 deg from -Y) + socket well + snap detent groove
  tilt_rad = math.radians(PLUG_TILT_DEG)
  plug_dir = (0.0, -math.cos(tilt_rad), math.sin(tilt_rad))
  plug_rot_x = ROT_90_DEG - PLUG_TILT_DEG
  r_plug_shank = max(PLUG_SHANK_MIN_R_MM, PLUG_SHANK_BASE_R_MM * s)
  r_plug_well = max(PLUG_WELL_MIN_R_MM, PLUG_WELL_BASE_R_MM * s)

  shank_end = 18.0 * s
  shank_len = shank_end + 4.0 * s
  shank_mid = HALF_FACTOR * (shank_end - 4.0 * s)
  well_start = 15.5 * s
  well_end = 75.0 * s
  well_mid = HALF_FACTOR * (well_start + well_end)

  def _plug_pt(s_val: float) -> Tuple[float, float, float]:
    return (0.0, PLUG_ORIGIN_Y_MM * s + plug_dir[1] * s_val, PLUG_ORIGIN_Z_MM * s + plug_dir[2] * s_val)

  with BuildPart() as plug_cut_bp:
    with Locations(_plug_pt(shank_mid)):
      Cylinder(radius=r_plug_shank + r_slip, height=shank_len, rotation=(plug_rot_x, 0.0, 0.0))
    with Locations(_plug_pt(11.0 * s)):
      Cylinder(radius=r_plug_shank + r_slip + 0.4, height=max(1.4, 2.0 * s), rotation=(plug_rot_x, 0.0, 0.0))
    with Locations(_plug_pt(well_mid)):
      Cylinder(radius=r_plug_well, height=well_end - well_start, rotation=(plug_rot_x, 0.0, 0.0))
  cutters.append(plug_cut_bp.part)

  # Open +Y timing tunnel + Idler-3 blind socket
  tun_d = (TIMING_TUNNEL_BACK_Y_MM - TIMING_TUNNEL_FLOOR_Y_MM) * s + 1.0
  with BuildPart() as tun_bp:
    with Locations((0.0, TIMING_TUNNEL_FLOOR_Y_MM * s + HALF_FACTOR * tun_d, HALF_FACTOR * head_h)):
      Box(2.0 * TIMING_TUNNEL_HALF_X_MM * s, tun_d, head_h + 4.0)
  cutters.append(tun_bp.part)

  sock_len = (TIMING_TUNNEL_FLOOR_Y_MM - IDLER_SOCKET_START_Y_MM) * s + 0.5
  sock_mid_y = IDLER_SOCKET_START_Y_MM * s + HALF_FACTOR * sock_len
  cutters.append(
      _teardrop_bore(_idler_pin_radius_mm(s) + r_slip, sock_len, axis="Y", crest_up_world=False).translate(
          (0.0, sock_mid_y, _idler_z_mm(2, s))
      )
  )

  # Snap/pin interfaces (cut relief slots before fusing recessed hooks!)
  add_solids: list[Part] = list(head_pins) + list(intake_pins)
  cutters.extend(tower_holes)
  cutters.extend(j.pocket_cutter for j in tower_snaps)
  for host, joint in timing_snaps:
    if host == "cylinder_head":
      cutters.append(joint.pocket_cutter)

  # Optional twin -Y valve viewing windows preserving center spark plug bridge
  if cfg.cutaway:
    z_bot = 10.0 * s
    z_top = 44.0 * s
    y_cut_depth = 34.0 * s  # Y in [-62*s, -28*s]
    for x_a, x_b in ((-40.0 * s, -14.0 * s), (14.0 * s, 40.0 * s)):
      w_win = x_b - x_a
      x_mid = HALF_FACTOR * (x_a + x_b)
      drop = min(HALF_FACTOR * w_win, 0.35 * (z_top - z_bot))
      with BuildSketch(Plane.XZ) as hwin_sk:
        Polygon([
            (x_a, z_top),
            (x_a, z_bot + drop),
            (x_mid, z_bot),
            (x_b, z_bot + drop),
            (x_b, z_top),
        ])
      with BuildPart() as hwin_bp:
        extrude(hwin_sk.sketch, amount=y_cut_depth)
      cutters.append(hwin_bp.part.translate((0.0, -28.0 * s, 0.0)))

  return _apply_male_snaps(head_bp.part, list(head_snaps) + list(intake_snaps), cutters, add_solids)


def _build_cam_tower(
    cfg: PrintKitConfig,
    tower_pins: Sequence[Part],
    tower_snaps: Sequence[_SnapJoint],
    cover_holes: Sequence[Part],
    cover_snaps: Sequence[_SnapJoint],
    timing_pins: Sequence[Tuple[str, Part, Part]],
    timing_snaps: Sequence[Tuple[str, _SnapJoint]],
) -> Part:
  """Generate kit_cam_tower with teardrop cam bores, rocker pockets, and snap interfaces."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  h_tower = (TOWER_MAX_Z_MM - TOWER_MIN_Z_MM) * s
  mid_z = HALF_FACTOR * (TOWER_MAX_Z_MM + TOWER_MIN_Z_MM) * s
  tower_w = TOWER_WIDTH_X_MM * s
  min_y = TOWER_MIN_Y_MM * s
  max_y = TIMING_TUNNEL_BACK_Y_MM * s
  depth_y = max_y - min_y
  mid_y = HALF_FACTOR * (min_y + max_y)

  with BuildPart() as tower_bp:
    with Locations((0.0, mid_y, mid_z)):
      Box(tower_w, depth_y, h_tower)

  cutters: list[Part] = []

  # Central valvetrain pocket
  with BuildPart() as bay_bp:
    with Locations((0.0, 0.0, mid_z)):
      Box(2.0 * TOWER_BAY_HALF_X_MM * s, 2.0 * TOWER_BAY_HALF_Y_MM * s, h_tower + 4.0)
  cutters.append(bay_bp.part)

  # Open +Y timing bay
  tun_d = (TIMING_TUNNEL_BACK_Y_MM - TIMING_TUNNEL_FLOOR_Y_MM) * s + 1.0
  with BuildPart() as tbay_bp:
    with Locations((0.0, TIMING_TUNNEL_FLOOR_Y_MM * s + HALF_FACTOR * tun_d, mid_z)):
      Box(2.0 * TIMING_TUNNEL_HALF_X_MM * s, tun_d, h_tower + 4.0)
  cutters.append(tbay_bp.part)

  # Teardrop cam journal bores across front and rear bulkheads
  cam_r = max(CAM_CORE_MIN_R_MM, CAM_CORE_BASE_R_MM * s) + r_run
  cam_span_y = (TIMING_TUNNEL_FLOOR_Y_MM - TOWER_MIN_Y_MM) * s + 2.0
  cam_mid_y = HALF_FACTOR * (TIMING_TUNNEL_FLOOR_Y_MM + TOWER_MIN_Y_MM) * s
  cutters.append(
      _teardrop_bore(cam_r, cam_span_y, axis="Y", crest_up_world=False).translate(
          (0.0, cam_mid_y, CAM_AXIS_Z_MM * s)
      )
  )

  # Rocker shaft pockets from Y=-30.5*s through front bay and into rear bulkhead to Y=+26*s
  rocker_r = max(ROCKER_SHAFT_MIN_R_MM, ROCKER_SHAFT_BASE_R_MM * s) + r_slip
  r_min_y = (TOWER_MIN_Y_MM - 0.5) * s
  r_max_y = ROCKER_BORE_END_Y_MM * s
  r_len = r_max_y - r_min_y
  r_mid_y = HALF_FACTOR * (r_min_y + r_max_y)
  for rx in (-ROCKER_PIVOT_X_MM * s, ROCKER_PIVOT_X_MM * s):
    cutters.append(
        _teardrop_bore(rocker_r, r_len, axis="Y", crest_up_world=False).translate(
            (rx, r_mid_y, ROCKER_PIVOT_Z_MM * s)
        )
    )

  # Interfaces
  add_solids: list[Part] = list(tower_pins) + [j.hook for j in tower_snaps]
  cutters.extend(j.male_relief_cutter for j in tower_snaps if j.male_relief_cutter is not None)
  cutters.extend(cover_holes)
  cutters.extend(j.pocket_cutter for j in cover_snaps)
  for host, _, hole in timing_pins:
    if host == "cam_tower":
      cutters.append(hole)
  for host, joint in timing_snaps:
    if host == "cam_tower":
      cutters.append(joint.pocket_cutter)

  return _apply_cut_list(_apply_fuse_list(tower_bp.part, add_solids), cutters)


def _build_cam_cover(
    cfg: PrintKitConfig,
    cover_pins: Sequence[Part],
    cover_snaps: Sequence[_SnapJoint],
) -> Part:
  """Generate kit_cam_cover capping valvetrain & extended rear timing bay."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)

  base_z = TOWER_MAX_Z_MM * s
  top_z = CAM_COVER_TOP_Z_MM * s
  cover_h = top_z - base_z
  mid_z = HALF_FACTOR * (base_z + top_z)
  cover_w = TOWER_WIDTH_X_MM * s
  min_y = TOWER_MIN_Y_MM * s
  max_y = _cover_outer_y_mm(cfg)
  depth_y = max_y - min_y
  mid_y = HALF_FACTOR * (min_y + max_y)

  cav_top_z = min(CAM_COVER_CAVITY_TOP_Z_MM * s, top_z - w_min)
  cav_h = cav_top_z - base_z + 1.0
  cav_mid_z = base_z + HALF_FACTOR * (cav_top_z - base_z) - 0.5

  with BuildPart() as cover_bp:
    with Locations((0.0, mid_y, mid_z)):
      Box(cover_w, depth_y, cover_h)

  cutters: list[Part] = []
  with BuildPart() as val_cav_bp:
    with Locations((0.0, 0.0, cav_mid_z)):
      Box(2.0 * TOWER_BAY_HALF_X_MM * s, 2.0 * TOWER_BAY_HALF_Y_MM * s, cav_h)
  cutters.append(val_cav_bp.part)

  pkt_min_y = 47.0 * s
  pkt_max_y = _timing_cover_pocket_max_y_mm(cfg)
  pkt_d = pkt_max_y - pkt_min_y
  with BuildPart() as tim_cav_bp:
    with Locations((0.0, pkt_min_y + HALF_FACTOR * pkt_d, cav_mid_z)):
      Box(2.0 * TIMING_POCKET_HALF_X_MM * s, pkt_d, cav_h)
  cutters.append(tim_cav_bp.part)

  return _apply_male_snaps(cover_bp.part, cover_snaps, cutters, cover_pins)


def _build_right_timing_cover(
    cfg: PrintKitConfig,
    timing_pins: Sequence[Tuple[str, Part, Part]],
    timing_snaps: Sequence[Tuple[str, _SnapJoint]],
) -> Part:
  """Generate full-height kit_right_timing_cover with 3 idler sockets, PTO bore, and 6 snaps."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  min_y = TIMING_TUNNEL_BACK_Y_MM * s
  max_y = _cover_outer_y_mm(cfg)
  cover_d = max_y - min_y
  cover_y = HALF_FACTOR * (min_y + max_y)
  top_z = TOWER_MAX_Z_MM * s
  bot_z = TIMING_COVER_BOT_Z_MM * s
  cover_h = top_z - bot_z
  cover_z = HALF_FACTOR * (top_z + bot_z)
  cover_w = TIMING_COVER_WIDTH_X_MM * s

  pkt_max_y = _timing_cover_pocket_max_y_mm(cfg)
  pkt_d = pkt_max_y - min_y + 1.0
  pkt_y = min_y - 0.5 + HALF_FACTOR * pkt_d
  pkt_bot_z = TIMING_COVER_POCKET_BOT_Z_MM * s
  pkt_h = top_z - pkt_bot_z + 1.0
  pkt_z = pkt_bot_z + HALF_FACTOR * pkt_h

  with BuildPart() as cover_bp:
    with Locations((0.0, cover_y, cover_z)):
      Box(cover_w, cover_d, cover_h)

  cutters: list[Part] = []
  with BuildPart() as pkt_bp:
    with Locations((0.0, pkt_y, pkt_z)):
      Box(2.0 * TIMING_POCKET_HALF_X_MM * s, pkt_d, pkt_h)
  cutters.append(pkt_bp.part)

  # PTO shaft clearance bore (20*s + 2*r_run)
  with BuildPart() as pto_bp:
    with Locations((0.0, cover_y, CRANK_AXIS_Z_MM * s)):
      Cylinder(radius=10.0 * s + r_run, height=cover_d + 4.0, rotation=(ROT_90_DEG, 0.0, 0.0))
  cutters.append(pto_bp.part)

  # 3 blind sockets for timing idler pins (Y in [64*s, 68*s])
  sock_end_y = TIMING_COVER_SOCKET_END_Y_MM * s
  sock_len = sock_end_y - pkt_max_y + 0.5
  sock_mid_y = pkt_max_y - 0.25 + HALF_FACTOR * sock_len
  pin_r_hole = _idler_pin_radius_mm(s) + r_slip
  for idx in range(3):
    with BuildPart() as sock_bp:
      with Locations((0.0, sock_mid_y, _idler_z_mm(idx, s))):
        Cylinder(radius=pin_r_hole, height=sock_len, rotation=(ROT_90_DEG, 0.0, 0.0))
    cutters.append(sock_bp.part)

  if cfg.cutaway:
    slot_bot = -128.0 * s
    slot_top = 96.0 * s
    with BuildPart() as slot_bp:
      with Locations((0.0, cover_y, HALF_FACTOR * (slot_bot + slot_top))):
        Box(64.0 * s, cover_d + 4.0, slot_top - slot_bot)
    cutters.append(slot_bp.part)

  add_solids: list[Part] = [pin for _, pin, _ in timing_pins]
  return _apply_male_snaps(
      cover_bp.part,
      [joint for _, joint in timing_snaps],
      cutters,
      add_solids,
  )


def _build_fan_shroud(cfg: PrintKitConfig, shroud_snaps: Sequence[_SnapJoint]) -> Part:
  """Generate kit_fan_shroud with 4 +Y standoff snap arms clearing 80*s flywheel."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)

  shroud_w = SHROUD_WIDTH_X_MM * s
  min_y = SHROUD_MIN_Y_MM * s
  max_y = max(SHROUD_MAX_Y_MM * s, min_y + w_min)
  shroud_d = max_y - min_y
  shroud_y = HALF_FACTOR * (min_y + max_y)
  shroud_h = (SHROUD_MAX_Z_MM - SHROUD_MIN_Z_MM) * s
  shroud_z = HALF_FACTOR * (SHROUD_MAX_Z_MM + SHROUD_MIN_Z_MM) * s

  arm_w = max(2.0 * w_min, 8.0 * s)
  arm_h = max(2.0 * w_min, 8.0 * s)
  case_front_y = -CASE_HALF_Y_MM * s
  arm_len = case_front_y - max_y
  arm_mid_y = max_y + HALF_FACTOR * arm_len - 0.25

  add_solids: list[Part] = []
  with BuildPart() as shroud_bp:
    with Locations((0.0, shroud_y, shroud_z)):
      Box(shroud_w, shroud_d, shroud_h)

  # Corner gusset ears connecting shroud perimeter smoothly to the 4 flywheel-clearing arms
  for ax, az in SHROUD_ARM_XZ_MM:
    x_pos = ax * s
    z_pos = az * s
    anchor_x = max(-HALF_FACTOR * shroud_w + HALF_FACTOR * arm_w, min(HALF_FACTOR * shroud_w - HALF_FACTOR * arm_w, x_pos))
    anchor_z = max(SHROUD_MIN_Z_MM * s + HALF_FACTOR * arm_h, min(SHROUD_MAX_Z_MM * s - HALF_FACTOR * arm_h, z_pos))
    with BuildPart() as arm_bp:
      with Locations((x_pos, arm_mid_y, z_pos)):
        Box(arm_w, arm_len + 0.5, arm_h)
      # Bridge lug inside shroud plane [min_y, max_y]
      x_lo = min(anchor_x, x_pos) - HALF_FACTOR * arm_w
      x_hi = max(anchor_x, x_pos) + HALF_FACTOR * arm_w
      z_lo = min(anchor_z, z_pos) - HALF_FACTOR * arm_h
      z_hi = max(anchor_z, z_pos) + HALF_FACTOR * arm_h
      with Locations((HALF_FACTOR * (x_lo + x_hi), shroud_y, HALF_FACTOR * (z_lo + z_hi))):
        Box(x_hi - x_lo, shroud_d, z_hi - z_lo)
    add_solids.append(arm_bp.part)

  add_solids.extend(j.hook for j in shroud_snaps)

  with BuildPart() as hole_bp:
    with Locations((0.0, shroud_y, CRANK_AXIS_Z_MM * s)):
      Cylinder(radius=max(SHROUD_CENTER_HOLE_R_MM * s, 10.0 * s), height=shroud_d + 4.0, rotation=(ROT_90_DEG, 0.0, 0.0))

  return _apply_cut_list(_apply_fuse_list(shroud_bp.part, add_solids), [hole_bp.part])


def _build_intake_carb_unit(
    cfg: PrintKitConfig,
    intake_holes: Sequence[Part],
    intake_snaps: Sequence[_SnapJoint],
) -> Part:
  """Generate merged kit_intake_carb_unit with self-supporting carb gussets and flange pockets."""
  s = cfg.scale
  w_min = min_printable_wall_mm(cfg, structural=True)

  x_start = INTAKE_START_X_MM * s
  x_carb = CARB_START_X_MM * s
  x_end = CARB_END_X_MM * s
  flange_t = max(w_min * 2.0, INTAKE_FLANGE_THICK_X_MM * s)
  flange_y = INTAKE_FLANGE_Y_MM * s
  flange_min_z = INTAKE_FLANGE_MIN_Z_MM * s
  flange_max_z = INTAKE_FLANGE_MAX_Z_MM * s
  flange_h = flange_max_z - flange_min_z
  flange_mid_z = HALF_FACTOR * (flange_min_z + flange_max_z)

  r_in = IN_FLANGE_R_MM * s
  r_venturi = CARB_VENTURI_R_MM * s
  r_neck = max(17.5 * s, r_venturi + w_min)
  neck_len = x_carb - x_start

  bowl_bot_z = CARB_BOWL_BOT_Z_MM * s
  bowl_w = CARB_BOWL_WIDTH_Y_MM * s
  bowl_len = x_end - x_carb
  bowl_mid_x = HALF_FACTOR * (x_carb + x_end)
  bowl_h = IN_FLANGE_Z_MM * s - bowl_bot_z

  tower_r = max(CARB_TOWER_R_MM * s, r_venturi)
  tower_top_z = CARB_SLIDE_TOP_Z_MM * s
  tower_h = tower_top_z - IN_FLANGE_Z_MM * s

  with BuildPart() as unit_bp:
    with Locations((x_start + HALF_FACTOR * flange_t, 0.0, flange_mid_z)):
      Box(flange_t, flange_y, flange_h)
    with Locations((x_start + 0.5 + HALF_FACTOR * (neck_len - 0.5), 0.0, IN_FLANGE_Z_MM * s)):
      Cylinder(radius=r_neck, height=neck_len, rotation=(0.0, ROT_90_DEG, 0.0))
    with Locations((bowl_mid_x, 0.0, IN_FLANGE_Z_MM * s)):
      Cylinder(radius=r_neck, height=bowl_len, rotation=(0.0, ROT_90_DEG, 0.0))
    with Locations((bowl_mid_x, 0.0, bowl_bot_z + HALF_FACTOR * bowl_h)):
      Box(bowl_len, bowl_w, bowl_h)
    with Locations((bowl_mid_x, 0.0, IN_FLANGE_Z_MM * s + HALF_FACTOR * tower_h)):
      Cylinder(radius=tower_r, height=tower_h)

    # <= 42-deg self-supporting gusset loft from flange base to float bowl bottom
    with BuildSketch(Plane.YZ.offset(x_start + flange_t - 0.5)) as fl_sk:
      Polygon([
          (-HALF_FACTOR * bowl_w, flange_min_z),
          (HALF_FACTOR * bowl_w, flange_min_z),
          (HALF_FACTOR * bowl_w, IN_FLANGE_Z_MM * s),
          (-HALF_FACTOR * bowl_w, IN_FLANGE_Z_MM * s),
      ])
    with BuildSketch(Plane.YZ.offset(x_carb + 0.5)) as bw_sk:
      Polygon([
          (-HALF_FACTOR * bowl_w, bowl_bot_z),
          (HALF_FACTOR * bowl_w, bowl_bot_z),
          (HALF_FACTOR * bowl_w, IN_FLANGE_Z_MM * s),
          (-HALF_FACTOR * bowl_w, IN_FLANGE_Z_MM * s),
      ])
    loft()

  cutters: list[Part] = []
  total_x = x_end - x_start
  with BuildPart() as bore_bp:
    with Locations((x_start + HALF_FACTOR * total_x, 0.0, IN_FLANGE_Z_MM * s)):
      Cylinder(radius=r_venturi, height=total_x + 4.0, rotation=(0.0, ROT_90_DEG, 0.0))
    with Locations((x_start - 0.5, 0.0, IN_FLANGE_Z_MM * s)):
      Cone(
          bottom_radius=r_in,
          top_radius=r_venturi,
          height=flange_t + 2.0,
          rotation=(0.0, ROT_90_DEG, 0.0),
          align=(Align.CENTER, Align.CENTER, Align.MIN),
      )
  cutters.append(bore_bp.part)
  cutters.extend(intake_holes)
  cutters.extend(j.pocket_cutter for j in intake_snaps)
  return _apply_cut_list(unit_bp.part, cutters)


def _build_spark_plug(cfg: PrintKitConfig) -> Part:
  """Generate cosmetic snap-in kit_spark_plug in scaled world coordinates."""
  s = cfg.scale
  r_slip = slip_clearance_radial_mm(cfg)
  r_shank = max(PLUG_SHANK_MIN_R_MM, PLUG_SHANK_BASE_R_MM * s)
  r_well = max(PLUG_WELL_MIN_R_MM, PLUG_WELL_BASE_R_MM * s)
  r_hex = min(r_well - max(0.35, r_slip + 0.15), max(3.4, 8.6 * s))
  r_ceramic = min(r_hex - 0.4, max(2.0, 5.2 * s))
  r_term = max(1.0, 2.8 * s)

  s1 = (PLUG_REACH_LEN_MM + 1.0) * s
  s2 = s1 + PLUG_HEX_LEN_MM * s
  s3 = s2 + PLUG_CERAMIC_LEN_MM * s
  s4 = s3 + PLUG_TERMINAL_LEN_MM * s
  chamfer_h = max(0.6, 1.15 * (r_hex - r_shank))

  with BuildPart() as plug_bp:
    with Locations((0.0, 0.0, HALF_FACTOR * s1)):
      Cylinder(radius=r_shank, height=s1)
    # Tapered friction-snap rib seated cleanly inside head bore detent groove
    with Locations((0.0, 0.0, 10.0 * s)):
      Cylinder(radius=r_shank + r_slip + 0.15, height=max(0.8, min(1.1, 1.2 * s)))
    # 45-deg self-supporting hex transition starting inside socket well (s_head >= 17*s)
    with Locations((0.0, 0.0, s1)):
      Cone(bottom_radius=r_shank, top_radius=r_hex, height=chamfer_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations((0.0, 0.0, s1 + chamfer_h + HALF_FACTOR * max(1.0, s2 - s1 - chamfer_h))):
      Cylinder(radius=r_hex, height=max(1.0, s2 - s1 - chamfer_h))
    with Locations((0.0, 0.0, s2 + HALF_FACTOR * (s3 - s2))):
      Cylinder(radius=r_ceramic, height=s3 - s2)
    with Locations((0.0, 0.0, s3 + HALF_FACTOR * (s4 - s3))):
      Cylinder(radius=r_term, height=s4 - s3)

  tilt_rad = math.radians(PLUG_TILT_DEG)
  plug_dir = (0.0, -math.cos(tilt_rad), math.sin(tilt_rad))
  s_off = 1.0 * s
  return (
      plug_bp.part.rotate(Axis.X, ROT_90_DEG - PLUG_TILT_DEG)
      .translate((
          0.0,
          PLUG_ORIGIN_Y_MM * s + plug_dir[1] * s_off,
          PLUG_ORIGIN_Z_MM * s + plug_dir[2] * s_off,
      ))
  )


# ---------------------------------------------------------------------------
# Public Entry Points
# ---------------------------------------------------------------------------

def build_kit_housings_parts(cfg: PrintKitConfig) -> Dict[str, KitPart]:
  """Build all 11 3D-printable engine housing, cover, manifold, and plug KitParts."""
  split_pins, split_holes, split_snaps = _crankcase_split_pins_and_snaps(cfg)
  sump_snaps = _sump_snap_joints(cfg)
  block_pins, block_holes, block_snaps = _block_to_case_pins_and_snaps(cfg)
  head_pins, head_holes, head_snaps, block_side_notches = _head_to_block_pins_and_snaps(cfg)
  tower_pins, tower_holes, tower_snaps = _tower_to_head_pins_and_snaps(cfg)
  cover_pins, cover_holes, cover_snaps = _cover_to_tower_pins_and_snaps(cfg)
  timing_pins, timing_snaps = _timing_cover_pins_and_snaps(cfg)
  shroud_snaps = _shroud_snap_joints(cfg)
  intake_pins, intake_holes, intake_snaps = _intake_pins_and_snaps(cfg)

  crankcase_left = _build_crankcase_half(
      cfg, False, split_pins, split_holes, split_snaps, sump_snaps,
      block_holes, block_snaps, timing_pins, timing_snaps, shroud_snaps,
  )
  crankcase_right = _build_crankcase_half(
      cfg, True, split_pins, split_holes, split_snaps, sump_snaps,
      block_holes, block_snaps, timing_pins, timing_snaps, shroud_snaps,
  )
  sump_pan = _build_sump_pan(cfg, sump_snaps)
  cylinder_block = _build_cylinder_block(cfg, block_pins, block_snaps, head_holes, block_side_notches)
  cylinder_head = _build_cylinder_head(
      cfg, head_pins, head_snaps, tower_holes, tower_snaps, timing_snaps, intake_pins, intake_snaps,
  )
  cam_tower = _build_cam_tower(cfg, tower_pins, tower_snaps, cover_holes, cover_snaps, timing_pins, timing_snaps)
  cam_cover = _build_cam_cover(cfg, cover_pins, cover_snaps)
  right_timing_cover = _build_right_timing_cover(cfg, timing_pins, timing_snaps)
  fan_shroud = _build_fan_shroud(cfg, shroud_snaps)
  intake_carb_unit = _build_intake_carb_unit(cfg, intake_holes, intake_snaps)
  spark_plug = _build_spark_plug(cfg)

  cut_note = " (cutaway display)" if cfg.cutaway else ""
  return {
      "kit_crankcase_left": KitPart(
          name="kit_crankcase_left",
          part=crankcase_left,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes=f"Left (-Y) crankcase half, top-deck down{cut_note}",
      ),
      "kit_crankcase_right": KitPart(
          name="kit_crankcase_right",
          part=crankcase_right,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes=f"Right (+Y) crankcase half w/ timing bay & Idler-1 socket{cut_note}",
      ),
      "kit_sump_pan": KitPart(
          name="kit_sump_pan",
          part=sump_pan,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(0.0, 0.0, 0.0),
          notes="Sloped wet-sump pan w/ locating lip & 6 perimeter snap hooks",
      ),
      "kit_cylinder_block": KitPart(
          name="kit_cylinder_block",
          part=cylinder_block,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes=f"Finned cylinder block + merged liner spigot, top-deck down{cut_note}",
      ),
      "kit_cylinder_head": KitPart(
          name="kit_cylinder_head",
          part=cylinder_head,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes=f"Pent-roof head w/ merged guides/seats & teardrop ports{cut_note}",
      ),
      "kit_cam_tower": KitPart(
          name="kit_cam_tower",
          part=cam_tower,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes="SOHC cam tower w/ teardrop cam bores & rocker shaft pockets",
      ),
      "kit_cam_cover": KitPart(
          name="kit_cam_cover",
          part=cam_cover,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_180_DEG, 0.0, 0.0),
          notes="Extended rear SOHC cam cover w/ 4 tool-free side snaps",
      ),
      "kit_right_timing_cover": KitPart(
          name="kit_right_timing_cover",
          part=right_timing_cover,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(-ROT_90_DEG, 0.0, 0.0),
          notes=f"Full-height 5-gear timing cover w/ 3 idler sockets & PTO bore{cut_note}",
      ),
      "kit_fan_shroud": KitPart(
          name="kit_fan_shroud",
          part=fan_shroud,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(ROT_90_DEG, 0.0, 0.0),
          notes="Flywheel cooling shroud w/ 4 standoff snap arms clearing 80*s flywheel",
      ),
      "kit_intake_carb_unit": KitPart(
          name="kit_intake_carb_unit",
          part=intake_carb_unit,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(0.0, -ROT_90_DEG, 0.0),
          notes="Merged intake manifold adapter & Mikuni VM26 carburetor body",
      ),
      "kit_spark_plug": KitPart(
          name="kit_spark_plug",
          part=spark_plug,
          quantity=1,
          category="housings",
          preferred_rotation_xyz_deg=(-(ROT_90_DEG - PLUG_TILT_DEG), 0.0, 0.0),
          notes="Cosmetic snap-in spark plug (CMR7H)",
      ),
  }


def build_kit_housings_assembly_parts(
    cfg: PrintKitConfig,
    crank_angle_deg: float = 0.0,
) -> Dict[str, KitPart]:
  """Return all posed housing KitParts in master assembly coordinates."""
  _ = crank_angle_deg
  return build_kit_housings_parts(cfg)
