"""Filament-specific tolerance and snap-fit calibration coupon generator."""

from typing import Tuple

from build123d import Align, Box, BuildPart, Cone, Cylinder, Locations, Part, Solid, chamfer
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Trsf

from openengine.printing.primitives import (
    make_alignment_hole_cutter,
    make_alignment_pin,
    make_cantilever_snap_hook,
    make_snap_pocket_cutter,
)
from openengine.printing.tolerances import (
    PrintKitConfig,
    SnapFitDesign,
    auto_size_cantilever_snap,
    press_interference_radial_mm,
    running_clearance_radial_mm,
    slip_clearance_radial_mm,
)

COUPON_LENGTH_X_MM: float = 50.0
COUPON_DEPTH_Y_MM: float = 35.0
PLATE_DEPTH_Y_MM: float = 19.0
PLATE_THICKNESS_Z_MM: float = 8.0
HALF_FACTOR: float = 0.5

PLATE_MIN_Y_MM: float = -HALF_FACTOR * COUPON_DEPTH_Y_MM
PLATE_MAX_Y_MM: float = PLATE_MIN_Y_MM + PLATE_DEPTH_Y_MM

BORE_NOMINAL_RADIUS_MM: float = 2.5
BORE_Y_MM: float = -7.5
BORE_X_POSITIONS_MM: Tuple[float, ...] = (-17.0, -8.5, 0.0, 8.5, 17.0)
BORE_CHAMFER_MM: float = 0.4
BORE_CUT_OVERSHOOT_MM: float = 0.2
MIN_PRESS_OFFSET_MM: float = 0.04
TALLY_Y_BASE_MM: float = -13.0
TALLY_Y_STEP_MM: float = 1.3
TALLY_X_STEP_MM: float = 1.4
TALLY_RADIUS_MM: float = 0.5
TALLY_DEPTH_MM: float = 0.8

RECEIVER_MIN_X_MM: float = -25.0
RECEIVER_MAX_X_MM: float = -5.0
RECEIVER_MIN_Y_MM: float = PLATE_MAX_Y_MM
RECEIVER_MAX_Y_MM: float = HALF_FACTOR * COUPON_DEPTH_Y_MM
RECEIVER_POCKET_Z_FLOOR_MM: float = 2.0
RECEIVER_POCKET_Y_MM: float = 9.5

SNAP_AVAILABLE_LEN_MM: float = 9.0
SNAP_BEAM_WIDTH_MM: float = 5.5

LATCH_GRIP_MIN_X_MM: float = 10.0
LATCH_GRIP_MAX_X_MM: float = 14.5
LATCH_GRIP_MIN_Y_MM: float = 5.5
LATCH_GRIP_MAX_Y_MM: float = 15.0
LATCH_HOOK_Y_MM: float = 9.5

PEG_TAB_MIN_X_MM: float = 17.0
PEG_TAB_MAX_X_MM: float = 25.0
PEG_TAB_MIN_Y_MM: float = 4.5
PEG_TAB_MAX_Y_MM: float = 16.5
PEG_TAB_HEIGHT_Z_MM: float = 3.5
PEG_TAB_CHAMFER_MM: float = 0.4
PEG_LENGTH_MM: float = 8.5
PEG_CHAMFER_MM: float = 0.5


def coupon_bore_radial_offsets_mm(cfg: PrintKitConfig) -> Tuple[float, float, float, float, float]:
  """Return the 5 radial bore clearance offsets (mm) calibrated around the filament profile.

  The sequence spans from slight press interference through nominal zero, half-slip,
  ``slip_clearance_radial_mm(cfg)``, and ``running_clearance_radial_mm(cfg)``.
  """
  r_press = -max(MIN_PRESS_OFFSET_MM, press_interference_radial_mm(cfg))
  r_zero = 0.0
  r_slip = slip_clearance_radial_mm(cfg)
  r_half_slip = round(HALF_FACTOR * r_slip, 4)
  r_run = running_clearance_radial_mm(cfg)
  return (r_press, r_zero, r_half_slip, r_slip, r_run)


def _place_snap_flat_neg_x(
    shape: Part,
    beam_width_mm: float,
    anchor_xyz_mm: Tuple[float, float, float],
) -> Part:
  """Place a local +Z-beam/+X-barb snap solid flat on Z=anchor_z pointing -X with barb on +Y.

  Uses an exact integer gp_Trsf matrix so coplanar Z=0 faces carry zero floating-point
  trigonometric skew when fused onto horizontal build-plate pads.
  """
  ax, ay, az = anchor_xyz_mm
  # Map local (x, y, z) -> (ax - z, ay + x, az + width/2 - y)
  trsf = gp_Trsf()
  trsf.SetValues(
      0.0, 0.0, -1.0, ax,
      1.0, 0.0, 0.0, ay,
      0.0, -1.0, 0.0, az + (HALF_FACTOR * beam_width_mm),
  )
  placed_solids = [
      Solid(TopoDS.Solid(BRepBuilderAPI_Transform(sol.wrapped, trsf, True).Shape()))
      for sol in shape.solids()
  ]
  return Part(children=placed_solids)


def _build_gauge_and_receiver(
    cfg: PrintKitConfig,
    snap_design: SnapFitDesign,
) -> Solid:
  """Build the 5-bore clearance gauge plate fused with the open-top snap-fit receiver."""
  offsets = coupon_bore_radial_offsets_mm(cfg)

  with BuildPart() as plate_builder:
    with Locations((0.0, HALF_FACTOR * (PLATE_MIN_Y_MM + PLATE_MAX_Y_MM), 0.0)):
      Box(
          COUPON_LENGTH_X_MM,
          PLATE_DEPTH_Y_MM,
          PLATE_THICKNESS_Z_MM,
          align=(Align.CENTER, Align.CENTER, Align.MIN),
      )

  rec_w = RECEIVER_MAX_X_MM - RECEIVER_MIN_X_MM
  rec_d = RECEIVER_MAX_Y_MM - RECEIVER_MIN_Y_MM
  rec_cx = HALF_FACTOR * (RECEIVER_MIN_X_MM + RECEIVER_MAX_X_MM)
  rec_cy = HALF_FACTOR * (RECEIVER_MIN_Y_MM + RECEIVER_MAX_Y_MM)
  with BuildPart() as rec_builder:
    with Locations((rec_cx, rec_cy, 0.0)):
      Box(
          rec_w,
          rec_d,
          PLATE_THICKNESS_Z_MM,
          align=(Align.CENTER, Align.CENTER, Align.MIN),
      )

  body = plate_builder.part.fuse(rec_builder.part)

  cutters: list[Part] = []
  for idx, (x_pos, offset_mm) in enumerate(zip(BORE_X_POSITIONS_MM, offsets), start=1):
    bore_r = BORE_NOMINAL_RADIUS_MM + offset_mm
    if offset_mm >= 0.0:
      bore_cut = make_alignment_hole_cutter(
          radius_mm=BORE_NOMINAL_RADIUS_MM,
          depth_mm=PLATE_THICKNESS_Z_MM + (2.0 * BORE_CUT_OVERSHOOT_MM),
          clearance_mm=offset_mm,
      )
    else:
      bore_cut = make_alignment_hole_cutter(
          radius_mm=bore_r,
          depth_mm=PLATE_THICKNESS_Z_MM + (2.0 * BORE_CUT_OVERSHOOT_MM),
          clearance_mm=0.0,
      )
    cutters.append(bore_cut.translate((x_pos, BORE_Y_MM, -BORE_CUT_OVERSHOOT_MM)))

    # Lead-in chamfer at the top entry of each calibration bore.
    with BuildPart() as lead_cone:
      with Locations((x_pos, BORE_Y_MM, PLATE_THICKNESS_Z_MM - BORE_CHAMFER_MM)):
        Cone(
            bottom_radius=bore_r,
            top_radius=bore_r + BORE_CHAMFER_MM,
            height=BORE_CHAMFER_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    cutters.append(lead_cone.part)

    # Index tally dimples (1..5) along the front rim under each bore.
    for dot_i in range(idx):
      row = dot_i // 3
      col = dot_i % 3
      row_count = min(3, idx - (row * 3))
      dx = (col - HALF_FACTOR * (row_count - 1)) * TALLY_X_STEP_MM
      dy = -row * TALLY_Y_STEP_MM
      with BuildPart() as dot_cut:
        with Locations((x_pos + dx, TALLY_Y_BASE_MM + dy, PLATE_THICKNESS_Z_MM - TALLY_DEPTH_MM)):
          Cylinder(
              radius=TALLY_RADIUS_MM,
              height=TALLY_DEPTH_MM + BORE_CUT_OVERSHOOT_MM,
              align=(Align.CENTER, Align.CENTER, Align.MIN),
          )
      cutters.append(dot_cut.part)

  # Cut the snap-fit receiver pocket open at +X and +Z over a solid 2.0 mm floor.
  pocket_local = make_snap_pocket_cutter(snap_design, snap_design.clearance_mm)
  pocket_positioned = _place_snap_flat_neg_x(
      pocket_local,
      beam_width_mm=snap_design.beam_width_mm,
      anchor_xyz_mm=(RECEIVER_MAX_X_MM, RECEIVER_POCKET_Y_MM, RECEIVER_POCKET_Z_FLOOR_MM),
  )
  cutters.append(pocket_positioned)
  cutters.append(pocket_positioned.translate((0.0, 0.0, 1.0)))

  result = body.cut(*cutters)
  solids = list(result.solids())
  if len(solids) != 1:
    raise RuntimeError(f"Expected 1 solid for coupon gauge block, got {len(solids)}")
  return solids[0]


def _build_snap_latch_solid(snap_design: SnapFitDesign) -> Solid:
  """Build the printable cantilever snap-fit latch handle sitting flat on Z=0."""
  grip_w = LATCH_GRIP_MAX_X_MM - LATCH_GRIP_MIN_X_MM
  grip_d = LATCH_GRIP_MAX_Y_MM - LATCH_GRIP_MIN_Y_MM
  grip_cx = HALF_FACTOR * (LATCH_GRIP_MIN_X_MM + LATCH_GRIP_MAX_X_MM)
  grip_cy = HALF_FACTOR * (LATCH_GRIP_MIN_Y_MM + LATCH_GRIP_MAX_Y_MM)

  with BuildPart() as grip_builder:
    with Locations((grip_cx, grip_cy, 0.0)):
      Box(
          grip_w,
          grip_d,
          snap_design.beam_width_mm,
          align=(Align.CENTER, Align.CENTER, Align.MIN),
      )

  hook_local = make_cantilever_snap_hook(snap_design)
  hook_placed = _place_snap_flat_neg_x(
      hook_local,
      beam_width_mm=snap_design.beam_width_mm,
      anchor_xyz_mm=(LATCH_GRIP_MIN_X_MM, LATCH_HOOK_Y_MM, 0.0),
  )

  fused = grip_builder.part.fuse(hook_placed)
  solids = list(fused.solids())
  if len(solids) != 1:
    raise RuntimeError(f"Expected 1 solid for coupon snap latch, got {len(solids)}")
  return solids[0]


def _build_calibration_peg_solid() -> Solid:
  """Build the removable upright nominal calibration peg on a chamfered thumb tab."""
  tab_w = PEG_TAB_MAX_X_MM - PEG_TAB_MIN_X_MM
  tab_d = PEG_TAB_MAX_Y_MM - PEG_TAB_MIN_Y_MM
  tab_cx = HALF_FACTOR * (PEG_TAB_MIN_X_MM + PEG_TAB_MAX_X_MM)
  tab_cy = HALF_FACTOR * (PEG_TAB_MIN_Y_MM + PEG_TAB_MAX_Y_MM)

  with BuildPart() as tab_builder:
    with Locations((tab_cx, tab_cy, 0.0)):
      Box(
          tab_w,
          tab_d,
          PEG_TAB_HEIGHT_Z_MM,
          align=(Align.CENTER, Align.CENTER, Align.MIN),
      )
  top_edges = tab_builder.part.edges().filter_by(lambda e: e.center().Z > PEG_TAB_HEIGHT_Z_MM - 0.1)
  tab_chamfered = Part(chamfer(top_edges, length=PEG_TAB_CHAMFER_MM).wrapped)

  pin = make_alignment_pin(
      radius_mm=BORE_NOMINAL_RADIUS_MM,
      length_mm=PEG_LENGTH_MM,
      chamfer_mm=PEG_CHAMFER_MM,
  ).translate((tab_cx, tab_cy, PEG_TAB_HEIGHT_Z_MM))

  fused = tab_chamfered.fuse(pin)
  solids = list(fused.solids())
  if len(solids) != 1:
    raise RuntimeError(f"Expected 1 solid for coupon calibration peg, got {len(solids)}")
  return solids[0]


def generate_tolerance_coupon(cfg: PrintKitConfig) -> Part:
  """Generate a compact (~50x35x12 mm) filament-specific bore & snap-fit calibration coupon.

  Contains three separated build-plate-ready solids sitting on ``Z = 0``:
  1. A 5-bore clearance gauge plate spanning ``coupon_bore_radial_offsets_mm(cfg)``
     plus an integrated open-top snap-fit receiver slot sized via ``snap_clearance_mm(cfg)``.
  2. A printable cantilever snap-fit latch handle generated from
     ``auto_size_cantilever_snap(cfg, ...)`` lying flat in XY for perimeter-strong bending.
  3. An upright 5.0 mm nominal test peg on a chamfered thumb tab for checking all 5 bores.
  """
  snap_design = auto_size_cantilever_snap(
      cfg,
      available_length_mm=SNAP_AVAILABLE_LEN_MM,
      width_mm=SNAP_BEAM_WIDTH_MM,
  )
  gauge_solid = _build_gauge_and_receiver(cfg, snap_design)
  latch_solid = _build_snap_latch_solid(snap_design)
  peg_solid = _build_calibration_peg_solid()

  coupon = Part(children=[gauge_solid, latch_solid, peg_solid])
  coupon.label = f"tolerance_coupon_{cfg.filament_profile.name.lower()}"
  return coupon
