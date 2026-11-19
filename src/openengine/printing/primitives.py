"""Reusable build123d FDM snap-fit, alignment, support-free bore, and flexure primitives.

All dimensions and returned B-Rep solids are in millimeters (mm) with predictable,
documented local coordinate frames so downstream model-kit builders can place and
Boolean-subtract cutters reliably.
"""

import math
from build123d import (
    Align,
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    CenterArc,
    Cylinder,
    Line,
    Part,
    Plane,
    Polyline,
    Rectangle,
    Shape,
    extrude,
    loft,
    make_face,
    revolve,
)

from openengine.printing.tolerances import SnapFitDesign

HALF_FACTOR: float = 0.5
MIN_GEOM_TOL_MM: float = 1.0e-4
CHAMFER_CLAMP_RATIO: float = 0.45
MIN_HOOK_NOSE_X_MM: float = 0.4
HOOK_NOSE_TIP_RATIO: float = 0.5
FILLET_LENGTH_CLAMP_RATIO: float = 0.4
DEFAULT_KEYED_CLEARANCE_MM: float = 0.20

TEARDROP_START_ANGLE_DEG: float = 135.0
TEARDROP_ARC_SPAN_DEG: float = 270.0
TEARDROP_SQRT2: float = math.sqrt(2.0)

FULL_CIRCLE_DEG: float = 360.0
RADIAL_CUT_MARGIN_MM: float = 1.0
RADIAL_CUT_CHORD_DEG: float = 10.0
MAX_WEDGE_STEP_DEG: float = 15.0
MIN_WEDGE_HEIGHT_MM: float = 0.06
MIN_SPRING_BEAM_T_MM: float = 0.45
TARGET_MIN_SPRING_BEAM_T_MM: float = 0.7
TARGET_MAX_SPRING_BEAM_T_MM: float = 1.4
MIN_SPRING_SLOT_H_MM: float = 0.5
SPRING_BEAM_HEIGHT_DIV_BASE: float = 1.2
SPRING_BEAM_HEIGHT_DIV_PER_ROW: float = 2.4
MIN_PILLAR_WIDTH_MM: float = 1.2
PILLAR_WIDTH_BEAM_MULT: float = 1.4
MAX_PILLAR_PERIOD_RATIO: float = 0.45
MIN_ROOF_WIN_RATIO: float = 0.22
MAX_ROOF_WIN_RATIO: float = 0.45


def _as_part(shape: Shape) -> Part:
  """Normalize a build123d Solid/Compound/Part into a valid Part without dropping volume."""
  if isinstance(shape, Part):
    return shape
  solids = list(shape.solids())
  if not solids:
    raise ValueError("Expected at least one solid in generated B-Rep shape")
  return Part(children=solids)


def _validate_positive(name: str, value: float) -> float:
  """Ensure a geometric scalar parameter is strictly positive."""
  val = float(value)
  if val <= 0.0:
    raise ValueError(f"{name} must be > 0, got {value}")
  return val


def _snap_hook_envelope_mm(design: SnapFitDesign) -> tuple[float, float, float, float, float]:
  """Compute shared hook profile Z/X milestones used by hook and pocket builders.

  Returns:
    Tuple ``(root_fillet_mm, z_catch_peak_mm, z_crest_end_mm, nose_x_mm, z_nose_mm)``.
  """
  beam_len = _validate_positive("design.beam_length_mm", design.beam_length_mm)
  h_tip = _validate_positive("design.tip_thickness_mm", design.tip_thickness_mm)
  undercut = _validate_positive("design.undercut_mm", design.undercut_mm)
  root_fillet = min(max(0.0, design.root_fillet_mm), FILLET_LENGTH_CLAMP_RATIO * beam_len)

  dz_ret = undercut / math.tan(math.radians(design.retention_angle_deg))
  z_catch_peak = beam_len + dz_ret
  z_crest_end = z_catch_peak + max(0.0, design.hook_tip_flat_mm)

  nose_x = min(h_tip, max(MIN_HOOK_NOSE_X_MM, HOOK_NOSE_TIP_RATIO * h_tip))
  dx_lead = (h_tip + undercut) - nose_x
  dz_lead = dx_lead / math.tan(math.radians(design.lead_angle_deg))
  z_nose = z_crest_end + dz_lead
  return root_fillet, z_catch_peak, z_crest_end, nose_x, z_nose


def make_cantilever_snap_hook(design: SnapFitDesign) -> Part:
  """Build a tapered cantilever snap-fit hook solid in documented local axes.

  Local Coordinate Frame:
    - ``+Z``: Beam length axis from root base (``z = 0.0``) to lead-in nose tip.
    - ``+X``: Beam thickness from flat backing wall (``x = 0.0``) toward latch barb
      protruding by ``design.undercut_mm`` beyond ``design.tip_thickness_mm``.
    - ``Y``: Width axis centered symmetrically on ``y = 0.0`` over
      ``[-beam_width_mm / 2, +beam_width_mm / 2]``.
  """
  h_root = _validate_positive("design.root_thickness_mm", design.root_thickness_mm)
  h_tip = _validate_positive("design.tip_thickness_mm", design.tip_thickness_mm)
  beam_len = _validate_positive("design.beam_length_mm", design.beam_length_mm)
  width = _validate_positive("design.beam_width_mm", design.beam_width_mm)
  undercut = _validate_positive("design.undercut_mm", design.undercut_mm)

  root_fillet, z_catch_peak, z_crest_end, nose_x, z_nose = _snap_hook_envelope_mm(design)
  x_barb = h_tip + undercut

  with BuildPart() as builder:
    with BuildSketch(Plane.XZ):
      with BuildLine():
        if root_fillet > MIN_GEOM_TOL_MM:
          Line((0.0, 0.0), (h_root + root_fillet, 0.0))
          CenterArc(
              (h_root + root_fillet, root_fillet),
              root_fillet,
              start_angle=-90.0,
              arc_size=-90.0,
          )
          Line((h_root, root_fillet), (h_tip, beam_len))
        else:
          Line((0.0, 0.0), (h_root, 0.0))
          Line((h_root, 0.0), (h_tip, beam_len))

        Line((h_tip, beam_len), (x_barb, z_catch_peak))
        if z_crest_end > z_catch_peak + MIN_GEOM_TOL_MM:
          Line((x_barb, z_catch_peak), (x_barb, z_crest_end))
        Line((x_barb, z_crest_end), (nose_x, z_nose))
        Line((nose_x, z_nose), (0.0, z_nose))
        Line((0.0, z_nose), (0.0, 0.0))
      make_face()
    extrude(amount=HALF_FACTOR * width, both=True)
  return builder.part


def make_snap_pocket_cutter(design: SnapFitDesign, clearance_mm: float) -> Part:
  """Build a mating snap-pocket Boolean cutter in the same local frame as the hook.

  Provides ``-X`` deflection relief behind the cantilever arm (``-(undercut + clearance)``),
  normal-offset passage clearance along the tapered arm, and a stepped catch recess past
  ``z = design.beam_length_mm - clearance_mm`` so the hook snaps into engagement without
  binding or overlapping the receiver wall.
  """
  if clearance_mm < 0.0:
    raise ValueError(f"clearance_mm must be >= 0, got {clearance_mm}")

  h_root = _validate_positive("design.root_thickness_mm", design.root_thickness_mm)
  h_tip = _validate_positive("design.tip_thickness_mm", design.tip_thickness_mm)
  beam_len = _validate_positive("design.beam_length_mm", design.beam_length_mm)
  width = _validate_positive("design.beam_width_mm", design.beam_width_mm)
  undercut = _validate_positive("design.undercut_mm", design.undercut_mm)

  root_fillet, _, _, _, z_nose = _snap_hook_envelope_mm(design)
  c = float(clearance_mm)
  x_back = -(undercut + c)
  z_bot = -c
  z_catch = max(z_bot + MIN_GEOM_TOL_MM, beam_len - c)
  z_top = z_nose + c

  taper_span_z = max(MIN_GEOM_TOL_MM, beam_len - root_fillet)
  taper_slope = abs(h_root - h_tip) / taper_span_z
  normal_x_pad = c * math.hypot(1.0, taper_slope)
  x_channel_root = h_root + normal_x_pad
  x_channel_tip = h_tip + (taper_slope * c) + normal_x_pad
  x_pocket_outer = h_tip + undercut + c

  pts: list[tuple[float, float]] = [
      (x_back, z_bot),
      (h_root + root_fillet + c, z_bot),
  ]
  if root_fillet > MIN_GEOM_TOL_MM:
    pts.append((x_channel_root, root_fillet))
  pts.extend([
      (x_channel_tip, z_catch),
      (x_pocket_outer, z_catch),
      (x_pocket_outer, z_top),
      (x_back, z_top),
  ])

  with BuildPart() as builder:
    with BuildSketch(Plane.XZ):
      with BuildLine():
        Polyline(*pts, close=True)
      make_face()
    extrude(amount=(HALF_FACTOR * width) + c, both=True)
  return builder.part


def make_alignment_pin(
    radius_mm: float,
    length_mm: float,
    chamfer_mm: float = 0.5,
) -> Part:
  """Build a cylindrical alignment dowel pin along ``+Z`` anchored at ``z = 0.0``.

  Adds a lead-in chamfer at ``z = length_mm`` for easy blind insertion into printed bores.
  """
  r = _validate_positive("radius_mm", radius_mm)
  length = _validate_positive("length_mm", length_mm)
  if chamfer_mm < 0.0:
    raise ValueError(f"chamfer_mm must be >= 0, got {chamfer_mm}")

  c = min(float(chamfer_mm), CHAMFER_CLAMP_RATIO * r, CHAMFER_CLAMP_RATIO * length)
  pts: list[tuple[float, float]] = [(0.0, 0.0), (r, 0.0)]
  if c > MIN_GEOM_TOL_MM:
    pts.append((r, length - c))
    pts.append((r - c, length))
  else:
    pts.append((r, length))
  pts.append((0.0, length))

  with BuildPart() as builder:
    with BuildSketch(Plane.XZ):
      with BuildLine():
        Polyline(*pts, close=True)
      make_face()
    revolve(axis=Axis.Z)
  return builder.part


def make_alignment_hole_cutter(
    radius_mm: float,
    depth_mm: float,
    clearance_mm: float,
) -> Part:
  """Build a cylindrical alignment bore cutter along ``+Z`` spanning ``z in [0, depth_mm]``."""
  r = _validate_positive("radius_mm", radius_mm)
  depth = _validate_positive("depth_mm", depth_mm)
  if clearance_mm < 0.0:
    raise ValueError(f"clearance_mm must be >= 0, got {clearance_mm}")

  with BuildPart() as builder:
    Cylinder(
        radius=r + float(clearance_mm),
        height=depth,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
  return builder.part


def make_teardrop_bore_cutter(
    radius_mm: float,
    length_mm: float,
    axis: str = "Y",
) -> Part:
  """Build a horizontal bore cutter with a 45-degree ``+Z`` roof crest for support-free printing.

  Centered on ``(0, 0, 0)`` and extruded symmetrically along ``axis`` (``"X"`` or ``"Y"``)
  over ``[-length_mm / 2, +length_mm / 2]``. The bottom 270 degrees form an exact circle
  of radius ``radius_mm`` while the top 90 degrees rise tangentially at 45 degrees to
  ``z = +radius_mm * sqrt(2)``.
  """
  r = _validate_positive("radius_mm", radius_mm)
  length = _validate_positive("length_mm", length_mm)
  axis_key = axis.strip().upper()
  if axis_key == "Y":
    sketch_plane = Plane.XZ
  elif axis_key == "X":
    sketch_plane = Plane.YZ
  else:
    raise ValueError(f"Teardrop bore axis must be 'X' or 'Y' (horizontal), got {axis!r}")

  shoulder = r / TEARDROP_SQRT2
  crest_z = r * TEARDROP_SQRT2
  with BuildPart() as builder:
    with BuildSketch(sketch_plane):
      with BuildLine():
        CenterArc(
            (0.0, 0.0),
            r,
            start_angle=TEARDROP_START_ANGLE_DEG,
            arc_size=TEARDROP_ARC_SPAN_DEG,
        )
        Line((shoulder, shoulder), (0.0, crest_z))
        Line((0.0, crest_z), (-shoulder, shoulder))
      make_face()
    extrude(amount=HALF_FACTOR * length, both=True)
  return builder.part


def _make_d_cylinder(radius_mm: float, length_mm: float, flat_depth_mm: float) -> Part:
  """Shared D-profile extrusion along ``+Z`` over ``[0, length_mm]`` with flat on ``+X``."""
  r = _validate_positive("radius_mm", radius_mm)
  length = _validate_positive("length_mm", length_mm)
  flat_d = _validate_positive("flat_depth_mm", flat_depth_mm)
  if flat_d >= r:
    raise ValueError(f"flat_depth_mm ({flat_d}) must be < radius_mm ({r})")

  x_flat = r - flat_d
  y_flat = math.sqrt(max(1.0e-12, (r * r) - (x_flat * x_flat)))
  ang_deg = math.degrees(math.atan2(y_flat, x_flat))

  with BuildPart() as builder:
    with BuildSketch(Plane.XY):
      with BuildLine():
        CenterArc(
            (0.0, 0.0),
            r,
            start_angle=ang_deg,
            arc_size=FULL_CIRCLE_DEG - (2.0 * ang_deg),
        )
        Line((x_flat, -y_flat), (x_flat, y_flat))
      make_face()
    extrude(amount=length)
  return builder.part


def make_keyed_shaft_solid(
    radius_mm: float,
    length_mm: float,
    flat_depth_mm: float,
) -> Part:
  """Build a D-flat keyed shaft along ``+Z`` spanning ``z in [0, length_mm]``.

  The key flat is cut on the ``+X`` side at ``x = radius_mm - flat_depth_mm``.
  """
  return _make_d_cylinder(radius_mm, length_mm, flat_depth_mm)


def make_keyed_bore_cutter(
    radius_mm: float,
    length_mm: float,
    flat_depth_mm: float,
    clearance_mm: float = DEFAULT_KEYED_CLEARANCE_MM,
) -> Part:
  """Build a mating D-flat bore cutter with uniform slip clearance on arc and flat.

  Using outer radius ``radius_mm + clearance_mm`` with the same ``flat_depth_mm`` places
  the bore flat at ``x = (radius_mm - flat_depth_mm) + clearance_mm``, maintaining uniform
  ``clearance_mm`` gap on both the cylindrical surface and the key flat.
  """
  if clearance_mm < 0.0:
    raise ValueError(f"clearance_mm must be >= 0, got {clearance_mm}")
  _ = _validate_positive("radius_mm", radius_mm)
  _ = _validate_positive("flat_depth_mm", flat_depth_mm)
  if flat_depth_mm >= radius_mm:
    raise ValueError(
        f"flat_depth_mm ({flat_depth_mm}) must be < nominal shaft radius_mm ({radius_mm})"
    )
  return _make_d_cylinder(radius_mm + float(clearance_mm), length_mm, flat_depth_mm)


def _window_z_bounds(
    theta: float,
    th0: float,
    th1: float,
    th2: float,
    th3: float,
    z_mid: float,
    slot_h: float,
) -> tuple[float, float]:
  """Evaluate lower/upper Z bounds for a single radial window cutter at angle ``theta``."""
  half_h = HALF_FACTOR * slot_h
  if theta <= th1:
    u = (theta - th0) / max(1.0e-6, th1 - th0)
    return z_mid - (half_h * u), z_mid + (half_h * u)
  if theta >= th2:
    u = (th3 - theta) / max(1.0e-6, th3 - th2)
    return z_mid - (half_h * u), z_mid + (half_h * u)
  return z_mid - half_h, z_mid + half_h


def _make_radial_wedge_cutter(
    r_mid_cut: float,
    dr_cut: float,
    th_a: float,
    z_lo_a: float,
    z_hi_a: float,
    th_b: float,
    z_lo_b: float,
    z_hi_b: float,
) -> Part:
  """Create a planar-faced ruled wedge solid spanning radial sector ``[th_a, th_b]``."""
  h_a = max(MIN_WEDGE_HEIGHT_MM, z_hi_a - z_lo_a)
  h_b = max(MIN_WEDGE_HEIGHT_MM, z_hi_b - z_lo_b)
  zc_a = HALF_FACTOR * (z_lo_a + z_hi_a)
  zc_b = HALF_FACTOR * (z_lo_b + z_hi_b)

  plane_a = Plane(
      origin=(r_mid_cut * math.cos(th_a), r_mid_cut * math.sin(th_a), zc_a),
      x_dir=(math.cos(th_a), math.sin(th_a), 0.0),
      z_dir=(-math.sin(th_a), math.cos(th_a), 0.0),
  )
  plane_b = Plane(
      origin=(r_mid_cut * math.cos(th_b), r_mid_cut * math.sin(th_b), zc_b),
      x_dir=(math.cos(th_b), math.sin(th_b), 0.0),
      z_dir=(-math.sin(th_b), math.cos(th_b), 0.0),
  )
  with BuildPart() as wedge:
    with BuildSketch(plane_a) as sk_a:
      Rectangle(dr_cut, h_a)
    with BuildSketch(plane_b) as sk_b:
      Rectangle(dr_cut, h_b)
    loft([sk_a.sketch.faces()[0], sk_b.sketch.faces()[0]], ruled=True)
  return wedge.part


def make_compliant_wave_spring(
    outer_radius_mm: float,
    inner_radius_mm: float,
    free_height_mm: float,
    wire_radial_mm: float,
    waves: int = 3,
    turns: int = 2,
) -> Part:
  """Build a flat-seated, support-printable serpentine axial wave flexure spring.

  Constructed as a monolithic annular flexure cylinder spanning ``z in [0, free_height_mm]``
  with flat bottom/top seating rings and ``turns + 1`` staggered rows of ``waves`` 45-deg
  shouldered axial relief windows. Centered on ``(0, 0)`` with coaxial bore along ``+Z``.
  """
  r_out = _validate_positive("outer_radius_mm", outer_radius_mm)
  r_in = _validate_positive("inner_radius_mm", inner_radius_mm)
  height = _validate_positive("free_height_mm", free_height_mm)
  w_rad = _validate_positive("wire_radial_mm", wire_radial_mm)
  if r_in >= r_out:
    raise ValueError(f"inner_radius_mm ({r_in}) must be < outer_radius_mm ({r_out})")
  if waves < 2:
    raise ValueError(f"waves must be >= 2, got {waves}")
  if turns < 1:
    raise ValueError(f"turns must be >= 1, got {turns}")

  r_mid = HALF_FACTOR * (r_in + r_out)
  r_band = min(w_rad, r_out - r_in)
  r0 = r_mid - (HALF_FACTOR * r_band)
  r1 = r_mid + (HALF_FACTOR * r_band)

  n_rows = turns + 1
  beam_t = max(
      TARGET_MIN_SPRING_BEAM_T_MM,
      min(
          TARGET_MAX_SPRING_BEAM_T_MM,
          height / ((SPRING_BEAM_HEIGHT_DIV_PER_ROW * n_rows) + SPRING_BEAM_HEIGHT_DIV_BASE),
      ),
  )
  slot_h = (height - ((n_rows + 1) * beam_t)) / n_rows
  if slot_h < MIN_SPRING_SLOT_H_MM:
    beam_t = max(MIN_SPRING_BEAM_T_MM, (height - (n_rows * MIN_SPRING_SLOT_H_MM)) / (n_rows + 1))
    slot_h = (height - ((n_rows + 1) * beam_t)) / n_rows
  if slot_h <= MIN_WEDGE_HEIGHT_MM:
    raise ValueError(
        f"free_height_mm ({height}) too short for {turns} turns of printable flexure slots"
    )

  with BuildPart() as outer_cyl:
    Cylinder(radius=r1, height=height, align=(Align.CENTER, Align.CENTER, Align.MIN))
  with BuildPart() as inner_cyl:
    Cylinder(radius=r0, height=height + (2.0 * RADIAL_CUT_MARGIN_MM), align=(Align.CENTER, Align.CENTER, Align.CENTER))
  shell = outer_cyl.part.cut(inner_cyl.part.translate((0.0, 0.0, HALF_FACTOR * height)))

  period = (2.0 * math.pi) / waves
  pillar_w_mm = max(MIN_PILLAR_WIDTH_MM, beam_t * PILLAR_WIDTH_BEAM_MULT)
  pillar_ang = min(MAX_PILLAR_PERIOD_RATIO * period, pillar_w_mm / r_mid)
  win_ang = period - pillar_ang
  roof_ang = min(
      MAX_ROOF_WIN_RATIO * win_ang,
      max(MIN_ROOF_WIN_RATIO * win_ang, (HALF_FACTOR * slot_h) / r_mid),
  )

  r_cut_in = max(0.05, r0 - RADIAL_CUT_MARGIN_MM)
  r_cut_out = (r1 + RADIAL_CUT_MARGIN_MM) / math.cos(math.radians(RADIAL_CUT_CHORD_DEG))
  r_cut_mid = HALF_FACTOR * (r_cut_in + r_cut_out)
  dr_cut = r_cut_out - r_cut_in

  max_step_rad = math.radians(MAX_WEDGE_STEP_DEG)
  cut_solids: list[Part] = []

  for row in range(n_rows):
    z_mid = (beam_t * (row + 1)) + (slot_h * (row + HALF_FACTOR))
    phase = (row % 2) * (HALF_FACTOR * period)
    for w_idx in range(waves):
      th_c = (w_idx * period) + phase
      th0 = th_c - (HALF_FACTOR * win_ang)
      th1 = th_c - ((HALF_FACTOR * win_ang) - roof_ang)
      th2 = th_c + ((HALF_FACTOR * win_ang) - roof_ang)
      th3 = th_c + (HALF_FACTOR * win_ang)

      pts_th = [th0, th1]
      n_mid = max(2, int(math.ceil((th2 - th1) / max_step_rad)))
      for seg_idx in range(1, n_mid):
        pts_th.append(th1 + ((th2 - th1) * seg_idx / n_mid))
      pts_th.extend([th2, th3])

      for seg_idx in range(len(pts_th) - 1):
        ta, tb = pts_th[seg_idx], pts_th[seg_idx + 1]
        la, ha = _window_z_bounds(ta, th0, th1, th2, th3, z_mid, slot_h)
        lb, hb = _window_z_bounds(tb, th0, th1, th2, th3, z_mid, slot_h)
        cut_solids.append(
            _make_radial_wedge_cutter(r_cut_mid, dr_cut, ta, la, ha, tb, lb, hb)
        )

  return _as_part(shell.cut(*cut_solids))
