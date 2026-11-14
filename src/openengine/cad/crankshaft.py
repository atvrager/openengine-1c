"""Parametric 3-Piece Press-Fit Billet Crankshaft Solid Model via build123d.

Generates a watertight B-Rep CAD model tailored for NestWorks C500 CNC machining
and wet-sump splash lubrication (1144 Stressproof steel webs + ground 4140 pins):
- Hollow ground 4140 crankpin (28.0 mm OD x 21.0 mm between webs, 15.0 mm bore)
- Dual 1144 profiled counterweight webs (<= 106 x 96 x 18 mm per web)
- Ground 4140 30.0 mm main journals mating with 6206-2Z bearings and Viton seals
- Right +Y timing sprocket (18T #25 chain at Y = +50.0 mm), seal land, and 20 mm PTO
- Left -Y flywheel and 36-1 trigger-wheel journal + M12 retention nose
"""

from dataclasses import dataclass
from typing import Dict, Optional
from build123d import (
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Rectangle,
    extrude,
    insert,
    make_hull,
)

MM_PER_M: float = 1000.0
CM3_PER_MM3: float = 1.0e-3
KG_PER_G: float = 1.0e-3
KG_TO_G_MM: float = 1.0e6
G_CM3_PER_KG_M3: float = 1.0e-3
FULL_ROTATION_Y_DEG: float = 90.0
HALF_FACTOR: float = 0.5

CRANKPIN_BOSS_MARGIN_MM: float = 4.0
MAIN_BOSS_MARGIN_MM: float = 4.0
CW_CUT_OFFSET_MM: float = -3.0
CW_CUT_HEIGHT_MM: float = 70.0
CW_CUT_WIDTH_MM: float = 120.0
CRANKPIN_PRESS_OVERLAP_MM: float = 2.0
MAIN_PRESS_OVERLAP_MM: float = 4.0
MIN_RECIP_BALANCE_FACTOR: float = 0.40
MAX_RECIP_BALANCE_FACTOR: float = 0.60
TARGET_BALANCE_FACTOR: float = 0.50

DEFAULT_STROKE_M: float = 0.054
DEFAULT_CRANKPIN_DIAM_M: float = 0.028
DEFAULT_MAIN_JOURNAL_DIAM_M: float = 0.030
DEFAULT_COUNTERWEIGHT_RADIUS_M: float = 0.0478
DEFAULT_WEB_THICKNESS_M: float = 0.0145
CRANK_SPROCKET_TEETH: int = 18


@dataclass
class CrankshaftParams:
  """Geometric and material design parameters for the 3-piece billet crankshaft."""

  stroke_m: float = DEFAULT_STROKE_M
  crankpin_diameter_m: float = DEFAULT_CRANKPIN_DIAM_M
  crankpin_width_m: float = 0.021
  crankpin_bore_diameter_m: float = 0.015
  main_journal_diameter_m: float = DEFAULT_MAIN_JOURNAL_DIAM_M
  left_main_length_m: float = 0.028
  right_main_length_m: float = 0.018
  web_thickness_m: float = DEFAULT_WEB_THICKNESS_M
  counterweight_radius_m: float = DEFAULT_COUNTERWEIGHT_RADIUS_M
  timing_journal_diameter_m: float = 0.020
  timing_journal_length_m: float = 0.014
  right_seal_collar_diam_m: float = 0.030
  right_seal_collar_length_m: float = 0.009
  pto_shaft_diameter_m: float = 0.020
  pto_shaft_length_m: float = 0.030
  flywheel_hub_diameter_m: float = 0.020
  flywheel_hub_length_m: float = 0.034
  nose_thread_diameter_m: float = 0.012
  nose_thread_length_m: float = 0.010
  steel_density_kg_m3: float = 7850.0


@dataclass
class CrankshaftParts:
  """Individual press-fit billet pieces of the 3-piece crankshaft."""

  left_web: Part
  right_web: Part
  crankpin: Part
  left_shaft: Part
  right_shaft: Part


def _build_web_sketch(p: CrankshaftParams) -> BuildSketch:
  """Construct the 1144 Steel counterweight web profile in the XZ plane."""
  r_throw = (p.stroke_m * MM_PER_M) * HALF_FACTOR
  d_pin = p.crankpin_diameter_m * MM_PER_M
  d_main = p.main_journal_diameter_m * MM_PER_M
  r_max = p.counterweight_radius_m * MM_PER_M
  cw_rect_y = CW_CUT_OFFSET_MM + (CW_CUT_HEIGHT_MM * HALF_FACTOR)

  with BuildSketch(Plane.XZ) as cheek_sketch:
    with BuildSketch(mode=Mode.PRIVATE) as arm_sk:
      with Locations((0.0, r_throw)):
        Circle(radius=(d_pin * HALF_FACTOR) + CRANKPIN_BOSS_MARGIN_MM)
      with Locations((0.0, 0.0)):
        Circle(radius=(d_main * HALF_FACTOR) + MAIN_BOSS_MARGIN_MM)
      make_hull()
    insert(arm_sk.sketch)
    with BuildSketch(mode=Mode.PRIVATE) as cw_sk:
      Circle(radius=r_max)
      with Locations((0.0, cw_rect_y)):
        Rectangle(CW_CUT_WIDTH_MM, CW_CUT_HEIGHT_MM, mode=Mode.SUBTRACT)
    insert(cw_sk.sketch)
  return cheek_sketch


def _extrude_web(cheek_sketch: BuildSketch, p: CrankshaftParams, side: int) -> Part:
  """Extrude one 1144 crank web with press-fit pin and main bores."""
  r_throw = (p.stroke_m * MM_PER_M) * HALF_FACTOR
  w_pin = p.crankpin_width_m * MM_PER_M
  t_web = p.web_thickness_m * MM_PER_M
  r_pin = (p.crankpin_diameter_m * MM_PER_M) * HALF_FACTOR
  r_main = (p.main_journal_diameter_m * MM_PER_M) * HALF_FACTOR
  y_start = (w_pin * HALF_FACTOR) * float(side)
  y_mid = y_start + (t_web * HALF_FACTOR) * float(side)

  with BuildPart() as web:
    extrude(
        cheek_sketch.sketch.translate((0.0, y_start, 0.0)),
        amount=t_web,
        dir=(0.0, float(side), 0.0),
    )
    with Locations((0.0, y_mid, r_throw)):
      Cylinder(
          radius=r_pin,
          height=t_web + CRANKPIN_PRESS_OVERLAP_MM,
          rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
          mode=Mode.SUBTRACT,
      )
    with Locations((0.0, y_mid, 0.0)):
      Cylinder(
          radius=r_main,
          height=t_web + CRANKPIN_PRESS_OVERLAP_MM,
          rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
          mode=Mode.SUBTRACT,
      )
  return web.part


def _build_left_shaft(p: CrankshaftParams) -> Part:
  """Construct the stepped ground 4140 left main, flywheel, and nose shaft."""
  w_pin = p.crankpin_width_m * MM_PER_M
  t_web = p.web_thickness_m * MM_PER_M
  y_web_inner = -(w_pin * HALF_FACTOR)
  r_main = (p.main_journal_diameter_m * MM_PER_M) * HALF_FACTOR
  l_left_main = p.left_main_length_m * MM_PER_M
  l_journal_press = t_web + l_left_main
  y_main_center = y_web_inner - (l_journal_press * HALF_FACTOR)

  y_left_edge = y_web_inner - l_journal_press
  r_fw = (p.flywheel_hub_diameter_m * MM_PER_M) * HALF_FACTOR
  l_fw = p.flywheel_hub_length_m * MM_PER_M
  r_nose = (p.nose_thread_diameter_m * MM_PER_M) * HALF_FACTOR
  l_nose = p.nose_thread_length_m * MM_PER_M

  with BuildPart() as left_shaft:
    with Locations((0.0, y_main_center, 0.0)):
      Cylinder(
          radius=r_main,
          height=l_journal_press,
          rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
      )
    with Locations((0.0, y_left_edge - (l_fw * HALF_FACTOR), 0.0)):
      Cylinder(
          radius=r_fw,
          height=l_fw,
          rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
      )
    with Locations((0.0, y_left_edge - l_fw - (l_nose * HALF_FACTOR), 0.0)):
      Cylinder(
          radius=r_nose,
          height=l_nose,
          rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
      )
  return left_shaft.part


def _build_right_shaft(p: CrankshaftParams) -> Part:
  """Construct the stepped ground 4140 right main, timing sprocket, seal, and PTO shaft."""
  w_pin = p.crankpin_width_m * MM_PER_M
  t_web = p.web_thickness_m * MM_PER_M
  y_web_inner = w_pin * HALF_FACTOR
  r_main = (p.main_journal_diameter_m * MM_PER_M) * HALF_FACTOR
  l_right_main = p.right_main_length_m * MM_PER_M
  l_journal_press = t_web + l_right_main
  y_main_center = y_web_inner + (l_journal_press * HALF_FACTOR)

  y_cursor = y_web_inner + l_journal_press
  r_timing = (p.timing_journal_diameter_m * MM_PER_M) * HALF_FACTOR
  l_timing = p.timing_journal_length_m * MM_PER_M
  y_timing = y_cursor + (l_timing * HALF_FACTOR)

  y_cursor += l_timing
  r_collar = (p.right_seal_collar_diam_m * MM_PER_M) * HALF_FACTOR
  l_collar = p.right_seal_collar_length_m * MM_PER_M
  y_collar = y_cursor + (l_collar * HALF_FACTOR)

  y_cursor += l_collar
  r_pto = (p.pto_shaft_diameter_m * MM_PER_M) * HALF_FACTOR
  l_pto = p.pto_shaft_length_m * MM_PER_M
  y_pto = y_cursor + (l_pto * HALF_FACTOR)

  with BuildPart() as right_shaft:
    with Locations((0.0, y_main_center, 0.0)):
      Cylinder(radius=r_main, height=l_journal_press, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
    with Locations((0.0, y_timing, 0.0)):
      Cylinder(radius=r_timing, height=l_timing, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
    with Locations((0.0, y_collar, 0.0)):
      Cylinder(radius=r_collar, height=l_collar, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
    with Locations((0.0, y_pto, 0.0)):
      Cylinder(radius=r_pto, height=l_pto, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
  return right_shaft.part


def _build_crankpin(p: CrankshaftParams) -> Part:
  """Construct the hollow ground 4140 crankpin pressing into both crank webs."""
  r_throw = (p.stroke_m * MM_PER_M) * HALF_FACTOR
  w_pin = p.crankpin_width_m * MM_PER_M
  t_web = p.web_thickness_m * MM_PER_M
  total_len = w_pin + (2.0 * t_web)
  r_pin = (p.crankpin_diameter_m * MM_PER_M) * HALF_FACTOR
  r_bore = (p.crankpin_bore_diameter_m * MM_PER_M) * HALF_FACTOR

  with BuildPart() as pin:
    with Locations((0.0, 0.0, r_throw)):
      Cylinder(radius=r_pin, height=total_len, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
      if r_bore > 0.0:
        Cylinder(
            radius=r_bore,
            height=total_len + MAIN_PRESS_OVERLAP_MM,
            rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
            mode=Mode.SUBTRACT,
        )
  return pin.part


def _build_shaft_sections(p: CrankshaftParams) -> None:
  """Add left and right stepped shaft sections inside an active BuildPart."""
  w_pin = p.crankpin_width_m * MM_PER_M
  t_web = p.web_thickness_m * MM_PER_M
  r_main = (p.main_journal_diameter_m * MM_PER_M) * HALF_FACTOR

  # Left (-Y): main journal + flywheel/trigger hub + M12 retention nose
  y_left_edge = -(w_pin * HALF_FACTOR) - t_web
  l_left_main = p.left_main_length_m * MM_PER_M
  r_fw = (p.flywheel_hub_diameter_m * MM_PER_M) * HALF_FACTOR
  l_fw = p.flywheel_hub_length_m * MM_PER_M
  r_nose = (p.nose_thread_diameter_m * MM_PER_M) * HALF_FACTOR
  l_nose = p.nose_thread_length_m * MM_PER_M

  with Locations((0.0, y_left_edge - (l_left_main * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_main, height=l_left_main, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
  with Locations((0.0, y_left_edge - l_left_main - (l_fw * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_fw, height=l_fw, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
  with Locations((0.0, y_left_edge - l_left_main - l_fw - (l_nose * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_nose, height=l_nose, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))

  # Right (+Y): main journal + 18T timing journal + 30mm seal collar + 20mm PTO
  y_cursor = (w_pin * HALF_FACTOR) + t_web
  l_right_main = p.right_main_length_m * MM_PER_M
  with Locations((0.0, y_cursor + (l_right_main * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_main, height=l_right_main, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))

  y_cursor += l_right_main
  r_timing = (p.timing_journal_diameter_m * MM_PER_M) * HALF_FACTOR
  l_timing = p.timing_journal_length_m * MM_PER_M
  with Locations((0.0, y_cursor + (l_timing * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_timing, height=l_timing, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))

  y_cursor += l_timing
  r_collar = (p.right_seal_collar_diam_m * MM_PER_M) * HALF_FACTOR
  l_collar = p.right_seal_collar_length_m * MM_PER_M
  with Locations((0.0, y_cursor + (l_collar * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_collar, height=l_collar, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))

  y_cursor += l_collar
  r_pto = (p.pto_shaft_diameter_m * MM_PER_M) * HALF_FACTOR
  l_pto = p.pto_shaft_length_m * MM_PER_M
  with Locations((0.0, y_cursor + (l_pto * HALF_FACTOR), 0.0)):
    Cylinder(radius=r_pto, height=l_pto, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))


def generate_crankshaft_parts(
    params: Optional[CrankshaftParams] = None,
) -> CrankshaftParts:
  """Generate the five individual C500 billet pieces of the 3-piece crankshaft."""
  p = params or CrankshaftParams()
  cheek_sketch = _build_web_sketch(p)
  return CrankshaftParts(
      left_web=_extrude_web(cheek_sketch, p, side=-1),
      right_web=_extrude_web(cheek_sketch, p, side=1),
      crankpin=_build_crankpin(p),
      left_shaft=_build_left_shaft(p),
      right_shaft=_build_right_shaft(p),
  )


def generate_crankshaft(params: Optional[CrankshaftParams] = None) -> Part:
  """Generate the assembled watertight B-Rep 3-piece billet crankshaft."""
  p = params or CrankshaftParams()
  cheek_sketch = _build_web_sketch(p)
  r_throw = (p.stroke_m * MM_PER_M) * HALF_FACTOR
  d_pin = p.crankpin_diameter_m * MM_PER_M
  w_pin = p.crankpin_width_m * MM_PER_M
  t_web = p.web_thickness_m * MM_PER_M
  r_bore = (p.crankpin_bore_diameter_m * MM_PER_M) * HALF_FACTOR

  with BuildPart() as crank:
    with Locations((0.0, 0.0, r_throw)):
      Cylinder(radius=d_pin * HALF_FACTOR, height=w_pin, rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0))
    extrude(cheek_sketch.sketch.translate((0.0, w_pin * HALF_FACTOR, 0.0)), amount=t_web, dir=(0.0, 1.0, 0.0))
    extrude(cheek_sketch.sketch.translate((0.0, -w_pin * HALF_FACTOR, 0.0)), amount=t_web, dir=(0.0, -1.0, 0.0))
    if r_bore > 0.0:
      with Locations((0.0, 0.0, r_throw)):
        Cylinder(
            radius=r_bore,
            height=w_pin + (2.0 * t_web) + MAIN_PRESS_OVERLAP_MM,
            rotation=(FULL_ROTATION_Y_DEG, 0.0, 0.0),
            mode=Mode.SUBTRACT,
        )
    _build_shaft_sections(p)
  return crank.part


def calculate_crankshaft_balance(
    params: Optional[CrankshaftParams] = None,
    rotating_mass_kg: Optional[float] = None,
    reciprocating_mass_kg: Optional[float] = None,
) -> Dict[str, float]:
  """Evaluate static unbalance and effective reciprocating balance factor."""
  if rotating_mass_kg is None or reciprocating_mass_kg is None:
    from openengine.cad.twin_metrics import get_cad_rotating_and_reciprocating_masses_kg

    cad_rot_kg, cad_recip_kg = get_cad_rotating_and_reciprocating_masses_kg()
    if rotating_mass_kg is None:
      rotating_mass_kg = cad_rot_kg
    if reciprocating_mass_kg is None:
      reciprocating_mass_kg = cad_recip_kg

  p = params or CrankshaftParams()
  crank = generate_crankshaft(p)

  vol_cm3 = float(crank.volume) * CM3_PER_MM3
  mass_kg = vol_cm3 * (p.steel_density_kg_m3 * G_CM3_PER_KG_M3) * KG_PER_G
  cm = crank.center()

  r_m = p.stroke_m * HALF_FACTOR
  unbalance_kg_m = mass_kg * abs(float(cm.Z)) / MM_PER_M
  rot_unbalance = rotating_mass_kg * r_m
  recip_target_full = reciprocating_mass_kg * r_m
  k_bal_achieved = (unbalance_kg_m - rot_unbalance) / recip_target_full

  return {
      "crankshaft_volume_cm3": vol_cm3,
      "crankshaft_mass_kg": mass_kg,
      "center_of_mass_z_mm": float(cm.Z),
      "counterweight_unbalance_kg_m": unbalance_kg_m,
      "counterweight_unbalance_g_mm": unbalance_kg_m * KG_TO_G_MM,
      "target_unbalance_50pct_g_mm": (rot_unbalance + TARGET_BALANCE_FACTOR * recip_target_full) * KG_TO_G_MM,
      "effective_balance_factor": float(k_bal_achieved),
      "balance_factor_percent": float(k_bal_achieved * 100.0),
      "is_balanced": MIN_RECIP_BALANCE_FACTOR <= k_bal_achieved <= MAX_RECIP_BALANCE_FACTOR,
  }
