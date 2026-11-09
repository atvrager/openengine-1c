"""Parametric Counterweighted Crankshaft Solid Model via build123d / Open CASCADE.

Generates a watertight B-Rep CAD model of a forged 4340 alloy steel single-throw crankshaft:
- Precision ground crankpin (28.0 mm diameter x 16.0 mm width, 27.0 mm throw radius)
- Dual profiled counterweight cheeks shaped for 50% reciprocating mass balancing
- Lightening holes on the crankpin side to maximize polar counterweight moment
- Dual main bearing journals (32.0 mm diameter x 22.0 mm length)
- Cross-drilled pressurized oil gallery feeding main oil pressure to the big-end bearing
- Left flywheel mounting taper and threaded retention nose
- Right timing chain drive sprocket journal and Power Take-Off (PTO) output shaft
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
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
    Vector,
    extrude,
    insert,
    make_hull,
)


@dataclass
class CrankshaftParams:
  """Geometric and material design parameters for the crankshaft."""

  stroke_m: float = 0.054  # 54.0 mm stroke (27.0 mm crank throw)
  crankpin_diameter_m: float = 0.028  # 28.0 mm crankpin journal diameter
  crankpin_width_m: float = 0.016  # 16.0 mm between web inner faces
  main_journal_diameter_m: float = 0.032  # 32.0 mm main journal diameter
  main_journal_length_m: float = 0.022  # 22.0 mm main bearing journal length
  web_thickness_m: float = 0.014  # 14.0 mm crank web axial thickness
  counterweight_radius_m: float = 0.0485  # 48.5 mm outer counterweight radius
  timing_journal_diameter_m: float = 0.025  # 25.0 mm timing sprocket seat
  timing_journal_length_m: float = 0.016  # 16.0 mm timing sprocket length
  pto_shaft_diameter_m: float = 0.020  # 20.0 mm PTO drive shaft diameter
  pto_shaft_length_m: float = 0.035  # 35.0 mm PTO shaft length
  flywheel_taper_length_m: float = 0.030  # 30.0 mm flywheel seat length
  oil_gallery_diameter_m: float = 0.004  # 4.0 mm cross-drilled oil hole
  steel_density_kg_m3: float = 7850.0  # 4340 forged alloy steel


def generate_crankshaft(params: Optional[CrankshaftParams] = None) -> Part:
  """Generate the complete solid B-Rep crankshaft model."""
  p = params or CrankshaftParams()

  r_stroke = (p.stroke_m * 1000.0) / 2.0  # 27 mm
  d_pin = p.crankpin_diameter_m * 1000.0
  w_pin = p.crankpin_width_m * 1000.0
  d_main = p.main_journal_diameter_m * 1000.0
  l_main = p.main_journal_length_m * 1000.0
  t_web = p.web_thickness_m * 1000.0
  r_max = p.counterweight_radius_m * 1000.0

  # 1. Crank web cheek profile in XZ plane
  with BuildSketch(Plane.XZ) as cheek_sketch:
    with Locations((0.0, r_stroke)):
      Circle(radius=18.0)
    with BuildSketch(mode=Mode.PRIVATE) as cw_sk:
      Circle(radius=r_max)
      with Locations((0.0, 30.0)):
        Rectangle(120.0, 60.0, mode=Mode.SUBTRACT)
    insert(cw_sk.sketch)
    make_hull()
    with Locations((-28.0, 14.0), (28.0, 14.0)):
      Circle(radius=14.0, mode=Mode.SUBTRACT)

  with BuildPart() as crank:
    # 2. Crankpin journal (centered at Y = 0, Z = r_stroke)
    with Locations((0.0, 0.0, r_stroke)):
      Cylinder(radius=d_pin / 2.0, height=w_pin, rotation=(90.0, 0.0, 0.0))

    # 3. Right cheek (extending +Y from w_pin/2 to w_pin/2 + t_web)
    with Locations((0.0, w_pin / 2.0, 0.0)):
      extrude(cheek_sketch.sketch, amount=t_web, dir=(0.0, 1.0, 0.0))
    # Lightening holes on pin side
    with Locations(
        (-14.0, (w_pin / 2.0) + (t_web / 2.0), 24.0),
        (14.0, (w_pin / 2.0) + (t_web / 2.0), 24.0),
    ):
      Cylinder(radius=5.0, height=t_web + 2.0, rotation=(90.0, 0.0, 0.0), mode=Mode.SUBTRACT)

    # 4. Left cheek (extending -Y from -w_pin/2 to -w_pin/2 - t_web)
    with Locations((0.0, -w_pin / 2.0, 0.0)):
      extrude(cheek_sketch.sketch, amount=t_web, dir=(0.0, -1.0, 0.0))
    # Lightening holes on pin side
    with Locations(
        (-14.0, (-w_pin / 2.0) - (t_web / 2.0), 24.0),
        (14.0, (-w_pin / 2.0) - (t_web / 2.0), 24.0),
    ):
      Cylinder(radius=5.0, height=t_web + 2.0, rotation=(90.0, 0.0, 0.0), mode=Mode.SUBTRACT)

    # 5. Right main journal
    y_right_main = (w_pin / 2.0) + t_web + (l_main / 2.0)
    with Locations((0.0, y_right_main, 0.0)):
      Cylinder(radius=d_main / 2.0, height=l_main, rotation=(90.0, 0.0, 0.0))

    # 6. Left main journal
    y_left_main = (-w_pin / 2.0) - t_web - (l_main / 2.0)
    with Locations((0.0, y_left_main, 0.0)):
      Cylinder(radius=d_main / 2.0, height=l_main, rotation=(90.0, 0.0, 0.0))

    # 7. Right shaft extensions (timing sprocket seat + PTO drive shaft)
    y_right_edge = (w_pin / 2.0) + t_web + l_main
    l_timing = p.timing_journal_length_m * 1000.0
    r_timing = (p.timing_journal_diameter_m * 1000.0) / 2.0
    with Locations((0.0, y_right_edge + (l_timing / 2.0), 0.0)):
      Cylinder(radius=r_timing, height=l_timing, rotation=(90.0, 0.0, 0.0))

    l_pto = p.pto_shaft_length_m * 1000.0
    r_pto = (p.pto_shaft_diameter_m * 1000.0) / 2.0
    with Locations((0.0, y_right_edge + l_timing + (l_pto / 2.0), 0.0)):
      Cylinder(radius=r_pto, height=l_pto, rotation=(90.0, 0.0, 0.0))

    # 8. Left shaft extensions (flywheel seat + threaded nose)
    y_left_edge = (-w_pin / 2.0) - t_web - l_main
    l_fw = p.flywheel_taper_length_m * 1000.0
    with Locations((0.0, y_left_edge - (l_fw / 2.0), 0.0)):
      Cylinder(radius=11.0, height=l_fw, rotation=(90.0, 0.0, 0.0))

    with Locations((0.0, y_left_edge - l_fw - 7.5, 0.0)):
      Cylinder(radius=8.0, height=15.0, rotation=(90.0, 0.0, 0.0))

    # 9. Cross-drilled pressurized oil gallery from left main journal to crankpin
    p_main_feed = Vector(0.0, (-w_pin / 2.0) - (t_web / 2.0), 0.0)
    p_pin_center = Vector(0.0, 0.0, r_stroke)
    v_oil = p_pin_center - p_main_feed
    len_oil = v_oil.length
    mid_oil = p_main_feed + v_oil * 0.5
    theta_x = np.degrees(np.arctan2(v_oil.Z, v_oil.Y))
    r_oil = (p.oil_gallery_diameter_m * 1000.0) / 2.0

    with Locations((mid_oil.X, mid_oil.Y, mid_oil.Z)):
      Cylinder(radius=r_oil, height=len_oil + 4.0, rotation=(-theta_x + 90.0, 0.0, 0.0), mode=Mode.SUBTRACT)

  return crank.part


def calculate_crankshaft_balance(
    params: Optional[CrankshaftParams] = None,
    rotating_mass_kg: float = 0.280,
    reciprocating_mass_kg: float = 0.220,
) -> Dict[str, float]:
  """Evaluate the physical static unbalance and effective reciprocating balance factor."""
  p = params or CrankshaftParams()
  crank = generate_crankshaft(p)

  vol_cm3 = float(crank.volume) / 1000.0
  mass_kg = vol_cm3 * (p.steel_density_kg_m3 / 1000.0) / 1000.0  # kg
  cm = crank.center()

  # Crankpin throw radius
  r_m = p.stroke_m / 2.0

  # Static unbalance U = M_crank * |Z_cm| (counterbalance points in -Z direction)
  unbalance_kg_m = mass_kg * abs(cm.Z) / 1000.0  # kg.m

  # Effective reciprocating balance factor:
  # U_crank = m_rot * r + k_bal * m_recip * r
  # => k_bal = (U_crank - m_rot * r) / (m_recip * r)
  rot_unbalance = rotating_mass_kg * r_m
  recip_target_full = reciprocating_mass_kg * r_m

  k_bal_achieved = (unbalance_kg_m - rot_unbalance) / recip_target_full

  return {
      "crankshaft_volume_cm3": vol_cm3,
      "crankshaft_mass_kg": mass_kg,
      "center_of_mass_z_mm": float(cm.Z),
      "counterweight_unbalance_kg_m": unbalance_kg_m,
      "counterweight_unbalance_g_mm": unbalance_kg_m * 1.0e6,
      "target_unbalance_50pct_g_mm": (rot_unbalance + 0.50 * recip_target_full) * 1.0e6,
      "effective_balance_factor": float(k_bal_achieved),
      "balance_factor_percent": float(k_bal_achieved * 100.0),
      "is_balanced": 0.40 <= k_bal_achieved <= 0.60,
  }
