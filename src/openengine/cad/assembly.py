"""Master Parametric Engine Assembly, Kinematic Mating, and Collision Verification via build123d.

Combines all engine B-Rep components into a fully constrained, parameterized master assembly:
- Forged 4340 crankshaft rotated by crank angle theta
- I-beam connecting rod tilted by obliquity angle beta(theta) and mated to the crankpin
- Slipper piston translated along cylinder bore axis Z to exact slider-crank position
- Cylindrical wrist pin mated inside the piston and small-end eye
- SOHC pent-roof cylinder head fixed at the deck datum (Z = 0)
- Cast iron liner and finned aluminum cylinder block fixed at Z <= 0
- Intake and exhaust poppet valves displaced according to the 4-5-6-7 polynomial cam profile
- Automated 720-degree kinematic sweep verifying valve-to-piston and rod-to-liner clearances
- Export to master multi-body STEP AP214 file
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from build123d import (
    Axis,
    BuildPart,
    Compound,
    Cylinder,
    Locations,
    Part,
    Vector,
)

from openengine.cad.connecting_rod import DEFAULT_CENTER_TO_CENTER_M, generate_connecting_rod
from openengine.cad.cots_parts import generate_wrist_pin_16x45
from openengine.cad.crankshaft import CrankshaftParams, DEFAULT_STROKE_M, generate_crankshaft
from openengine.cad.cylinder_block import (
    CylinderBlockParams,
    DEFAULT_BORE_M,
    generate_cylinder_block,
    generate_cylinder_liner,
)
from openengine.cad.cylinder_head import (
    CylinderHeadParams,
    DEFAULT_VALVE_HEAD_EX_M,
    DEFAULT_VALVE_HEAD_IN_M,
    generate_cylinder_head,
    generate_poppet_valve,
)
from openengine.cad.piston import generate_piston
from openengine.cad.step_export import style_named_parts_as_compound

MM_PER_M: float = 1000.0
HALF_FACTOR: float = 0.5
DEFAULT_CON_ROD_LENGTH_M: float = DEFAULT_CENTER_TO_CENTER_M
DEFAULT_COMPRESSION_HEIGHT_M: float = 0.024
VALVE_SEAT_Z_MM: float = 6.0
VALVE_TILT_DEG: float = 15.0
VALVE_X_OFFSET_MM: float = 16.0
INTAKE_VALVE_DIAM_M: float = DEFAULT_VALVE_HEAD_IN_M
EXHAUST_VALVE_DIAM_M: float = DEFAULT_VALVE_HEAD_EX_M
LINER_BOTTOM_Z_MM: float = -105.0
ROD_FLANGE_HALF_W_MM: float = 8.5
DEFAULT_ROD_CLEARANCE_MM: float = 20.0
TARGET_PV_CLEARANCE_MM: float = 1.50
TARGET_ROD_CLEARANCE_MM: float = 5.00
PISTON_PIN_OFFSET_Z_MM: float = 3.0
ROT_90_DEG: float = 90.0
FULL_CYCLE_DEG: float = 720.0
DEFAULT_INTERFERENCE_STEP_DEG: float = 15.0
DEFAULT_INTERFERENCE_VOL_TOL_MM3: float = 1.0e-3
DEFAULT_SWEEP_STEP_DEG: float = 2.0
SWEEP_INTERFERENCE_STEP_DEG: float = 30.0
INIT_MIN_CLEARANCE_MM: float = 999.0

MASTER_ASSEMBLY_MATERIALS: Dict[str, str] = {
    "cylinder_head": "ALUMINUM_6061_T6",
    "cylinder_block": "ALUMINUM_6061_T6",
    "cylinder_liner": "DOM_STEEL_A513_T5",
    "crankshaft": "STEEL_4340",
    "connecting_rod": "ALUMINUM_7075_T6",
    "piston": "ALUMINUM_4032_T6",
    "wrist_pin": "STEEL_4140_PREHARD",
    "valve_intake": "STEEL_4140_PREHARD",
    "valve_exhaust": "STEEL_4140_PREHARD",
}


@dataclass
class EngineAssemblyState:
  """Instantaneous kinematic configuration and spatial coordinates at crank angle theta."""

  crank_angle_deg: float
  crankpin_x_mm: float
  crankpin_z_mm: float
  con_rod_angle_deg: float
  wrist_pin_z_mm: float
  piston_crown_z_mm: float
  intake_valve_lift_mm: float
  exhaust_valve_lift_mm: float
  intake_pv_clearance_mm: float
  exhaust_pv_clearance_mm: float
  rod_to_liner_clearance_mm: float
  is_safe: bool


@dataclass
class ClearanceReport:
  """Summary of minimum mechanical clearances evaluated across the full 720-degree cycle."""

  min_intake_pv_clearance_mm: float
  min_intake_pv_angle_deg: float
  min_exhaust_pv_clearance_mm: float
  min_exhaust_pv_angle_deg: float
  min_rod_to_liner_clearance_mm: float
  min_rod_to_liner_angle_deg: float
  target_pv_clearance_mm: float = 1.50
  target_rod_clearance_mm: float = 5.00
  interference_hits: int = 0
  is_clearance_safe: bool = True


class EngineAssembly:
  """Parametric 4-stroke engine assembly builder and kinematic verification engine."""

  def __init__(
      self,
      stroke_m: float = DEFAULT_STROKE_M,
      con_rod_length_m: float = DEFAULT_CON_ROD_LENGTH_M,
      compression_height_m: float = DEFAULT_COMPRESSION_HEIGHT_M,
      bore_m: float = DEFAULT_BORE_M,
  ) -> None:
    self.stroke_mm = stroke_m * MM_PER_M
    self.r_mm = self.stroke_mm * HALF_FACTOR  # 27 mm
    self.l_rod_mm = con_rod_length_m * MM_PER_M  # 108 mm
    self.comp_h_mm = compression_height_m * MM_PER_M  # 24 mm
    self.bore_r_mm = (bore_m * MM_PER_M) * HALF_FACTOR  # 34 mm
    self.lam = self.r_mm / self.l_rod_mm  # 0.25

    # Main crankshaft axis located at Z = -159 mm so piston crown hits Z = 0 at TDC
    self.z_crank_main_mm = -(self.comp_h_mm + self.r_mm + self.l_rod_mm)  # -159 mm

    from openengine.dynamics.sohc_cam import SOHCCamshaft

    self.cam = SOHCCamshaft()

  def solve_kinematics(self, crank_angle_deg: float) -> EngineAssemblyState:
    """Solve analytical kinematics and instantaneous component clearances."""
    th = np.radians(crank_angle_deg)

    # Crankpin position in XZ plane
    x_pin = self.r_mm * np.sin(th)
    z_pin = self.z_crank_main_mm + self.r_mm * np.cos(th)

    # Connecting rod obliquity angle beta: sin(beta) = lambda * sin(theta)
    sin_beta = self.lam * np.sin(th)
    beta = np.arcsin(sin_beta)
    beta_deg = float(np.degrees(beta))

    # Wrist pin center
    z_wrist = z_pin + self.l_rod_mm * np.cos(beta)
    z_crown = z_wrist + self.comp_h_mm

    # Valve lifts from 4-5-6-7 polynomial cam profile
    th_arr = np.array([crank_angle_deg])
    y_in_mm = float(self.cam.evaluate_valve_lift_crank_angle(self.cam.intake_params, th_arr)[0] * MM_PER_M)
    y_ex_mm = float(self.cam.evaluate_valve_lift_crank_angle(self.cam.exhaust_params, th_arr)[0] * MM_PER_M)

    # P-V Clearances: valve seats sit at Z = 6.0 mm, tilted at 15 deg
    # When open by lift y, valve head bottom face is at Z = 6.0 - y * cos(15)
    z_in_valve_bottom = VALVE_SEAT_Z_MM - y_in_mm * np.cos(np.radians(VALVE_TILT_DEG))
    z_ex_valve_bottom = VALVE_SEAT_Z_MM - y_ex_mm * np.cos(np.radians(VALVE_TILT_DEG))

    clr_in = z_in_valve_bottom - z_crown
    clr_ex = z_ex_valve_bottom - z_crown

    # Rod-to-liner clearance at bottom of 105 mm DOM cylinder liner (Z = -105 mm)
    if z_pin < LINER_BOTTOM_Z_MM < z_wrist:
      frac = (LINER_BOTTOM_Z_MM - z_wrist) / (z_pin - z_wrist)
      x_rod = frac * x_pin
      clr_rod = self.bore_r_mm - (abs(x_rod) + ROD_FLANGE_HALF_W_MM)  # 8.5 mm half-width of 17 mm rod flange
    else:
      clr_rod = DEFAULT_ROD_CLEARANCE_MM

    is_safe = (
        (clr_in >= TARGET_PV_CLEARANCE_MM)
        and (clr_ex >= TARGET_PV_CLEARANCE_MM)
        and (clr_rod >= TARGET_ROD_CLEARANCE_MM)
    )

    return EngineAssemblyState(
        crank_angle_deg=crank_angle_deg,
        crankpin_x_mm=float(x_pin),
        crankpin_z_mm=float(z_pin),
        con_rod_angle_deg=beta_deg,
        wrist_pin_z_mm=float(z_wrist),
        piston_crown_z_mm=float(z_crown),
        intake_valve_lift_mm=y_in_mm,
        exhaust_valve_lift_mm=y_ex_mm,
        intake_pv_clearance_mm=float(clr_in),
        exhaust_pv_clearance_mm=float(clr_ex),
        rod_to_liner_clearance_mm=float(clr_rod),
        is_safe=is_safe,
    )

  def _base_moving_solids(self) -> Dict[str, Part]:
    """Generate canonical CAD solids once for reuse across assembly poses and interference sweeps."""
    return {
        "crank": generate_crankshaft(),
        "rod": generate_connecting_rod().rotate(Axis.Z, ROT_90_DEG).rotate(Axis.Y, -ROT_90_DEG),
        "piston": generate_piston(),
        "wrist_pin": generate_wrist_pin_16x45(),
        "in_valve": generate_poppet_valve(head_diam_m=INTAKE_VALVE_DIAM_M),
        "ex_valve": generate_poppet_valve(head_diam_m=EXHAUST_VALVE_DIAM_M),
    }

  def _pose_moving_parts(
      self,
      crank_angle_deg: float,
      base: Dict[str, Part],
  ) -> Tuple[EngineAssemblyState, Dict[str, Part]]:
    """Transform moving solids to their instantaneous kinematic pose at crank_angle_deg."""
    st = self.solve_kinematics(crank_angle_deg)
    crank = base["crank"].rotate(Axis.Y, crank_angle_deg).translate((0.0, 0.0, self.z_crank_main_mm))
    rod = base["rod"].rotate(Axis.Y, -st.con_rod_angle_deg).translate((0.0, 0.0, st.wrist_pin_z_mm))
    # Piston pin bore is at local Z = -3.0 mm; translate to align with z_wrist
    piston = base["piston"].translate((0.0, 0.0, st.wrist_pin_z_mm + PISTON_PIN_OFFSET_Z_MM))
    wrist_pin = base["wrist_pin"].translate((0.0, 0.0, st.wrist_pin_z_mm))

    d_in_x = -st.intake_valve_lift_mm * np.sin(np.radians(VALVE_TILT_DEG))
    d_in_z = -st.intake_valve_lift_mm * np.cos(np.radians(VALVE_TILT_DEG))
    in_valve = base["in_valve"].rotate(Axis.Y, VALVE_TILT_DEG).translate(
        (VALVE_X_OFFSET_MM + d_in_x, 0.0, VALVE_SEAT_Z_MM + d_in_z)
    )

    d_ex_x = st.exhaust_valve_lift_mm * np.sin(np.radians(VALVE_TILT_DEG))
    d_ex_z = -st.exhaust_valve_lift_mm * np.cos(np.radians(VALVE_TILT_DEG))
    ex_valve = base["ex_valve"].rotate(Axis.Y, -VALVE_TILT_DEG).translate(
        (-VALVE_X_OFFSET_MM + d_ex_x, 0.0, VALVE_SEAT_Z_MM + d_ex_z)
    )

    return st, {
        "crank": crank,
        "rod": rod,
        "piston": piston,
        "wrist_pin": wrist_pin,
        "in_valve": in_valve,
        "ex_valve": ex_valve,
    }

  def check_3d_interference(
      self,
      step_deg: float = DEFAULT_INTERFERENCE_STEP_DEG,
      vol_tol_mm3: float = DEFAULT_INTERFERENCE_VOL_TOL_MM3,
  ) -> List[str]:
    """Verify zero 3D B-Rep overlap across moving-component pairs over the 720-degree cycle."""
    base = self._base_moving_solids()
    liner = generate_cylinder_liner()
    hits: List[str] = []

    if len(base["crank"].solids()) != 1:
      hits.append(f"crankshaft has {len(base['crank'].solids())} disconnected solids (expected 1)")

    pair_names = [
        ("crank", "rod"),
        ("crank", "piston"),
        ("crank", "wrist_pin"),
        ("crank", "liner"),
        ("rod", "piston"),
        ("rod", "wrist_pin"),
        ("rod", "liner"),
        ("piston", "wrist_pin"),
        ("piston", "liner"),
        ("piston", "in_valve"),
        ("piston", "ex_valve"),
    ]

    for angle_deg in np.arange(0.0, FULL_CYCLE_DEG, step_deg):
      _, posed = self._pose_moving_parts(float(angle_deg), base)
      parts = {**posed, "liner": liner}
      for name_a, name_b in pair_names:
        inter = parts[name_a].intersect(parts[name_b])
        vol = float(sum(s.volume for s in inter)) if inter else 0.0
        if vol > vol_tol_mm3:
          hits.append(f"CA={angle_deg:.1f} deg {name_a}-{name_b}: {vol:.2f} mm^3")

    return hits

  def sweep_clearances(
      self,
      step_deg: float = DEFAULT_SWEEP_STEP_DEG,
      interference_step_deg: float = SWEEP_INTERFERENCE_STEP_DEG,
  ) -> ClearanceReport:
    """Evaluate analytical clearances and 3D B-Rep interference across the full 720-degree cycle."""
    angles = np.arange(0.0, FULL_CYCLE_DEG, step_deg)

    min_in_clr = INIT_MIN_CLEARANCE_MM
    min_in_ang = 0.0
    min_ex_clr = INIT_MIN_CLEARANCE_MM
    min_ex_ang = 0.0
    min_rod_clr = INIT_MIN_CLEARANCE_MM
    min_rod_ang = 0.0

    for a in angles:
      st = self.solve_kinematics(a)
      if st.intake_pv_clearance_mm < min_in_clr:
        min_in_clr = st.intake_pv_clearance_mm
        min_in_ang = a
      if st.exhaust_pv_clearance_mm < min_ex_clr:
        min_ex_clr = st.exhaust_pv_clearance_mm
        min_ex_ang = a
      if st.rod_to_liner_clearance_mm < min_rod_clr:
        min_rod_clr = st.rod_to_liner_clearance_mm
        min_rod_ang = a

    interference_hits = self.check_3d_interference(step_deg=interference_step_deg)
    is_safe = (
        (min_in_clr >= TARGET_PV_CLEARANCE_MM)
        and (min_ex_clr >= TARGET_PV_CLEARANCE_MM)
        and (min_rod_clr >= TARGET_ROD_CLEARANCE_MM)
        and (len(interference_hits) == 0)
    )

    return ClearanceReport(
        min_intake_pv_clearance_mm=float(min_in_clr),
        min_intake_pv_angle_deg=float(min_in_ang),
        min_exhaust_pv_clearance_mm=float(min_ex_clr),
        min_exhaust_pv_angle_deg=float(min_ex_ang),
        min_rod_to_liner_clearance_mm=float(min_rod_clr),
        min_rod_to_liner_angle_deg=float(min_rod_ang),
        target_pv_clearance_mm=TARGET_PV_CLEARANCE_MM,
        target_rod_clearance_mm=TARGET_ROD_CLEARANCE_MM,
        interference_hits=len(interference_hits),
        is_clearance_safe=is_safe,
    )

  def build_assembly(self, crank_angle_deg: float = 0.0) -> Compound:
    """Generate and mate all engine components into a master multi-body Compound at angle theta."""
    block = generate_cylinder_block()
    liner = generate_cylinder_liner()
    head = generate_cylinder_head()

    _, posed = self._pose_moving_parts(crank_angle_deg, self._base_moving_solids())

    ordered_parts: Dict[str, Part] = {
        "cylinder_head": head,
        "cylinder_block": block,
        "cylinder_liner": liner,
        "crankshaft": posed["crank"],
        "connecting_rod": posed["rod"],
        "piston": posed["piston"],
        "wrist_pin": posed["wrist_pin"],
        "valve_intake": posed["in_valve"],
        "valve_exhaust": posed["ex_valve"],
    }
    return style_named_parts_as_compound(
        "engine_assembly",
        ordered_parts,
        MASTER_ASSEMBLY_MATERIALS.__getitem__,
    )


def generate_engine_assembly(crank_angle_deg: float = 0.0) -> Compound:
  """Helper to construct the complete multi-body engine assembly at crank angle theta."""
  asm = EngineAssembly()
  return asm.build_assembly(crank_angle_deg)


def verify_assembly_clearances() -> ClearanceReport:
  """Helper to run the full 720-degree cycle clearance verification."""
  asm = EngineAssembly()
  return asm.sweep_clearances()
