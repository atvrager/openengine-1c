"""SOHC Camshaft Profile Synthesis, Valvetrain Kinematics, and Valve Float Analysis.

Implements the 4-5-6-7 polynomial motion law for overhead camshafts, guaranteeing
zero velocity, zero acceleration, and zero jerk at cam lobe boundaries to prevent
valvetrain shock loading. Evaluates follower dynamics, valve float safety margins,
and Hertzian contact stress.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CamLobeParams:
  """Geometric and timing specifications for a single camshaft lobe."""

  name: str
  max_valve_lift_m: float  # Maximum lift at the valve, e.g. 0.0085 m (8.5 mm)
  rocker_ratio: float = 1.15  # Valve lift / cam lift
  duration_crank_deg: float = 220.0  # Duration in crank angle degrees
  open_crank_deg: float = 345.0  # Crank angle of valve opening (e.g. 15 deg BTDC)
  base_circle_radius_m: float = 0.016  # Base circle radius R_0 (16 mm)
  roller_radius_m: float = 0.009  # Roller follower radius (9 mm)
  lobe_width_m: float = 0.012  # Cam lobe contact face width (12 mm)


@dataclass
class ValvetrainMassSpring:
  """Mechanical parameters for valvetrain dynamics and spring sizing."""

  valve_mass_kg: float = 0.022  # Lightweight titanium valve (22 g)
  retainer_mass_kg: float = 0.005  # Titanium retainer + locks (5 g)
  spring_mass_kg: float = 0.021  # Beehive valve spring (21 g)
  rocker_inertia_equiv_kg: float = 0.008  # Aluminum roller rocker inertia at valve (8 g)
  seat_force_preload_n: float = 320.0  # Installed seat force at 0 lift (320 N)
  spring_rate_n_m: float = 62000.0  # Beehive spring stiffness k (62.0 N/mm)

  @property
  def equivalent_valve_mass_kg(self) -> float:
    """Effective reciprocating mass acting at the valve stem.

    Includes 1/3 of the spring mass (Rayleigh energy approximation).
    """
    return (
        self.valve_mass_kg
        + self.retainer_mass_kg
        + (1.0 / 3.0) * self.spring_mass_kg
        + self.rocker_inertia_equiv_kg
    )


@dataclass
class CamKinematics:
  """Instantaneous kinematic state of the valve and cam lobe."""

  cam_angle_deg: np.ndarray
  valve_lift_m: np.ndarray
  cam_lift_m: np.ndarray
  valve_velocity_m_s: np.ndarray
  valve_accel_m_s2: np.ndarray
  valve_jerk_m_s3: np.ndarray
  spring_force_n: np.ndarray
  inertia_force_n: np.ndarray
  follower_contact_force_n: np.ndarray
  hertz_stress_mpa: np.ndarray


@dataclass
class ValveFloatReport:
  """Summary of valve float margin and contact stresses."""

  cam_name: str
  rated_rpm: float
  redline_rpm: float
  test_rpm: float  # 1.2 x redline
  min_contact_force_at_rated_n: float
  min_contact_force_at_redline_n: float
  min_contact_force_at_test_n: float
  valve_float_rpm: float
  float_margin_percent: float
  peak_hertz_stress_mpa: float
  max_valve_acceleration_m_s2: float
  max_valve_deceleration_m_s2: float
  is_safe: bool


def polynomial_4567(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Normalized 4-5-6-7 polynomial rise motion law.

  Args:
    u: Normalized angle variable u in [0, 1].

  Returns:
    y_norm: Normalized lift (0 -> 1)
    dy_du: First derivative d(y_norm)/du
    d2y_du2: Second derivative d^2(y_norm)/du^2
    d3y_du3: Third derivative d^3(y_norm)/du^3
  """
  u_clamped = np.clip(u, 0.0, 1.0)
  # y = 35*u^4 - 84*u^5 + 70*u^6 - 20*u^7
  y = 35.0 * (u_clamped**4) - 84.0 * (u_clamped**5) + 70.0 * (u_clamped**6) - 20.0 * (u_clamped**7)

  # dy/du = 140*u^3 * (1 - u)^3 = 140*u^3 - 420*u^4 + 420*u^5 - 140*u^6
  dy = 140.0 * (u_clamped**3) - 420.0 * (u_clamped**4) + 420.0 * (u_clamped**5) - 140.0 * (u_clamped**6)

  # d2y/du2 = 420*u^2 - 1680*u^3 + 2100*u^4 - 840*u^5
  d2y = 420.0 * (u_clamped**2) - 1680.0 * (u_clamped**3) + 2100.0 * (u_clamped**4) - 840.0 * (u_clamped**5)

  # d3y/du3 = 840*u - 5040*u^2 + 8400*u^3 - 4200*u^4
  d3y = 840.0 * u_clamped - 5040.0 * (u_clamped**2) + 8400.0 * (u_clamped**3) - 4200.0 * (u_clamped**4)

  return y, dy, d2y, d3y


class SOHCCamshaft:
  """Parametric 4-5-6-7 polynomial SOHC valvetrain model."""

  def __init__(
      self,
      intake_params: Optional[CamLobeParams] = None,
      exhaust_params: Optional[CamLobeParams] = None,
      spring_params: Optional[ValvetrainMassSpring] = None,
      steel_youngs_modulus_pa: float = 2.10e11,
      steel_poisson_ratio: float = 0.30,
  ):
    self.intake_params = intake_params or CamLobeParams(
        name="Intake",
        max_valve_lift_m=0.0085,  # 8.5 mm lift
        rocker_ratio=1.15,
        duration_crank_deg=245.0,
        open_crank_deg=345.0,  # 15 deg BTDC overlap
    )
    self.exhaust_params = exhaust_params or CamLobeParams(
        name="Exhaust",
        max_valve_lift_m=0.0075,  # 7.5 mm lift
        rocker_ratio=1.15,
        duration_crank_deg=245.0,
        open_crank_deg=135.0,  # 45 deg BBDC
    )
    self.spring = spring_params or ValvetrainMassSpring()
    self.e_mod = steel_youngs_modulus_pa
    self.nu = steel_poisson_ratio

    # Equivalent elastic modulus for Hertz contact: E* = E / (2 * (1 - nu^2))
    self.e_star = self.e_mod / (2.0 * (1.0 - self.nu**2))

  def evaluate_lobe_cam_angle(
      self,
      params: CamLobeParams,
      rpm: float,
      num_points: int = 360,
  ) -> CamKinematics:
    """Evaluate kinematics and contact stresses over a full 360 cam rotation."""
    cam_angles_deg = np.linspace(0.0, 360.0, num_points, endpoint=False)
    cam_angles_rad = np.radians(cam_angles_deg)

    # Cam rotates at half engine speed
    omega_cam = (rpm * 2.0 * np.pi / 60.0) / 2.0  # rad/s

    # Half duration for opening, half for closing
    duration_cam_deg = params.duration_crank_deg / 2.0
    duration_cam_rad = np.radians(duration_cam_deg)
    half_dur_rad = duration_cam_rad / 2.0

    # Max cam lift
    h_cam_max = params.max_valve_lift_m / params.rocker_ratio
    h_valve_max = params.max_valve_lift_m

    # Center the lobe at cam_angle = 180 deg
    phi_center_deg = 180.0
    phi_rel_deg = cam_angles_deg - phi_center_deg
    phi_rel_rad = np.radians(phi_rel_deg)

    valve_lift = np.zeros_like(cam_angles_deg)
    dy_dphi = np.zeros_like(cam_angles_deg)  # m / rad
    d2y_dphi2 = np.zeros_like(cam_angles_deg)  # m / rad^2
    d3y_dphi3 = np.zeros_like(cam_angles_deg)  # m / rad^3

    for i, phi in enumerate(phi_rel_rad):
      if -half_dur_rad <= phi <= 0.0:
        # Opening flank: u runs from 0 (at -half_dur) to 1 (at center)
        u = (phi + half_dur_rad) / half_dur_rad
        y_n, dy_n, d2y_n, d3y_n = polynomial_4567(np.array([u]))
        valve_lift[i] = h_valve_max * y_n[0]
        dy_dphi[i] = (h_valve_max / half_dur_rad) * dy_n[0]
        d2y_dphi2[i] = (h_valve_max / (half_dur_rad**2)) * d2y_n[0]
        d3y_dphi3[i] = (h_valve_max / (half_dur_rad**3)) * d3y_n[0]
      elif 0.0 < phi <= half_dur_rad:
        # Closing flank: v runs from 0 (at center) to 1 (at +half_dur)
        v = phi / half_dur_rad
        y_n, dy_n, d2y_n, d3y_n = polynomial_4567(np.array([v]))
        valve_lift[i] = h_valve_max * (1.0 - y_n[0])
        dy_dphi[i] = -(h_valve_max / half_dur_rad) * dy_n[0]
        d2y_dphi2[i] = -(h_valve_max / (half_dur_rad**2)) * d2y_n[0]
        d3y_dphi3[i] = -(h_valve_max / (half_dur_rad**3)) * d3y_n[0]
      else:
        # Base circle
        valve_lift[i] = 0.0
        dy_dphi[i] = 0.0
        d2y_dphi2[i] = 0.0
        d3y_dphi3[i] = 0.0

    cam_lift = valve_lift / params.rocker_ratio

    # Time derivatives
    v_valve = omega_cam * dy_dphi
    a_valve = (omega_cam**2) * d2y_dphi2
    j_valve = (omega_cam**3) * d3y_dphi3

    # Spring force: F_s = F_seat + k * y_valve
    f_spring = self.spring.seat_force_preload_n + self.spring.spring_rate_n_m * valve_lift

    # Valve inertia force: F_inertia = m_eq * a_valve
    m_eq = self.spring.equivalent_valve_mass_kg
    f_inertia = m_eq * a_valve

    # Contact force between valve / rocker follower:
    # On the lobe: follower must exert force to compress spring and accelerate mass:
    #   F_contact = F_spring + m_eq * a_valve
    # When decelerating (a_valve < 0 over the nose), F_contact reduces toward 0 (float).
    # On the base circle: valve rests on the head seat, so cam contact force is zero.
    on_lobe_mask = (valve_lift > 1.0e-5) | (np.abs(d2y_dphi2) > 1.0e-4)
    f_contact_valve = np.zeros_like(valve_lift)
    f_contact_valve[on_lobe_mask] = f_spring[on_lobe_mask] + f_inertia[on_lobe_mask]

    # Normal force on cam lobe = F_contact_valve * rocker_ratio
    f_contact_cam = np.maximum(0.0, f_contact_valve * params.rocker_ratio)

    # Hertzian contact stress calculation
    # Cam radius of curvature: rho_cam(phi) = R_base + y_cam + d^2 y_cam / d phi^2
    d2y_cam_dphi2 = d2y_dphi2 / params.rocker_ratio
    rho_cam = params.base_circle_radius_m + cam_lift + d2y_cam_dphi2
    # Ensure physical positive curvature
    rho_cam = np.maximum(0.003, rho_cam)

    # Equivalent radius: 1 / rho_eq = 1 / rho_cam + 1 / R_roller
    inv_rho_eq = (1.0 / rho_cam) + (1.0 / params.roller_radius_m)
    rho_eq = 1.0 / inv_rho_eq

    # Peak compressive Hertz stress: sigma_H = sqrt( (F / (pi * b)) * (E* / rho_eq) )
    b_width = params.lobe_width_m
    stress_pa = np.zeros_like(f_contact_cam)
    positive_mask = f_contact_cam > 0.0
    stress_pa[positive_mask] = np.sqrt(
        (f_contact_cam[positive_mask] / (np.pi * b_width))
        * (self.e_star / rho_eq[positive_mask])
    )
    stress_mpa = stress_pa / 1.0e6

    return CamKinematics(
        cam_angle_deg=cam_angles_deg,
        valve_lift_m=valve_lift,
        cam_lift_m=cam_lift,
        valve_velocity_m_s=v_valve,
        valve_accel_m_s2=a_valve,
        valve_jerk_m_s3=j_valve,
        spring_force_n=f_spring,
        inertia_force_n=f_inertia,
        follower_contact_force_n=f_contact_cam,
        hertz_stress_mpa=stress_mpa,
    )

  def evaluate_valve_lift_crank_angle(
      self,
      params: CamLobeParams,
      theta_crank_deg: np.ndarray,
  ) -> np.ndarray:
    """Evaluate instantaneous valve lift [m] for an array of crank angles [0..720]."""
    duration = params.duration_crank_deg
    half_dur = duration / 2.0
    open_deg = params.open_crank_deg
    close_deg = open_deg + duration

    lifts = np.zeros_like(theta_crank_deg, dtype=float)

    for i, th in enumerate(theta_crank_deg):
      # Handle cycle wrap-around (720 deg periodicity)
      th_mod = th % 720.0
      # Check if within open window
      # Window might wrap around 0 / 720 deg
      if close_deg <= 720.0:
        is_open = (open_deg <= th_mod <= close_deg)
        d_th = th_mod - open_deg
      else:
        # Wraps around 720 deg (e.g. opens at 680, closes at 900 -> 180)
        close_wrapped = close_deg % 720.0
        is_open = (th_mod >= open_deg) or (th_mod <= close_wrapped)
        if th_mod >= open_deg:
          d_th = th_mod - open_deg
        else:
          d_th = (720.0 - open_deg) + th_mod

      if is_open and 0.0 <= d_th <= duration:
        if d_th <= half_dur:
          # Opening
          u = d_th / half_dur
          y_n, _, _, _ = polynomial_4567(np.array([u]))
          lifts[i] = params.max_valve_lift_m * y_n[0]
        else:
          # Closing
          v = (d_th - half_dur) / half_dur
          y_n, _, _, _ = polynomial_4567(np.array([v]))
          lifts[i] = params.max_valve_lift_m * (1.0 - y_n[0])
      else:
        lifts[i] = 0.0

    return lifts

  def calculate_valve_float_rpm(self, params: CamLobeParams) -> Tuple[float, float, float]:
    """Calculate the exact valve float RPM and peak deceleration.

    Returns:
      float_rpm: The engine RPM where contact force reaches 0.
      max_decel_m_s2_at_rated: Peak negative valve deceleration at 5500 RPM.
      max_accel_m_s2_at_rated: Peak positive valve acceleration at 5500 RPM.
    """
    # Evaluate at reference 1000 RPM to find geometric d2y/dphi2
    ref_rpm = 1000.0
    kin = self.evaluate_lobe_cam_angle(params, rpm=ref_rpm, num_points=720)

    # Identify points where valve acceleration is negative (decelerating)
    decel_mask = kin.valve_accel_m_s2 < -1.0e-3
    if not np.any(decel_mask):
      return 99999.0, 0.0, 0.0

    m_eq = self.spring.equivalent_valve_mass_kg

    # At any RPM, a_valve(phi) = a_valve_ref(phi) * (RPM / ref_rpm)^2
    # Float occurs when F_spring(phi) - m_eq * |a_valve(phi)| = 0
    # => (RPM / ref_rpm)^2 = F_spring(phi) / (m_eq * |a_ref(phi)|)
    # => RPM = ref_rpm * sqrt( F_spring(phi) / (m_eq * |a_ref(phi)|) )
    a_decel = np.abs(kin.valve_accel_m_s2[decel_mask])
    f_spr = kin.spring_force_n[decel_mask]

    rpm_ratios = np.sqrt(f_spr / (m_eq * a_decel))
    float_rpm = float(ref_rpm * np.min(rpm_ratios))

    # Evaluate at rated 5500 RPM for acceleration reporting
    rated_kin = self.evaluate_lobe_cam_angle(params, rpm=5500.0, num_points=720)
    max_accel = float(np.max(rated_kin.valve_accel_m_s2))
    max_decel = float(np.min(rated_kin.valve_accel_m_s2))

    return float_rpm, max_decel, max_accel

  def analyze_valvetrain_safety(
      self,
      params: CamLobeParams,
      rated_rpm: float = 5500.0,
      redline_rpm: float = 7000.0,
  ) -> ValveFloatReport:
    """Perform full valvetrain safety verification against valve float and contact fatigue."""
    test_rpm = 1.2 * redline_rpm  # 8400 RPM
    float_rpm, max_decel, max_accel = self.calculate_valve_float_rpm(params)

    # Contact forces at the three speeds
    kin_rated = self.evaluate_lobe_cam_angle(params, rpm=rated_rpm, num_points=720)
    kin_redline = self.evaluate_lobe_cam_angle(params, rpm=redline_rpm, num_points=720)
    kin_test = self.evaluate_lobe_cam_angle(params, rpm=test_rpm, num_points=720)

    min_contact_rated = float(np.min(kin_rated.follower_contact_force_n[kin_rated.valve_lift_m > 1e-4]))
    min_contact_redline = float(np.min(kin_redline.follower_contact_force_n[kin_redline.valve_lift_m > 1e-4]))
    min_contact_test = float(np.min(kin_test.follower_contact_force_n[kin_test.valve_lift_m > 1e-4]))

    peak_hertz = float(np.max(kin_test.hertz_stress_mpa))

    # Float margin relative to 1.2x redline
    float_margin = ((float_rpm - test_rpm) / test_rpm) * 100.0
    is_safe = (float_margin >= 30.0) and (min_contact_test > 0.0) and (peak_hertz < 1500.0)

    return ValveFloatReport(
        cam_name=params.name,
        rated_rpm=rated_rpm,
        redline_rpm=redline_rpm,
        test_rpm=test_rpm,
        min_contact_force_at_rated_n=min_contact_rated,
        min_contact_force_at_redline_n=min_contact_redline,
        min_contact_force_at_test_n=min_contact_test,
        valve_float_rpm=float_rpm,
        float_margin_percent=float_margin,
        peak_hertz_stress_mpa=peak_hertz,
        max_valve_acceleration_m_s2=max_accel,
        max_valve_deceleration_m_s2=max_decel,
        is_safe=is_safe,
    )
