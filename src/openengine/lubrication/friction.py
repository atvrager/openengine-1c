"""Four-Zone Mechanical Friction (FMEP) Solver for Wet-Sump Rolling-Crank Engines.

Computes cycle-integrated mechanical friction losses across four tribological
subsystems:
  1. Piston ring pack and skirt mixed-hydrodynamic Stribeck friction (including
     4-stroke reciprocating inertia and gas side-thrust loads).
  2. Crankshaft 6206 deep-groove ball bearings, big-end needle roller bearing,
     and Viton radial shaft seals (SKF / Palmgren load + viscous churning).
  3. SOHC cam-lobe, roller-rocker pivot, valve guide/seal, and #25 roller chain
     guide shoe friction using dynamic loads from SOHCCamshaft.
  4. Connecting-rod splash-dipper hydrodynamic momentum drag, crank-web mist
     windage, and single-cylinder crankcase underside pumping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Optional
import numpy as np
from openengine.cad.crankshaft import (
    DEFAULT_COUNTERWEIGHT_RADIUS_M,
    DEFAULT_MAIN_JOURNAL_DIAM_M,
    DEFAULT_WEB_THICKNESS_M,
)
from openengine.cad.twin_metrics import get_cad_reciprocating_mass_kg
from openengine.lubrication.rheology import OilGradeSpec, OilState
from openengine.lubrication.splash import (
    SplashSystemParams,
    SplashTransportReport,
    evaluate_splash_transport,
)

if TYPE_CHECKING:
  from openengine.dynamics.sohc_cam import SOHCCamshaft
  from openengine.thermo.cycle import EngineGeometry

SECONDS_PER_MINUTE: float = 60.0
CYCLES_PER_TWO_REV_DIVISOR: float = 120.0
TWO_PI: float = 2.0 * math.pi
FOUR_PI: float = 4.0 * math.pi
DEG_TO_RAD: float = math.pi / 180.0
FULL_CYCLE_DEG: float = 720.0
MM_TO_M: float = 1.0e-3
PA_TO_BAR: float = 1.0e-5
N_MM_TO_N_M: float = 1.0e-3
HALF_FACTOR: float = 0.5
PARABOLIC_MEAN_DEPTH_FACTOR: float = 2.0 / 3.0
DEFAULT_ATM_PRESSURE_PA: float = 101325.0
DEFAULT_LINER_WALL_TEMP_C: float = 155.0
MIN_SPEED_EPSILON: float = 1.0e-4
MIN_CAM_EVAL_RPM: float = 100.0
MOTORED_TRACE_POLYTROPIC_EXP: float = 1.32

# Zone 1: Piston ring-pack, Thin-film Couette shear, and 4-stroke side thrust
TOP_RING_AXIAL_WIDTH_M: float = 0.0012
SECOND_RING_AXIAL_WIDTH_M: float = 0.0012
OIL_RING_LAND_WIDTH_M: float = 0.0010
TOP_RING_ELASTIC_PA: float = 165.0e3
SECOND_RING_ELASTIC_PA: float = 150.0e3
OIL_RING_ELASTIC_PA: float = 340.0e3
TOP_RING_GAS_BACKING_FRAC: float = 0.55
SECOND_RING_GAS_BACKING_FRAC: float = 0.18
STARVED_RING_FILM_M: float = 1.8e-6
BOUNDARY_FRICTION_COEFF: float = 0.12
STRIBECK_CRITICAL_DUTY: float = 2.2e-5
HYDRO_RING_SHEAR_COEFF: float = 4.2
SKIRT_LENGTH_TO_BORE_RATIO: float = 0.58
SKIRT_WETTED_CIRCUM_FRAC: float = 0.62
HOT_SKIRT_RADIAL_CLEARANCE_M: float = 24.0e-6
SKIRT_MIXED_FRICTION_COEFF: float = 0.042

# Zone 2: Rolling bearings (2x 6206 ball + K28x33x17 needle) and Viton seals
NUM_MAIN_BALL_BEARINGS: float = 2.0
MAIN_6206_PITCH_DIAM_MM: float = 46.0
MAIN_6206_F0_SPLASH: float = 2.4
MAIN_6206_F1_LOAD: float = 0.0006
BIG_END_NEEDLE_PITCH_DIAM_MM: float = 30.5
BIG_END_NEEDLE_F0_SPLASH: float = 2.8
BIG_END_NEEDLE_F1_LOAD: float = 0.0009
PALMGREN_VISCOUS_PREFACTOR: float = 1.0e-7
PALMGREN_VISCOUS_EXP: float = 2.0 / 3.0
PALMGREN_DIAM_EXP: float = 3.0
MIN_PALMGREN_NU_N: float = 2000.0
NUM_CRANK_VITON_SEALS: float = 2.0
SEAL_SHAFT_DIAM_M: float = DEFAULT_MAIN_JOURNAL_DIAM_M
SEAL_LIP_LINE_LOAD_N_M: float = 115.0
SEAL_FRICTION_COEFF: float = 0.075

# Zone 3: SOHC cam, roller rocker, valve stem guides/seals, #25 chain guide shoe
ROLLER_FOLLOWER_FRICTION_COEFF: float = 0.016
ROCKER_PIVOT_RADIUS_M: float = 0.006
ROCKER_PIVOT_FRICTION_COEFF: float = 0.035
VALVE_GUIDE_SIDE_FRICTION_FRAC: float = 0.024
VALVE_STEM_SEAL_DRAG_N: float = 14.0
TWO_VALVE_STROKES_FACTOR: float = 2.0
CAM_JOURNAL_DRAG_TORQUE_NM_REF: float = 0.095
CHAIN_SHOE_NORMAL_LOAD_N: float = 18.0
CHAIN_SHOE_FRICTION_COEFF: float = 0.065
CHAIN_MESH_EFFICIENCY_LOSS: float = 0.022
VALVETRAIN_VISCOUS_REF_PA_S: float = 0.011
VALVETRAIN_VISCOUS_EXP: float = 0.28

# Zone 4: Dipper scoop + splash jet acceleration + crank windage + bay pumping
DIPPER_DRAG_COEFF: float = 1.35
SPLASH_JET_ENTRAINMENT_FACTOR: float = 2.2
NUM_CRANK_WEBS: float = 2.0
CRANK_WEB_RADIUS_M: float = DEFAULT_COUNTERWEIGHT_RADIUS_M
CRANK_WEB_TOTAL_WIDTH_M: float = NUM_CRANK_WEBS * DEFAULT_WEB_THICKNESS_M
MIST_DENSITY_BASE_KG_M3: float = 1.45
MIST_OIL_LOADING_COEFF: float = 0.0085
CRANK_WINDAGE_COEFF: float = 0.095
WINDAGE_RADIUS_EXP: float = 4.0
UNDERSIDE_BREATHER_LOSS_COEFF: float = 1.6
UNDERSIDE_WINDOW_AREA_M2: float = 1.45e-3


@dataclass(frozen=True, slots=True)
class FrictionBreakdownResult:
  """Cycle friction work, 4-zone FMEP breakdown, torque loss, and heat power."""

  rpm: float
  oil_grade: str
  oil_temp_c: float
  oil_viscosity_cst: float
  ring_skirt_fmep_pa: float
  rolling_bearings_fmep_pa: float
  valvetrain_fmep_pa: float
  dipper_windage_fmep_pa: float
  total_fmep_pa: float
  total_fmep_bar: float
  friction_work_per_cycle_j: float
  friction_torque_nm: float
  ring_skirt_heat_w: float
  rolling_bearings_heat_w: float
  valvetrain_heat_w: float
  dipper_windage_heat_w: float
  total_friction_power_w: float


def _piston_kinematics_and_thrust(
    geom: EngineGeometry,
    theta_deg: np.ndarray,
    pressure_pa: np.ndarray,
    rpm: float,
    reciprocating_mass_kg: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute |ds/dtheta|, piston speed |V_p|, and skirt side-thrust |F_side|."""
  m_recip_kg = (
      get_cad_reciprocating_mass_kg()
      if reciprocating_mass_kg is None
      else reciprocating_mass_kg
  )
  th_rad = np.radians(theta_deg)
  omega = rpm * (TWO_PI / SECONDS_PER_MINUTE)
  r_crank = geom.crank_radius
  rod_ratio = geom.rod_to_crank_ratio
  inv_r = 1.0 / rod_ratio

  sin_th = np.sin(th_rad)
  cos_th = np.cos(th_rad)
  cos_2th = np.cos(2.0 * th_rad)
  root_term = np.sqrt(np.maximum(MIN_SPEED_EPSILON, rod_ratio**2 - sin_th**2))

  ds_dth = np.abs(r_crank * sin_th * (1.0 + cos_th / root_term))
  v_piston = ds_dth * omega
  a_piston = (omega**2) * r_crank * (cos_th + inv_r * cos_2th)

  piston_area = math.pi * ((geom.bore * HALF_FACTOR) ** 2)
  f_gas = (pressure_pa - DEFAULT_ATM_PRESSURE_PA) * piston_area
  f_inertia = -m_recip_kg * a_piston
  tan_beta = np.abs(sin_th) / root_term
  f_side = np.abs(f_gas + f_inertia) * tan_beta
  return ds_dth, v_piston, f_side


def _compute_ring_skirt_work_j(
    geom: EngineGeometry,
    theta_deg: np.ndarray,
    pressure_pa: np.ndarray,
    rpm: float,
    oil_spec: OilGradeSpec,
    oil_temp_c: float,
    liner_wall_temp_c: float,
    reciprocating_mass_kg: Optional[float] = None,
) -> float:
  """Integrate Zone 1 piston 3-ring pack + skirt mixed Stribeck work [J/cycle]."""
  film_temp_c = HALF_FACTOR * (oil_temp_c + liner_wall_temp_c)
  mu_film = oil_spec.dynamic_viscosity_walther_pa_s(film_temp_c)

  ds_dth, v_piston, f_side = _piston_kinematics_and_thrust(
      geom,
      theta_deg,
      pressure_pa,
      rpm,
      reciprocating_mass_kg=reciprocating_mass_kg,
  )
  p_gage = np.maximum(0.0, pressure_pa - DEFAULT_ATM_PRESSURE_PA)

  circum = math.pi * geom.bore
  rings = (
      (TOP_RING_AXIAL_WIDTH_M, TOP_RING_ELASTIC_PA + TOP_RING_GAS_BACKING_FRAC * p_gage),
      (
          SECOND_RING_AXIAL_WIDTH_M,
          SECOND_RING_ELASTIC_PA + SECOND_RING_GAS_BACKING_FRAC * p_gage,
      ),
      (OIL_RING_LAND_WIDTH_M, np.full_like(p_gage, OIL_RING_ELASTIC_PA)),
  )

  f_rings = np.zeros_like(theta_deg, dtype=float)
  for width_m, p_load_pa in rings:
    duty = (mu_film * v_piston) / np.maximum(1.0, p_load_pa * width_m)
    mu_boundary = BOUNDARY_FRICTION_COEFF * np.exp(
        -np.sqrt(duty / STRIBECK_CRITICAL_DUTY)
    )
    mu_hydro = HYDRO_RING_SHEAR_COEFF * np.sqrt(duty)
    f_normal = p_load_pa * width_m * circum
    f_couette = mu_film * (width_m * circum) * (v_piston / STARVED_RING_FILM_M)
    f_rings += (mu_boundary + mu_hydro) * f_normal + f_couette

  skirt_len_m = SKIRT_LENGTH_TO_BORE_RATIO * geom.bore
  skirt_area_m2 = SKIRT_WETTED_CIRCUM_FRAC * circum * skirt_len_m
  f_skirt_visc = mu_film * skirt_area_m2 * (v_piston / HOT_SKIRT_RADIAL_CLEARANCE_M)
  f_skirt_mixed = SKIRT_MIXED_FRICTION_COEFF * f_side

  return float(
      np.trapezoid(
          (f_rings + f_skirt_visc + f_skirt_mixed) * ds_dth,
          np.radians(theta_deg),
      )
  )


def _compute_rolling_bearings_work_j(
    geom: EngineGeometry,
    theta_deg: np.ndarray,
    pressure_pa: np.ndarray,
    rpm: float,
    oil: OilState,
    reciprocating_mass_kg: Optional[float] = None,
) -> float:
  """Compute Zone 2 6206 main ball, big-end needle, and Viton seal work [J/cycle]."""
  m_recip_kg = (
      get_cad_reciprocating_mass_kg()
      if reciprocating_mass_kg is None
      else reciprocating_mass_kg
  )
  _, _, f_side = _piston_kinematics_and_thrust(
      geom,
      theta_deg,
      pressure_pa,
      rpm,
      reciprocating_mass_kg=m_recip_kg,
  )
  piston_area = math.pi * ((geom.bore * HALF_FACTOR) ** 2)
  omega = rpm * (TWO_PI / SECONDS_PER_MINUTE)
  th_rad = np.radians(theta_deg)
  f_gas = (pressure_pa - DEFAULT_ATM_PRESSURE_PA) * piston_area
  f_inertia = (
      m_recip_kg
      * (omega**2)
      * geom.crank_radius
      * np.cos(th_rad)
  )
  mean_radial_load_n = float(np.mean(np.hypot(f_gas - f_inertia, f_side)))

  nu_n = max(MIN_PALMGREN_NU_N, oil.kinematic_viscosity_cst * max(1.0, rpm))
  visc_factor = PALMGREN_VISCOUS_PREFACTOR * (nu_n**PALMGREN_VISCOUS_EXP)

  m0_main_nm = (
      visc_factor
      * MAIN_6206_F0_SPLASH
      * (MAIN_6206_PITCH_DIAM_MM**PALMGREN_DIAM_EXP)
      * N_MM_TO_N_M
  )
  m1_main_nm = (
      MAIN_6206_F1_LOAD
      * (HALF_FACTOR * mean_radial_load_n)
      * (MAIN_6206_PITCH_DIAM_MM * MM_TO_M)
  )
  torque_mains_nm = NUM_MAIN_BALL_BEARINGS * (m0_main_nm + m1_main_nm)

  m0_needle_nm = (
      visc_factor
      * BIG_END_NEEDLE_F0_SPLASH
      * (BIG_END_NEEDLE_PITCH_DIAM_MM**PALMGREN_DIAM_EXP)
      * N_MM_TO_N_M
  )
  m1_needle_nm = (
      BIG_END_NEEDLE_F1_LOAD
      * mean_radial_load_n
      * (BIG_END_NEEDLE_PITCH_DIAM_MM * MM_TO_M)
  )

  seal_circum = math.pi * SEAL_SHAFT_DIAM_M
  seal_radius = HALF_FACTOR * SEAL_SHAFT_DIAM_M
  torque_seals_nm = (
      NUM_CRANK_VITON_SEALS
      * SEAL_LIP_LINE_LOAD_N_M
      * seal_circum
      * SEAL_FRICTION_COEFF
      * seal_radius
  )

  total_bearing_torque_nm = torque_mains_nm + m0_needle_nm + m1_needle_nm + torque_seals_nm
  return float(total_bearing_torque_nm * FOUR_PI)


def compute_valvetrain_mechanical_work_j(
    rpm: float,
    camshaft: SOHCCamshaft,
    splash_params: SplashSystemParams,
) -> float:
  """Compute temperature-independent cam lobe, guide, seal, and shoe work [J/cycle]."""
  kin_in = camshaft.evaluate_lobe_cam_angle(
      camshaft.intake_params, rpm=max(MIN_CAM_EVAL_RPM, rpm)
  )
  kin_ex = camshaft.evaluate_lobe_cam_angle(
      camshaft.exhaust_params, rpm=max(MIN_CAM_EVAL_RPM, rpm)
  )

  phi_rad = np.radians(kin_in.cam_angle_deg)
  r_base_in = camshaft.intake_params.base_circle_radius_m
  r_base_ex = camshaft.exhaust_params.base_circle_radius_m

  torque_in = kin_in.follower_contact_force_n * (
      ROLLER_FOLLOWER_FRICTION_COEFF * r_base_in
      + ROCKER_PIVOT_FRICTION_COEFF * ROCKER_PIVOT_RADIUS_M
  )
  torque_ex = kin_ex.follower_contact_force_n * (
      ROLLER_FOLLOWER_FRICTION_COEFF * r_base_ex
      + ROCKER_PIVOT_FRICTION_COEFF * ROCKER_PIVOT_RADIUS_M
  )
  lobe_work_j = float(np.trapezoid(torque_in + torque_ex, phi_rad))

  # Valve stem guide + Viton stem seal sliding over opening + closing lift strokes
  mean_f_in = float(np.mean(kin_in.follower_contact_force_n))
  mean_f_ex = float(np.mean(kin_ex.follower_contact_force_n))
  guide_in_j = (
      (VALVE_GUIDE_SIDE_FRICTION_FRAC * mean_f_in + VALVE_STEM_SEAL_DRAG_N)
      * TWO_VALVE_STROKES_FACTOR
      * camshaft.intake_params.max_valve_lift_m
  )
  guide_ex_j = (
      (VALVE_GUIDE_SIDE_FRICTION_FRAC * mean_f_ex + VALVE_STEM_SEAL_DRAG_N)
      * TWO_VALVE_STROKES_FACTOR
      * camshaft.exhaust_params.max_valve_lift_m
  )

  # #25 chain travel over 2 crankshaft revolutions per cycle against tensioner shoe
  chain_travel_per_cycle_m = (
      TWO_VALVE_STROKES_FACTOR
      * splash_params.crank_sprocket_teeth
      * splash_params.chain_pitch_m
  )
  chain_shoe_work_j = (
      CHAIN_SHOE_NORMAL_LOAD_N
      * CHAIN_SHOE_FRICTION_COEFF
      * chain_travel_per_cycle_m
  )
  return float(lobe_work_j + guide_in_j + guide_ex_j + chain_shoe_work_j)


def _compute_valvetrain_work_j(
    rpm: float,
    oil: OilState,
    camshaft: SOHCCamshaft,
    splash_params: SplashSystemParams,
    precomputed_mech_work_j: Optional[float] = None,
) -> float:
  """Compute Zone 3 SOHC cam, rocker, stem guide/seal, and #25 chain work [J/cycle]."""
  mech_work_j = (
      precomputed_mech_work_j
      if precomputed_mech_work_j is not None
      else compute_valvetrain_mechanical_work_j(rpm, camshaft, splash_params)
  )
  visc_scale = (
      oil.dynamic_viscosity_pa_s / VALVETRAIN_VISCOUS_REF_PA_S
  ) ** VALVETRAIN_VISCOUS_EXP
  cam_journal_work_j = TWO_PI * CAM_JOURNAL_DRAG_TORQUE_NM_REF * visc_scale
  return float((mech_work_j + cam_journal_work_j) * (1.0 + CHAIN_MESH_EFFICIENCY_LOSS))


def _compute_dipper_and_windage_work_j(
    geom: EngineGeometry,
    theta_deg: np.ndarray,
    rpm: float,
    oil: OilState,
    splash_params: SplashSystemParams,
    splash_report: SplashTransportReport,
) -> float:
  """Compute Zone 4 dipper impact, crank mist windage, and bay pumping [J/cycle]."""
  omega = max(0.0, rpm) * (TWO_PI / SECONDS_PER_MINUTE)
  r_dip_m = splash_params.dipper_tip_radius_mm * MM_TO_M
  w_dip_m = splash_params.dipper_width_mm * MM_TO_M
  h_dyn_m = splash_report.dynamic_immersion_mm * MM_TO_M

  v_tip = omega * r_dip_m
  mean_submerged_area_m2 = w_dip_m * (PARABOLIC_MEAN_DEPTH_FACTOR * h_dyn_m)
  f_dipper_mean_n = (
      HALF_FACTOR
      * DIPPER_DRAG_COEFF
      * SPLASH_JET_ENTRAINMENT_FACTOR
      * oil.density_kg_m3
      * mean_submerged_area_m2
      * (v_tip**2)
  )

  arc_rad_per_cycle = splash_report.submerged_arc_per_cycle_deg * DEG_TO_RAD
  dipper_path_m = r_dip_m * arc_rad_per_cycle
  dipper_work_j = f_dipper_mean_n * dipper_path_m

  rho_mist = MIST_DENSITY_BASE_KG_M3 + MIST_OIL_LOADING_COEFF * oil.density_kg_m3 * (
      splash_report.oil_in_flight_l / max(MIN_SPEED_EPSILON, splash_params.sump_volume_l)
  )
  windage_torque_nm = (
      CRANK_WINDAGE_COEFF
      * rho_mist
      * (omega**2)
      * (CRANK_WEB_RADIUS_M**WINDAGE_RADIUS_EXP)
      * CRANK_WEB_TOTAL_WIDTH_M
  )
  windage_work_j = windage_torque_nm * FOUR_PI

  # Single-cylinder underside piston bay air-mist pumping across crankcase windows
  piston_area = math.pi * ((geom.bore * HALF_FACTOR) ** 2)
  th_rad = np.radians(theta_deg)
  r_crank = geom.crank_radius
  rod_ratio = geom.rod_to_crank_ratio
  sin_th = np.sin(th_rad)
  cos_th = np.cos(th_rad)
  root_term = np.sqrt(np.maximum(MIN_SPEED_EPSILON, rod_ratio**2 - sin_th**2))
  ds_dth = np.abs(r_crank * sin_th * (1.0 + cos_th / root_term))
  v_piston = ds_dth * omega
  v_window = v_piston * (piston_area / UNDERSIDE_WINDOW_AREA_M2)
  dp_bay = HALF_FACTOR * UNDERSIDE_BREATHER_LOSS_COEFF * rho_mist * (v_window**2)
  bay_pump_work_j = float(np.trapezoid(dp_bay * piston_area * ds_dth, th_rad))

  return float(dipper_work_j + windage_work_j + bay_pump_work_j)


def evaluate_engine_friction(
    geom: EngineGeometry,
    rpm: float,
    oil_temp_c: float,
    theta_deg: Optional[np.ndarray] = None,
    pressure_pa: Optional[np.ndarray] = None,
    oil_spec: Optional[OilGradeSpec] = None,
    splash_params: Optional[SplashSystemParams] = None,
    camshaft: Optional[SOHCCamshaft] = None,
    liner_wall_temp_c: float = DEFAULT_LINER_WALL_TEMP_C,
    precomputed_valvetrain_mech_work_j: Optional[float] = None,
    reciprocating_mass_kg: Optional[float] = None,
) -> FrictionBreakdownResult:
  """Calculate 4-zone FMEP, friction torque, and heat generation over 720 deg."""
  spec = oil_spec or OilGradeSpec.sae_10w40()
  splash_cfg = splash_params or SplashSystemParams()
  if camshaft is None:
    from openengine.dynamics.sohc_cam import SOHCCamshaft

    cam_model = SOHCCamshaft()
  else:
    cam_model = camshaft
  m_recip_kg = (
      get_cad_reciprocating_mass_kg()
      if reciprocating_mass_kg is None
      else reciprocating_mass_kg
  )
  oil_state = spec.evaluate(oil_temp_c)

  if theta_deg is None or pressure_pa is None:
    th_arr = np.linspace(0.0, FULL_CYCLE_DEG, int(FULL_CYCLE_DEG) + 1)
    v_max = geom.displacement + geom.clearance_volume
    v_arr = np.asarray(geom.volume(np.radians(th_arr)), dtype=float)
    p_arr = DEFAULT_ATM_PRESSURE_PA * ((v_max / v_arr) ** MOTORED_TRACE_POLYTROPIC_EXP)
  else:
    th_arr = np.asarray(theta_deg, dtype=float)
    p_arr = np.asarray(pressure_pa, dtype=float)

  splash_rep = evaluate_splash_transport(
      rpm=rpm,
      oil_temp_c=oil_temp_c,
      oil_spec=spec,
      params=splash_cfg,
  )

  w_ring_j = _compute_ring_skirt_work_j(
      geom=geom,
      theta_deg=th_arr,
      pressure_pa=p_arr,
      rpm=rpm,
      oil_spec=spec,
      oil_temp_c=oil_temp_c,
      liner_wall_temp_c=liner_wall_temp_c,
      reciprocating_mass_kg=m_recip_kg,
  )
  w_brg_j = _compute_rolling_bearings_work_j(
      geom=geom,
      theta_deg=th_arr,
      pressure_pa=p_arr,
      rpm=rpm,
      oil=oil_state,
      reciprocating_mass_kg=m_recip_kg,
  )
  w_valve_j = _compute_valvetrain_work_j(
      rpm=rpm,
      oil=oil_state,
      camshaft=cam_model,
      splash_params=splash_cfg,
      precomputed_mech_work_j=precomputed_valvetrain_mech_work_j,
  )
  w_dip_j = _compute_dipper_and_windage_work_j(
      geom=geom,
      theta_deg=th_arr,
      rpm=rpm,
      oil=oil_state,
      splash_params=splash_cfg,
      splash_report=splash_rep,
  )

  vd = geom.displacement
  fmep_ring = w_ring_j / vd
  fmep_brg = w_brg_j / vd
  fmep_valve = w_valve_j / vd
  fmep_dip = w_dip_j / vd
  fmep_total = fmep_ring + fmep_brg + fmep_valve + fmep_dip

  w_total_j = w_ring_j + w_brg_j + w_valve_j + w_dip_j
  cycle_freq_hz = max(0.0, rpm) / CYCLES_PER_TWO_REV_DIVISOR

  return FrictionBreakdownResult(
      rpm=float(rpm),
      oil_grade=spec.grade,
      oil_temp_c=float(oil_state.temperature_c),
      oil_viscosity_cst=float(oil_state.kinematic_viscosity_cst),
      ring_skirt_fmep_pa=float(fmep_ring),
      rolling_bearings_fmep_pa=float(fmep_brg),
      valvetrain_fmep_pa=float(fmep_valve),
      dipper_windage_fmep_pa=float(fmep_dip),
      total_fmep_pa=float(fmep_total),
      total_fmep_bar=float(fmep_total * PA_TO_BAR),
      friction_work_per_cycle_j=float(w_total_j),
      friction_torque_nm=float(w_total_j / FOUR_PI),
      ring_skirt_heat_w=float(w_ring_j * cycle_freq_hz),
      rolling_bearings_heat_w=float(w_brg_j * cycle_freq_hz),
      valvetrain_heat_w=float(w_valve_j * cycle_freq_hz),
      dipper_windage_heat_w=float(w_dip_j * cycle_freq_hz),
      total_friction_power_w=float(w_total_j * cycle_freq_hz),
  )
