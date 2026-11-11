"""Steady-State Wet-Sump Oil Thermal Balance and Piston Splash Cooling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Optional
import numpy as np
from openengine.cad.cylinder_block import THERMAL_BUDGET_W
from openengine.lubrication.friction import (
    FrictionBreakdownResult,
    compute_valvetrain_mechanical_work_j,
    evaluate_engine_friction,
)
from openengine.lubrication.rheology import (
    EngineLubricationConfigRoot,
    OilGradeSpec,
    OilState,
)
from openengine.lubrication.splash import (
    SplashSystemParams,
    SplashTransportReport,
    evaluate_splash_transport,
)
from scipy.optimize import root_scalar

if TYPE_CHECKING:
  from openengine.dynamics.sohc_cam import SOHCCamshaft
  from openengine.thermo.cycle import EngineGeometry

DEFAULT_AMBIENT_TEMP_C: float = 27.0
DEFAULT_SUMP_FIN_AREA_M2: float = 0.185
DEFAULT_COOLING_AIRFLOW_M_S: float = 13.5
DEFAULT_FIN_EFFICIENCY: float = 0.88
DEFAULT_MAX_OIL_TEMP_C: float = 130.0
MIN_SAFE_HOT_VISCOSITY_CST: float = 6.5

# Dry baseline piston and barrel thermal references (matching FEA / CHT defaults)
DRY_PISTON_CROWN_TEMP_C: float = 250.0
DRY_PISTON_RING_LAND_TEMP_C: float = 190.0
DRY_PISTON_SKIRT_TEMP_C: float = 140.0
DRY_BARREL_HEAT_BUDGET_W: float = THERMAL_BUDGET_W
MIN_BARREL_HEAT_BUDGET_W: float = 600.0

# Heat partitioning and convective correlations
RING_FRICTION_TO_OIL_FRACTION: float = 0.35
DIPPER_WINDAGE_TO_OIL_FRACTION: float = 0.70
CRANKCASE_BARREL_CONDUCTION_W: float = 55.0
NATURAL_AIR_HTC_W_M2_K: float = 12.0
FORCED_AIR_HTC_COEFF: float = 11.8
FORCED_AIR_VELOCITY_EXP: float = 0.80
INTERNAL_SPLASH_OIL_HTC_W_M2_K: float = 380.0

# Piston under-crown splash impingement cooling parameters
HALF_FACTOR: float = 0.5
UNDERCROWN_RIB_AREA_FACTOR: float = 1.30
SPLASH_HTC_BASE_W_M2_K: float = 95.0
SPLASH_HTC_THROW_COEFF: float = 34.0
SPLASH_THROW_VELOCITY_EXP: float = 0.65
SPLASH_REF_VISC_CST: float = 14.0
SPLASH_VISC_EXP: float = -0.22
PISTON_CROWN_CONDUCTION_R_K_W: float = 0.085
CROWN_COOLING_SENSITIVITY_K_W: float = 0.082
RING_LAND_COOLING_SENSITIVITY_K_W: float = 0.055
SKIRT_COOLING_SENSITIVITY_K_W: float = 0.048
BARREL_RELIEF_FRACTION: float = 0.65

# Root-finding temperature bracket
SOLVER_UPPER_TEMP_C: float = 240.0
MIN_TEMP_MARGIN_EPSILON: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class SumpThermalParams:
  """Thermal parameters for the finned wet-sump pan and piston oil cooling."""

  ambient_temp_c: float = DEFAULT_AMBIENT_TEMP_C
  sump_fin_area_m2: float = DEFAULT_SUMP_FIN_AREA_M2
  cooling_airflow_m_s: float = DEFAULT_COOLING_AIRFLOW_M_S
  fin_efficiency: float = DEFAULT_FIN_EFFICIENCY
  max_oil_temp_c: float = DEFAULT_MAX_OIL_TEMP_C
  min_safe_viscosity_cst: float = MIN_SAFE_HOT_VISCOSITY_CST
  dry_crown_temp_c: float = DRY_PISTON_CROWN_TEMP_C
  dry_ring_land_temp_c: float = DRY_PISTON_RING_LAND_TEMP_C
  dry_skirt_temp_c: float = DRY_PISTON_SKIRT_TEMP_C
  dry_barrel_heat_budget_w: float = DRY_BARREL_HEAT_BUDGET_W

  @classmethod
  def from_config(cls, cfg: EngineLubricationConfigRoot) -> "SumpThermalParams":
    """Build SumpThermalParams from a parsed engine_specs.yaml mapping."""
    lub = cfg.get("lubrication")
    if lub is None:
      return cls()
    return cls(
        sump_fin_area_m2=float(lub.get("sump_fin_area_m2", DEFAULT_SUMP_FIN_AREA_M2)),
        cooling_airflow_m_s=float(
            lub.get("cooling_airflow_m_s", DEFAULT_COOLING_AIRFLOW_M_S)
        ),
        max_oil_temp_c=float(lub.get("max_oil_temp_c", DEFAULT_MAX_OIL_TEMP_C)),
    )


@dataclass(frozen=True, slots=True)
class SumpThermalBalanceResult:
  """Equilibrium sump oil temperature, heat fluxes, and cooled boundary targets."""

  rpm: float
  oil_grade: str
  equilibrium_oil_temp_c: float
  max_allowable_oil_temp_c: float
  oil_temp_margin_c: float
  equilibrium_viscosity_cst: float
  friction_heat_to_oil_w: float
  piston_splash_heat_w: float
  crankcase_conduction_w: float
  total_oil_heat_load_w: float
  sump_overall_ua_w_k: float
  sump_heat_rejected_w: float
  undercrown_htc_w_m2_k: float
  cooled_piston_crown_temp_c: float
  cooled_piston_ring_land_temp_c: float
  cooled_piston_skirt_temp_c: float
  adjusted_barrel_heat_budget_w: float
  friction: FrictionBreakdownResult
  splash: SplashTransportReport
  is_oil_temp_safe: bool
  is_viscosity_safe: bool
  is_thermal_safe: bool


def compute_sump_overall_ua_w_k(params: SumpThermalParams) -> float:
  """Compute overall thermal conductance UA [W/K] from sump oil to cooling air."""
  v_air = max(0.0, params.cooling_airflow_m_s)
  h_air = NATURAL_AIR_HTC_W_M2_K + FORCED_AIR_HTC_COEFF * (
      v_air**FORCED_AIR_VELOCITY_EXP
  )
  inv_u = (1.0 / (params.fin_efficiency * h_air)) + (
      1.0 / INTERNAL_SPLASH_OIL_HTC_W_M2_K
  )
  u_overall = 1.0 / inv_u
  return float(u_overall * params.sump_fin_area_m2)


def compute_piston_splash_cooling(
    geom: EngineGeometry,
    oil_state: OilState,
    splash_report: SplashTransportReport,
    thermal_params: SumpThermalParams,
) -> tuple[float, float]:
  """Calculate under-crown impingement HTC [W/(m2.K)] and extracted heat [W]."""
  piston_area = math.pi * ((geom.bore * HALF_FACTOR) ** 2)
  undercrown_area = UNDERCROWN_RIB_AREA_FACTOR * piston_area

  v_throw = max(0.0, splash_report.tip_throw_velocity_m_s)
  visc_term = (
      max(MIN_TEMP_MARGIN_EPSILON, oil_state.kinematic_viscosity_cst)
      / SPLASH_REF_VISC_CST
  ) ** SPLASH_VISC_EXP
  h_splash = (
      SPLASH_HTC_BASE_W_M2_K
      + SPLASH_HTC_THROW_COEFF * (v_throw**SPLASH_THROW_VELOCITY_EXP) * visc_term
  )

  delta_t = max(0.0, thermal_params.dry_crown_temp_c - oil_state.temperature_c)
  r_conv = 1.0 / max(MIN_TEMP_MARGIN_EPSILON, h_splash * undercrown_area)
  q_splash_w = delta_t / (PISTON_CROWN_CONDUCTION_R_K_W + r_conv)
  return float(h_splash), float(q_splash_w)


def _evaluate_heat_terms_at_temp(
    oil_temp_c: float,
    geom: EngineGeometry,
    rpm: float,
    theta_deg: Optional[np.ndarray],
    pressure_pa: Optional[np.ndarray],
    oil_spec: OilGradeSpec,
    splash_params: SplashSystemParams,
    thermal_params: SumpThermalParams,
    ua_sump_w_k: float,
    camshaft: SOHCCamshaft,
    vt_mech_work_j: float,
) -> tuple[
    OilState,
    FrictionBreakdownResult,
    SplashTransportReport,
    float,
    float,
    float,
    float,
]:
  """Evaluate all sump heat-in and heat-out terms at candidate oil_temp_c."""
  oil_state = oil_spec.evaluate(oil_temp_c)
  splash_rep = evaluate_splash_transport(
      rpm=rpm,
      oil_temp_c=oil_temp_c,
      oil_spec=oil_spec,
      params=splash_params,
  )
  fric = evaluate_engine_friction(
      geom=geom,
      rpm=rpm,
      oil_temp_c=oil_temp_c,
      theta_deg=theta_deg,
      pressure_pa=pressure_pa,
      oil_spec=oil_spec,
      splash_params=splash_params,
      camshaft=camshaft,
      precomputed_valvetrain_mech_work_j=vt_mech_work_j,
  )

  q_fric_oil = (
      fric.rolling_bearings_heat_w
      + fric.valvetrain_heat_w
      + DIPPER_WINDAGE_TO_OIL_FRACTION * fric.dipper_windage_heat_w
      + RING_FRICTION_TO_OIL_FRACTION * fric.ring_skirt_heat_w
  )
  h_splash, q_piston_splash = compute_piston_splash_cooling(
      geom=geom,
      oil_state=oil_state,
      splash_report=splash_rep,
      thermal_params=thermal_params,
  )
  q_in_total = q_fric_oil + q_piston_splash + CRANKCASE_BARREL_CONDUCTION_W
  q_reject = ua_sump_w_k * (oil_temp_c - thermal_params.ambient_temp_c)
  return (
      oil_state,
      fric,
      splash_rep,
      h_splash,
      q_fric_oil,
      q_piston_splash,
      q_in_total - q_reject,
  )


def solve_sump_thermal_balance(
    geom: EngineGeometry,
    rpm: float,
    theta_deg: Optional[np.ndarray] = None,
    pressure_pa: Optional[np.ndarray] = None,
    oil_spec: Optional[OilGradeSpec] = None,
    splash_params: Optional[SplashSystemParams] = None,
    thermal_params: Optional[SumpThermalParams] = None,
    camshaft: Optional[SOHCCamshaft] = None,
) -> SumpThermalBalanceResult:
  """Solve self-consistent steady-state sump oil temperature and cooling splits."""
  spec = oil_spec or OilGradeSpec.sae_10w40()
  s_params = splash_params or SplashSystemParams()
  t_params = thermal_params or SumpThermalParams()
  if camshaft is None:
    from openengine.dynamics.sohc_cam import SOHCCamshaft

    cam_model = SOHCCamshaft()
  else:
    cam_model = camshaft
  vt_mech_work_j = compute_valvetrain_mechanical_work_j(rpm, cam_model, s_params)
  ua_sump = compute_sump_overall_ua_w_k(t_params)

  def residual(candidate_temp_c: float) -> float:
    """Return net heat imbalance Q_in(T) - Q_out(T) [W]."""
    _, _, _, _, _, _, net_q = _evaluate_heat_terms_at_temp(
        oil_temp_c=candidate_temp_c,
        geom=geom,
        rpm=rpm,
        theta_deg=theta_deg,
        pressure_pa=pressure_pa,
        oil_spec=spec,
        splash_params=s_params,
        thermal_params=t_params,
        ua_sump_w_k=ua_sump,
        camshaft=cam_model,
        vt_mech_work_j=vt_mech_work_j,
    )
    return net_q

  t_low = t_params.ambient_temp_c
  t_high = SOLVER_UPPER_TEMP_C
  r_low = residual(t_low)
  r_high = residual(t_high)

  if r_low <= 0.0:
    t_eq = t_low
  elif r_high >= 0.0:
    t_eq = t_high
  else:
    sol = root_scalar(residual, bracket=[t_low, t_high], method="brentq")
    t_eq = float(sol.root)

  (
      oil_eq,
      fric_eq,
      splash_eq,
      h_undercrown,
      q_fric_oil,
      q_piston_splash,
      _,
  ) = _evaluate_heat_terms_at_temp(
      oil_temp_c=t_eq,
      geom=geom,
      rpm=rpm,
      theta_deg=theta_deg,
      pressure_pa=pressure_pa,
      oil_spec=spec,
      splash_params=s_params,
      thermal_params=t_params,
      ua_sump_w_k=ua_sump,
      camshaft=cam_model,
      vt_mech_work_j=vt_mech_work_j,
  )

  q_total_in = q_fric_oil + q_piston_splash + CRANKCASE_BARREL_CONDUCTION_W
  q_rejected = ua_sump * (t_eq - t_params.ambient_temp_c)

  cooled_crown = t_params.dry_crown_temp_c - (
      CROWN_COOLING_SENSITIVITY_K_W * q_piston_splash
  )
  cooled_ring = t_params.dry_ring_land_temp_c - (
      RING_LAND_COOLING_SENSITIVITY_K_W * q_piston_splash
  )
  cooled_skirt = t_params.dry_skirt_temp_c - (
      SKIRT_COOLING_SENSITIVITY_K_W * q_piston_splash
  )
  adjusted_barrel_w = max(
      MIN_BARREL_HEAT_BUDGET_W,
      t_params.dry_barrel_heat_budget_w - BARREL_RELIEF_FRACTION * q_piston_splash,
  )

  temp_ok = t_eq <= t_params.max_oil_temp_c
  visc_ok = oil_eq.kinematic_viscosity_cst >= t_params.min_safe_viscosity_cst

  return SumpThermalBalanceResult(
      rpm=float(rpm),
      oil_grade=spec.grade,
      equilibrium_oil_temp_c=float(t_eq),
      max_allowable_oil_temp_c=float(t_params.max_oil_temp_c),
      oil_temp_margin_c=float(t_params.max_oil_temp_c - t_eq),
      equilibrium_viscosity_cst=float(oil_eq.kinematic_viscosity_cst),
      friction_heat_to_oil_w=float(q_fric_oil),
      piston_splash_heat_w=float(q_piston_splash),
      crankcase_conduction_w=CRANKCASE_BARREL_CONDUCTION_W,
      total_oil_heat_load_w=float(q_total_in),
      sump_overall_ua_w_k=float(ua_sump),
      sump_heat_rejected_w=float(q_rejected),
      undercrown_htc_w_m2_k=float(h_undercrown),
      cooled_piston_crown_temp_c=float(cooled_crown),
      cooled_piston_ring_land_temp_c=float(cooled_ring),
      cooled_piston_skirt_temp_c=float(cooled_skirt),
      adjusted_barrel_heat_budget_w=float(adjusted_barrel_w),
      friction=fric_eq,
      splash=splash_eq,
      is_oil_temp_safe=temp_ok,
      is_viscosity_safe=visc_ok,
      is_thermal_safe=temp_ok and visc_ok and splash_eq.is_transport_safe,
  )
