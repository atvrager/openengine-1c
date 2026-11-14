"""Transient Inertia Dynamometer Pull, Thermal Warm-Up, and Crank Press-Fit Audit.

Couples live CAD flywheel/cranktrain polar inertia plus virtual roller drum
inertia (`2000 -> 7000 RPM` WOT pull + closed-throttle overrun coast-down) with:
  - Steady-state MBT WOT torque curve & SliderCrankDynamics peak firing torque
  - Coupled Walther oil viscosity warm-up thinning -> dynamic FMEP reduction
  - Cylinder head thermal rise (ALUMINUM_6061_T6 CAD head + barrel mass)
  - FlywheelDynamics speed fluctuation C_s and redline burst safety factor
  - 3-piece crankshaft ISO 286 H6/x6 Lamé shrink-press fit torsional slip SF
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Optional, Tuple
import numpy as np

from openengine.cad.crankshaft import (
    CRANKPIN_BOSS_MARGIN_MM,
    CrankshaftParams,
    MAIN_BOSS_MARGIN_MM,
)
from openengine.cad.twin_metrics import (
    get_cad_flywheel_and_crank_inertia,
    get_cad_head_and_barrel_masses_kg,
    get_cad_sump_splash_metrics,
)
from openengine.dynamics.flywheel import FlywheelDimensions, FlywheelDynamics
from openengine.dynamics.slider_crank import SliderCrankDynamics
from openengine.dynamics.sohc_cam import SOHCCamshaft
from openengine.dyno.steady_sweep import (
    ROUND_ONE_DECIMAL,
    ROUND_TWO_DECIMALS,
    ROUND_VALUE_DECIMALS,
    SteadyDynoSweepResult,
    build_live_twin_simulator,
)
from openengine.fea.calculix import STEEL_1144_STRESSPROOF, STEEL_4140_PREHARD
from openengine.lubrication.friction import (
    compute_valvetrain_mechanical_work_j,
    evaluate_engine_friction,
)
from openengine.lubrication.rheology import OilGradeSpec
from openengine.lubrication.splash import evaluate_splash_transport
from openengine.lubrication.thermal import (
    CRANKCASE_BARREL_CONDUCTION_W,
    DIPPER_WINDAGE_TO_OIL_FRACTION,
    RING_FRICTION_TO_OIL_FRACTION,
    compute_piston_splash_cooling,
    compute_sump_overall_ua_w_k,
)
from openengine.thermo.cycle import EngineGeometry
from openengine.thermo.gas_exchange import FourStrokeSimulator

# General mathematical & unit conversion constants
ZERO_FLOAT: float = 0.0
HALF_FACTOR: float = 0.5
TWO_FACTOR: float = 2.0
THREE_FACTOR: float = 3.0
FOUR_FACTOR: float = 4.0
SIX_FACTOR: float = 6.0
TWO_PI: float = TWO_FACTOR * math.pi
FOUR_PI: float = FOUR_FACTOR * math.pi
SECONDS_PER_MINUTE: float = 60.0
BAR_TO_PA: float = 1.0e5
MPA_TO_PA: float = 1.0e6
PA_TO_MPA: float = 1.0e-6
MM_TO_M: float = 1.0e-3
KG_PER_L_APPROX_DENSITY: float = 1.0e-3
WATT_TO_KW: float = 1000.0
KW_TO_MECHANICAL_HP: float = 1.34102209
NM_TO_LB_FT: float = 0.73756215
TONNE_MM3_TO_KG_M3: float = 1.0e12
MIN_EPSILON: float = 1.0e-9
ROUND_PRESSURE_MPA: int = 2

# Inertia dyno pull & coast-down configuration
DEFAULT_DRUM_INERTIA_KG_M2: float = 0.258
DEFAULT_PULL_START_RPM: float = 2000.0
DEFAULT_PULL_END_RPM: float = 7000.0
DEFAULT_OVERRUN_DURATION_S: float = 2.5
DEFAULT_DT_S: float = 0.02
MAX_PULL_DURATION_S: float = 45.0
MIN_OMEGA_RAD_S: float = 25.0
EXPORT_TRACE_SAMPLES: int = 150
CLOSED_THROTTLE_FRACTION: float = 0.0

# Thermal & warm-up parameters (6061-T6 Cp = 896 J/kg*K)
AL_6061_CP_J_KG_K: float = 896.0
HEAD_COOLING_SHARE: float = 0.58
HEAD_AIR_UA_BASE_W_K: float = 4.2
HEAD_AIR_UA_RPM_SLOPE_W_K: float = 1.8
HEAD_AIR_REF_RPM: float = 3000.0
DEFAULT_INITIAL_HEAD_TEMP_C: float = 85.0
DEFAULT_INITIAL_OIL_TEMP_C: float = 55.0
DEFAULT_AMBIENT_AIR_TEMP_C: float = 25.0
OIL_CP_J_KG_K: float = 2050.0
OVERRUN_PUMPING_BASE_BAR: float = 0.52
OVERRUN_PUMPING_RPM_BAR: float = 0.28
OVERRUN_PUMPING_REF_RPM: float = 7000.0
DRUM_BEARING_DRAG_NM: float = 0.18
DRUM_WINDAGE_COEFF: float = 1.6e-4

# Precomputed FMEP / oil-heat grid dimensions
FMEP_GRID_RPM_POINTS: int = 7
FMEP_GRID_TEMP_MIN_C: float = 40.0
FMEP_GRID_TEMP_MAX_C: float = 135.0
FMEP_GRID_TEMP_POINTS: int = 6

# Crankshaft ISO 286 18..30 mm H6/x6 thermal shrink-press fit limits & safety policy
H6_X6_MIN_DIAM_INTERFERENCE_M: float = 51.0e-6
H6_X6_NOM_DIAM_INTERFERENCE_M: float = 64.0e-6
H6_X6_MAX_DIAM_INTERFERENCE_M: float = 77.0e-6
SHRINK_FIT_FRICTION_COEFF: float = 0.20
ROOM_ASSEMBLY_TEMP_C: float = 20.0
MIN_PRESS_FIT_SF_THRESHOLD: float = 1.20
TWO_WEB_CRANKPIN_SHARE: float = 0.50
FLYWHEEL_TARGET_CS: float = 0.035
STATE_THETA_IDX: int = 0
STATE_OMEGA_IDX: int = 1
STATE_HEAD_TEMP_IDX: int = 2
STATE_OIL_TEMP_IDX: int = 3
LAST_INDEX: int = -1


@dataclass(frozen=True, slots=True)
class CrankPressFitSpec:
  """ISO 286 shrink-press fit interference envelope and dry friction coefficient."""

  fit_label: str = "H6/x6"
  min_diametral_interference_m: float = H6_X6_MIN_DIAM_INTERFERENCE_M
  nominal_diametral_interference_m: float = H6_X6_NOM_DIAM_INTERFERENCE_M
  max_diametral_interference_m: float = H6_X6_MAX_DIAM_INTERFERENCE_M
  friction_coeff: float = SHRINK_FIT_FRICTION_COEFF
  reference_temp_c: float = ROOM_ASSEMBLY_TEMP_C


@dataclass(frozen=True, slots=True)
class JointPressFitMargin:
  """Lamé contact pressure, Von Mises stress, and slip SF for a single crank joint."""

  joint_name: str
  rpm: float
  torque_demand_nm: float
  min_contact_pressure_mpa: float
  nom_contact_pressure_mpa: float
  max_bore_von_mises_mpa: float
  min_slip_torque_nm: float
  nom_slip_torque_nm: float
  min_slip_safety_factor: float
  nom_slip_safety_factor: float
  bore_yield_safety_factor: float
  is_safe: bool


@dataclass(frozen=True, slots=True)
class CrankPressFitReport:
  """Governing 3-piece crankshaft shrink-press fit margins across main & pin joints."""

  fit_label: str
  main_journal: JointPressFitMargin
  crankpin: JointPressFitMargin
  governing_joint: str
  min_slip_safety_factor: float
  nom_slip_safety_factor: float
  bore_yield_safety_factor: float
  is_safe: bool

  def to_dict(self) -> Dict[str, object]:
    """Serialize governing crankshaft press-fit margins to dictionary."""
    return {
        "fit_label": self.fit_label,
        "governing_joint": self.governing_joint,
        "min_slip_safety_factor": round(self.min_slip_safety_factor, ROUND_TWO_DECIMALS),
        "nom_slip_safety_factor": round(self.nom_slip_safety_factor, ROUND_TWO_DECIMALS),
        "bore_yield_safety_factor": round(self.bore_yield_safety_factor, ROUND_TWO_DECIMALS),
        "is_safe": self.is_safe,
        "main_journal": {
            "torque_demand_nm": round(self.main_journal.torque_demand_nm, ROUND_ONE_DECIMAL),
            "min_contact_pressure_mpa": round(self.main_journal.min_contact_pressure_mpa, ROUND_PRESSURE_MPA),
            "max_bore_von_mises_mpa": round(self.main_journal.max_bore_von_mises_mpa, ROUND_PRESSURE_MPA),
            "min_slip_torque_nm": round(self.main_journal.min_slip_torque_nm, ROUND_ONE_DECIMAL),
            "min_slip_safety_factor": round(self.main_journal.min_slip_safety_factor, ROUND_TWO_DECIMALS),
        },
        "crankpin": {
            "torque_demand_nm": round(self.crankpin.torque_demand_nm, ROUND_ONE_DECIMAL),
            "min_contact_pressure_mpa": round(self.crankpin.min_contact_pressure_mpa, ROUND_PRESSURE_MPA),
            "max_bore_von_mises_mpa": round(self.crankpin.max_bore_von_mises_mpa, ROUND_PRESSURE_MPA),
            "min_slip_torque_nm": round(self.crankpin.min_slip_torque_nm, ROUND_ONE_DECIMAL),
            "min_slip_safety_factor": round(self.crankpin.min_slip_safety_factor, ROUND_TWO_DECIMALS),
        },
    }


@dataclass(frozen=True, slots=True)
class InertiaPullConfig:
  """Configuration for transient virtual roller-drum inertia pull & overrun."""

  start_rpm: float = DEFAULT_PULL_START_RPM
  end_rpm: float = DEFAULT_PULL_END_RPM
  drum_inertia_kg_m2: float = DEFAULT_DRUM_INERTIA_KG_M2
  overrun_duration_s: float = DEFAULT_OVERRUN_DURATION_S
  dt_s: float = DEFAULT_DT_S
  initial_head_temp_c: float = DEFAULT_INITIAL_HEAD_TEMP_C
  initial_oil_temp_c: float = DEFAULT_INITIAL_OIL_TEMP_C
  ambient_air_temp_c: float = DEFAULT_AMBIENT_AIR_TEMP_C
  press_fit_spec: CrankPressFitSpec = CrankPressFitSpec()


@dataclass(frozen=True, slots=True)
class InertiaPullResult:
  """Complete time-domain pull trajectory, thermal/viscosity state, and structural SFs."""

  config: InertiaPullConfig
  cad_flywheel_mass_kg: float
  cad_flywheel_inertia_kg_m2: float
  cad_crank_mass_kg: float
  cad_cranktrain_inertia_kg_m2: float
  effective_rotational_inertia_kg_m2: float
  wot_pull_duration_s: float
  total_clip_duration_s: float
  peak_wheel_hp: float
  peak_wheel_kw: float
  peak_wheel_torque_lb_ft: float
  peak_wheel_torque_nm: float
  peak_wheel_rpm: float
  head_temp_start_c: float
  head_temp_end_c: float
  head_temp_rise_c: float
  oil_temp_start_c: float
  oil_temp_end_c: float
  oil_viscosity_start_cst: float
  oil_viscosity_end_cst: float
  fmep_start_bar: float
  fmep_end_bar: float
  flywheel_cs_coeff: float
  flywheel_burst_sf_redline: float
  peak_firing_torque_nm: float
  min_crank_press_fit_sf: float
  crank_press_fit_report: CrankPressFitReport
  time_s: np.ndarray
  theta_rad: np.ndarray
  rpm: np.ndarray
  throttle: np.ndarray
  wheel_power_hp: np.ndarray
  wheel_torque_nm: np.ndarray
  head_temp_c: np.ndarray
  oil_temp_c: np.ndarray
  oil_viscosity_cst: np.ndarray
  evo_pressure_pa: np.ndarray

  def to_dict(self) -> Dict[str, object]:
    """Serialize scalar KPIs and downsampled pull traces to JSON-safe dictionary."""
    sample_idx = np.linspace(0, len(self.time_s) - 1, min(EXPORT_TRACE_SAMPLES, len(self.time_s)), dtype=int)
    return {
        "drum_inertia_kg_m2": round(self.config.drum_inertia_kg_m2, ROUND_VALUE_DECIMALS),
        "cad_flywheel_mass_kg": round(self.cad_flywheel_mass_kg, ROUND_VALUE_DECIMALS),
        "cad_flywheel_inertia_kg_m2": round(self.cad_flywheel_inertia_kg_m2, ROUND_VALUE_DECIMALS),
        "cad_crank_mass_kg": round(self.cad_crank_mass_kg, ROUND_VALUE_DECIMALS),
        "cad_cranktrain_inertia_kg_m2": round(self.cad_cranktrain_inertia_kg_m2, ROUND_VALUE_DECIMALS),
        "effective_rotational_inertia_kg_m2": round(self.effective_rotational_inertia_kg_m2, ROUND_VALUE_DECIMALS),
        "wot_pull_duration_s": round(self.wot_pull_duration_s, ROUND_TWO_DECIMALS),
        "total_clip_duration_s": round(self.total_clip_duration_s, ROUND_TWO_DECIMALS),
        "peak_wheel_hp": round(self.peak_wheel_hp, ROUND_TWO_DECIMALS),
        "peak_wheel_kw": round(self.peak_wheel_kw, ROUND_TWO_DECIMALS),
        "peak_wheel_torque_lb_ft": round(self.peak_wheel_torque_lb_ft, ROUND_TWO_DECIMALS),
        "peak_wheel_torque_nm": round(self.peak_wheel_torque_nm, ROUND_TWO_DECIMALS),
        "peak_wheel_rpm": round(self.peak_wheel_rpm, ROUND_ONE_DECIMAL),
        "head_temp_start_c": round(self.head_temp_start_c, ROUND_ONE_DECIMAL),
        "head_temp_end_c": round(self.head_temp_end_c, ROUND_ONE_DECIMAL),
        "head_temp_rise_c": round(self.head_temp_rise_c, ROUND_TWO_DECIMALS),
        "oil_temp_start_c": round(self.oil_temp_start_c, ROUND_ONE_DECIMAL),
        "oil_temp_end_c": round(self.oil_temp_end_c, ROUND_ONE_DECIMAL),
        "oil_viscosity_start_cst": round(self.oil_viscosity_start_cst, ROUND_TWO_DECIMALS),
        "oil_viscosity_end_cst": round(self.oil_viscosity_end_cst, ROUND_TWO_DECIMALS),
        "fmep_start_bar": round(self.fmep_start_bar, ROUND_VALUE_DECIMALS),
        "fmep_end_bar": round(self.fmep_end_bar, ROUND_VALUE_DECIMALS),
        "flywheel_cs_coeff": round(self.flywheel_cs_coeff, ROUND_VALUE_DECIMALS),
        "flywheel_burst_sf_redline": round(self.flywheel_burst_sf_redline, ROUND_TWO_DECIMALS),
        "peak_firing_torque_nm": round(self.peak_firing_torque_nm, ROUND_TWO_DECIMALS),
        "min_crank_press_fit_sf": round(self.min_crank_press_fit_sf, ROUND_TWO_DECIMALS),
        "crank_press_fit": self.crank_press_fit_report.to_dict(),
        "trace": {
            "time_s": [round(float(self.time_s[i]), ROUND_TWO_DECIMALS) for i in sample_idx],
            "rpm": [round(float(self.rpm[i]), ROUND_ONE_DECIMAL) for i in sample_idx],
            "throttle": [round(float(self.throttle[i]), ROUND_TWO_DECIMALS) for i in sample_idx],
            "wheel_power_hp": [round(float(self.wheel_power_hp[i]), ROUND_TWO_DECIMALS) for i in sample_idx],
            "wheel_torque_nm": [round(float(self.wheel_torque_nm[i]), ROUND_TWO_DECIMALS) for i in sample_idx],
            "head_temp_c": [round(float(self.head_temp_c[i]), ROUND_ONE_DECIMAL) for i in sample_idx],
            "oil_temp_c": [round(float(self.oil_temp_c[i]), ROUND_ONE_DECIMAL) for i in sample_idx],
            "oil_viscosity_cst": [round(float(self.oil_viscosity_cst[i]), ROUND_TWO_DECIMALS) for i in sample_idx],
        },
    }


def _evaluate_single_joint_press_fit(
    joint_name: str,
    outer_diam_m: float,
    inner_bore_diam_m: float,
    web_boss_outer_radius_m: float,
    engagement_length_m: float,
    rpm: float,
    web_temp_c: float,
    torque_demand_nm: float,
    spec: CrankPressFitSpec,
) -> JointPressFitMargin:
  """Compute Lamé interference pressure, centrifugal/thermal shifts, and slip SF."""
  r = HALF_FACTOR * outer_diam_m
  r_i = HALF_FACTOR * max(ZERO_FLOAT, inner_bore_diam_m)
  r_o = web_boss_outer_radius_m

  e_web = STEEL_1144_STRESSPROOF.youngs_modulus_mpa * MPA_TO_PA
  nu_web = STEEL_1144_STRESSPROOF.poissons_ratio
  alpha_web = STEEL_1144_STRESSPROOF.thermal_expansion_coeff
  rho_web = STEEL_1144_STRESSPROOF.density_tonne_mm3 * TONNE_MM3_TO_KG_M3

  e_pin = STEEL_4140_PREHARD.youngs_modulus_mpa * MPA_TO_PA
  nu_pin = STEEL_4140_PREHARD.poissons_ratio
  alpha_pin = STEEL_4140_PREHARD.thermal_expansion_coeff

  k_web = ((r_o * r_o) + (r * r)) / max(MIN_EPSILON, (r_o * r_o) - (r * r))
  if r_i > ZERO_FLOAT:
    k_pin = ((r * r) + (r_i * r_i)) / max(MIN_EPSILON, (r * r) - (r_i * r_i))
  else:
    k_pin = 1.0
  radial_comp = r * (((k_web + nu_web) / e_web) + ((k_pin - nu_pin) / e_pin))

  omega = max(ZERO_FLOAT, rpm) * (TWO_PI / SECONDS_PER_MINUTE)
  du_hub = ((rho_web * omega * omega * r) / (FOUR_FACTOR * e_web)) * (
      ((THREE_FACTOR + nu_web) * r_o * r_o) + ((1.0 - nu_web) * r * r)
  )
  du_pin = ((rho_web * omega * omega * r) / (FOUR_FACTOR * e_pin)) * (
      ((THREE_FACTOR + nu_pin) * r_i * r_i) + ((1.0 - nu_pin) * r * r)
  )
  du_cent = max(ZERO_FLOAT, du_hub - du_pin)
  du_therm = r * (alpha_pin - alpha_web) * (web_temp_c - spec.reference_temp_c)

  dr_min = max(ZERO_FLOAT, (HALF_FACTOR * spec.min_diametral_interference_m) - du_cent + du_therm)
  dr_nom = max(ZERO_FLOAT, (HALF_FACTOR * spec.nominal_diametral_interference_m) - du_cent + du_therm)
  dr_max = max(ZERO_FLOAT, (HALF_FACTOR * spec.max_diametral_interference_m) - du_cent + du_therm)

  p_min = dr_min / radial_comp
  p_nom = dr_nom / radial_comp
  p_max = dr_max / radial_comp

  t_coeff = TWO_PI * (r * r) * engagement_length_m * spec.friction_coeff
  t_slip_min = t_coeff * p_min
  t_slip_nom = t_coeff * p_nom

  denom_lame = max(MIN_EPSILON, (r_o * r_o) - (r * r))
  vm_factor = math.sqrt((THREE_FACTOR * (r_o**FOUR_FACTOR)) + (r**FOUR_FACTOR)) / denom_lame
  vm_max_mpa = (p_max * PA_TO_MPA) * vm_factor

  sf_min = t_slip_min / max(MIN_EPSILON, torque_demand_nm)
  sf_nom = t_slip_nom / max(MIN_EPSILON, torque_demand_nm)
  yield_sf = STEEL_1144_STRESSPROOF.yield_strength_mpa / max(MIN_EPSILON, vm_max_mpa)
  return JointPressFitMargin(
      joint_name=joint_name,
      rpm=float(rpm),
      torque_demand_nm=float(torque_demand_nm),
      min_contact_pressure_mpa=float(p_min * PA_TO_MPA),
      nom_contact_pressure_mpa=float(p_nom * PA_TO_MPA),
      max_bore_von_mises_mpa=float(vm_max_mpa),
      min_slip_torque_nm=float(t_slip_min),
      nom_slip_torque_nm=float(t_slip_nom),
      min_slip_safety_factor=float(sf_min),
      nom_slip_safety_factor=float(sf_nom),
      bore_yield_safety_factor=float(yield_sf),
      is_safe=(sf_min >= MIN_PRESS_FIT_SF_THRESHOLD) and (yield_sf >= MIN_PRESS_FIT_SF_THRESHOLD),
  )


def compute_crank_press_fit_margin(
    rpm: float,
    web_temp_c: float,
    peak_firing_torque_nm: float,
    spec: Optional[CrankPressFitSpec] = None,
    crank_params: Optional[CrankshaftParams] = None,
) -> CrankPressFitReport:
  """Evaluate 3-piece CAD crankshaft main-journal & crankpin press-fit slip margins."""
  fit = spec or CrankPressFitSpec()
  cp = crank_params or CrankshaftParams()

  r_main_boss_o = (HALF_FACTOR * cp.main_journal_diameter_m) + (MAIN_BOSS_MARGIN_MM * MM_TO_M)
  main_margin = _evaluate_single_joint_press_fit(
      joint_name="main_journal_30mm",
      outer_diam_m=cp.main_journal_diameter_m,
      inner_bore_diam_m=ZERO_FLOAT,
      web_boss_outer_radius_m=r_main_boss_o,
      engagement_length_m=cp.web_thickness_m,
      rpm=rpm,
      web_temp_c=web_temp_c,
      torque_demand_nm=peak_firing_torque_nm,
      spec=fit,
  )

  r_pin_boss_o = (HALF_FACTOR * cp.crankpin_diameter_m) + (CRANKPIN_BOSS_MARGIN_MM * MM_TO_M)
  pin_margin = _evaluate_single_joint_press_fit(
      joint_name="hollow_crankpin_28mm",
      outer_diam_m=cp.crankpin_diameter_m,
      inner_bore_diam_m=cp.crankpin_bore_diameter_m,
      web_boss_outer_radius_m=r_pin_boss_o,
      engagement_length_m=cp.web_thickness_m,
      rpm=rpm,
      web_temp_c=web_temp_c,
      torque_demand_nm=TWO_WEB_CRANKPIN_SHARE * peak_firing_torque_nm,
      spec=fit,
  )

  gov = main_margin if main_margin.min_slip_safety_factor <= pin_margin.min_slip_safety_factor else pin_margin
  yield_sf = min(main_margin.bore_yield_safety_factor, pin_margin.bore_yield_safety_factor)
  return CrankPressFitReport(
      fit_label=fit.fit_label,
      main_journal=main_margin,
      crankpin=pin_margin,
      governing_joint=gov.joint_name,
      min_slip_safety_factor=gov.min_slip_safety_factor,
      nom_slip_safety_factor=gov.nom_slip_safety_factor,
      bore_yield_safety_factor=yield_sf,
      is_safe=main_margin.is_safe and pin_margin.is_safe,
  )


@dataclass(frozen=True, slots=True)
class _DynoPlantModel:
  """Pre-tabulated indicated torque, 2D FMEP/oil-heat lookup, and thermal capacities."""

  geom: EngineGeometry
  oil_spec: OilGradeSpec
  i_eff_kg_m2: float
  head_thermal_cap_j_k: float
  oil_thermal_cap_j_k: float
  sump_ua_w_k: float
  wot_rpms: np.ndarray
  wot_ind_torque_nm: np.ndarray
  wot_cool_w: np.ndarray
  wot_evo_pa: np.ndarray
  grid_rpms: np.ndarray
  grid_temps_c: np.ndarray
  fmep_bar_grid: np.ndarray
  oil_heat_w_grid: np.ndarray


def _interp_2d(
    x_val: float,
    y_val: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
) -> float:
  """Bilinearly interpolate on ascending 1D grids x_grid (rows) and y_grid (cols)."""
  xc = float(np.clip(x_val, x_grid[0], x_grid[-1]))
  yc = float(np.clip(y_val, y_grid[0], y_grid[-1]))
  ix = min(len(x_grid) - 2, max(0, int(np.searchsorted(x_grid, xc, side="right")) - 1))
  iy = min(len(y_grid) - 2, max(0, int(np.searchsorted(y_grid, yc, side="right")) - 1))
  fx = (xc - float(x_grid[ix])) / max(MIN_EPSILON, float(x_grid[ix + 1] - x_grid[ix]))
  fy = (yc - float(y_grid[iy])) / max(MIN_EPSILON, float(y_grid[iy + 1] - y_grid[iy]))
  z00 = float(z_grid[ix, iy])
  z10 = float(z_grid[ix + 1, iy])
  z01 = float(z_grid[ix, iy + 1])
  z11 = float(z_grid[ix + 1, iy + 1])
  return ((1.0 - fx) * (((1.0 - fy) * z00) + (fy * z01))) + (fx * (((1.0 - fy) * z10) + (fy * z11)))


def _build_plant_model(
    steady: SteadyDynoSweepResult,
    sim: FourStrokeSimulator,
    config: InertiaPullConfig,
) -> _DynoPlantModel:
  """Precompute indicated tables and 2D FMEP/oil-heat lookup from live CAD & sweep."""
  _, fw_inertia, _, crank_inertia = get_cad_flywheel_and_crank_inertia()
  head_mass_kg, barrel_mass_kg = get_cad_head_and_barrel_masses_kg()
  sump_vol_l, dipper_imm_mm, dipper_rad_mm = get_cad_sump_splash_metrics()
  i_eff = fw_inertia + crank_inertia + config.drum_inertia_kg_m2

  thermal_mass_kg = head_mass_kg + (HALF_FACTOR * barrel_mass_kg)
  head_cap_j_k = thermal_mass_kg * AL_6061_CP_J_KG_K
  oil_ref_state = sim.oil_spec.evaluate(config.initial_oil_temp_c)
  oil_mass_kg = max(MIN_EPSILON, sump_vol_l * oil_ref_state.density_kg_m3 * KG_PER_L_APPROX_DENSITY)
  oil_cap_j_k = oil_mass_kg * OIL_CP_J_KG_K
  sump_ua = compute_sump_overall_ua_w_k(sim.thermal_params)

  vd = sim.geom.displacement
  wot_rpms = np.array([pt.rpm for pt in steady.wot_points], dtype=float)
  wot_ind_torque = np.array(
      [((pt.nimep_bar + pt.pmep_bar) * BAR_TO_PA * vd) / FOUR_PI for pt in steady.wot_points],
      dtype=float,
  )
  wot_cool_w = np.array([pt.mean_cooling_power_w for pt in steady.wot_points], dtype=float)
  wot_evo_pa = np.array([pt.evo_pressure_pa for pt in steady.wot_points], dtype=float)

  grid_rpms = np.linspace(wot_rpms[0], wot_rpms[-1], FMEP_GRID_RPM_POINTS)
  grid_temps = np.linspace(FMEP_GRID_TEMP_MIN_C, FMEP_GRID_TEMP_MAX_C, FMEP_GRID_TEMP_POINTS)
  fmep_grid = np.zeros((FMEP_GRID_RPM_POINTS, FMEP_GRID_TEMP_POINTS), dtype=float)
  heat_grid = np.zeros((FMEP_GRID_RPM_POINTS, FMEP_GRID_TEMP_POINTS), dtype=float)
  th_ref = np.array(steady.wot_cycle_theta_deg, dtype=float)
  splash_cfg = replace(
      sim.splash_params,
      sump_volume_l=sump_vol_l,
      dipper_immersion_mm=dipper_imm_mm,
      dipper_tip_radius_mm=dipper_rad_mm,
  )
  cam_model = SOHCCamshaft()

  for ix, rpm_val in enumerate(grid_rpms):
    closest_rpm = float(wot_rpms[int(np.argmin(np.abs(wot_rpms - rpm_val)))])
    p_trace = np.array(steady.wot_cycle_pressures_pa[closest_rpm], dtype=float)
    vt_mech_j = compute_valvetrain_mechanical_work_j(
        rpm=float(rpm_val),
        camshaft=cam_model,
        splash_params=splash_cfg,
    )
    for iy, t_oil in enumerate(grid_temps):
      oil_st = sim.oil_spec.evaluate(float(t_oil))
      splash_rep = evaluate_splash_transport(rpm=float(rpm_val), oil_temp_c=float(t_oil), oil_spec=sim.oil_spec, params=splash_cfg)
      fric = evaluate_engine_friction(
          geom=sim.geom,
          rpm=float(rpm_val),
          oil_temp_c=float(t_oil),
          theta_deg=th_ref,
          pressure_pa=p_trace,
          oil_spec=sim.oil_spec,
          splash_params=splash_cfg,
          camshaft=cam_model,
          precomputed_valvetrain_mech_work_j=vt_mech_j,
      )
      _, q_splash = compute_piston_splash_cooling(sim.geom, oil_st, splash_rep, sim.thermal_params)
      q_oil_fric = (
          fric.rolling_bearings_heat_w
          + fric.valvetrain_heat_w
          + (DIPPER_WINDAGE_TO_OIL_FRACTION * fric.dipper_windage_heat_w)
          + (RING_FRICTION_TO_OIL_FRACTION * fric.ring_skirt_heat_w)
      )
      fmep_grid[ix, iy] = fric.total_fmep_pa / BAR_TO_PA
      heat_grid[ix, iy] = q_oil_fric + q_splash + CRANKCASE_BARREL_CONDUCTION_W

  return _DynoPlantModel(
      geom=sim.geom,
      oil_spec=sim.oil_spec,
      i_eff_kg_m2=float(i_eff),
      head_thermal_cap_j_k=float(head_cap_j_k),
      oil_thermal_cap_j_k=float(oil_cap_j_k),
      sump_ua_w_k=float(sump_ua),
      wot_rpms=wot_rpms,
      wot_ind_torque_nm=wot_ind_torque,
      wot_cool_w=wot_cool_w,
      wot_evo_pa=wot_evo_pa,
      grid_rpms=grid_rpms,
      grid_temps_c=grid_temps,
      fmep_bar_grid=fmep_grid,
      oil_heat_w_grid=heat_grid,
  )


def _eval_plant_derivatives(
    plant: _DynoPlantModel,
    config: InertiaPullConfig,
    state: np.ndarray,
    is_wot: bool,
) -> Tuple[np.ndarray, float, float]:
  """Return d/dt[theta, omega, T_head, T_oil], wheel_torque_nm, and evo_pa."""
  omega = max(MIN_OMEGA_RAD_S, float(state[STATE_OMEGA_IDX]))
  t_head = float(state[STATE_HEAD_TEMP_IDX])
  t_oil = float(state[STATE_OIL_TEMP_IDX])
  rpm = omega * (SECONDS_PER_MINUTE / TWO_PI)

  fmep_bar = _interp_2d(rpm, t_oil, plant.grid_rpms, plant.grid_temps_c, plant.fmep_bar_grid)
  oil_heat_in = _interp_2d(rpm, t_oil, plant.grid_rpms, plant.grid_temps_c, plant.oil_heat_w_grid)
  fric_nm = (fmep_bar * BAR_TO_PA * plant.geom.displacement) / FOUR_PI
  roller_drag = DRUM_BEARING_DRAG_NM + (DRUM_WINDAGE_COEFF * omega)

  if is_wot:
    t_ind = float(np.interp(rpm, plant.wot_rpms, plant.wot_ind_torque_nm))
    q_comb_wall = HEAD_COOLING_SHARE * float(np.interp(rpm, plant.wot_rpms, plant.wot_cool_w))
    evo_pa = float(np.interp(rpm, plant.wot_rpms, plant.wot_evo_pa))
  else:
    p_pump_bar = OVERRUN_PUMPING_BASE_BAR + (OVERRUN_PUMPING_RPM_BAR * (rpm / OVERRUN_PUMPING_REF_RPM))
    t_ind = -(p_pump_bar * BAR_TO_PA * plant.geom.displacement) / FOUR_PI
    q_comb_wall = ZERO_FLOAT
    evo_pa = float(plant.wot_evo_pa[0]) * HALF_FACTOR

  t_wheel = t_ind - fric_nm - roller_drag
  d_omega = t_wheel / plant.i_eff_kg_m2
  ua_head = HEAD_AIR_UA_BASE_W_K + (HEAD_AIR_UA_RPM_SLOPE_W_K * (rpm / HEAD_AIR_REF_RPM))
  d_head = (q_comb_wall - (ua_head * (t_head - config.ambient_air_temp_c))) / plant.head_thermal_cap_j_k
  d_oil = (oil_heat_in - (plant.sump_ua_w_k * (t_oil - config.ambient_air_temp_c))) / plant.oil_thermal_cap_j_k
  return np.array([omega, d_omega, d_head, d_oil], dtype=float), float(t_wheel), evo_pa


def _rk4_step_state(
    plant: _DynoPlantModel,
    config: InertiaPullConfig,
    state: np.ndarray,
    dt_s: float,
    is_wot: bool,
) -> Tuple[np.ndarray, float, float]:
  """Advance dyno state vector by one RK4 step."""
  k1, tq_now, evo_now = _eval_plant_derivatives(plant, config, state, is_wot)
  k2, _, _ = _eval_plant_derivatives(plant, config, state + (HALF_FACTOR * dt_s * k1), is_wot)
  k3, _, _ = _eval_plant_derivatives(plant, config, state + (HALF_FACTOR * dt_s * k2), is_wot)
  k4, _, _ = _eval_plant_derivatives(plant, config, state + (dt_s * k3), is_wot)
  nxt = state + ((dt_s / SIX_FACTOR) * (k1 + (TWO_FACTOR * k2) + (TWO_FACTOR * k3) + k4))
  nxt[STATE_OMEGA_IDX] = max(MIN_OMEGA_RAD_S, float(nxt[STATE_OMEGA_IDX]))
  return nxt, tq_now, evo_now


def _evaluate_flywheel_and_press_fit(
    steady: SteadyDynoSweepResult,
    geom: EngineGeometry,
    fw_plus_crank_inertia: float,
    end_head_temp_c: float,
    config: InertiaPullConfig,
) -> Tuple[float, float, float, CrankPressFitReport]:
  """Compute redline FlywheelDynamics C_s/burst SF and governing crank shrink-fit SF."""
  dyn = SliderCrankDynamics(geom=geom)
  th_arr = np.array(steady.wot_cycle_theta_deg, dtype=float)
  fw_dyn = FlywheelDynamics(FlywheelDimensions())

  worst_report: Optional[CrankPressFitReport] = None
  peak_firing_tq = ZERO_FLOAT
  for pt in steady.wot_points:
    p_arr = np.array(steady.wot_cycle_pressures_pa[pt.rpm], dtype=float)
    forces = dyn.solve_forces(rpm=pt.rpm, theta_deg=th_arr, pressure_pa=p_arr)
    peak_firing_tq = max(peak_firing_tq, float(forces.peak_torque_nm))
    rep = compute_crank_press_fit_margin(
        rpm=pt.rpm,
        web_temp_c=end_head_temp_c,
        peak_firing_torque_nm=float(forces.peak_torque_nm),
        spec=config.press_fit_spec,
    )
    if worst_report is None or rep.min_slip_safety_factor < worst_report.min_slip_safety_factor:
      worst_report = rep

  peak_rpm = steady.peak_power_rpm
  p_peak = np.array(steady.wot_cycle_pressures_pa[peak_rpm], dtype=float)
  forces_peak = dyn.solve_forces(rpm=peak_rpm, theta_deg=th_arr, pressure_pa=p_peak)
  e_fluc, _, _ = fw_dyn.integrate_energy(th_arr, forces_peak.crankshaft_torque_nm)
  _, _, actual_cs = fw_dyn.evaluate_speed_trajectory(
      mean_rpm=peak_rpm,
      energy_j=e_fluc,
      total_inertia_kg_m2=fw_plus_crank_inertia,
  )
  fw_sizing = fw_dyn.size_flywheel(
      rated_rpm=peak_rpm,
      theta_deg=th_arr,
      torque_nm=forces_peak.crankshaft_torque_nm,
      target_coeff_fluctuation=FLYWHEEL_TARGET_CS,
      redline_rpm=config.end_rpm,
  )
  assert worst_report is not None
  return actual_cs, fw_sizing.burst_safety_factor_redline, peak_firing_tq, worst_report


def run_inertia_dyno_pull(
    steady_result: SteadyDynoSweepResult,
    config: Optional[InertiaPullConfig] = None,
    sim: Optional[FourStrokeSimulator] = None,
) -> InertiaPullResult:
  """Integrate transient WOT inertia pull (2000->7000 RPM) and closed-throttle overrun."""
  cfg = config or InertiaPullConfig()
  simulator = sim or build_live_twin_simulator()
  plant = _build_plant_model(steady=steady_result, sim=simulator, config=cfg)

  omega_start = cfg.start_rpm * (TWO_PI / SECONDS_PER_MINUTE)
  omega_end = cfg.end_rpm * (TWO_PI / SECONDS_PER_MINUTE)
  state = np.array([ZERO_FLOAT, omega_start, cfg.initial_head_temp_c, cfg.initial_oil_temp_c], dtype=float)

  t_curr = ZERO_FLOAT
  times, thetas, rpms, throttles = [], [], [], []
  whp_list, tq_list, thead_list, toil_list, visc_list, evo_list = [], [], [], [], [], []

  def _record_step(st: np.ndarray, t_val: float, thr: float, tq_nm: float, evo_val: float) -> None:
    rpm_v = float(st[STATE_OMEGA_IDX]) * (SECONDS_PER_MINUTE / TWO_PI)
    p_kw = (tq_nm * float(st[STATE_OMEGA_IDX])) / WATT_TO_KW
    times.append(t_val)
    thetas.append(float(st[STATE_THETA_IDX]))
    rpms.append(rpm_v)
    throttles.append(thr)
    tq_list.append(tq_nm)
    whp_list.append(p_kw * KW_TO_MECHANICAL_HP)
    thead_list.append(float(st[STATE_HEAD_TEMP_IDX]))
    toil_list.append(float(st[STATE_OIL_TEMP_IDX]))
    visc_list.append(float(plant.oil_spec.evaluate(float(st[STATE_OIL_TEMP_IDX])).kinematic_viscosity_cst))
    evo_list.append(evo_val)

  while state[STATE_OMEGA_IDX] < omega_end and t_curr < MAX_PULL_DURATION_S:
    nxt, tq_now, evo_now = _rk4_step_state(plant, cfg, state, cfg.dt_s, is_wot=True)
    _record_step(state, t_curr, 1.0, tq_now, evo_now)
    state = nxt
    t_curr += cfg.dt_s

  wot_duration_s = t_curr
  overrun_end_t = wot_duration_s + cfg.overrun_duration_s
  while t_curr <= overrun_end_t:
    nxt, tq_now, evo_now = _rk4_step_state(plant, cfg, state, cfg.dt_s, is_wot=False)
    _record_step(state, t_curr, CLOSED_THROTTLE_FRACTION, tq_now, evo_now)
    state = nxt
    t_curr += cfg.dt_s

  fw_mass, fw_I, crank_mass, crank_I = get_cad_flywheel_and_crank_inertia()
  cs_coeff, burst_sf, peak_firing_tq, press_rep = _evaluate_flywheel_and_press_fit(
      steady=steady_result,
      geom=simulator.geom,
      fw_plus_crank_inertia=fw_I + crank_I,
      end_head_temp_c=thead_list[LAST_INDEX],
      config=cfg,
  )

  whp_arr = np.asarray(whp_list, dtype=float)
  tq_arr = np.asarray(tq_list, dtype=float)
  rpm_arr = np.asarray(rpms, dtype=float)
  pk_hp_idx = int(np.argmax(whp_arr))
  peak_hp = float(whp_arr[pk_hp_idx])
  peak_tq_nm = float(np.max(tq_arr))
  fmep_start = _interp_2d(cfg.start_rpm, toil_list[0], plant.grid_rpms, plant.grid_temps_c, plant.fmep_bar_grid)
  fmep_end = _interp_2d(rpm_arr[-1], toil_list[-1], plant.grid_rpms, plant.grid_temps_c, plant.fmep_bar_grid)

  return InertiaPullResult(
      config=cfg,
      cad_flywheel_mass_kg=fw_mass,
      cad_flywheel_inertia_kg_m2=fw_I,
      cad_crank_mass_kg=crank_mass,
      cad_cranktrain_inertia_kg_m2=crank_I,
      effective_rotational_inertia_kg_m2=plant.i_eff_kg_m2,
      wot_pull_duration_s=float(wot_duration_s),
      total_clip_duration_s=float(times[-1]),
      peak_wheel_hp=peak_hp,
      peak_wheel_kw=peak_hp / KW_TO_MECHANICAL_HP,
      peak_wheel_torque_lb_ft=peak_tq_nm * NM_TO_LB_FT,
      peak_wheel_torque_nm=peak_tq_nm,
      peak_wheel_rpm=float(rpm_arr[pk_hp_idx]),
      head_temp_start_c=float(thead_list[0]),
      head_temp_end_c=float(thead_list[-1]),
      head_temp_rise_c=float(thead_list[-1] - thead_list[0]),
      oil_temp_start_c=float(toil_list[0]),
      oil_temp_end_c=float(toil_list[-1]),
      oil_viscosity_start_cst=float(visc_list[0]),
      oil_viscosity_end_cst=float(visc_list[-1]),
      fmep_start_bar=float(fmep_start),
      fmep_end_bar=float(fmep_end),
      flywheel_cs_coeff=float(cs_coeff),
      flywheel_burst_sf_redline=float(burst_sf),
      peak_firing_torque_nm=float(peak_firing_tq),
      min_crank_press_fit_sf=float(press_rep.min_slip_safety_factor),
      crank_press_fit_report=press_rep,
      time_s=np.asarray(times, dtype=float),
      theta_rad=np.asarray(thetas, dtype=float),
      rpm=rpm_arr,
      throttle=np.asarray(throttles, dtype=float),
      wheel_power_hp=whp_arr,
      wheel_torque_nm=tq_arr,
      head_temp_c=np.asarray(thead_list, dtype=float),
      oil_temp_c=np.asarray(toil_list, dtype=float),
      oil_viscosity_cst=np.asarray(visc_list, dtype=float),
      evo_pressure_pa=np.asarray(evo_list, dtype=float),
  )
