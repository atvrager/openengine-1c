"""Two-Zone Adiabatic End-Gas and Livengood-Wu Auto-Ignition Knock Model.

References:
  - Livengood, J. C., and Wu, P. C., 'Correlation of Autoignition Phenomena in
    Internal Combustion Engines and Rapid Compression Machines,' Symposium
    (International) on Combustion, Vol. 5, pp. 347-356, 1955.
  - Douaud, A. M., and Eyzat, P., 'Four-Octane-Number Method for Predicting the
    Anti-Knock Behavior of Fuels and Engines,' SAE Technical Paper 780080, 1978.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from openengine.thermo.cantera_kinetics import (
    ETHANOL_LATENT_HEAT_J_KG,
    HYDROCARBON_LATENT_HEAT_J_KG,
    PERCENT_SCALE,
    STD_ATM_PA,
    US_87_AKI,
    FuelBlendSpec,
)
from openengine.thermo.cycle import wiebe_burned_fraction

# Douaud-Eyzat (SAE 780080) auto-ignition delay constants
DOUAUD_EYZAT_PREFACTOR_S: float = 0.01768
DOUAUD_EYZAT_OCTANE_EXP: float = 3.402
DOUAUD_EYZAT_PRESSURE_EXP: float = -1.7
DOUAUD_EYZAT_ACTIVATION_TEMP_K: float = 3800.0
OCTANE_REFERENCE: float = 100.0

# Livengood-Wu auto-ignition thresholds
KNOCK_INTEGRAL_CRITICAL: float = 1.0
DEFAULT_KNOCK_SAFETY_CEILING: float = 0.88
END_GAS_BURN_CUTOFF: float = 0.98

# Unburned gasoline-air mixture gamma(T) fit calibrated to Cantera PRF+E10
UNBURNED_GAMMA_REF_TEMP_K: float = 300.0
UNBURNED_GAMMA_AT_REF: float = 1.354
UNBURNED_GAMMA_SLOPE_PER_K: float = 8.5e-5
UNBURNED_GAMMA_MIN: float = 1.245
UNBURNED_GAMMA_MAX: float = 1.390

# Cycle window and numerical bounds
FULL_CYCLE_DEG: float = 720.0
HALF_CYCLE_DEG: float = 360.0
EXPANSION_WINDOW_END_DEG: float = 90.0
DEFAULT_IVC_DEG: float = 585.0
SECONDS_PER_MINUTE: float = 60.0
TWO_PI: float = 2.0 * float(np.pi)
MIN_PRESSURE_ATM: float = 0.5
MIN_END_GAS_TEMP_K: float = 250.0
MIN_PRESSURE_PA: float = 1000.0
MIN_OCTANE_NUMBER: float = 10.0
MIN_RPM: float = 100.0
CHARGE_CP_REF_J_KG_K: float = 1050.0
VAPORIZATION_CHARGE_FRACTION: float = 0.45
IGNITION_DELAY_DEG_OFFSET: float = 4.0
INTERP_EPSILON: float = 1.0e-12
TRAPEZOID_HALF_WEIGHT: float = 0.5
DEFAULT_WIEBE_SOC_DEG: float = -15.0
DEFAULT_WIEBE_DURATION_DEG: float = 42.0
DEFAULT_WIEBE_A: float = 5.0
DEFAULT_WIEBE_M: float = 2.0


@dataclass(frozen=True, slots=True)
class KnockEvaluationResult:
  """End-gas temperature history and Livengood-Wu auto-ignition evaluation."""

  crank_angle_deg: np.ndarray
  pressure_pa: np.ndarray
  end_gas_temp_k: np.ndarray
  burned_fraction: np.ndarray
  ignition_delay_s: np.ndarray
  knock_integral_trace: np.ndarray
  peak_end_gas_temp_k: float
  knock_integral: float
  knock_margin: float
  is_knocking: bool
  knock_onset_crank_angle_deg: Optional[float]
  unburned_fraction_at_knock: float
  fuel_aki: float


@dataclass(frozen=True, slots=True)
class KnockSweepPoint:
  """Single (compression_ratio, rpm, spark_timing) knock and output point."""

  compression_ratio: float
  rpm: float
  spark_advance_deg_btdc: float
  wiebe_soc_deg: float
  nimep_pa: float
  power_kw: float
  torque_nm: float
  p_max_pa: float
  end_gas_temp_max_k: float
  knock_integral: float
  knock_margin: float
  is_knocking: bool


@dataclass(frozen=True, slots=True)
class KnockSafeOperatingMap:
  """Knock-limited compression ratio and speed-scheduled spark advance map."""

  selected_compression_ratio: float
  spark_schedule_deg_btdc: Dict[float, float]
  soc_schedule_deg: Dict[float, float]
  worst_selected_knock_integral: float
  points: Tuple[KnockSweepPoint, ...]


def unburned_mixture_gamma(temp_k: np.ndarray | float) -> np.ndarray | float:
  """Evaluate temperature-dependent unburned E10-air specific heat ratio."""
  t_arr = np.asarray(temp_k, dtype=float)
  delta_t = np.maximum(0.0, t_arr - UNBURNED_GAMMA_REF_TEMP_K)
  gamma_val = UNBURNED_GAMMA_AT_REF - UNBURNED_GAMMA_SLOPE_PER_K * delta_t
  clipped = np.clip(gamma_val, UNBURNED_GAMMA_MIN, UNBURNED_GAMMA_MAX)
  if np.ndim(temp_k) == 0:
    return float(clipped)
  return clipped


def ethanol_evaporative_cooling_delta_k(fuel: FuelBlendSpec) -> float:
  """Compute extra intake charge evaporative cooling [K] from ethanol content."""
  delta_h_fg = max(0.0, fuel.latent_heat_j_kg - HYDROCARBON_LATENT_HEAT_J_KG)
  fuel_mass_per_kg_charge = 1.0 / (1.0 + fuel.stoich_afr)
  return float(
      (VAPORIZATION_CHARGE_FRACTION * fuel_mass_per_kg_charge * delta_h_fg)
      / CHARGE_CP_REF_J_KG_K
  )


def douaud_eyzat_ignition_delay_s(
    pressure_pa: np.ndarray | float,
    end_gas_temp_k: np.ndarray | float,
    octane_number: float = US_87_AKI,
) -> np.ndarray | float:
  """Calculate auto-ignition delay tau [s] via the Douaud-Eyzat correlation."""
  p_atm = np.maximum(MIN_PRESSURE_ATM, np.asarray(pressure_pa, dtype=float) / STD_ATM_PA)
  t_clamped = np.maximum(MIN_END_GAS_TEMP_K, np.asarray(end_gas_temp_k, dtype=float))
  aki_clamped = max(MIN_OCTANE_NUMBER, float(octane_number))

  octane_term = (aki_clamped / OCTANE_REFERENCE) ** DOUAUD_EYZAT_OCTANE_EXP
  pressure_term = p_atm ** DOUAUD_EYZAT_PRESSURE_EXP
  arrhenius_term = np.exp(DOUAUD_EYZAT_ACTIVATION_TEMP_K / t_clamped)

  tau_s = DOUAUD_EYZAT_PREFACTOR_S * octane_term * pressure_term * arrhenius_term
  if np.ndim(pressure_pa) == 0 and np.ndim(end_gas_temp_k) == 0:
    return float(tau_s)
  return tau_s


def compute_adiabatic_end_gas_temperature(
    pressure_pa: np.ndarray,
    t_initial_k: float,
) -> np.ndarray:
  """Integrate adiabatic unburned end-gas core temperature T_u(theta) [K]."""
  p_arr = np.maximum(MIN_PRESSURE_PA, np.asarray(pressure_pa, dtype=float))
  tu = np.empty_like(p_arr, dtype=float)
  tu[0] = max(MIN_END_GAS_TEMP_K, float(t_initial_k))

  for idx in range(len(p_arr) - 1):
    gamma_u = float(unburned_mixture_gamma(tu[idx]))
    exponent = (gamma_u - 1.0) / gamma_u
    tu[idx + 1] = tu[idx] * ((p_arr[idx + 1] / p_arr[idx]) ** exponent)

  return tu


def _extract_high_pressure_window(
    theta_deg: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    ivc_deg: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
  """Extract monotonic [-180, +90] deg CA high-pressure window from cycle traces."""
  th = np.asarray(theta_deg, dtype=float)
  p = np.asarray(pressure_pa, dtype=float)
  t = np.asarray(temperature_k, dtype=float)

  if float(np.max(th)) > HALF_CYCLE_DEG:
    # Full 0..720 deg cycle where 0/720 is Firing TDC
    comp_mask = th >= ivc_deg
    exp_mask = th <= EXPANSION_WINDOW_END_DEG
    th_hp = np.concatenate([th[comp_mask] - FULL_CYCLE_DEG, th[exp_mask]])
    p_hp = np.concatenate([p[comp_mask], p[exp_mask]])
    t_hp = np.concatenate([t[comp_mask], t[exp_mask]])
  else:
    # Closed-cycle trace already centered around Firing TDC (0 deg)
    ivc_closed = ivc_deg - FULL_CYCLE_DEG if ivc_deg > HALF_CYCLE_DEG else ivc_deg
    mask = (th >= ivc_closed) & (th <= EXPANSION_WINDOW_END_DEG)
    th_hp = th[mask]
    p_hp = p[mask]
    t_hp = t[mask]

  order = np.argsort(th_hp)
  th_sorted = th_hp[order]
  p_sorted = p_hp[order]
  t_sorted = t_hp[order]

  unique_mask = np.concatenate(([True], np.diff(th_sorted) > 0.0))
  return th_sorted[unique_mask], p_sorted[unique_mask], float(t_sorted[unique_mask][0])


def _find_knock_onset(
    crank_angle_deg: np.ndarray,
    knock_integral_trace: np.ndarray,
    burned_fraction: np.ndarray,
) -> Tuple[Optional[float], float]:
  """Interpolate crank angle and unburned fraction where knock integral hits 1.0."""
  exceed_indices = np.flatnonzero(knock_integral_trace >= KNOCK_INTEGRAL_CRITICAL)
  if len(exceed_indices) == 0:
    return None, 0.0

  idx = int(exceed_indices[0])
  if idx == 0:
    return float(crank_angle_deg[0]), float(1.0 - burned_fraction[0])

  i0 = float(knock_integral_trace[idx - 1])
  i1 = float(knock_integral_trace[idx])
  denom = max(INTERP_EPSILON, i1 - i0)
  frac = float(np.clip((KNOCK_INTEGRAL_CRITICAL - i0) / denom, 0.0, 1.0))

  th_onset = float(
      crank_angle_deg[idx - 1] + frac * (crank_angle_deg[idx] - crank_angle_deg[idx - 1])
  )
  xb_onset = float(
      burned_fraction[idx - 1] + frac * (burned_fraction[idx] - burned_fraction[idx - 1])
  )
  return th_onset, float(max(0.0, 1.0 - xb_onset))


def evaluate_cycle_knock(
    theta_deg: np.ndarray,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    rpm: float,
    fuel: Optional[FuelBlendSpec] = None,
    wiebe_soc_deg: float = DEFAULT_WIEBE_SOC_DEG,
    wiebe_duration_deg: float = DEFAULT_WIEBE_DURATION_DEG,
    wiebe_a: float = DEFAULT_WIEBE_A,
    wiebe_m: float = DEFAULT_WIEBE_M,
    ivc_deg: float = DEFAULT_IVC_DEG,
) -> KnockEvaluationResult:
  """Evaluate end-gas temperature and Livengood-Wu knock integral over a cycle."""
  fuel_spec = fuel or FuelBlendSpec.us_87_e10()
  th_hp, p_hp, t_ivc = _extract_high_pressure_window(
      theta_deg=theta_deg,
      pressure_pa=pressure_pa,
      temperature_k=temperature_k,
      ivc_deg=ivc_deg,
  )

  evap_cooling_k = ethanol_evaporative_cooling_delta_k(fuel_spec)
  t_ivc_adjusted = max(MIN_END_GAS_TEMP_K, t_ivc - evap_cooling_k)
  tu_trace = compute_adiabatic_end_gas_temperature(p_hp, t_ivc_adjusted)

  xb_trace = np.asarray(
      wiebe_burned_fraction(
          th_hp,
          theta_soc_deg=wiebe_soc_deg,
          duration_deg=wiebe_duration_deg,
          a=wiebe_a,
          m=wiebe_m,
      ),
      dtype=float,
  )

  tau_trace = np.asarray(
      douaud_eyzat_ignition_delay_s(p_hp, tu_trace, octane_number=fuel_spec.aki),
      dtype=float,
  )

  unburned_mask = xb_trace < END_GAS_BURN_CUTOFF
  inv_tau = np.where(unburned_mask, 1.0 / tau_trace, 0.0)

  omega_rad_s = max(MIN_RPM, float(rpm)) * (TWO_PI / SECONDS_PER_MINUTE)
  time_s = np.radians(th_hp) / omega_rad_s

  lw_trace = np.zeros_like(th_hp, dtype=float)
  if len(th_hp) > 1:
    dt_steps = np.diff(time_s)
    mid_rates = TRAPEZOID_HALF_WEIGHT * (inv_tau[:-1] + inv_tau[1:])
    lw_trace[1:] = np.cumsum(mid_rates * dt_steps)

  peak_tu = (
      float(np.max(tu_trace[unburned_mask]))
      if np.any(unburned_mask)
      else float(np.max(tu_trace))
  )
  final_integral = float(lw_trace[-1]) if len(lw_trace) > 0 else 0.0
  is_knocking = final_integral >= KNOCK_INTEGRAL_CRITICAL
  onset_deg, unburned_at_knock = _find_knock_onset(th_hp, lw_trace, xb_trace)

  return KnockEvaluationResult(
      crank_angle_deg=th_hp,
      pressure_pa=p_hp,
      end_gas_temp_k=tu_trace,
      burned_fraction=xb_trace,
      ignition_delay_s=tau_trace,
      knock_integral_trace=lw_trace,
      peak_end_gas_temp_k=peak_tu,
      knock_integral=final_integral,
      knock_margin=float(KNOCK_INTEGRAL_CRITICAL - final_integral),
      is_knocking=is_knocking,
      knock_onset_crank_angle_deg=onset_deg,
      unburned_fraction_at_knock=unburned_at_knock,
      fuel_aki=fuel_spec.aki,
  )


def soc_from_spark_advance(
    spark_advance_deg_btdc: float,
    ignition_delay_deg: float = IGNITION_DELAY_DEG_OFFSET,
) -> float:
  """Convert spark advance [deg BTDC] to Wiebe start-of-combustion angle [deg CA]."""
  return float(-spark_advance_deg_btdc + ignition_delay_deg)


@dataclass(frozen=True, slots=True)
class _IndexedSweepPoints:
  """Pre-indexed O(1) lookup tables over evaluated knock sweep points."""

  crs_desc: Tuple[float, ...]
  rpms_asc: Tuple[float, ...]
  best_safe_by_key: Dict[Tuple[float, float], KnockSweepPoint]
  safest_by_key: Dict[Tuple[float, float], KnockSweepPoint]


def _index_sweep_points(
    points: Sequence[KnockSweepPoint],
    max_knock_integral: float,
) -> _IndexedSweepPoints:
  """Build O(1) per-(CR, RPM) lookup dicts for best-NIMEP and safest points."""
  crs_set = set()
  rpms_set = set()
  best_safe: Dict[Tuple[float, float], KnockSweepPoint] = {}
  safest: Dict[Tuple[float, float], KnockSweepPoint] = {}

  for pt in points:
    crs_set.add(pt.compression_ratio)
    rpms_set.add(pt.rpm)
    key = (pt.compression_ratio, pt.rpm)

    prev_safest = safest.get(key)
    if prev_safest is None or pt.knock_integral < prev_safest.knock_integral:
      safest[key] = pt

    if pt.knock_integral <= max_knock_integral:
      prev_best = best_safe.get(key)
      if prev_best is None or pt.nimep_pa > prev_best.nimep_pa:
        best_safe[key] = pt

  return _IndexedSweepPoints(
      crs_desc=tuple(sorted(crs_set, reverse=True)),
      rpms_asc=tuple(sorted(rpms_set)),
      best_safe_by_key=best_safe,
      safest_by_key=safest,
  )


def _build_schedule_for_cr(
    cr: float,
    rpms: Sequence[float],
    lookup: Dict[Tuple[float, float], KnockSweepPoint],
    points: Sequence[KnockSweepPoint],
) -> Optional[KnockSafeOperatingMap]:
  """Assemble a speed-scheduled spark map if every speed has an entry in lookup."""
  spark_sched: Dict[float, float] = {}
  soc_sched: Dict[float, float] = {}
  worst_lw = 0.0

  for rpm in rpms:
    chosen = lookup.get((cr, rpm))
    if chosen is None:
      return None
    spark_sched[rpm] = chosen.spark_advance_deg_btdc
    soc_sched[rpm] = chosen.wiebe_soc_deg
    worst_lw = max(worst_lw, chosen.knock_integral)

  return KnockSafeOperatingMap(
      selected_compression_ratio=cr,
      spark_schedule_deg_btdc=spark_sched,
      soc_schedule_deg=soc_sched,
      worst_selected_knock_integral=worst_lw,
      points=tuple(points),
  )


def select_best_knock_safe_map(
    points: Sequence[KnockSweepPoint],
    max_knock_integral: float = DEFAULT_KNOCK_SAFETY_CEILING,
) -> KnockSafeOperatingMap:
  """Select highest-CR knock-safe operating map from evaluated sweep points.

  A compression ratio qualifies if every evaluated engine speed has at least one
  spark timing whose Livengood-Wu knock integral stays at or below
  max_knock_integral. At each speed, the spark timing that maximizes NIMEP
  within the knock ceiling wins.
  """
  if not points:
    raise ValueError("Sweep point sequence must not be empty.")

  indexed = _index_sweep_points(points, max_knock_integral=max_knock_integral)
  for cr in indexed.crs_desc:
    candidate = _build_schedule_for_cr(
        cr=cr,
        rpms=indexed.rpms_asc,
        lookup=indexed.best_safe_by_key,
        points=points,
    )
    if candidate is not None:
      return candidate

  fallback = _build_schedule_for_cr(
      cr=indexed.crs_desc[-1],
      rpms=indexed.rpms_asc,
      lookup=indexed.safest_by_key,
      points=points,
  )
  assert fallback is not None
  return fallback
