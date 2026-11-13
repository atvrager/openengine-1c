"""Hydrodynamic Journal Bearing Lubrication and PV Orbit Analysis.

Implements Ocvirk's short-bearing approximation (L/D <= 0.5) for engine connecting rod
big-end bearings and crankshaft main journal bearings. Solves instantaneous load capacity,
Sommerfeld and Ocvirk numbers, journal eccentricity ratio, minimum oil film thickness,
and Lambda ratio across the 720-degree engine cycle.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import root_scalar


@dataclass
class BearingGeometry:
  """Dimensions and clearances for a cylindrical journal bearing."""

  name: str
  journal_diameter_m: float  # e.g. 0.030 m (30 mm)
  bearing_width_m: float  # e.g. 0.016 m (16 mm)
  radial_clearance_m: float = 20.0e-6  # e.g. 20 um (c = 0.020 mm)
  journal_roughness_rms_m: float = 0.20e-6  # Ra/Rq ~ 0.2 um ground & polished steel
  bearing_roughness_rms_m: float = 0.40e-6  # Ra/Rq ~ 0.4 um bi-metal / tri-metal liner
  max_allowable_pressure_mpa: float = 75.0  # Peak allowable pressure (tri-metal / Al-Sn)

  @property
  def radius_m(self) -> float:
    return self.journal_diameter_m / 2.0

  @property
  def length_to_diameter_ratio(self) -> float:
    return self.bearing_width_m / self.journal_diameter_m

  @property
  def projected_area_m2(self) -> float:
    return self.journal_diameter_m * self.bearing_width_m

  @property
  def composite_roughness_m(self) -> float:
    """RMS composite surface roughness R_q = sqrt(R_j^2 + R_b^2)."""
    return float(np.sqrt(self.journal_roughness_rms_m**2 + self.bearing_roughness_rms_m**2))


@dataclass
class OilProperties:
  """Engine lubricant physical properties."""

  name: str = "SAE 10W-40 @ 100C"
  dynamic_viscosity_pa_s: float = 0.010  # 10 mPa.s (10 cP) at 100 deg C
  density_kg_m3: float = 850.0  # 850 kg/m^3


@dataclass
class BearingCycleState:
  """Instantaneous bearing lubrication state as a function of crank angle."""

  theta_deg: np.ndarray
  radial_load_n: np.ndarray
  pressure_mpa: np.ndarray
  sliding_velocity_m_s: float
  pv_factor_mpa_m_s: np.ndarray
  eccentricity_ratio: np.ndarray
  film_thickness_um: np.ndarray
  lambda_ratio: np.ndarray


@dataclass
class BearingReport:
  """Summary performance and safety assessment for an engine journal bearing."""

  bearing_name: str
  rpm: float
  peak_load_n: float
  peak_pressure_mpa: float
  allowable_pressure_mpa: float
  mean_pressure_mpa: float
  sliding_velocity_m_s: float
  peak_pv_mpa_m_s: float
  mean_pv_mpa_m_s: float
  min_film_thickness_um: float
  min_lambda_ratio: float
  peak_eccentricity_ratio: float
  is_hydrodynamic: bool
  is_safe: bool


def ocvirk_load_capacity(eps: float) -> float:
  """Ocvirk short bearing capacity function O_n(eps).

  O_n = ( (1 - eps^2)^2 ) / ( pi * eps * sqrt(16 * eps^2 + pi^2 * (1 - eps^2)) )
  """
  eps_clamped = np.clip(eps, 1.0e-5, 0.9999)
  num = (1.0 - eps_clamped**2)**2
  denom = np.pi * eps_clamped * np.sqrt(16.0 * (eps_clamped**2) + (np.pi**2) * (1.0 - eps_clamped**2))
  return float(num / denom)


class HydrodynamicBearing:
  """Ocvirk narrow-bearing lubrication solver."""

  def __init__(
      self,
      geom: Optional[BearingGeometry] = None,
      oil: Optional[OilProperties] = None,
  ):
    self.geom = geom or BearingGeometry(
        name="ConnectingRodBigEnd",
        journal_diameter_m=0.028,
        bearing_width_m=0.020,
        radial_clearance_m=20.0e-6,
    )
    self.oil = oil or OilProperties()

  def solve_eccentricity(self, ocvirk_number: float) -> float:
    """Solve for journal eccentricity ratio eps in (0, 1) given the Ocvirk number."""
    if ocvirk_number <= 0.0:
      return 0.0
    if ocvirk_number >= 100.0:
      return 0.999

    def objective(eps: float) -> float:
      return ocvirk_load_capacity(eps) - ocvirk_number

    try:
      res = root_scalar(objective, bracket=[1.0e-4, 0.999], method="brentq")
      return float(np.clip(res.root, 0.0, 0.999))
    except (ValueError, RuntimeError):
      # Fallback approximation for high loads
      return 0.98

  def evaluate_cycle(
      self,
      rpm: float,
      theta_deg: np.ndarray,
      radial_load_n: np.ndarray,
  ) -> BearingCycleState:
    """Evaluate hydrodynamic lubrication over the 720-degree cycle."""
    n_rev_per_s = rpm / 60.0
    v_slide = np.pi * self.geom.journal_diameter_m * n_rev_per_s

    # Projected bearing pressure: P = F / (D * W)
    proj_area = self.geom.projected_area_m2
    # Ensure minimum positive load for numerical stability
    f_rad = np.maximum(50.0, np.abs(radial_load_n))
    p_pa = f_rad / proj_area
    p_mpa = p_pa / 1.0e6

    # PV factor
    pv = p_mpa * v_slide

    # Sommerfeld number S = (r / c)^2 * (mu * N / P)
    r_over_c = self.geom.radius_m / self.geom.radial_clearance_m
    w_over_d = self.geom.bearing_width_m / self.geom.journal_diameter_m

    eps_arr = np.zeros_like(theta_deg, dtype=float)
    h_min_um = np.zeros_like(theta_deg, dtype=float)
    lambda_arr = np.zeros_like(theta_deg, dtype=float)

    rq = self.geom.composite_roughness_m

    for i, p in enumerate(p_pa):
      sommerfeld = (r_over_c**2) * (self.oil.dynamic_viscosity_pa_s * n_rev_per_s / p)
      ocvirk_num = sommerfeld * (w_over_d**2)

      eps = self.solve_eccentricity(ocvirk_num)
      h_min = self.geom.radial_clearance_m * (1.0 - eps)

      eps_arr[i] = eps
      h_min_um[i] = h_min * 1.0e6
      lambda_arr[i] = h_min / rq

    return BearingCycleState(
        theta_deg=theta_deg,
        radial_load_n=radial_load_n,
        pressure_mpa=p_mpa,
        sliding_velocity_m_s=v_slide,
        pv_factor_mpa_m_s=pv,
        eccentricity_ratio=eps_arr,
        film_thickness_um=h_min_um,
        lambda_ratio=lambda_arr,
    )

  def analyze_bearing_safety(
      self,
      rpm: float,
      theta_deg: np.ndarray,
      radial_load_n: np.ndarray,
  ) -> BearingReport:
    """Analyze bearing safety factors, PV limits, and hydrodynamic fluid film regime."""
    state = self.evaluate_cycle(rpm, theta_deg, radial_load_n)

    peak_load = float(np.max(np.abs(radial_load_n)))
    peak_press = float(np.max(state.pressure_mpa))
    mean_press = float(np.mean(state.pressure_mpa))
    peak_pv = float(np.max(state.pv_factor_mpa_m_s))
    mean_pv = float(np.mean(state.pv_factor_mpa_m_s))
    min_film = float(np.min(state.film_thickness_um))
    min_lam = float(np.min(state.lambda_ratio))
    peak_eps = float(np.max(state.eccentricity_ratio))

    # Hydrodynamic regime check: Lambda >= 2.5 ensures full fluid-film across firing pulse
    is_hydro = min_lam >= 2.5
    is_safe = (peak_press <= self.geom.max_allowable_pressure_mpa) and (min_lam >= 2.0)

    return BearingReport(
        bearing_name=self.geom.name,
        rpm=rpm,
        peak_load_n=peak_load,
        peak_pressure_mpa=peak_press,
        allowable_pressure_mpa=self.geom.max_allowable_pressure_mpa,
        mean_pressure_mpa=mean_press,
        sliding_velocity_m_s=state.sliding_velocity_m_s,
        peak_pv_mpa_m_s=peak_pv,
        mean_pv_mpa_m_s=mean_pv,
        min_film_thickness_um=min_film,
        min_lambda_ratio=min_lam,
        peak_eccentricity_ratio=peak_eps,
        is_hydrodynamic=is_hydro,
        is_safe=is_safe,
    )
