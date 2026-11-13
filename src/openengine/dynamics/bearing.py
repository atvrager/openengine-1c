"""Hydrodynamic Journal Bearing Lubrication and Dynamic Squeeze-Film Orbit Analysis.

Implements a finite-length hydrodynamic journal bearing solver combining:
- Ocvirk short-bearing capacity and half-Sommerfeld long-bearing capacity via
  harmonic series resistance blending (Reason & Narang, valid across short,
  finite L/D ~ 0.7, and long bearings)
- Booker transient 2D squeeze-film orbit integration capturing crank-angle load
  reversals, connecting-rod angular velocity modulation, and firing-pulse
  squeeze-film cushioning across the 720-degree engine cycle
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
  eccentricity_x: Optional[np.ndarray] = None
  eccentricity_z: Optional[np.ndarray] = None


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
  """Ocvirk short bearing capacity function O_n(eps) = S * (L/D)^2.

  O_n = ( (1 - eps^2)^2 ) / ( pi * eps * sqrt(16 * eps^2 + pi^2 * (1 - eps^2)) )
  """
  eps_clamped = np.clip(eps, 1.0e-5, 0.9999)
  num = (1.0 - eps_clamped**2)**2
  denom = np.pi * eps_clamped * np.sqrt(16.0 * (eps_clamped**2) + (np.pi**2) * (1.0 - eps_clamped**2))
  return float(num / denom)


def sommerfeld_long_load_capacity(eps: float) -> float:
  """Half-Sommerfeld (pi-film) infinitely-long bearing Sommerfeld number S_inf(eps)."""
  e = float(np.clip(eps, 1.0e-5, 0.9999))
  return float(
      ((2.0 + e**2) * (1.0 - e**2))
      / (6.0 * np.pi * e * np.sqrt((np.pi**2) * (1.0 - e**2) + 4.0 * (e**2)))
  )


def finite_bearing_sommerfeld(eps: float, length_to_diameter_ratio: float) -> float:
  """Finite-length journal bearing Sommerfeld number via short/long harmonic series blend.

  Uses Reason & Narang series impedance blending: S_finite = O_n(eps)/(L/D)^2 + S_inf(eps).
  """
  lod = max(1.0e-3, float(length_to_diameter_ratio))
  return (ocvirk_load_capacity(eps) / (lod**2)) + sommerfeld_long_load_capacity(eps)


def _build_squeeze_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Precompute short- and long-bearing dimensionless pure-squeeze resistance vs eccentricity."""
  eps_nodes = np.array([
      0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.90, 0.92, 0.94, 0.96, 0.975, 0.985, 0.992, 0.996
  ])
  th = np.linspace(0.0, 0.5 * np.pi, 256)
  cth = np.cos(th)
  sth = np.sin(th)
  dth = th[1] - th[0]

  short_vals = np.zeros_like(eps_nodes)
  long_vals = np.zeros_like(eps_nodes)
  for idx, eps in enumerate(eps_nodes):
    h3 = (1.0 - eps * cth) ** 3
    short_vals[idx] = 4.0 * np.trapezoid((cth**2) / h3, th)
    dp = -12.0 * sth / h3
    seg = 0.5 * (dp[:-1] + dp[1:]) * dth
    p = np.zeros_like(th)
    p[:-1] = np.cumsum(seg[::-1])[::-1] * (-1.0)
    long_vals[idx] = np.trapezoid(p * cth, th)

  return eps_nodes, short_vals, long_vals


_EPS_SQUEEZE_NODES, _SQ_SHORT_NODES, _SQ_LONG_NODES = _build_squeeze_tables()
_ORBIT_THETA = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
_ORBIT_COS = np.cos(_ORBIT_THETA)
_ORBIT_SIN = np.sin(_ORBIT_THETA)
_ORBIT_DTH = 2.0 * np.pi / len(_ORBIT_THETA)


class HydrodynamicBearing:
  """Finite-width hydrodynamic journal bearing and dynamic squeeze-film orbit solver."""

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

  @staticmethod
  def _solve_monotone_eccentricity(capacity_fn, target: float) -> float:
    """Invert a monotonically decreasing load capacity curve S(eps) = target on (0, 1)."""
    if target <= 0.0:
      return 0.999
    if target >= capacity_fn(1.0e-4):
      return 0.0
    if target <= capacity_fn(0.999):
      return 0.999
    try:
      res = root_scalar(lambda eps: capacity_fn(eps) - target, bracket=[1.0e-4, 0.999], method="brentq")
      return float(np.clip(res.root, 0.0, 0.999))
    except (ValueError, RuntimeError):
      return 0.98

  def solve_eccentricity(self, ocvirk_number: float) -> float:
    """Solve for steady-state eccentricity ratio eps in (0, 1) from short-bearing Ocvirk number."""
    return self._solve_monotone_eccentricity(ocvirk_load_capacity, ocvirk_number)

  def solve_finite_eccentricity(self, sommerfeld_number: float) -> float:
    """Solve for steady-state eccentricity ratio eps in (0, 1) using finite-width blending."""
    lod = self.geom.length_to_diameter_ratio
    return self._solve_monotone_eccentricity(
        lambda eps: finite_bearing_sommerfeld(eps, lod),
        sommerfeld_number,
    )

  def _finite_width_bearing_force(
      self,
      ex: float,
      ez: float,
      vx: float,
      vz: float,
      base_scale: float,
      lod: float,
  ) -> Tuple[float, float]:
    """Compute finite-width hydrodynamic film reaction [N] for eccentricity (ex, ez) and squeeze velocity."""
    h = np.maximum(1.0e-3, 1.0 - ex * _ORBIT_COS - ez * _ORBIT_SIN)
    sq = vx * _ORBIT_COS + vz * _ORBIT_SIN
    p = np.maximum(0.0, sq) / (h**3)
    fx_short = 2.0 * float(np.sum(p * _ORBIT_COS)) * _ORBIT_DTH * (lod**2)
    fz_short = 2.0 * float(np.sum(p * _ORBIT_SIN)) * _ORBIT_DTH * (lod**2)

    raw_eps = float(np.hypot(ex, ez))
    if raw_eps > 1.0e-4:
      erx, erz = ex / raw_eps, ez / raw_eps
    else:
      erx, erz = 1.0, 0.0
    eps = float(np.clip(raw_eps, 1.0e-4, 0.996))
    v_rad = vx * erx + vz * erz
    v_tan = -vx * erz + vz * erx

    # Weight short-to-finite impedance attenuation between pure squeeze (radial) and wedge rotation (tangential)
    c_sh = float(np.interp(eps, _EPS_SQUEEZE_NODES, _SQ_SHORT_NODES)) * (lod**2)
    c_lg = float(np.interp(eps, _EPS_SQUEEZE_NODES, _SQ_LONG_NODES))
    att_rad = 1.0 / (1.0 + c_sh / c_lg)

    s_sh = ocvirk_load_capacity(eps) / (lod**2)
    s_lg = sommerfeld_long_load_capacity(eps)
    att_tan = s_sh / (s_sh + s_lg)

    v2 = v_rad**2 + v_tan**2 + 1.0e-12
    att = (att_rad * (v_rad**2) + att_tan * (v_tan**2)) / v2
    return base_scale * att * fx_short, base_scale * att * fz_short

  def _solve_vsq_newton(
      self,
      ex: float,
      ez: float,
      fx_target: float,
      fz_target: float,
      vx_guess: float,
      vz_guess: float,
      base_scale: float,
      lod: float,
  ) -> Tuple[float, float]:
    """Solve for instantaneous dimensionless squeeze velocity V_sq balancing applied bearing load."""
    vx, vz = vx_guess, vz_guess
    tol = 1.0e-3 * max(1.0, float(np.hypot(fx_target, fz_target)))
    for _ in range(8):
      f0x, f0z = self._finite_width_bearing_force(ex, ez, vx, vz, base_scale, lod)
      rx, rz = f0x - fx_target, f0z - fz_target
      if np.hypot(rx, rz) < tol:
        break
      dv = max(1.0e-2, 1.0e-4 * (1.0 + float(np.hypot(vx, vz))))
      f1x, f1z = self._finite_width_bearing_force(ex, ez, vx + dv, vz, base_scale, lod)
      f2x, f2z = self._finite_width_bearing_force(ex, ez, vx, vz + dv, base_scale, lod)
      j11, j21 = (f1x - f0x) / dv, (f1z - f0z) / dv
      j12, j22 = (f2x - f0x) / dv, (f2z - f0z) / dv
      det = j11 * j22 - j12 * j21
      if abs(det) < 1.0e-12:
        break
      vx += -(j22 * rx - j12 * rz) / det
      vz += -(-j21 * rx + j11 * rz) / det
    return vx, vz

  def _integrate_dynamic_orbit(
      self,
      rpm: float,
      theta_deg: np.ndarray,
      fx_arr: np.ndarray,
      fz_arr: np.ndarray,
      sleeve_omega_rad_s: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate periodic 2D journal center orbit (ex, ez) with transient squeeze-film damping."""
    omega_crank = rpm * (2.0 * np.pi / 60.0)
    lod = self.geom.length_to_diameter_ratio
    r_over_c = self.geom.radius_m / self.geom.radial_clearance_m
    base_scale = self.oil.dynamic_viscosity_pa_s * (r_over_c**2) * self.geom.projected_area_m2

    dth_deg = np.diff(theta_deg)
    n_pts = len(theta_deg)
    ex_hist = np.zeros(n_pts, dtype=float)
    ez_hist = np.zeros(n_pts, dtype=float)

    ex, ez = 0.0, 0.0
    vx_g, vz_g = 0.0, 0.0
    prev_cycle_end = (0.0, 0.0)

    for cyc in range(4):
      for i in range(n_pts - 1):
        ex_hist[i] = ex
        ez_hist[i] = ez
        dt = np.radians(dth_deg[i]) / omega_crank
        if dt <= 0.0:
          continue
        omega_bar = 0.5 * (omega_crank + 0.5 * (sleeve_omega_rad_s[i] + sleeve_omega_rad_s[i + 1]))
        fx_t = 0.5 * (fx_arr[i] + fx_arr[i + 1])
        fz_t = 0.5 * (fz_arr[i] + fz_arr[i + 1])

        n_sub = 4
        dts = dt / n_sub
        for _ in range(n_sub):
          vx_g, vz_g = self._solve_vsq_newton(ex, ez, fx_t, fz_t, vx_g, vz_g, base_scale, lod)
          ex += (vx_g - omega_bar * ez) * dts
          ez += (vz_g + omega_bar * ex) * dts
          mag = float(np.hypot(ex, ez))
          if mag > 0.995:
            scale = 0.995 / mag
            ex *= scale
            ez *= scale
      ex_hist[-1] = ex
      ez_hist[-1] = ez

      if cyc >= 1 and np.hypot(ex - prev_cycle_end[0], ez - prev_cycle_end[1]) < 1.0e-4:
        break
      prev_cycle_end = (ex, ez)

    return ex_hist, ez_hist

  def evaluate_cycle(
      self,
      rpm: float,
      theta_deg: np.ndarray,
      radial_load_n: np.ndarray,
      load_x_n: Optional[np.ndarray] = None,
      load_z_n: Optional[np.ndarray] = None,
      sleeve_omega_rad_s: Optional[np.ndarray] = None,
  ) -> BearingCycleState:
    """Evaluate finite-width hydrodynamic lubrication over the engine cycle."""
    n_rev_per_s = rpm / 60.0
    v_slide = np.pi * self.geom.journal_diameter_m * n_rev_per_s

    proj_area = self.geom.projected_area_m2
    f_rad = np.maximum(50.0, np.abs(radial_load_n))
    p_pa = f_rad / proj_area
    p_mpa = p_pa / 1.0e6
    pv = p_mpa * v_slide

    r_over_c = self.geom.radius_m / self.geom.radial_clearance_m
    rq = self.geom.composite_roughness_m
    theta_arr = np.asarray(theta_deg, dtype=float)

    use_dynamic = len(theta_arr) >= 4 and (theta_arr[-1] - theta_arr[0]) >= 10.0 and rpm > 0.0
    if use_dynamic:
      fx_arr = np.asarray(load_x_n, dtype=float) if load_x_n is not None else np.zeros_like(theta_arr)
      fz_arr = np.asarray(load_z_n, dtype=float) if load_z_n is not None else -np.abs(np.asarray(radial_load_n, dtype=float))
      sleeve_arr = np.asarray(sleeve_omega_rad_s, dtype=float) if sleeve_omega_rad_s is not None else np.zeros_like(theta_arr)
      ex_arr, ez_arr = self._integrate_dynamic_orbit(rpm, theta_arr, fx_arr, fz_arr, sleeve_arr)
      eps_arr = np.hypot(ex_arr, ez_arr)
    else:
      ex_arr = None
      ez_arr = None
      eps_arr = np.zeros_like(theta_arr, dtype=float)
      for i, p in enumerate(p_pa):
        sommerfeld = (r_over_c**2) * (self.oil.dynamic_viscosity_pa_s * n_rev_per_s / p)
        eps_arr[i] = self.solve_finite_eccentricity(sommerfeld)

    h_min = self.geom.radial_clearance_m * (1.0 - eps_arr)
    h_min_um = h_min * 1.0e6
    lambda_arr = h_min / rq

    return BearingCycleState(
        theta_deg=theta_arr,
        radial_load_n=radial_load_n,
        pressure_mpa=p_mpa,
        sliding_velocity_m_s=v_slide,
        pv_factor_mpa_m_s=pv,
        eccentricity_ratio=eps_arr,
        film_thickness_um=h_min_um,
        lambda_ratio=lambda_arr,
        eccentricity_x=ex_arr,
        eccentricity_z=ez_arr,
    )

  def analyze_bearing_safety(
      self,
      rpm: float,
      theta_deg: np.ndarray,
      radial_load_n: np.ndarray,
      load_x_n: Optional[np.ndarray] = None,
      load_z_n: Optional[np.ndarray] = None,
      sleeve_omega_rad_s: Optional[np.ndarray] = None,
  ) -> BearingReport:
    """Analyze bearing safety factors, PV limits, and hydrodynamic fluid film regime."""
    state = self.evaluate_cycle(
        rpm,
        theta_deg,
        radial_load_n,
        load_x_n=load_x_n,
        load_z_n=load_z_n,
        sleeve_omega_rad_s=sleeve_omega_rad_s,
    )

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
