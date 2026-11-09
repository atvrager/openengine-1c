"""0D/1D Thermodynamic Cycle and In-Cylinder Combustion Models."""

from dataclasses import dataclass
import numpy as np


@dataclass
class EngineGeometry:
  """Basic geometric dimensions of a slider-crank engine."""

  bore: float  # [m]
  stroke: float  # [m]
  con_rod_length: float  # [m]
  compression_ratio: float  # [-]

  @property
  def crank_radius(self) -> float:
    return self.stroke / 2.0

  @property
  def displacement(self) -> float:
    return np.pi * (self.bore / 2.0) ** 2 * self.stroke

  @property
  def clearance_volume(self) -> float:
    return self.displacement / (self.compression_ratio - 1.0)

  @property
  def rod_to_crank_ratio(self) -> float:
    return self.con_rod_length / self.crank_radius

  def volume(self, theta_rad: np.ndarray | float) -> np.ndarray | float:
    """Calculate instantaneous cylinder volume V(theta).

    theta = 0 corresponds to Top Dead Center (TDC).
    """
    R = self.rod_to_crank_ratio
    disp_norm = 0.5 * (1.0 - np.cos(theta_rad) + R - np.sqrt(R**2 - np.sin(theta_rad) ** 2))
    return self.clearance_volume + self.displacement * disp_norm

  def piston_distance_from_tdc(
      self, theta_rad: np.ndarray | float
  ) -> np.ndarray | float:
    """Distance of piston crown from TDC position."""
    R = self.rod_to_crank_ratio
    return self.crank_radius * (
        (1.0 - np.cos(theta_rad)) + R - np.sqrt(R**2 - np.sin(theta_rad) ** 2)
    )


def wiebe_burned_fraction(
    theta_deg: np.ndarray | float,
    theta_soc_deg: float = -15.0,
    duration_deg: float = 42.0,
    a: float = 5.0,
    m: float = 2.0,
) -> np.ndarray | float:
  """Calculate Mass Fraction Burned (MFB) x_b(theta) using the Wiebe function."""
  theta = np.asarray(theta_deg)
  scalar_input = theta.ndim == 0
  theta = np.atleast_1d(theta)

  xb = np.zeros_like(theta, dtype=float)
  active_mask = (theta >= theta_soc_deg) & (
      theta <= (theta_soc_deg + duration_deg)
  )
  completed_mask = theta > (theta_soc_deg + duration_deg)

  progress = (theta[active_mask] - theta_soc_deg) / duration_deg
  xb[active_mask] = 1.0 - np.exp(-a * (progress ** (m + 1.0)))
  xb[completed_mask] = 1.0

  return float(xb[0]) if scalar_input else xb


@dataclass
class CycleResults:
  """Summary outputs of a closed-cycle simulation."""

  theta_deg: np.ndarray
  pressure_pa: np.ndarray
  temperature_k: np.ndarray
  volume_m3: np.ndarray
  p_max_pa: float
  t_max_k: float
  imep_pa: float
  indicated_work_j: float
  thermal_efficiency: float


class ClosedCycleSimulator:
  """Single-zone thermodynamic engine cycle simulator."""

  def __init__(
      self,
      geom: EngineGeometry,
      gamma: float = 1.32,
      q_fuel_j_per_kg: float = 44.0e6,
      air_fuel_ratio: float = 14.7,
  ):
    self.geom = geom
    self.gamma = gamma
    self.q_fuel = q_fuel_j_per_kg
    self.afr = air_fuel_ratio

  def solve_cycle(
      self,
      p_init_pa: float = 101325.0,
      t_init_k: float = 330.0,
      n_points: int = 720,
  ) -> CycleResults:
    """Solve the closed portion of the 4-stroke cycle (-180 to +180 deg CA)."""
    theta = np.linspace(-180.0, 180.0, n_points)
    theta_rad = np.radians(theta)
    v = self.geom.volume(theta_rad)

    r_gas = 287.05  # J/(kg*K)
    m_charge = (p_init_pa * v[0]) / (r_gas * t_init_k)
    m_fuel = m_charge / (1.0 + self.afr)
    total_heat_release = m_fuel * self.q_fuel * 0.90  # 90% combustion efficiency

    xb = wiebe_burned_fraction(theta)

    p = np.zeros_like(theta)
    t = np.zeros_like(theta)
    p[0] = p_init_pa
    t[0] = t_init_k

    cv = r_gas / (self.gamma - 1.0)

    for i in range(len(theta) - 1):
      dv = v[i + 1] - v[i]
      dxb = xb[i + 1] - xb[i]
      dq_comb = dxb * total_heat_release

      # Energy balance: m * cv * dT = dQ - P * dV
      p_mid = p[i]
      dt = (dq_comb - p_mid * dv) / (m_charge * cv)
      t[i + 1] = t[i] + dt
      p[i + 1] = (m_charge * r_gas * t[i + 1]) / v[i + 1]

    # Compute Indicated Work: integral(P dV)
    w_ind = np.trapezoid(p, v)
    imep = w_ind / self.geom.displacement
    eta = w_ind / total_heat_release

    return CycleResults(
        theta_deg=theta,
        pressure_pa=p,
        temperature_k=t,
        volume_m3=v,
        p_max_pa=float(np.max(p)),
        t_max_k=float(np.max(t)),
        imep_pa=float(imep),
        indicated_work_j=float(w_ind),
        thermal_efficiency=float(eta),
    )
