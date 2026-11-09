"""Slider-Crank Mechanism Dynamics, Balancing, and Flywheel Sizing."""

from dataclasses import dataclass
import numpy as np
from openengine.thermo.cycle import EngineGeometry


@dataclass
class DynamicForces:
  """Resolved forces and torque as a function of crank angle."""

  theta_deg: np.ndarray
  piston_acceleration_m_s2: np.ndarray
  gas_force_n: np.ndarray
  inertia_force_n: np.ndarray
  net_wrist_pin_force_n: np.ndarray
  cylinder_side_thrust_n: np.ndarray
  crankshaft_torque_nm: np.ndarray
  mean_torque_nm: float
  peak_torque_nm: float


@dataclass
class BalancerOutputs:
  """Counterweight and flywheel design values."""

  counterweight_mass_kg: float
  counterweight_radius_m: float
  flywheel_inertia_kg_m2: float
  flywheel_mass_kg: float  # Sized as a solid disc


class SliderCrankDynamics:
  """Kinematic force balancer and mechanism solver."""

  def __init__(
      self,
      geom: EngineGeometry,
      reciprocating_mass_kg: float = 0.220,  # Piston + pin + rings + 1/3 rod
      rotating_mass_kg: float = 0.280,  # Crankpin + 2/3 rod
      counterweight_radius_m: float = 0.038,  # Typical cheek radius
      flywheel_outer_radius_m: float = 0.085,  # 170 mm diameter flywheel
  ):
    self.geom = geom
    self.m_recip = reciprocating_mass_kg
    self.m_rot = rotating_mass_kg
    self.r_cw = counterweight_radius_m
    self.r_fw = flywheel_outer_radius_m

  def solve_forces(
      self,
      rpm: float,
      theta_deg: np.ndarray,
      pressure_pa: np.ndarray,
  ) -> DynamicForces:
    """Compute all instantaneous mechanism forces and torque."""
    omega = rpm * (2.0 * np.pi / 60.0)
    theta_rad = np.radians(theta_deg)
    r = self.geom.crank_radius
    l_rod = self.geom.con_rod_length
    lam = r / l_rod

    # Piston acceleration: a ≈ r * omega^2 * (cos(theta) + lambda * cos(2*theta))
    accel = r * (omega**2) * (np.cos(theta_rad) + lam * np.cos(2.0 * theta_rad))

    # Gas force
    piston_area = np.pi * (self.geom.bore / 2.0) ** 2
    f_gas = pressure_pa * piston_area

    # Reciprocating inertia force: F_inertia = -m_recip * a
    f_inertia = -self.m_recip * accel

    # Net wrist pin force
    f_net = f_gas + f_inertia

    # Connecting rod angular obliquity beta: sin(beta) = lambda * sin(theta)
    sin_beta = lam * np.sin(theta_rad)
    cos_beta = np.sqrt(1.0 - sin_beta**2)
    tan_beta = sin_beta / cos_beta

    # Side thrust against cylinder wall
    f_side = f_net * tan_beta

    # Crankshaft output torque
    torque = f_net * r * (np.sin(theta_rad) + (lam * np.sin(2.0 * theta_rad)) / (2.0 * cos_beta))

    return DynamicForces(
        theta_deg=theta_deg,
        piston_acceleration_m_s2=accel,
        gas_force_n=f_gas,
        inertia_force_n=f_inertia,
        net_wrist_pin_force_n=f_net,
        cylinder_side_thrust_n=f_side,
        crankshaft_torque_nm=torque,
        mean_torque_nm=float(np.mean(torque)),
        peak_torque_nm=float(np.max(torque)),
    )

  def size_balancer_and_flywheel(
      self,
      rpm: float,
      forces: DynamicForces,
      speed_fluctuation_coeff: float = 0.035,
      recip_balance_factor: float = 0.50,
  ) -> BalancerOutputs:
    """Size counterweights and flywheel inertia."""
    omega = rpm * (2.0 * np.pi / 60.0)
    r = self.geom.crank_radius

    # Counterweight sizing:
    # m_cw * r_cw = m_rot * r + (factor * m_recip * r)
    m_cw = (self.m_rot * r + recip_balance_factor * self.m_recip * r) / self.r_cw

    # Flywheel sizing via torque integration:
    theta_rad = np.radians(forces.theta_deg)
    delta_t = forces.crankshaft_torque_nm - forces.mean_torque_nm

    # Cumulative energy fluctuation
    energy = np.zeros_like(theta_rad)
    for i in range(len(theta_rad) - 1):
      dth = theta_rad[i + 1] - theta_rad[i]
      energy[i + 1] = energy[i] + 0.5 * (delta_t[i] + delta_t[i + 1]) * dth

    delta_e = float(np.max(energy) - np.min(energy))

    # I_flywheel = delta_e / (C_s * omega^2)
    i_fw = delta_e / (speed_fluctuation_coeff * (omega**2))

    # Equivalent solid disc mass: I = 0.5 * M * R^2 -> M = 2 * I / R^2
    m_fw = (2.0 * i_fw) / (self.r_fw**2)

    return BalancerOutputs(
        counterweight_mass_kg=float(m_cw),
        counterweight_radius_m=self.r_cw,
        flywheel_inertia_kg_m2=float(i_fw),
        flywheel_mass_kg=float(m_fw),
    )
