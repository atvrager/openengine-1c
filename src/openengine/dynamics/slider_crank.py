"""Slider-Crank Mechanism Dynamics, Balancing, and Flywheel Sizing."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from openengine.thermo.cycle import EngineGeometry


@dataclass
class DynamicForces:
  """Resolved forces, shaking vectors, and torque as a function of crank angle."""

  theta_deg: np.ndarray
  piston_acceleration_m_s2: np.ndarray
  gas_force_n: np.ndarray
  inertia_force_n: np.ndarray
  net_wrist_pin_force_n: np.ndarray
  cylinder_side_thrust_n: np.ndarray
  crankshaft_torque_nm: np.ndarray
  mean_torque_nm: float
  peak_torque_nm: float
  shaking_force_x_n: Optional[np.ndarray] = None
  shaking_force_y_n: Optional[np.ndarray] = None
  shaking_force_mag_n: Optional[np.ndarray] = None
  peak_shaking_force_n: float = 0.0
  rms_shaking_force_n: float = 0.0


@dataclass
class BalanceSweepResult:
  """Sweep of reciprocating balance factors and resultant shaking forces."""

  balance_factors: np.ndarray
  peak_shaking_forces_n: np.ndarray
  rms_shaking_forces_n: np.ndarray
  optimal_factor_peak: float
  optimal_factor_rms: float


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
      recip_balance_factor: float = 0.50,
  ) -> DynamicForces:
    """Compute all instantaneous mechanism forces, torque, and polar shaking vectors."""
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

    # Polar shaking forces transmitted to engine mounts:
    # Counterweight balances 100% rotating unbalance plus a fraction of reciprocating mass
    # Net unbalance rotating vector in block coordinates:
    f_net_rot_x = recip_balance_factor * self.m_recip * r * (omega**2) * np.sin(theta_rad)
    f_net_rot_y = recip_balance_factor * self.m_recip * r * (omega**2) * np.cos(theta_rad)

    # Transverse shaking force: side thrust + net horizontal centrifugal component
    f_shake_x = f_side + f_net_rot_x
    # Longitudinal (cylinder axis) shaking force: reciprocating inertia + vertical centrifugal component
    f_shake_y = f_inertia + f_net_rot_y
    f_shake_mag = np.sqrt(f_shake_x**2 + f_shake_y**2)

    peak_shake = float(np.max(f_shake_mag))
    rms_shake = float(np.sqrt(np.mean(f_shake_mag**2)))

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
        shaking_force_x_n=f_shake_x,
        shaking_force_y_n=f_shake_y,
        shaking_force_mag_n=f_shake_mag,
        peak_shaking_force_n=peak_shake,
        rms_shaking_force_n=rms_shake,
    )

  def sweep_balance_factors(
      self,
      rpm: float,
      forces: DynamicForces,
      factors: Optional[np.ndarray] = None,
  ) -> BalanceSweepResult:
    """Evaluate peak and RMS shaking forces across a range of reciprocating balance factors."""
    if factors is None:
      factors = np.linspace(0.0, 1.0, 21)  # 0% to 100% in 5% steps

    omega = rpm * (2.0 * np.pi / 60.0)
    theta_rad = np.radians(forces.theta_deg)
    r = self.geom.crank_radius

    peak_forces = np.zeros_like(factors)
    rms_forces = np.zeros_like(factors)

    for i, bf in enumerate(factors):
      f_rot_x = bf * self.m_recip * r * (omega**2) * np.sin(theta_rad)
      f_rot_y = bf * self.m_recip * r * (omega**2) * np.cos(theta_rad)

      f_sx = forces.cylinder_side_thrust_n + f_rot_x
      f_sy = forces.inertia_force_n + f_rot_y
      f_mag = np.sqrt(f_sx**2 + f_sy**2)

      peak_forces[i] = float(np.max(f_mag))
      rms_forces[i] = float(np.sqrt(np.mean(f_mag**2)))

    opt_peak_idx = int(np.argmin(peak_forces))
    opt_rms_idx = int(np.argmin(rms_forces))

    return BalanceSweepResult(
        balance_factors=factors,
        peak_shaking_forces_n=peak_forces,
        rms_shaking_forces_n=rms_forces,
        optimal_factor_peak=float(factors[opt_peak_idx]),
        optimal_factor_rms=float(factors[opt_rms_idx]),
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
