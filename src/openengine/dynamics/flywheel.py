"""Flywheel Sizing, Crankshaft Kinetic Energy Integration, and Speed Fluctuation.

Computes the torque fluctuation across the 720-degree 4-stroke cycle, sizes the required
flywheel polar moment of inertia to maintain the speed fluctuation coefficient C_s <= 0.035,
resolves instantaneous angular velocity omega(theta), and calculates rim hoop stresses
and burst safety factors at redline.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from openengine.cad.covers_and_aux import (
    FLYWHEEL_HUB_BORE_RADIUS_MM,
    FLYWHEEL_OUTER_RADIUS_MM,
)
from openengine.fea.calculix import STEEL_1144_STRESSPROOF


@dataclass
class FlywheelDimensions:
  """Geometric and material boundaries for flywheel sizing."""

  outer_radius_m: float = FLYWHEEL_OUTER_RADIUS_MM * 1.0e-3
  inner_bore_radius_m: float = FLYWHEEL_HUB_BORE_RADIUS_MM * 1.0e-3
  density_kg_m3: float = STEEL_1144_STRESSPROOF.density_tonne_mm3 * 1.0e12
  allowable_tensile_stress_mpa: float = 220.0  # Fatigue/yield tensile limit


@dataclass
class FlywheelEnergyState:
  """Torque and kinetic energy fluctuation across the cycle."""

  theta_deg: np.ndarray
  torque_nm: np.ndarray
  mean_torque_nm: float
  delta_torque_nm: np.ndarray
  cumulative_energy_j: np.ndarray
  delta_energy_j: float
  instantaneous_speed_rad_s: np.ndarray
  instantaneous_rpm: np.ndarray


@dataclass
class FlywheelSizingReport:
  """Flywheel engineering dimensions, inertia, and stress verification."""

  rated_rpm: float
  redline_rpm: float
  target_fluctuation_coeff: float
  actual_fluctuation_coeff: float
  delta_energy_j: float
  flywheel_inertia_kg_m2: float
  flywheel_mass_kg: float
  flywheel_thickness_mm: float
  min_rpm_at_rated: float
  max_rpm_at_rated: float
  rim_velocity_rated_m_s: float
  rim_velocity_redline_m_s: float
  hoop_stress_rated_mpa: float
  hoop_stress_redline_mpa: float
  burst_safety_factor_redline: float
  is_safe: bool


class FlywheelDynamics:
  """Kinetic energy integrator and flywheel sizing engine."""

  def __init__(self, dimensions: Optional[FlywheelDimensions] = None):
    self.dims = dimensions or FlywheelDimensions()

  def integrate_energy(
      self,
      theta_deg: np.ndarray,
      torque_nm: np.ndarray,
  ) -> Tuple[np.ndarray, float, float]:
    """Integrate torque excess T(theta) - T_mean to determine cumulative energy fluctuation.

    Returns:
      cumulative_energy_j: E(theta) array in Joules
      delta_energy_j: Max peak-to-peak energy fluctuation Delta E
      mean_torque_nm: Mean cycle torque
    """
    mean_t = float(np.mean(torque_nm))
    delta_t = torque_nm - mean_t
    th_rad = np.radians(theta_deg)

    energy = np.zeros_like(th_rad)
    for i in range(len(th_rad) - 1):
      dth = th_rad[i + 1] - th_rad[i]
      energy[i + 1] = energy[i] + 0.5 * (delta_t[i] + delta_t[i + 1]) * dth

    # Zero-center the energy fluctuation
    energy -= np.mean(energy)
    delta_e = float(np.max(energy) - np.min(energy))
    return energy, delta_e, mean_t

  def evaluate_speed_trajectory(
      self,
      mean_rpm: float,
      energy_j: np.ndarray,
      total_inertia_kg_m2: float,
  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate instantaneous angular velocity omega(theta) and RPM across the cycle.

    Governing equation: 0.5 * I * (omega^2 - omega_0^2) = E(theta)
    """
    omega_0 = mean_rpm * (2.0 * np.pi / 60.0)
    # omega(theta) = sqrt( omega_0^2 + 2 * E(theta) / I )
    omega_sq = (omega_0**2) + (2.0 * energy_j / total_inertia_kg_m2)
    # Numerical safeguard
    omega_sq = np.maximum(1.0, omega_sq)
    omega = np.sqrt(omega_sq)
    rpm = omega * (60.0 / (2.0 * np.pi))

    c_s = float((np.max(omega) - np.min(omega)) / omega_0)
    return omega, rpm, c_s

  def size_flywheel(
      self,
      rated_rpm: float,
      theta_deg: np.ndarray,
      torque_nm: np.ndarray,
      target_coeff_fluctuation: float = 0.035,
      redline_rpm: float = 7000.0,
  ) -> FlywheelSizingReport:
    """Size flywheel inertia, solid rim thickness, and evaluate burst stresses."""
    omega_rated = rated_rpm * (2.0 * np.pi / 60.0)
    omega_redline = redline_rpm * (2.0 * np.pi / 60.0)

    energy_j, delta_e, mean_t = self.integrate_energy(theta_deg, torque_nm)

    # Required polar moment of inertia: I = Delta E / (C_s * omega^2)
    i_required = delta_e / (target_coeff_fluctuation * (omega_rated**2))

    # Calculate disc geometry:
    # I = 0.5 * pi * rho * thickness * (R_out^4 - R_in^4)
    r_out = self.dims.outer_radius_m
    r_in = self.dims.inner_bore_radius_m
    rho = self.dims.density_kg_m3

    inertia_per_meter = 0.5 * np.pi * rho * (r_out**4 - r_in**4)
    thickness_m = i_required / inertia_per_meter
    thickness_mm = thickness_m * 1000.0

    # Flywheel mass: M = rho * pi * (R_out^2 - R_in^2) * thickness
    area = np.pi * (r_out**2 - r_in**2)
    mass_kg = rho * area * thickness_m

    # Actual speed fluctuation with this sized inertia
    omega_inst, rpm_inst, actual_cs = self.evaluate_speed_trajectory(
        mean_rpm=rated_rpm,
        energy_j=energy_j,
        total_inertia_kg_m2=i_required,
    )

    # Rim hoop stress: sigma_hoop = rho * v_rim^2
    v_rim_rated = omega_rated * r_out
    v_rim_redline = omega_redline * r_out

    hoop_stress_rated_pa = rho * (v_rim_rated**2)
    hoop_stress_redline_pa = rho * (v_rim_redline**2)

    hoop_rated_mpa = hoop_stress_rated_pa / 1.0e6
    hoop_redline_mpa = hoop_stress_redline_pa / 1.0e6

    burst_sf = self.dims.allowable_tensile_stress_mpa / max(1e-3, hoop_redline_mpa)
    is_safe = (actual_cs <= target_coeff_fluctuation * 1.05) and (burst_sf >= 4.0)

    return FlywheelSizingReport(
        rated_rpm=rated_rpm,
        redline_rpm=redline_rpm,
        target_fluctuation_coeff=target_coeff_fluctuation,
        actual_fluctuation_coeff=actual_cs,
        delta_energy_j=delta_e,
        flywheel_inertia_kg_m2=float(i_required),
        flywheel_mass_kg=float(mass_kg),
        flywheel_thickness_mm=float(thickness_mm),
        min_rpm_at_rated=float(np.min(rpm_inst)),
        max_rpm_at_rated=float(np.max(rpm_inst)),
        rim_velocity_rated_m_s=float(v_rim_rated),
        rim_velocity_redline_m_s=float(v_rim_redline),
        hoop_stress_rated_mpa=float(hoop_rated_mpa),
        hoop_stress_redline_mpa=float(hoop_redline_mpa),
        burst_safety_factor_redline=float(burst_sf),
        is_safe=is_safe,
    )
