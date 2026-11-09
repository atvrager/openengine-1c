"""Woschni In-Cylinder Convective Heat Transfer Correlation (1967)."""

from dataclasses import dataclass
import numpy as np
from openengine.thermo.cycle import EngineGeometry


@dataclass
class HeatTransferOutputs:
  """Instantaneous heat transfer variables at a specific crank angle."""

  heat_transfer_coeff_w_m2_k: float  # h_c
  wall_area_m2: float  # A_wall
  heat_flux_w: float  # dQ/dt to walls
  gas_velocity_m_s: float  # w(theta)


class WoschniHeatTransfer:
  """Calculates in-cylinder convective wall heat loss using Woschni's correlation.

  Reference:
      Woschni, G., 'A Universally Applicable Equation for the Instantaneous
      Heat Transfer Coefficient in the Internal Combustion Engine,'
      SAE Technical Paper 670931, 1967.
  """

  def __init__(
      self,
      geom: EngineGeometry,
      t_wall_k: float = 430.0,  # ~157 C average cylinder metal temperature
      c1_scavenge: float = 6.18,
      c1_compression: float = 2.28,
      c1_combustion: float = 2.28,
      c2_combustion: float = 3.24e-3,  # [m / (s * K)]
      chamber_roof_angle_deg: float = 15.0,  # Pent-roof half angle
  ):
    self.geom = geom
    self.t_wall_k = t_wall_k
    self.c1_scav = c1_scavenge
    self.c1_comp = c1_compression
    self.c1_comb = c1_combustion
    self.c2_comb = c2_combustion

    b = self.geom.bore
    # Piston crown flat area
    self.a_piston = np.pi * (b / 2.0) ** 2
    # Pent-roof cylinder head surface area
    self.a_head = self.a_piston / np.cos(np.radians(chamber_roof_angle_deg))

  def surface_area(self, theta_rad: float) -> float:
    """Instantaneous total heat transfer surface area A_wall(theta).

    A_wall = A_head + A_piston + pi * Bore * piston_distance_from_tdc
    """
    s = self.geom.piston_distance_from_tdc(theta_rad)
    a_liner = np.pi * self.geom.bore * s
    return float(self.a_head + self.a_piston + a_liner)

  def characteristic_velocity(
      self,
      theta_deg: float,
      rpm: float,
      p_pa: float,
      p_motored_pa: float,
      t_ref_k: float,
      p_ref_pa: float,
      v_ref_m3: float,
  ) -> float:
    """Calculate Woschni characteristic gas velocity w(theta) [m/s].

    w = C1 * S_mean_piston + C2 * (V_d * T_ref / (P_ref * V_ref)) * (P - P_motored)
    """
    th = theta_deg % 720.0
    # Mean piston speed: S_p = 2 * Stroke * (RPM / 60)
    s_p = 2.0 * self.geom.stroke * (rpm / 60.0)

    # Determine engine cycle phase
    if 180.0 <= th < 540.0:
      # Gas exchange (exhaust and intake strokes)
      c1 = self.c1_scav
      c2 = 0.0
    elif 540.0 <= th < 685.0:
      # Compression stroke before ignition
      c1 = self.c1_comp
      c2 = 0.0
    else:
      # Combustion and expansion stroke (685..720 and 0..180)
      c1 = self.c1_comb
      c2 = self.c2_comb

    # Pressure rise due to combustion
    delta_p = max(0.0, p_pa - p_motored_pa)
    vd = self.geom.displacement

    term_turb = c2 * ((vd * t_ref_k) / (p_ref_pa * v_ref_m3)) * delta_p
    return float(c1 * s_p + term_turb)

  def calculate_heat_transfer(
      self,
      theta_deg: float,
      rpm: float,
      p_pa: float,
      t_gas_k: float,
      p_motored_pa: float = 0.0,
      t_ref_k: float = 300.0,
      p_ref_pa: float = 101325.0,
      v_ref_m3: float = 0.0002,
  ) -> HeatTransferOutputs:
    """Calculate convective heat transfer coefficient and wall heat flux."""
    th_rad = np.radians(theta_deg % 720.0)
    area = self.surface_area(th_rad)

    w = self.characteristic_velocity(
        theta_deg=theta_deg,
        rpm=rpm,
        p_pa=p_pa,
        p_motored_pa=p_motored_pa,
        t_ref_k=t_ref_k,
        p_ref_pa=p_ref_pa,
        v_ref_m3=v_ref_m3,
    )

    b = self.geom.bore
    # Woschni formula: h_c = 3.26 * B^(-0.2) * P^0.8 * T^(-0.55) * w^0.8
    # Pressures in kPa for standard Woschni coefficient 3.26:
    # Converting Pa to kPa: (p / 1000)^0.8
    p_kpa = max(10.0, p_pa / 1000.0)
    t = max(250.0, t_gas_k)
    w_clamped = max(0.5, w)

    hc = 3.26 * (b**-0.2) * (p_kpa**0.8) * (t**-0.55) * (w_clamped**0.8)

    # Convective heat loss rate: dQ/dt = h_c * Area * (T_gas - T_wall)
    q_dot = hc * area * (t - self.t_wall_k)

    return HeatTransferOutputs(
        heat_transfer_coeff_w_m2_k=float(hc),
        wall_area_m2=float(area),
        heat_flux_w=float(q_dot),
        gas_velocity_m_s=float(w),
    )
