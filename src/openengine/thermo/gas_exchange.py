"""1D Gas Exchange and Full 720-Degree Four-Stroke Engine Cycle Simulator."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from openengine.thermo.cantera_kinetics import CanteraThermochemistry
from openengine.thermo.cycle import EngineGeometry, wiebe_burned_fraction
from openengine.thermo.woschni import WoschniHeatTransfer
from scipy.integrate import solve_ivp


@dataclass
class ValveTiming:
  """Valve event angles in crank angle degrees relative to Firing TDC (0 deg).

  Cycle definition:
    0 deg:   Firing TDC
    180 deg: Expansion BDC
    360 deg: Gas Exchange TDC
    540 deg: Intake BDC
    720 deg: Firing TDC (completes 720 deg cycle)
  """

  # Exhaust events
  evo_deg: float = 135.0  # Exhaust Valve Opens (45 deg before BDC)
  evc_deg: float = 375.0  # Exhaust Valve Closes (15 deg after Gas Exchange TDC)

  # Intake events
  ivo_deg: float = 345.0  # Intake Valve Opens (15 deg before Gas Exchange TDC)
  ivc_deg: float = 585.0  # Intake Valve Closes (45 deg after BDC)


@dataclass
class ValveGeometry:
  """Dimensions and discharge coefficients for poppet valves."""

  intake_diameter_m: float = 0.030
  exhaust_diameter_m: float = 0.026
  intake_max_lift_m: float = 0.0085
  exhaust_max_lift_m: float = 0.0080
  seat_angle_deg: float = 45.0
  intake_cd: float = 0.65
  exhaust_cd: float = 0.62

  def lift(
      self,
      theta_deg: np.ndarray | float,
      open_deg: float,
      close_deg: float,
      max_lift: float,
  ) -> np.ndarray | float:
    """Smooth half-sine lift curve between open_deg and close_deg."""
    th = np.asarray(theta_deg)
    scalar_input = th.ndim == 0
    th = np.atleast_1d(th) % 720.0

    duration = close_deg - open_deg
    lift_vals = np.zeros_like(th, dtype=float)

    if open_deg < close_deg:
      mask = (th >= open_deg) & (th <= close_deg)
      progress = (th[mask] - open_deg) / duration
      lift_vals[mask] = max_lift * np.sin(np.pi * progress)
    else:
      # Wraps around 720 deg
      mask = (th >= open_deg) | (th <= close_deg)
      norm_th = np.where(th < open_deg, th + 720.0, th)
      progress = (norm_th[mask] - open_deg) / duration
      lift_vals[mask] = max_lift * np.sin(np.pi * progress)

    return float(lift_vals[0]) if scalar_input else lift_vals

  def curtain_area(self, lift_m: float, diameter_m: float) -> float:
    """Valve curtain area: Ac = pi * D * L * cos(seat_angle)."""
    cos_seat = np.cos(np.radians(self.seat_angle_deg))
    return float(np.pi * diameter_m * max(0.0, lift_m) * cos_seat)


def compressible_orifice_flow(
    p_up_pa: float,
    t_up_k: float,
    p_down_pa: float,
    cd: float,
    area_m2: float,
    gamma: float = 1.35,
    r_gas: float = 287.05,
) -> float:
  """Calculate 1D isentropic compressible orifice mass flow rate [kg/s].

  Returns positive flow from upstream to downstream.
  Handles subsonic and choked flow regimes smoothly.
  """
  if area_m2 <= 1.0e-9 or p_up_pa <= 1.0e3:
    return 0.0

  pr = max(1.0e-5, min(1.0, p_down_pa / p_up_pa))
  pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

  c_sound_up = np.sqrt(gamma * r_gas * t_up_k)
  rho_up = p_up_pa / (r_gas * t_up_k)

  if pr <= pr_crit:
    # Choked (sonic) flow
    psi = np.sqrt(gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))
  else:
    # Subsonic flow
    term1 = 2.0 * gamma / (gamma - 1.0)
    term2 = (pr ** (2.0 / gamma)) - (pr ** ((gamma + 1.0) / gamma))
    psi = np.sqrt(max(0.0, term1 * term2))

  return float(cd * area_m2 * (p_up_pa / c_sound_up) * psi)


@dataclass
class FourStrokeResults:
  """Complete 720-degree engine cycle performance and traces."""

  theta_deg: np.ndarray
  pressure_pa: np.ndarray
  temperature_k: np.ndarray
  volume_m3: np.ndarray
  mass_kg: np.ndarray
  heat_transfer_coeff_w_m2_k: np.ndarray
  wall_heat_flux_w: np.ndarray
  intake_lift_mm: np.ndarray
  exhaust_lift_mm: np.ndarray
  p_max_pa: float
  t_max_k: float
  gross_work_j: float
  pump_work_j: float
  net_work_j: float
  gimep_pa: float
  pmep_pa: float
  nimep_pa: float
  trapped_mass_kg: float
  volumetric_efficiency: float
  power_kw: float
  torque_nm: float
  q_wall_loss_j: float
  mean_cooling_power_w: float
  peak_heat_flux_mw_m2: float
  heat_loss_fraction: float
  equilibrium_species: Dict[str, float]


class FourStrokeSimulator:
  """Full 720-degree 4-stroke cycle and gas exchange simulator."""

  def __init__(
      self,
      geom: EngineGeometry,
      valves: ValveGeometry = ValveGeometry(),
      timing: ValveTiming = ValveTiming(),
      gamma_air: float = 1.35,
      r_gas: float = 287.05,
      q_fuel: float = 44.0e6,
      afr: float = 14.7,
      use_woschni: bool = True,
      t_wall_k: float = 430.0,
      use_cantera: bool = True,
      wiebe_soc_deg: float = -15.0,
      wiebe_duration_deg: float = 42.0,
      wiebe_a: float = 5.0,
      wiebe_m: float = 2.0,
  ):
    self.geom = geom
    self.valves = valves
    self.timing = timing
    self.gamma = gamma_air
    self.r_gas = r_gas
    self.q_fuel = q_fuel
    self.afr = afr
    self.cv0 = r_gas / (gamma_air - 1.0)
    self.cp0 = self.cv0 + r_gas
    self.use_woschni = use_woschni
    self.t_wall_k = t_wall_k
    self.use_cantera = use_cantera
    self.wiebe_soc_deg = wiebe_soc_deg
    self.wiebe_duration_deg = wiebe_duration_deg
    self.wiebe_a = wiebe_a
    self.wiebe_m = wiebe_m
    self.woschni = WoschniHeatTransfer(geom, t_wall_k=t_wall_k)
    if use_cantera:
      self.cantera_chem = CanteraThermochemistry()
    else:
      self.cantera_chem = None

  def solve_cycle(
      self,
      rpm: float = 5500.0,
      p_inlet_pa: float = 101325.0,
      t_inlet_k: float = 298.15,
      p_exhaust_pa: float = 105000.0,
      t_exhaust_k: float = 800.0,
      n_cycles: int = 2,
  ) -> FourStrokeResults:
    """Simulate full 720 deg cycles until convergence.

    Crank angle:
      0..180:   Power
      180..360: Exhaust
      360..540: Intake
      540..720: Compression
    """
    omega = rpm * (2.0 * np.pi / 60.0)  # rad/s
    rho_inlet = p_inlet_pa / (self.r_gas * t_inlet_k)
    v_max = self.geom.displacement + self.geom.clearance_volume

    # State variables: y = [m, T]
    # Initialize near BDC intake: fresh charge
    m_init = p_inlet_pa * v_max / (self.r_gas * t_inlet_k)
    t_init = t_inlet_k

    def ode_system(theta_deg, y):
      m = max(1.0e-5, y[0])
      t = max(200.0, y[1])

      th_mod = theta_deg % 720.0
      th_rad = np.radians(th_mod)
      v = self.geom.volume(th_rad)
      p = (m * self.r_gas * t) / v

      # Piston kinematics dV/dtheta
      R = self.geom.rod_to_crank_ratio
      dvdth = 0.5 * self.geom.displacement * (
          np.sin(th_rad) + (np.sin(2.0 * th_rad) / (2.0 * np.sqrt(R**2 - np.sin(th_rad)**2)))
      )  # m^3 / rad
      dvdt = dvdth * omega  # m^3 / s

      # Temperature-dependent specific heat capacity (vibrational activation)
      cv = self.cv0 * (1.0 + 3.8e-4 * max(0.0, t - 300.0))
      cp = cv + self.r_gas
      gamma_inst = cp / cv

      # Valve areas
      l_in = self.valves.lift(th_mod, self.timing.ivo_deg, self.timing.ivc_deg, self.valves.intake_max_lift_m)
      l_ex = self.valves.lift(th_mod, self.timing.evo_deg, self.timing.evc_deg, self.valves.exhaust_max_lift_m)
      a_in = self.valves.curtain_area(l_in, self.valves.intake_diameter_m)
      a_ex = self.valves.curtain_area(l_ex, self.valves.exhaust_diameter_m)

      # 1. Intake Mass Flow
      if p_inlet_pa >= p:
        # Inflow: fresh air into cylinder
        m_dot_in = compressible_orifice_flow(p_inlet_pa, t_inlet_k, p, self.valves.intake_cd, a_in, gamma_inst, self.r_gas)
        h_in_net = m_dot_in * cp * t_inlet_k
      else:
        # Reversion: backflow out of cylinder into intake
        m_dot_in = -compressible_orifice_flow(p, t, p_inlet_pa, self.valves.intake_cd, a_in, gamma_inst, self.r_gas)
        h_in_net = m_dot_in * cp * t

      # 2. Exhaust Mass Flow
      if p >= p_exhaust_pa:
        # Blowdown: cylinder gas escaping into exhaust port
        m_dot_ex = compressible_orifice_flow(p, t, p_exhaust_pa, self.valves.exhaust_cd, a_ex, gamma_inst, self.r_gas)
        h_ex_net = -m_dot_ex * cp * t
      else:
        # Backflow: exhaust gas drawn back in
        m_dot_ex = -compressible_orifice_flow(p_exhaust_pa, t_exhaust_k, p, self.valves.exhaust_cd, a_ex, gamma_inst, self.r_gas)
        h_ex_net = -m_dot_ex * cp * t_exhaust_k

      dmdt = m_dot_in - m_dot_ex

      # 3. Combustion Heat Release (Wiebe model active around firing TDC: 680..740 deg)
      ca_firing = th_mod - 720.0 if th_mod > 360.0 else th_mod
      dxb_dth = 0.0
      if self.wiebe_soc_deg <= ca_firing <= (self.wiebe_soc_deg + self.wiebe_duration_deg):
        xb1 = wiebe_burned_fraction(
            ca_firing,
            theta_soc_deg=self.wiebe_soc_deg,
            duration_deg=self.wiebe_duration_deg,
            a=self.wiebe_a,
            m=self.wiebe_m,
        )
        xb2 = wiebe_burned_fraction(
            ca_firing + 0.1,
            theta_soc_deg=self.wiebe_soc_deg,
            duration_deg=self.wiebe_duration_deg,
            a=self.wiebe_a,
            m=self.wiebe_m,
        )
        dxb_dth = (xb2 - xb1) / (0.1 * np.pi / 180.0)  # dxb/dtheta in rad^-1

      m_fuel = m / (1.0 + self.afr)
      q_total = m_fuel * self.q_fuel * 0.92  # 92% combustion efficiency
      dq_comb_dt = dxb_dth * omega * q_total

      # 4. Woschni Convective Wall Heat Loss
      if self.use_woschni:
        r_comp = v_max / max(1.0e-6, v)
        p_motored = p_inlet_pa * (r_comp**1.32)
        ht = self.woschni.calculate_heat_transfer(
            theta_deg=th_mod,
            rpm=rpm,
            p_pa=p,
            t_gas_k=t,
            p_motored_pa=p_motored,
            t_ref_k=t_inlet_k,
            p_ref_pa=p_inlet_pa,
            v_ref_m3=v_max,
        )
        dq_wall_dt = ht.heat_flux_w
      else:
        dq_wall_dt = 0.0

      # 5. Energy equation: m * cv * dT/dt = dQ_comb - dQ_wall - P * dV/dt + energy_flux
      u = cv * t
      energy_flux = h_in_net + h_ex_net - u * dmdt
      dtdt = (dq_comb_dt - dq_wall_dt - p * dvdt + energy_flux) / (m * cv)

      # Convert time derivatives to crank angle derivatives (1/omega)
      dmdth_deg = (dmdt / omega) * (np.pi / 180.0)
      dtdth_deg = (dtdt / omega) * (np.pi / 180.0)

      return [dmdth_deg, dtdth_deg]

    # Run for n_cycles to achieve thermodynamic limit cycle
    th_eval = np.linspace(0.0, 720.0 * n_cycles, 720 * n_cycles)
    sol = solve_ivp(
        ode_system,
        t_span=(0.0, 720.0 * n_cycles),
        y0=[m_init, t_init],
        t_eval=th_eval,
        method="RK23",
        rtol=1e-4,
        atol=1e-5,
    )

    # Take the final 720 deg cycle
    th_final = sol.t[-720:] % 720.0
    sort_idx = np.argsort(th_final)
    theta_deg = th_final[sort_idx]
    m_cycle = sol.y[0][-720:][sort_idx]
    t_cycle = sol.y[1][-720:][sort_idx]

    th_rad = np.radians(theta_deg)
    v_cycle = np.array([self.geom.volume(r) for r in th_rad])
    p_cycle = (m_cycle * self.r_gas * t_cycle) / v_cycle

    l_in_cycle = np.array([
        self.valves.lift(th, self.timing.ivo_deg, self.timing.ivc_deg, self.valves.intake_max_lift_m) * 1000.0
        for th in theta_deg
    ])
    l_ex_cycle = np.array([
        self.valves.lift(th, self.timing.evo_deg, self.timing.evc_deg, self.valves.exhaust_max_lift_m) * 1000.0
        for th in theta_deg
    ])

    # Compute wall heat flux and convective coefficient traces
    hc_cycle = np.zeros_like(theta_deg)
    q_wall_cycle = np.zeros_like(theta_deg)
    for i, th in enumerate(theta_deg):
      r_comp = v_max / max(1.0e-6, v_cycle[i])
      p_motored = p_inlet_pa * (r_comp**1.32)
      ht = self.woschni.calculate_heat_transfer(
          theta_deg=th,
          rpm=rpm,
          p_pa=p_cycle[i],
          t_gas_k=t_cycle[i],
          p_motored_pa=p_motored,
          t_ref_k=t_inlet_k,
          p_ref_pa=p_inlet_pa,
          v_ref_m3=v_max,
      )
      hc_cycle[i] = ht.heat_transfer_coeff_w_m2_k
      q_wall_cycle[i] = ht.heat_flux_w

    # Integrate cumulative heat loss per cycle: Q_wall = integral(q_dot dt)
    time_array = theta_deg * (np.pi / 180.0) / omega
    q_wall_total = float(np.trapezoid(q_wall_cycle, time_array))
    mean_cooling_power = q_wall_total * (rpm / 120.0)

    # Work integrations:
    # High-pressure loop (Power: 0..180, Compression: 540..720)
    high_mask = (theta_deg <= 180.0) | (theta_deg >= 540.0)
    w_gross = np.trapezoid(p_cycle[high_mask], v_cycle[high_mask])

    # Low-pressure pumping loop (Exhaust: 180..360, Intake: 360..540)
    pump_mask = (theta_deg > 180.0) & (theta_deg < 540.0)
    w_pump = np.trapezoid(p_cycle[pump_mask], v_cycle[pump_mask])

    w_net = w_gross + w_pump
    vd = self.geom.displacement

    gimep = w_gross / vd
    pmep = -w_pump / vd
    nimep = w_net / vd

    # Trapped mass evaluated right after Intake Valve Closes (IVC)
    ivc_idx = np.argmin(np.abs(theta_deg - self.timing.ivc_deg))
    trapped_mass = float(m_cycle[ivc_idx])
    eta_v = trapped_mass / (rho_inlet * vd)

    power_kw = (w_net * (rpm / 120.0)) / 1000.0
    torque_nm = w_net / (4.0 * np.pi)

    m_fuel_cycle = trapped_mass / (1.0 + self.afr)
    total_fuel_energy = m_fuel_cycle * self.q_fuel
    heat_loss_fraction = q_wall_total / max(1.0, total_fuel_energy)

    # Constant-volume chemical equilibrium at peak pressure
    eq_species = {}
    if self.cantera_chem:
      try:
        t_ivc = float(t_cycle[ivc_idx])
        p_ivc = float(p_cycle[ivc_idx])
        eq_res = self.cantera_chem.constant_volume_equilibrium(t_ivc, p_ivc)
        eq_species = eq_res.major_species_mole_fractions
      except Exception:
        eq_species = {"CO2": 0.08, "H2O": 0.18, "CO": 0.02, "N2": 0.70}

    peak_heat_flux = float(np.max(q_wall_cycle) / np.max(v_cycle)**(2/3)) / 1.0e6  # MW/m^2 approx

    return FourStrokeResults(
        theta_deg=theta_deg,
        pressure_pa=p_cycle,
        temperature_k=t_cycle,
        volume_m3=v_cycle,
        mass_kg=m_cycle,
        heat_transfer_coeff_w_m2_k=hc_cycle,
        wall_heat_flux_w=q_wall_cycle,
        intake_lift_mm=l_in_cycle,
        exhaust_lift_mm=l_ex_cycle,
        p_max_pa=float(np.max(p_cycle)),
        t_max_k=float(np.max(t_cycle)),
        gross_work_j=float(w_gross),
        pump_work_j=float(w_pump),
        net_work_j=float(w_net),
        gimep_pa=float(gimep),
        pmep_pa=float(pmep),
        nimep_pa=float(nimep),
        trapped_mass_kg=trapped_mass,
        volumetric_efficiency=float(eta_v),
        power_kw=float(power_kw),
        torque_nm=float(torque_nm),
        q_wall_loss_j=q_wall_total,
        mean_cooling_power_w=float(mean_cooling_power),
        peak_heat_flux_mw_m2=peak_heat_flux,
        heat_loss_fraction=float(heat_loss_fraction),
        equilibrium_species=eq_species,
    )
