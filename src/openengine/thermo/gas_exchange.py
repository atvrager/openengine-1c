"""1D Gas Exchange and Full 720-Degree Four-Stroke Engine Cycle Simulator."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict
import cantera as ct
import numpy as np
from openengine.thermo.cantera_kinetics import (
    E10_LHV_J_KG,
    E10_STOICH_AFR,
    US_87_AKI,
    CanteraThermochemistry,
    FuelBlendSpec,
    FuelConfigSection,
)
from openengine.thermo.cycle import EngineGeometry, wiebe_burned_fraction
from openengine.thermo.knock import (
    DEFAULT_KNOCK_SAFETY_CEILING,
    KnockEvaluationResult,
    KnockSafeOperatingMap,
    KnockSweepPoint,
    evaluate_cycle_knock,
    select_best_knock_safe_map,
    soc_from_spark_advance,
)
from openengine.thermo.woschni import WoschniHeatTransfer
from scipy.integrate import solve_ivp

FULL_CYCLE_DEG: float = 720.0
HALF_CYCLE_DEG: float = 360.0
POWER_STROKE_END_DEG: float = 180.0
COMPRESSION_STROKE_START_DEG: float = 540.0
DEFAULT_EVO_DEG: float = 135.0
DEFAULT_EVC_DEG: float = 375.0
DEFAULT_IVO_DEG: float = 345.0
DEFAULT_IVC_DEG: float = 585.0

DEFAULT_INTAKE_DIAM_M: float = 0.030
DEFAULT_EXHAUST_DIAM_M: float = 0.026
DEFAULT_INTAKE_LIFT_M: float = 0.0085
DEFAULT_EXHAUST_LIFT_M: float = 0.0080
DEFAULT_SEAT_ANGLE_DEG: float = 45.0
DEFAULT_INTAKE_CD: float = 0.65
DEFAULT_EXHAUST_CD: float = 0.62

DEFAULT_GAMMA_AIR: float = 1.35
DEFAULT_R_GAS_J_KG_K: float = 287.05
DEFAULT_T_WALL_K: float = 430.0
DEFAULT_WIEBE_SOC_DEG: float = -15.0
DEFAULT_WIEBE_DURATION_DEG: float = 42.0
DEFAULT_WIEBE_A: float = 5.0
DEFAULT_WIEBE_M: float = 2.0
COMBUSTION_EFFICIENCY: float = 0.92
VIBRATIONAL_CV_SLOPE_PER_K: float = 3.8e-4
CV_REF_TEMP_K: float = 300.0
MOTORED_POLYTROPIC_EXP: float = 1.32

MIN_ORIFICE_AREA_M2: float = 1.0e-9
MIN_ORIFICE_PRESSURE_PA: float = 1.0e3
MIN_PRESSURE_RATIO: float = 1.0e-5
MIN_MASS_KG: float = 1.0e-5
MIN_GAS_TEMP_K: float = 200.0
MIN_VOLUME_M3: float = 1.0e-6
FD_STEP_DEG: float = 0.1
DEG_TO_RAD: float = float(np.pi) / 180.0
TWO_PI: float = 2.0 * float(np.pi)
FOUR_PI: float = 4.0 * float(np.pi)
SECONDS_PER_MINUTE: float = 60.0
CYCLES_PER_TWO_REV_DIVISOR: float = 120.0
WATT_TO_KW: float = 1000.0
METER_TO_MM: float = 1000.0
WATT_TO_MW: float = 1.0e6
SURFACE_VOLUME_EXP: float = 2.0 / 3.0
ODE_RTOL: float = 1.0e-4
ODE_ATOL: float = 1.0e-5
DEFAULT_RATED_RPM: float = 5500.0
DEFAULT_INLET_PRESSURE_PA: float = 101325.0
DEFAULT_INLET_TEMP_K: float = 298.15
DEFAULT_EXHAUST_PRESSURE_PA: float = 105000.0
DEFAULT_EXHAUST_TEMP_K: float = 800.0
DEFAULT_N_CYCLES: int = 2
HALF_DISPLACEMENT_FACTOR: float = 0.5
FALLBACK_CO2_MOLE_FRAC: float = 0.10
FALLBACK_H2O_MOLE_FRAC: float = 0.13
FALLBACK_CO_MOLE_FRAC: float = 0.02
FALLBACK_N2_MOLE_FRAC: float = 0.71
DEFAULT_SWEEP_CRS: Tuple[float, ...] = (8.6, 9.0, 9.4, 9.8)
DEFAULT_SWEEP_RPMS: Tuple[float, ...] = (3500.0, 5500.0)
DEFAULT_SWEEP_SPARKS_DEG_BTDC: Tuple[float, ...] = (14.0, 16.0, 18.0, 22.0, 26.0)


class GeometryConfigSection(TypedDict):
  """Typed YAML schema for engine geometry configuration."""

  bore: float
  stroke: float
  con_rod_length: float
  compression_ratio: float


class ValvesConfigSection(TypedDict, total=False):
  """Typed YAML schema for poppet valve configuration."""

  intake_diameter: float
  exhaust_diameter: float
  max_lift: float
  exhaust_max_lift: float


class OperatingConfigSection(TypedDict, total=False):
  """Typed YAML schema for engine operating conditions."""

  air_fuel_ratio: float


class CombustionConfigSection(TypedDict, total=False):
  """Typed YAML schema for 0D Wiebe combustion settings."""

  a: float
  m: float
  duration_deg: float
  start_deg_btdc: float


class EngineSpecsConfig(TypedDict, total=False):
  """Typed root YAML schema for config/engine_specs.yaml."""

  geometry: GeometryConfigSection
  valves: ValvesConfigSection
  fuel: FuelConfigSection
  operating_conditions: OperatingConfigSection
  combustion_model: CombustionConfigSection


@dataclass(slots=True)
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
  evo_deg: float = DEFAULT_EVO_DEG  # Exhaust Valve Opens (45 deg before BDC)
  evc_deg: float = DEFAULT_EVC_DEG  # Exhaust Valve Closes (15 deg after Gas Exchange TDC)

  # Intake events
  ivo_deg: float = DEFAULT_IVO_DEG  # Intake Valve Opens (15 deg before Gas Exchange TDC)
  ivc_deg: float = DEFAULT_IVC_DEG  # Intake Valve Closes (45 deg after BDC)


@dataclass(slots=True)
class ValveGeometry:
  """Dimensions and discharge coefficients for poppet valves."""

  intake_diameter_m: float = DEFAULT_INTAKE_DIAM_M
  exhaust_diameter_m: float = DEFAULT_EXHAUST_DIAM_M
  intake_max_lift_m: float = DEFAULT_INTAKE_LIFT_M
  exhaust_max_lift_m: float = DEFAULT_EXHAUST_LIFT_M
  seat_angle_deg: float = DEFAULT_SEAT_ANGLE_DEG
  intake_cd: float = DEFAULT_INTAKE_CD
  exhaust_cd: float = DEFAULT_EXHAUST_CD

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
    th = np.atleast_1d(th) % FULL_CYCLE_DEG

    duration = close_deg - open_deg
    lift_vals = np.zeros_like(th, dtype=float)

    if open_deg < close_deg:
      mask = (th >= open_deg) & (th <= close_deg)
      progress = (th[mask] - open_deg) / duration
      lift_vals[mask] = max_lift * np.sin(np.pi * progress)
    else:
      # Wraps around 720 deg
      mask = (th >= open_deg) | (th <= close_deg)
      norm_th = np.where(th < open_deg, th + FULL_CYCLE_DEG, th)
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
    gamma: float = DEFAULT_GAMMA_AIR,
    r_gas: float = DEFAULT_R_GAS_J_KG_K,
) -> float:
  """Calculate 1D isentropic compressible orifice mass flow rate [kg/s].

  Returns positive flow from upstream to downstream.
  Handles subsonic and choked flow regimes smoothly.
  """
  if area_m2 <= MIN_ORIFICE_AREA_M2 or p_up_pa <= MIN_ORIFICE_PRESSURE_PA:
    return 0.0

  pr = max(MIN_PRESSURE_RATIO, min(1.0, p_down_pa / p_up_pa))
  pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

  c_sound_up = np.sqrt(gamma * r_gas * t_up_k)

  if pr <= pr_crit:
    # Choked (sonic) flow
    psi = np.sqrt(
        gamma * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
    )
  else:
    # Subsonic flow
    term1 = 2.0 * gamma / (gamma - 1.0)
    term2 = (pr ** (2.0 / gamma)) - (pr ** ((gamma + 1.0) / gamma))
    psi = np.sqrt(max(0.0, term1 * term2))

  return float(cd * area_m2 * (p_up_pa / c_sound_up) * psi)


@dataclass(slots=True)
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
  fuel_aki: float = US_87_AKI
  end_gas_temp_max_k: float = 0.0
  knock_integral: float = 0.0
  knock_margin: float = 1.0
  is_knocking: bool = False
  knock_onset_deg: Optional[float] = None


class FourStrokeSimulator:
  """Full 720-degree 4-stroke cycle and gas exchange simulator."""

  def __init__(
      self,
      geom: EngineGeometry,
      valves: Optional[ValveGeometry] = None,
      timing: Optional[ValveTiming] = None,
      gamma_air: float = DEFAULT_GAMMA_AIR,
      r_gas: float = DEFAULT_R_GAS_J_KG_K,
      q_fuel: Optional[float] = None,
      afr: Optional[float] = None,
      use_woschni: bool = True,
      t_wall_k: float = DEFAULT_T_WALL_K,
      use_cantera: bool = True,
      wiebe_soc_deg: float = DEFAULT_WIEBE_SOC_DEG,
      wiebe_duration_deg: float = DEFAULT_WIEBE_DURATION_DEG,
      wiebe_a: float = DEFAULT_WIEBE_A,
      wiebe_m: float = DEFAULT_WIEBE_M,
      fuel: Optional[FuelBlendSpec] = None,
  ) -> None:
    self.geom = geom
    self.valves = valves if valves is not None else ValveGeometry()
    self.timing = timing if timing is not None else ValveTiming()
    self.fuel: FuelBlendSpec = fuel if fuel is not None else FuelBlendSpec.us_87_e10()
    self.gamma = gamma_air
    self.r_gas = r_gas
    self.q_fuel: float = float(q_fuel) if q_fuel is not None else self.fuel.lhv_j_kg
    self.afr: float = float(afr) if afr is not None else self.fuel.stoich_afr
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
      self.cantera_chem: Optional[CanteraThermochemistry] = CanteraThermochemistry(
          fuel_blend=self.fuel
      )
    else:
      self.cantera_chem = None

  @classmethod
  def from_config(
      cls,
      cfg: EngineSpecsConfig,
      t_wall_k: float = DEFAULT_T_WALL_K,
      use_woschni: bool = True,
      use_cantera: bool = True,
  ) -> "FourStrokeSimulator":
    """Construct a FourStrokeSimulator from a parsed engine_specs.yaml mapping."""
    g_cfg = cfg["geometry"]
    geom = EngineGeometry(
        bore=float(g_cfg["bore"]),
        stroke=float(g_cfg["stroke"]),
        con_rod_length=float(g_cfg["con_rod_length"]),
        compression_ratio=float(g_cfg["compression_ratio"]),
    )

    v_cfg = cfg.get("valves", {})
    max_lift = float(v_cfg.get("max_lift", DEFAULT_INTAKE_LIFT_M))
    valves = ValveGeometry(
        intake_diameter_m=float(v_cfg.get("intake_diameter", DEFAULT_INTAKE_DIAM_M)),
        exhaust_diameter_m=float(v_cfg.get("exhaust_diameter", DEFAULT_EXHAUST_DIAM_M)),
        intake_max_lift_m=max_lift,
        exhaust_max_lift_m=float(v_cfg.get("exhaust_max_lift", max_lift)),
    )

    fuel_spec = FuelBlendSpec.from_config(cfg)
    op_cfg = cfg.get("operating_conditions", {})
    afr_val = float(op_cfg.get("air_fuel_ratio", fuel_spec.stoich_afr))

    comb_cfg = cfg.get("combustion_model", {})
    start_btdc = float(comb_cfg.get("start_deg_btdc", -DEFAULT_WIEBE_SOC_DEG))

    return cls(
        geom=geom,
        valves=valves,
        q_fuel=fuel_spec.lhv_j_kg,
        afr=afr_val,
        use_woschni=use_woschni,
        t_wall_k=t_wall_k,
        use_cantera=use_cantera,
        wiebe_soc_deg=-start_btdc,
        wiebe_duration_deg=float(comb_cfg.get("duration_deg", DEFAULT_WIEBE_DURATION_DEG)),
        wiebe_a=float(comb_cfg.get("a", DEFAULT_WIEBE_A)),
        wiebe_m=float(comb_cfg.get("m", DEFAULT_WIEBE_M)),
        fuel=fuel_spec,
    )

  def _valve_mass_and_enthalpy_fluxes(
      self,
      th_mod: float,
      p: float,
      t: float,
      cp: float,
      gamma_inst: float,
      p_inlet_pa: float,
      t_inlet_k: float,
      p_exhaust_pa: float,
      t_exhaust_k: float,
  ) -> Tuple[float, float]:
    """Compute net mass flow rate dmdt and port enthalpy rate [W]."""
    l_in = float(
        self.valves.lift(
            th_mod,
            self.timing.ivo_deg,
            self.timing.ivc_deg,
            self.valves.intake_max_lift_m,
        )
    )
    l_ex = float(
        self.valves.lift(
            th_mod,
            self.timing.evo_deg,
            self.timing.evc_deg,
            self.valves.exhaust_max_lift_m,
        )
    )
    a_in = self.valves.curtain_area(l_in, self.valves.intake_diameter_m)
    a_ex = self.valves.curtain_area(l_ex, self.valves.exhaust_diameter_m)

    if p_inlet_pa >= p:
      m_dot_in = compressible_orifice_flow(
          p_inlet_pa, t_inlet_k, p, self.valves.intake_cd, a_in, gamma_inst, self.r_gas
      )
      h_in_net = m_dot_in * cp * t_inlet_k
    else:
      m_dot_in = -compressible_orifice_flow(
          p, t, p_inlet_pa, self.valves.intake_cd, a_in, gamma_inst, self.r_gas
      )
      h_in_net = m_dot_in * cp * t

    if p >= p_exhaust_pa:
      m_dot_ex = compressible_orifice_flow(
          p, t, p_exhaust_pa, self.valves.exhaust_cd, a_ex, gamma_inst, self.r_gas
      )
      h_ex_net = -m_dot_ex * cp * t
    else:
      m_dot_ex = -compressible_orifice_flow(
          p_exhaust_pa, t_exhaust_k, p, self.valves.exhaust_cd, a_ex, gamma_inst, self.r_gas
      )
      h_ex_net = -m_dot_ex * cp * t_exhaust_k

    return m_dot_in - m_dot_ex, h_in_net + h_ex_net

  def _combustion_heat_release_rate(
      self,
      th_mod: float,
      m: float,
      omega: float,
  ) -> float:
    """Compute instantaneous Wiebe combustion heat release rate [W]."""
    ca_firing = th_mod - FULL_CYCLE_DEG if th_mod > HALF_CYCLE_DEG else th_mod
    if not (
        self.wiebe_soc_deg
        <= ca_firing
        <= (self.wiebe_soc_deg + self.wiebe_duration_deg)
    ):
      return 0.0

    xb1 = float(
        wiebe_burned_fraction(
            ca_firing,
            theta_soc_deg=self.wiebe_soc_deg,
            duration_deg=self.wiebe_duration_deg,
            a=self.wiebe_a,
            m=self.wiebe_m,
        )
    )
    xb2 = float(
        wiebe_burned_fraction(
            ca_firing + FD_STEP_DEG,
            theta_soc_deg=self.wiebe_soc_deg,
            duration_deg=self.wiebe_duration_deg,
            a=self.wiebe_a,
            m=self.wiebe_m,
        )
    )
    dxb_dth = (xb2 - xb1) / (FD_STEP_DEG * DEG_TO_RAD)
    m_fuel = m / (1.0 + self.afr)
    q_total = m_fuel * self.q_fuel * COMBUSTION_EFFICIENCY
    return float(dxb_dth * omega * q_total)

  def _evaluate_equilibrium_species(
      self,
      t_ivc: float,
      p_ivc: float,
  ) -> Dict[str, float]:
    """Evaluate constant-volume Cantera equilibrium products from IVC state."""
    if self.cantera_chem is None:
      return {}
    try:
      eq_res = self.cantera_chem.constant_volume_equilibrium(t_ivc, p_ivc)
      return eq_res.major_species_mole_fractions
    except (RuntimeError, ValueError, ct.CanteraError):
      return {
          "CO2": FALLBACK_CO2_MOLE_FRAC,
          "H2O": FALLBACK_H2O_MOLE_FRAC,
          "CO": FALLBACK_CO_MOLE_FRAC,
          "N2": FALLBACK_N2_MOLE_FRAC,
      }

  def solve_cycle(
      self,
      rpm: float = DEFAULT_RATED_RPM,
      p_inlet_pa: float = DEFAULT_INLET_PRESSURE_PA,
      t_inlet_k: float = DEFAULT_INLET_TEMP_K,
      p_exhaust_pa: float = DEFAULT_EXHAUST_PRESSURE_PA,
      t_exhaust_k: float = DEFAULT_EXHAUST_TEMP_K,
      n_cycles: int = DEFAULT_N_CYCLES,
  ) -> FourStrokeResults:
    """Simulate full 720 deg cycles until convergence.

    Crank angle:
      0..180:   Power
      180..360: Exhaust
      360..540: Intake
      540..720: Compression
    """
    omega = rpm * (TWO_PI / SECONDS_PER_MINUTE)
    rho_inlet = p_inlet_pa / (self.r_gas * t_inlet_k)
    v_max = self.geom.displacement + self.geom.clearance_volume

    # State variables: y = [m, T]
    # Initialize near BDC intake: fresh charge
    m_init = p_inlet_pa * v_max / (self.r_gas * t_inlet_k)
    t_init = t_inlet_k

    def ode_system(theta_deg: float, y: Sequence[float]) -> List[float]:
      """Evaluate crank-angle derivatives [dm/dtheta, dT/dtheta] for RK23."""
      m = max(MIN_MASS_KG, y[0])
      t = max(MIN_GAS_TEMP_K, y[1])

      th_mod = theta_deg % FULL_CYCLE_DEG
      th_rad = np.radians(th_mod)
      v = float(self.geom.volume(th_rad))
      p = (m * self.r_gas * t) / v

      # Piston kinematics dV/dtheta
      rod_ratio = self.geom.rod_to_crank_ratio
      dvdth = HALF_DISPLACEMENT_FACTOR * self.geom.displacement * (
          np.sin(th_rad)
          + (
              np.sin(2.0 * th_rad)
              / (2.0 * np.sqrt(rod_ratio**2 - np.sin(th_rad) ** 2))
          )
      )
      dvdt = dvdth * omega

      # Temperature-dependent specific heat capacity (vibrational activation)
      cv = self.cv0 * (1.0 + VIBRATIONAL_CV_SLOPE_PER_K * max(0.0, t - CV_REF_TEMP_K))
      cp = cv + self.r_gas
      gamma_inst = cp / cv

      dmdt, h_port_net = self._valve_mass_and_enthalpy_fluxes(
          th_mod=th_mod,
          p=p,
          t=t,
          cp=cp,
          gamma_inst=gamma_inst,
          p_inlet_pa=p_inlet_pa,
          t_inlet_k=t_inlet_k,
          p_exhaust_pa=p_exhaust_pa,
          t_exhaust_k=t_exhaust_k,
      )
      dq_comb_dt = self._combustion_heat_release_rate(th_mod=th_mod, m=m, omega=omega)

      if self.use_woschni:
        r_comp = v_max / max(MIN_VOLUME_M3, v)
        p_motored = p_inlet_pa * (r_comp**MOTORED_POLYTROPIC_EXP)
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

      # Energy equation: m * cv * dT/dt = dQ_comb - dQ_wall - P * dV/dt + energy_flux
      u = cv * t
      energy_flux = h_port_net - u * dmdt
      dtdt = (dq_comb_dt - dq_wall_dt - p * dvdt + energy_flux) / (m * cv)

      dmdth_deg = (dmdt / omega) * DEG_TO_RAD
      dtdth_deg = (dtdt / omega) * DEG_TO_RAD
      return [float(dmdth_deg), float(dtdth_deg)]

    n_eval = int(FULL_CYCLE_DEG) * n_cycles
    th_eval = np.linspace(0.0, FULL_CYCLE_DEG * n_cycles, n_eval)
    sol = solve_ivp(
        ode_system,
        t_span=(0.0, FULL_CYCLE_DEG * n_cycles),
        y0=[m_init, t_init],
        t_eval=th_eval,
        method="RK23",
        rtol=ODE_RTOL,
        atol=ODE_ATOL,
    )

    cycle_pts = int(FULL_CYCLE_DEG)
    th_final = sol.t[-cycle_pts:] % FULL_CYCLE_DEG
    sort_idx = np.argsort(th_final)
    theta_deg = th_final[sort_idx]
    m_cycle = sol.y[0][-cycle_pts:][sort_idx]
    t_cycle = sol.y[1][-cycle_pts:][sort_idx]

    th_rad = np.radians(theta_deg)
    v_cycle = np.array([self.geom.volume(r) for r in th_rad])
    p_cycle = (m_cycle * self.r_gas * t_cycle) / v_cycle

    l_in_cycle = np.array([
        self.valves.lift(
            th,
            self.timing.ivo_deg,
            self.timing.ivc_deg,
            self.valves.intake_max_lift_m,
        )
        * METER_TO_MM
        for th in theta_deg
    ])
    l_ex_cycle = np.array([
        self.valves.lift(
            th,
            self.timing.evo_deg,
            self.timing.evc_deg,
            self.valves.exhaust_max_lift_m,
        )
        * METER_TO_MM
        for th in theta_deg
    ])

    hc_cycle = np.zeros_like(theta_deg)
    q_wall_cycle = np.zeros_like(theta_deg)
    for i, th in enumerate(theta_deg):
      r_comp = v_max / max(MIN_VOLUME_M3, v_cycle[i])
      p_motored = p_inlet_pa * (r_comp**MOTORED_POLYTROPIC_EXP)
      ht = self.woschni.calculate_heat_transfer(
          theta_deg=float(th),
          rpm=rpm,
          p_pa=float(p_cycle[i]),
          t_gas_k=float(t_cycle[i]),
          p_motored_pa=p_motored,
          t_ref_k=t_inlet_k,
          p_ref_pa=p_inlet_pa,
          v_ref_m3=v_max,
      )
      hc_cycle[i] = ht.heat_transfer_coeff_w_m2_k
      q_wall_cycle[i] = ht.heat_flux_w

    time_array = theta_deg * DEG_TO_RAD / omega
    q_wall_total = float(np.trapezoid(q_wall_cycle, time_array))
    mean_cooling_power = q_wall_total * (rpm / CYCLES_PER_TWO_REV_DIVISOR)

    high_mask = (theta_deg <= POWER_STROKE_END_DEG) | (
        theta_deg >= COMPRESSION_STROKE_START_DEG
    )
    w_gross = np.trapezoid(p_cycle[high_mask], v_cycle[high_mask])

    pump_mask = (theta_deg > POWER_STROKE_END_DEG) & (
        theta_deg < COMPRESSION_STROKE_START_DEG
    )
    w_pump = np.trapezoid(p_cycle[pump_mask], v_cycle[pump_mask])

    w_net = w_gross + w_pump
    vd = self.geom.displacement

    gimep = w_gross / vd
    pmep = -w_pump / vd
    nimep = w_net / vd

    ivc_idx = int(np.argmin(np.abs(theta_deg - self.timing.ivc_deg)))
    trapped_mass = float(m_cycle[ivc_idx])
    eta_v = trapped_mass / (rho_inlet * vd)

    power_kw = (w_net * (rpm / CYCLES_PER_TWO_REV_DIVISOR)) / WATT_TO_KW
    torque_nm = w_net / FOUR_PI

    m_fuel_cycle = trapped_mass / (1.0 + self.afr)
    total_fuel_energy = m_fuel_cycle * self.q_fuel
    heat_loss_fraction = q_wall_total / max(1.0, total_fuel_energy)

    eq_species = self._evaluate_equilibrium_species(
        t_ivc=float(t_cycle[ivc_idx]),
        p_ivc=float(p_cycle[ivc_idx]),
    )

    peak_heat_flux = (
        float(np.max(q_wall_cycle) / (np.max(v_cycle) ** SURFACE_VOLUME_EXP))
        / WATT_TO_MW
    )

    knock_eval: KnockEvaluationResult = evaluate_cycle_knock(
        theta_deg=theta_deg,
        pressure_pa=p_cycle,
        temperature_k=t_cycle,
        rpm=rpm,
        fuel=self.fuel,
        wiebe_soc_deg=self.wiebe_soc_deg,
        wiebe_duration_deg=self.wiebe_duration_deg,
        wiebe_a=self.wiebe_a,
        wiebe_m=self.wiebe_m,
        ivc_deg=self.timing.ivc_deg,
    )

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
        fuel_aki=knock_eval.fuel_aki,
        end_gas_temp_max_k=knock_eval.peak_end_gas_temp_k,
        knock_integral=knock_eval.knock_integral,
        knock_margin=knock_eval.knock_margin,
        is_knocking=knock_eval.is_knocking,
        knock_onset_deg=knock_eval.knock_onset_crank_angle_deg,
    )

  def sweep_knock_safe_operating_map(
      self,
      compression_ratios: Sequence[float] = DEFAULT_SWEEP_CRS,
      rpms: Sequence[float] = DEFAULT_SWEEP_RPMS,
      spark_advances_deg_btdc: Sequence[float] = DEFAULT_SWEEP_SPARKS_DEG_BTDC,
      max_knock_integral: float = DEFAULT_KNOCK_SAFETY_CEILING,
  ) -> KnockSafeOperatingMap:
    """Sweep CR, RPM, and spark advance to select a knock-safe operating map."""
    sweep_points: List[KnockSweepPoint] = []

    for cr in compression_ratios:
      geom_cr = replace(self.geom, compression_ratio=float(cr))
      for spark_btdc in spark_advances_deg_btdc:
        soc_deg = soc_from_spark_advance(float(spark_btdc))
        sim_pt = FourStrokeSimulator(
            geom=geom_cr,
            valves=self.valves,
            timing=self.timing,
            gamma_air=self.gamma,
            r_gas=self.r_gas,
            q_fuel=self.q_fuel,
            afr=self.afr,
            use_woschni=self.use_woschni,
            t_wall_k=self.t_wall_k,
            use_cantera=False,
            wiebe_soc_deg=soc_deg,
            wiebe_duration_deg=self.wiebe_duration_deg,
            wiebe_a=self.wiebe_a,
            wiebe_m=self.wiebe_m,
            fuel=self.fuel,
        )
        for rpm_val in rpms:
          res = sim_pt.solve_cycle(rpm=float(rpm_val))
          sweep_points.append(
              KnockSweepPoint(
                  compression_ratio=float(cr),
                  rpm=float(rpm_val),
                  spark_advance_deg_btdc=float(spark_btdc),
                  wiebe_soc_deg=soc_deg,
                  nimep_pa=res.nimep_pa,
                  power_kw=res.power_kw,
                  torque_nm=res.torque_nm,
                  p_max_pa=res.p_max_pa,
                  end_gas_temp_max_k=res.end_gas_temp_max_k,
                  knock_integral=res.knock_integral,
                  knock_margin=res.knock_margin,
                  is_knocking=res.is_knocking,
              )
          )

    return select_best_knock_safe_map(
        points=sweep_points,
        max_knock_integral=max_knock_integral,
    )
