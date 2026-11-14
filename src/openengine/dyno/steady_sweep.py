"""Steady-State Virtual Brake Dynamometer Sweep and 2D BSFC Fuel Map.

Couples compressible Mikuni VM26 throttle flow, 1D Helmholtz intake ram tuning,
quarter-wave exhaust primary wave scavenging, two-stage MBT knock-safe spark
optimization on US 87-AKI E10 fuel, SAE J1349 atmospheric correction, and
coupled wet-sump splash lubrication (`FourStrokeSimulator`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
import numpy as np
from scipy.optimize import brentq
import yaml

from openengine.cad.connecting_rod import DEFAULT_CENTER_TO_CENTER_M
from openengine.cad.crankshaft import DEFAULT_STROKE_M
from openengine.cad.cylinder_block import DEFAULT_BORE_M
from openengine.cad.cylinder_head import (
    DEFAULT_VALVE_HEAD_EX_M,
    DEFAULT_VALVE_HEAD_IN_M,
    DEFAULT_VALVE_SEAT_ANGLE_DEG,
)
from openengine.cad.twin_metrics import (
    get_cad_compression_ratio,
    get_cad_runner_geometry,
    get_cad_sump_splash_metrics,
)
from openengine.lubrication.rheology import SAE_10W40_NAME, OilGradeSpec
from openengine.lubrication.splash import SplashSystemParams
from openengine.lubrication.thermal import SumpThermalParams
from openengine.thermo.cantera_kinetics import FuelBlendSpec
from openengine.thermo.cycle import EngineGeometry
from openengine.thermo.gas_exchange import (
    DEFAULT_OIL_TEMP_C,
    DEFAULT_WIEBE_A,
    DEFAULT_WIEBE_DURATION_DEG,
    DEFAULT_WIEBE_M,
    FourStrokeSimulator,
    ValveGeometry,
    ValveTiming,
    compressible_orifice_flow,
)
from openengine.thermo.knock import (
    DEFAULT_KNOCK_SAFETY_CEILING,
    evaluate_cycle_knock,
    soc_from_spark_advance,
)

# Unit conversions and physical constants
ZERO_FLOAT: float = 0.0
HALF_FACTOR: float = 0.5
TWO_FACTOR: float = 2.0
FOUR_FACTOR: float = 4.0
FOUR_PI: float = FOUR_FACTOR * float(np.pi)
TWO_PI: float = TWO_FACTOR * float(np.pi)
MM_PER_M: float = 1000.0
WATT_TO_KW: float = 1000.0
PA_TO_BAR: float = 1.0e-5
PA_TO_HPA: float = 0.01
KW_TO_MECHANICAL_HP: float = 1.34102209
NM_TO_LB_FT: float = 0.73756215
SECONDS_PER_MINUTE: float = 60.0
CYCLES_PER_TWO_REV_DIVISOR: float = 120.0
KELVIN_OFFSET: float = 273.15
DEG_TO_RAD: float = float(np.pi) / 180.0
FULL_CYCLE_DEG: float = 720.0
HALF_CYCLE_DEG: float = 360.0
POWER_STROKE_END_DEG: float = 180.0
DEFAULT_EXHAUST_TEMP_K: float = 800.0
DEFAULT_T_WALL_K: float = 430.0
HIGH_PRESSURE_EVO_END_DEG: float = 135.0
ROUND_VALUE_DECIMALS: int = 4
ROUND_ONE_DECIMAL: int = 1
ROUND_TWO_DECIMALS: int = 2

# SAE J1349 standard reference atmosphere and humidity constants
SAE_J1349_REF_DRY_PRESSURE_PA: float = 99000.0
SAE_J1349_REF_TEMP_K: float = 298.15
SAE_J1349_COEFF_A: float = 1.180
SAE_J1349_COEFF_B: float = 0.180
SAE_TEMP_EXP: float = 0.5
DEFAULT_AMBIENT_PRESSURE_PA: float = 101325.0
DEFAULT_AMBIENT_TEMP_K: float = 298.15
DEFAULT_REL_HUMIDITY: float = 0.30
MAGNUS_SAT_BASE_PA: float = 610.78
MAGNUS_COEFF_A: float = 17.27
MAGNUS_COEFF_B_C: float = 237.3
MIN_DRY_PRESSURE_PA: float = 20000.0

# 1D Helmholtz intake and quarter-wave exhaust tuning constants
DEFAULT_BELLMOUTH_LENGTH_M: float = 0.110
DEFAULT_EXHAUST_PRIMARY_LENGTH_M: float = 0.620
OPEN_END_CORRECTION_RADIUS_FACTOR: float = 0.85
INTAKE_GAMMA: float = 1.40
EXHAUST_GAMMA: float = 1.32
INTAKE_R_GAS: float = 287.05
EXHAUST_R_GAS: float = 289.0
EXHAUST_PIPE_MEAN_TEMP_K: float = 740.0
HELMHOLTZ_FOUR_STROKE_ORDER: float = 2.65
EXHAUST_QUARTER_WAVE_ORDER: float = 2.25
CARB_SLIDE_CD: float = 0.88
CARB_IDLE_AREA_FRAC: float = 0.015
CARB_SLIDE_EXP: float = 2.75
BASE_VE_PUMPING_ESTIMATE: float = 0.86
MIN_MAP_FRACTION: float = 0.12
MAX_MAP_FRACTION: float = 0.999
MAP_EQ_RESIDUAL_TOL_KG_S: float = 1.0e-7
INTAKE_RAM_GAIN_PRIMARY: float = 0.065
INTAKE_RAM_WIDTH_PRIMARY: float = 0.24
INTAKE_RAM_DIP_RATIO: float = 0.60
INTAKE_RAM_DIP_AMP: float = 0.025
INTAKE_RAM_DIP_WIDTH: float = 0.20
INTAKE_RAM_GAIN_HARMONIC2: float = 0.022
INTAKE_RAM_HARMONIC2_RATIO: float = 0.48
INTAKE_RAM_WIDTH2: float = 0.16
THROTTLE_WAVE_DAMP_EXP: float = 2.0
BASE_EXHAUST_DELTA_P_PA: float = 1500.0
EXHAUST_MUFFLER_QUAD_DROP_PA: float = 4200.0
EXHAUST_REF_RPM: float = 5500.0
EXHAUST_SCAVENGE_AMP_PA: float = 4500.0
EXHAUST_SCAVENGE_WIDTH: float = 0.26
MIN_INLET_PRESSURE_PA: float = 18000.0
MIN_EXHAUST_PRESSURE_PA: float = 85000.0
MIN_RPM: float = 400.0

# Spark advance auto-tuner and closed high-pressure RK4 constants
DEFAULT_SPEC_YAML_PATH: Path = Path("config/engine_specs.yaml")
DEFAULT_DYNO_RPMS: Tuple[float, ...] = (
    1400.0,
    2000.0,
    2500.0,
    3000.0,
    3500.0,
    4000.0,
    4500.0,
    5000.0,
    5500.0,
    6000.0,
    6500.0,
    7000.0,
)
DEFAULT_THROTTLE_FRACTIONS: Tuple[float, ...] = (0.22, 0.45, 0.72, 1.00)
BASELINE_SPARK_BTDC: float = 14.0
MIN_SPARK_DEG_BTDC: float = -6.0
MAX_SPARK_DEG_BTDC: float = 36.0
SPARK_GRID_STEP_DEG: float = 2.0
COARSE_SPARK_STEP_DEG: float = 6.0
CLOSED_RK4_STEP_DEG: float = 2.0
SEED_CYCLE_COUNT: int = 1
WOT_CYCLE_COUNT: int = 2
STAGE_C_KNOCK_RETARD_MAX_STEPS: int = 8
ONE_RADIAN_STEP_HALF: float = 0.5
RK4_MID_WEIGHT: float = 2.0
RK4_SIXTH_WEIGHT: float = 6.0
COMBUSTION_EFFICIENCY: float = 0.92
VIBRATIONAL_CV_SLOPE_PER_K: float = 3.8e-4
CV_REF_TEMP_K: float = 300.0
MOTORED_POLYTROPIC_EXP: float = 1.32
FD_STEP_DEG: float = 0.1
MIN_VOLUME_M3: float = 1.0e-6
MIN_GAS_TEMP_K: float = 200.0
KNOCK_BORDER_TOL_DEG: float = 0.1
DEFAULT_BSFC_SENTINEL: float = 9999.0
MIN_POSITIVE_BMEP_BAR: float = 0.05
OVERRUN_BSFC_CAP_G_KWH: float = 1500.0
SENTINEL_LARGE_VALUE: float = 1.0e12


@dataclass(frozen=True, slots=True)
class DynoRoomConditions:
  """Dyno cell atmospheric pressure, temperature, and relative humidity."""

  ambient_pressure_pa: float = DEFAULT_AMBIENT_PRESSURE_PA
  ambient_temp_k: float = DEFAULT_AMBIENT_TEMP_K
  relative_humidity: float = DEFAULT_REL_HUMIDITY


@dataclass(frozen=True, slots=True)
class HelmholtzRunnerSpec:
  """1D acoustic intake tract and exhaust primary geometry bound to CAD."""

  cad_intake_tract_length_m: float
  bellmouth_length_m: float
  intake_runner_diam_m: float
  carb_venturi_diam_m: float
  exhaust_primary_length_m: float
  exhaust_pipe_diam_m: float

  @classmethod
  def from_cad(
      cls,
      bellmouth_length_m: float = DEFAULT_BELLMOUTH_LENGTH_M,
      exhaust_primary_length_m: float = DEFAULT_EXHAUST_PRIMARY_LENGTH_M,
  ) -> "HelmholtzRunnerSpec":
    """Construct runner acoustic dimensions directly from live CAD B-Rep geometry."""
    tract_m, in_diam_m, carb_diam_m, ex_diam_m = get_cad_runner_geometry()
    return cls(
        cad_intake_tract_length_m=tract_m,
        bellmouth_length_m=bellmouth_length_m,
        intake_runner_diam_m=in_diam_m,
        carb_venturi_diam_m=carb_diam_m,
        exhaust_primary_length_m=exhaust_primary_length_m,
        exhaust_pipe_diam_m=ex_diam_m,
    )


@dataclass(frozen=True, slots=True)
class SteadyDynoConfig:
  """Configuration for a steady-state multi-throttle virtual dynamometer sweep."""

  rpms: Tuple[float, ...] = DEFAULT_DYNO_RPMS
  throttles: Tuple[float, ...] = DEFAULT_THROTTLE_FRACTIONS
  room: DynoRoomConditions = DynoRoomConditions()
  runner: Optional[HelmholtzRunnerSpec] = None
  max_knock_integral: float = DEFAULT_KNOCK_SAFETY_CEILING
  min_spark_deg_btdc: float = MIN_SPARK_DEG_BTDC
  max_spark_deg_btdc: float = MAX_SPARK_DEG_BTDC
  spark_step_deg: float = SPARK_GRID_STEP_DEG
  config_path: Path = DEFAULT_SPEC_YAML_PATH
  use_live_twin: bool = True


@dataclass(frozen=True, slots=True)
class DynoOperatingPoint:
  """Converged brake performance, knock safety, and port state at one (RPM, throttle)."""

  rpm: float
  throttle: float
  manifold_pressure_pa: float
  exhaust_backpressure_pa: float
  helmholtz_ram_ratio: float
  spark_advance_deg_btdc: float
  is_knock_limited: bool
  knock_integral: float
  knock_margin: float
  volumetric_efficiency: float
  nimep_bar: float
  pmep_bar: float
  fmep_bar: float
  bmep_bar: float
  mechanical_efficiency: float
  oil_temp_c: float
  observed_brake_torque_nm: float
  observed_brake_power_kw: float
  sae_correction_factor: float
  sae_brake_torque_nm: float
  sae_brake_torque_lb_ft: float
  sae_brake_power_kw: float
  sae_brake_power_hp: float
  bsfc_g_kwh: float
  p_max_pa: float
  mean_cooling_power_w: float
  evo_pressure_pa: float
  evo_temp_k: float

  def to_dict(self) -> Dict[str, object]:
    """Serialize operating point metrics into a JSON-safe dictionary."""
    return {
        "rpm": round(self.rpm, ROUND_ONE_DECIMAL),
        "throttle": round(self.throttle, ROUND_VALUE_DECIMALS),
        "manifold_pressure_pa": round(self.manifold_pressure_pa, ROUND_ONE_DECIMAL),
        "exhaust_backpressure_pa": round(self.exhaust_backpressure_pa, ROUND_ONE_DECIMAL),
        "helmholtz_ram_ratio": round(self.helmholtz_ram_ratio, ROUND_VALUE_DECIMALS),
        "spark_advance_deg_btdc": round(self.spark_advance_deg_btdc, ROUND_ONE_DECIMAL),
        "is_knock_limited": self.is_knock_limited,
        "knock_integral": round(self.knock_integral, ROUND_VALUE_DECIMALS),
        "knock_margin": round(self.knock_margin, ROUND_VALUE_DECIMALS),
        "volumetric_efficiency": round(self.volumetric_efficiency, ROUND_VALUE_DECIMALS),
        "nimep_bar": round(self.nimep_bar, ROUND_VALUE_DECIMALS),
        "pmep_bar": round(self.pmep_bar, ROUND_VALUE_DECIMALS),
        "fmep_bar": round(self.fmep_bar, ROUND_VALUE_DECIMALS),
        "bmep_bar": round(self.bmep_bar, ROUND_VALUE_DECIMALS),
        "mechanical_efficiency": round(self.mechanical_efficiency, ROUND_VALUE_DECIMALS),
        "oil_temp_c": round(self.oil_temp_c, ROUND_TWO_DECIMALS),
        "observed_brake_torque_nm": round(self.observed_brake_torque_nm, ROUND_VALUE_DECIMALS),
        "observed_brake_power_kw": round(self.observed_brake_power_kw, ROUND_VALUE_DECIMALS),
        "sae_correction_factor": round(self.sae_correction_factor, ROUND_VALUE_DECIMALS),
        "sae_brake_torque_nm": round(self.sae_brake_torque_nm, ROUND_VALUE_DECIMALS),
        "sae_brake_torque_lb_ft": round(self.sae_brake_torque_lb_ft, ROUND_VALUE_DECIMALS),
        "sae_brake_power_kw": round(self.sae_brake_power_kw, ROUND_VALUE_DECIMALS),
        "sae_brake_power_hp": round(self.sae_brake_power_hp, ROUND_VALUE_DECIMALS),
        "bsfc_g_kwh": round(self.bsfc_g_kwh, ROUND_TWO_DECIMALS),
        "p_max_pa": round(self.p_max_pa, ROUND_ONE_DECIMAL),
        "mean_cooling_power_w": round(self.mean_cooling_power_w, ROUND_ONE_DECIMAL),
        "evo_pressure_pa": round(self.evo_pressure_pa, ROUND_ONE_DECIMAL),
        "evo_temp_k": round(self.evo_temp_k, ROUND_ONE_DECIMAL),
    }


@dataclass(frozen=True, slots=True)
class BsfcMapGrid:
  """Structured 2D Brake Specific Fuel Consumption map across RPM and throttle."""

  rpms: Tuple[float, ...]
  throttles: Tuple[float, ...]
  bmep_bar_grid: Tuple[Tuple[float, ...], ...]
  bsfc_g_kwh_grid: Tuple[Tuple[float, ...], ...]
  sae_power_hp_grid: Tuple[Tuple[float, ...], ...]
  spark_deg_grid: Tuple[Tuple[float, ...], ...]
  min_bsfc_g_kwh: float
  min_bsfc_rpm: float
  min_bsfc_throttle: float
  min_bsfc_bmep_bar: float

  def to_dict(self) -> Dict[str, object]:
    """Serialize 2D BSFC map grids and optimum point into a JSON-safe dictionary."""
    return {
        "rpms": list(self.rpms),
        "throttles": list(self.throttles),
        "bmep_bar_grid": [list(row) for row in self.bmep_bar_grid],
        "bsfc_g_kwh_grid": [list(row) for row in self.bsfc_g_kwh_grid],
        "sae_power_hp_grid": [list(row) for row in self.sae_power_hp_grid],
        "spark_deg_grid": [list(row) for row in self.spark_deg_grid],
        "min_bsfc_g_kwh": round(self.min_bsfc_g_kwh, ROUND_TWO_DECIMALS),
        "min_bsfc_rpm": round(self.min_bsfc_rpm, ROUND_ONE_DECIMAL),
        "min_bsfc_throttle": round(self.min_bsfc_throttle, ROUND_VALUE_DECIMALS),
        "min_bsfc_bmep_bar": round(self.min_bsfc_bmep_bar, ROUND_VALUE_DECIMALS),
    }


@dataclass(frozen=True, slots=True)
class SteadyDynoSweepResult:
  """Complete steady-state dyno sweep deliverables, SAE peak curves, and BSFC map."""

  sae_j1349_factor: float
  helmholtz_tuned_rpm: float
  exhaust_scavenge_rpm: float
  wot_points: Tuple[DynoOperatingPoint, ...]
  bsfc_map: BsfcMapGrid
  peak_sae_power_hp: float
  peak_sae_power_kw: float
  peak_power_rpm: float
  peak_sae_torque_lb_ft: float
  peak_sae_torque_nm: float
  peak_torque_rpm: float
  wot_cycle_theta_deg: Tuple[float, ...]
  wot_cycle_pressures_pa: Dict[float, Tuple[float, ...]]

  def to_dict(self) -> Dict[str, object]:
    """Serialize sweep curves and 2D BSFC map to JSON-safe dictionary."""
    return {
        "sae_j1349_factor": round(self.sae_j1349_factor, ROUND_VALUE_DECIMALS),
        "helmholtz_tuned_rpm": round(self.helmholtz_tuned_rpm, ROUND_ONE_DECIMAL),
        "exhaust_scavenge_rpm": round(self.exhaust_scavenge_rpm, ROUND_ONE_DECIMAL),
        "peak_sae_power_hp": round(self.peak_sae_power_hp, ROUND_VALUE_DECIMALS),
        "peak_sae_power_kw": round(self.peak_sae_power_kw, ROUND_VALUE_DECIMALS),
        "peak_power_rpm": round(self.peak_power_rpm, ROUND_ONE_DECIMAL),
        "peak_sae_torque_lb_ft": round(self.peak_sae_torque_lb_ft, ROUND_VALUE_DECIMALS),
        "peak_sae_torque_nm": round(self.peak_sae_torque_nm, ROUND_VALUE_DECIMALS),
        "peak_torque_rpm": round(self.peak_torque_rpm, ROUND_ONE_DECIMAL),
        "wot_points": [pt.to_dict() for pt in self.wot_points],
        "bsfc_map": self.bsfc_map.to_dict(),
    }


def compute_sae_j1349_factor(room: DynoRoomConditions) -> float:
  """Compute SAE J1349 spark-ignition atmospheric correction factor."""
  temp_c = room.ambient_temp_k - KELVIN_OFFSET
  p_sat_pa = MAGNUS_SAT_BASE_PA * math.exp(
      (MAGNUS_COEFF_A * temp_c) / max(1.0, temp_c + MAGNUS_COEFF_B_C)
  )
  p_vapor_pa = float(np.clip(room.relative_humidity, ZERO_FLOAT, 1.0)) * p_sat_pa
  p_dry_pa = max(MIN_DRY_PRESSURE_PA, room.ambient_pressure_pa - p_vapor_pa)
  air_ratio = SAE_J1349_REF_DRY_PRESSURE_PA / p_dry_pa
  temp_ratio = (room.ambient_temp_k / SAE_J1349_REF_TEMP_K) ** SAE_TEMP_EXP
  return float((SAE_J1349_COEFF_A * air_ratio * temp_ratio) - SAE_J1349_COEFF_B)


def compute_runner_tuned_speeds(
    geom: EngineGeometry,
    runner: HelmholtzRunnerSpec,
    ambient_temp_k: float = DEFAULT_AMBIENT_TEMP_K,
) -> Tuple[float, float]:
  """Return primary Helmholtz intake tuned RPM and quarter-wave exhaust scavenge RPM."""
  d_eff = math.sqrt(runner.carb_venturi_diam_m * runner.intake_runner_diam_m)
  a_in = math.pi * HALF_FACTOR * HALF_FACTOR * (d_eff * d_eff)
  l_in_eff = (
      runner.cad_intake_tract_length_m
      + runner.bellmouth_length_m
      + (OPEN_END_CORRECTION_RADIUS_FACTOR * runner.intake_runner_diam_m * HALF_FACTOR)
  )
  v_eff = geom.clearance_volume + (HALF_FACTOR * geom.displacement)
  c_in = math.sqrt(INTAKE_GAMMA * INTAKE_R_GAS * ambient_temp_k)
  f_helmholtz = (c_in / TWO_PI) * math.sqrt(a_in / max(MIN_VOLUME_M3 * MIN_VOLUME_M3, l_in_eff * v_eff))
  tuned_in_rpm = (SECONDS_PER_MINUTE * f_helmholtz) / HELMHOLTZ_FOUR_STROKE_ORDER

  c_ex = math.sqrt(EXHAUST_GAMMA * EXHAUST_R_GAS * EXHAUST_PIPE_MEAN_TEMP_K)
  l_ex_eff = runner.exhaust_primary_length_m + (OPEN_END_CORRECTION_RADIUS_FACTOR * runner.exhaust_pipe_diam_m * HALF_FACTOR)
  f_quarter = c_ex / (FOUR_FACTOR * l_ex_eff)
  scavenge_rpm = (SECONDS_PER_MINUTE * f_quarter) / EXHAUST_QUARTER_WAVE_ORDER
  return float(tuned_in_rpm), float(scavenge_rpm)


def evaluate_runner_boundary_pressures(
    rpm: float,
    throttle: float,
    geom: EngineGeometry,
    runner: HelmholtzRunnerSpec,
    room: DynoRoomConditions,
) -> Tuple[float, float, float]:
  """Solve coupled throttle MAP, 1D Helmholtz ram boost, and exhaust backpressure."""
  alpha = float(np.clip(throttle, ZERO_FLOAT, 1.0))
  a_carb = math.pi * ((runner.carb_venturi_diam_m * HALF_FACTOR) ** TWO_FACTOR)
  a_throttle = CARB_SLIDE_CD * a_carb * (
      CARB_IDLE_AREA_FRAC + ((1.0 - CARB_IDLE_AREA_FRAC) * (alpha**CARB_SLIDE_EXP))
  )

  p_amb = room.ambient_pressure_pa
  t_amb = room.ambient_temp_k
  pump_conductance = (
      BASE_VE_PUMPING_ESTIMATE * geom.displacement * max(MIN_RPM, rpm)
  ) / (INTAKE_R_GAS * t_amb * CYCLES_PER_TWO_REV_DIVISOR)

  def _flow_balance(p_map_candidate: float) -> float:
    m_dot_th = compressible_orifice_flow(
        p_up_pa=p_amb,
        t_up_k=t_amb,
        p_down_pa=p_map_candidate,
        cd=1.0,
        area_m2=a_throttle,
        gamma=INTAKE_GAMMA,
        r_gas=INTAKE_R_GAS,
    )
    return m_dot_th - (pump_conductance * p_map_candidate)

  p_low = MIN_MAP_FRACTION * p_amb
  p_high = MAX_MAP_FRACTION * p_amb
  if _flow_balance(p_high) >= ZERO_FLOAT:
    p_map_base = p_high
  elif _flow_balance(p_low) <= ZERO_FLOAT:
    p_map_base = p_low
  else:
    p_map_base = float(brentq(_flow_balance, p_low, p_high, xtol=1.0))

  tuned_rpm, scavenge_rpm = compute_runner_tuned_speeds(geom, runner, ambient_temp_k=t_amb)
  rpm_safe = max(MIN_RPM, rpm)
  ram_wave = (
      (INTAKE_RAM_GAIN_PRIMARY * math.exp(-((math.log(rpm_safe / tuned_rpm) / INTAKE_RAM_WIDTH_PRIMARY) ** TWO_FACTOR)))
      - (
          INTAKE_RAM_DIP_AMP
          * math.exp(-((math.log(rpm_safe / (tuned_rpm * INTAKE_RAM_DIP_RATIO)) / INTAKE_RAM_DIP_WIDTH) ** TWO_FACTOR))
      )
      + (
          INTAKE_RAM_GAIN_HARMONIC2
          * math.exp(-((math.log(rpm_safe / (tuned_rpm * INTAKE_RAM_HARMONIC2_RATIO)) / INTAKE_RAM_WIDTH2) ** TWO_FACTOR))
      )
  )
  wave_authority = (p_map_base / p_amb) ** THROTTLE_WAVE_DAMP_EXP
  helmholtz_ratio = 1.0 + (ram_wave * wave_authority)
  p_inlet_eff = max(MIN_INLET_PRESSURE_PA, p_map_base * helmholtz_ratio)

  flow_ratio = (rpm_safe / EXHAUST_REF_RPM) * (p_inlet_eff / p_amb)
  scavenge_suction = EXHAUST_SCAVENGE_AMP_PA * wave_authority * math.exp(
      -((math.log(rpm_safe / scavenge_rpm) / EXHAUST_SCAVENGE_WIDTH) ** TWO_FACTOR)
  )
  p_exhaust_eff = max(
      MIN_EXHAUST_PRESSURE_PA,
      p_amb + BASE_EXHAUST_DELTA_P_PA + (EXHAUST_MUFFLER_QUAD_DROP_PA * (flow_ratio**TWO_FACTOR)) - scavenge_suction,
  )
  return float(p_inlet_eff), float(p_exhaust_eff), float(helmholtz_ratio)


def _load_twin_yaml_sections(
    config_path: Path,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
  """Load normalized YAML config sections without repeated dict-type checks."""
  raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
  cfg: Dict[str, object] = raw if isinstance(raw, dict) else {}
  comb = cfg.get("combustion_model")
  op = cfg.get("operating_conditions")
  lub = cfg.get("lubrication")
  valves = cfg.get("valves")
  return (
      cfg,
      comb if isinstance(comb, dict) else {},
      op if isinstance(op, dict) else {},
      lub if isinstance(lub, dict) else {},
      valves if isinstance(valves, dict) else {},
  )


def build_live_twin_simulator(
    config_path: Path = DEFAULT_SPEC_YAML_PATH,
    use_live_twin: bool = True,
) -> FourStrokeSimulator:
  """Construct a FourStrokeSimulator bound directly to live CAD and engine_specs.yaml."""
  cfg, comb_cfg, op_cfg, lub_cfg, valves_cfg = _load_twin_yaml_sections(config_path)
  if not use_live_twin and cfg:
    return FourStrokeSimulator.from_config(cfg, use_cantera=False)

  geom = EngineGeometry(
      bore=DEFAULT_BORE_M,
      stroke=DEFAULT_STROKE_M,
      con_rod_length=DEFAULT_CENTER_TO_CENTER_M,
      compression_ratio=get_cad_compression_ratio(),
  )
  fuel_spec = FuelBlendSpec.from_config(cfg) if cfg else FuelBlendSpec.us_87_e10()

  max_in_lift = float(valves_cfg.get("max_lift", ValveGeometry.intake_max_lift_m))
  max_ex_lift = float(valves_cfg.get("exhaust_max_lift", max_in_lift))
  valves = ValveGeometry(
      intake_diameter_m=DEFAULT_VALVE_HEAD_IN_M,
      exhaust_diameter_m=DEFAULT_VALVE_HEAD_EX_M,
      intake_max_lift_m=max_in_lift,
      exhaust_max_lift_m=max_ex_lift,
      seat_angle_deg=DEFAULT_VALVE_SEAT_ANGLE_DEG,
  )

  oil_grade = str(lub_cfg.get("oil_grade", SAE_10W40_NAME))
  sump_vol_l, dipper_imm_mm, dipper_rad_mm = get_cad_sump_splash_metrics()
  base_splash = SplashSystemParams.from_config(cfg) if cfg else SplashSystemParams()
  splash_params = replace(
      base_splash,
      sump_volume_l=sump_vol_l,
      dipper_immersion_mm=dipper_imm_mm,
      dipper_tip_radius_mm=dipper_rad_mm,
  )
  thermal_params = SumpThermalParams.from_config(cfg) if cfg else SumpThermalParams()

  start_btdc = float(comb_cfg.get("start_deg_btdc", BASELINE_SPARK_BTDC))
  return FourStrokeSimulator(
      geom=geom,
      valves=valves,
      q_fuel=fuel_spec.lhv_j_kg,
      afr=float(op_cfg.get("air_fuel_ratio", fuel_spec.stoich_afr)),
      use_woschni=True,
      t_wall_k=DEFAULT_T_WALL_K,
      use_cantera=False,
      wiebe_soc_deg=-start_btdc,
      wiebe_duration_deg=float(comb_cfg.get("duration_deg", DEFAULT_WIEBE_DURATION_DEG)),
      wiebe_a=float(comb_cfg.get("a", DEFAULT_WIEBE_A)),
      wiebe_m=float(comb_cfg.get("m", DEFAULT_WIEBE_M)),
      fuel=fuel_spec,
      oil_spec=OilGradeSpec.from_name(oil_grade),
      splash_params=splash_params,
      thermal_params=thermal_params,
  )


@dataclass(frozen=True, slots=True)
class _ClosedCycleTemplate:
  """Precomputed closed-cycle kinematics and compression baseline for fast spark search."""

  sim: FourStrokeSimulator
  rpm: float
  omega: float
  m_ivc_kg: float
  q_total_j: float
  p_ref_pa: float
  t_ref_k: float
  v_max: float
  th_start: float
  n_steps: int
  th_grid: np.ndarray
  th_mids: np.ndarray
  v_nodes: np.ndarray
  v_mids: np.ndarray
  dvdth_nodes: np.ndarray
  dvdth_mids: np.ndarray
  p_mot_nodes: np.ndarray
  p_mot_mids: np.ndarray
  t_comp: np.ndarray
  p_comp: np.ndarray


def _vectorized_volume_and_slope(geom: EngineGeometry, th_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluate cylinder volume and dV/dtheta over crank angle array."""
  volumes = np.array([geom.volume(float(r)) for r in th_rad], dtype=float)
  rod_ratio = geom.rod_to_crank_ratio
  sin_th = np.sin(th_rad)
  denom = TWO_FACTOR * np.sqrt((rod_ratio * rod_ratio) - (sin_th * sin_th))
  dvdth = HALF_FACTOR * geom.displacement * (sin_th + (np.sin(TWO_FACTOR * th_rad) / denom))
  return volumes, dvdth


def _eval_closed_temp_slope(
    tmpl: _ClosedCycleTemplate,
    th_deg: float,
    temp_k: float,
    vol_m3: float,
    dvdth_rad: float,
    p_mot_pa: float,
    soc_deg: float,
) -> float:
  """Evaluate dT/dtheta (K/deg) for closed high-pressure energy balance."""
  sim = tmpl.sim
  t_safe = max(MIN_GAS_TEMP_K, temp_k)
  p_pa = (tmpl.m_ivc_kg * sim.r_gas * t_safe) / vol_m3
  dvdt = dvdth_rad * tmpl.omega
  cv = sim.cv0 * (1.0 + VIBRATIONAL_CV_SLOPE_PER_K * max(ZERO_FLOAT, t_safe - CV_REF_TEMP_K))

  dq_comb_dt = ZERO_FLOAT
  if soc_deg <= th_deg <= (soc_deg + sim.wiebe_duration_deg):
    u = (th_deg - soc_deg) / sim.wiebe_duration_deg
    m_plus_1 = sim.wiebe_m + 1.0
    dxb_dth_deg = ((sim.wiebe_a * m_plus_1) / sim.wiebe_duration_deg) * (u**sim.wiebe_m) * math.exp(-sim.wiebe_a * (u**m_plus_1))
    dq_comb_dt = (dxb_dth_deg / DEG_TO_RAD) * tmpl.omega * tmpl.q_total_j

  ht = sim.woschni.calculate_heat_transfer(
      theta_deg=th_deg % FULL_CYCLE_DEG,
      rpm=tmpl.rpm,
      p_pa=p_pa,
      t_gas_k=t_safe,
      p_motored_pa=p_mot_pa,
      t_ref_k=tmpl.t_ref_k,
      p_ref_pa=tmpl.p_ref_pa,
      v_ref_m3=tmpl.v_max,
  )
  dtdt = (dq_comb_dt - ht.heat_flux_w - (p_pa * dvdt)) / (tmpl.m_ivc_kg * cv)
  return (dtdt / tmpl.omega) * DEG_TO_RAD


def _integrate_rk4_range(
    tmpl: _ClosedCycleTemplate,
    start_idx: int,
    soc_deg: float,
    t_trace: np.ndarray,
    p_trace: np.ndarray,
) -> None:
  """Advance closed-cycle RK4 energy trace from start_idx through EVO cutoff."""
  h_deg = CLOSED_RK4_STEP_DEG
  half_h = ONE_RADIAN_STEP_HALF * h_deg
  m_r_gas = tmpl.m_ivc_kg * tmpl.sim.r_gas
  t_curr = float(t_trace[start_idx])

  for idx in range(start_idx, tmpl.n_steps):
    th0 = float(tmpl.th_grid[idx])
    thm = float(tmpl.th_mids[idx])
    th1 = float(tmpl.th_grid[idx + 1])
    v0 = float(tmpl.v_nodes[idx])
    vm = float(tmpl.v_mids[idx])
    v1 = float(tmpl.v_nodes[idx + 1])

    k1 = _eval_closed_temp_slope(tmpl, th0, t_curr, v0, float(tmpl.dvdth_nodes[idx]), float(tmpl.p_mot_nodes[idx]), soc_deg)
    k2 = _eval_closed_temp_slope(tmpl, thm, t_curr + (half_h * k1), vm, float(tmpl.dvdth_mids[idx]), float(tmpl.p_mot_mids[idx]), soc_deg)
    k3 = _eval_closed_temp_slope(tmpl, thm, t_curr + (half_h * k2), vm, float(tmpl.dvdth_mids[idx]), float(tmpl.p_mot_mids[idx]), soc_deg)
    k4 = _eval_closed_temp_slope(tmpl, th1, t_curr + (h_deg * k3), v1, float(tmpl.dvdth_nodes[idx + 1]), float(tmpl.p_mot_nodes[idx + 1]), soc_deg)

    t_curr = max(MIN_GAS_TEMP_K, t_curr + ((h_deg / RK4_SIXTH_WEIGHT) * (k1 + (RK4_MID_WEIGHT * k2) + (RK4_MID_WEIGHT * k3) + k4)))
    t_trace[idx + 1] = t_curr
    p_trace[idx + 1] = (m_r_gas * t_curr) / v1


def _build_closed_cycle_template(
    sim: FourStrokeSimulator,
    rpm: float,
    m_ivc_kg: float,
    t_ivc_k: float,
    p_ref_pa: float,
    t_ref_k: float,
) -> _ClosedCycleTemplate:
  """Precompute geometry kinematics and pure-compression baseline for spark candidates."""
  omega = max(MIN_RPM, rpm) * (TWO_PI / SECONDS_PER_MINUTE)
  th_start = sim.timing.ivc_deg - FULL_CYCLE_DEG
  n_steps = int(round((HIGH_PRESSURE_EVO_END_DEG - th_start) / CLOSED_RK4_STEP_DEG))
  th_grid = np.linspace(th_start, HIGH_PRESSURE_EVO_END_DEG, n_steps + 1)
  th_mids = th_grid[:-1] + (ONE_RADIAN_STEP_HALF * CLOSED_RK4_STEP_DEG)
  v_nodes, dvdth_nodes = _vectorized_volume_and_slope(sim.geom, np.radians(th_grid))
  _, dvdth_mids = _vectorized_volume_and_slope(sim.geom, np.radians(th_mids))
  v_mids = HALF_FACTOR * (v_nodes[:-1] + v_nodes[1:])
  v_max = sim.geom.displacement + sim.geom.clearance_volume

  r_comp_nodes = v_max / np.maximum(MIN_VOLUME_M3, v_nodes)
  r_comp_mids = v_max / np.maximum(MIN_VOLUME_M3, v_mids)
  p_mot_nodes = p_ref_pa * (r_comp_nodes**MOTORED_POLYTROPIC_EXP)
  p_mot_mids = p_ref_pa * (r_comp_mids**MOTORED_POLYTROPIC_EXP)

  m_fuel = m_ivc_kg / (1.0 + sim.afr)
  q_total_j = m_fuel * sim.q_fuel * COMBUSTION_EFFICIENCY
  t_comp = np.empty_like(th_grid)
  p_comp = np.empty_like(th_grid)
  t_comp[0] = t_ivc_k
  p_comp[0] = (m_ivc_kg * sim.r_gas * t_ivc_k) / float(v_nodes[0])

  tmpl = _ClosedCycleTemplate(
      sim=sim,
      rpm=rpm,
      omega=omega,
      m_ivc_kg=m_ivc_kg,
      q_total_j=q_total_j,
      p_ref_pa=p_ref_pa,
      t_ref_k=t_ref_k,
      v_max=v_max,
      th_start=th_start,
      n_steps=n_steps,
      th_grid=th_grid,
      th_mids=th_mids,
      v_nodes=v_nodes,
      v_mids=v_mids,
      dvdth_nodes=dvdth_nodes,
      dvdth_mids=dvdth_mids,
      p_mot_nodes=p_mot_nodes,
      p_mot_mids=p_mot_mids,
      t_comp=t_comp,
      p_comp=p_comp,
  )
  _integrate_rk4_range(tmpl, start_idx=0, soc_deg=HIGH_PRESSURE_EVO_END_DEG + CLOSED_RK4_STEP_DEG, t_trace=t_comp, p_trace=p_comp)
  return tmpl


def _evaluate_spark_on_template(tmpl: _ClosedCycleTemplate, spark_btdc_deg: float) -> Tuple[float, float]:
  """Resume closed-cycle integration from SOC using precomputed compression prefix."""
  soc_deg = soc_from_spark_advance(spark_btdc_deg)
  start_idx = max(0, min(tmpl.n_steps, int(math.floor((soc_deg - tmpl.th_start) / CLOSED_RK4_STEP_DEG))))
  t_trace = tmpl.t_comp.copy()
  p_trace = tmpl.p_comp.copy()
  _integrate_rk4_range(tmpl, start_idx=start_idx, soc_deg=soc_deg, t_trace=t_trace, p_trace=p_trace)

  work_j = float(np.trapezoid(p_trace, tmpl.v_nodes))
  knock_res = evaluate_cycle_knock(
      theta_deg=tmpl.th_grid,
      pressure_pa=p_trace,
      temperature_k=t_trace,
      rpm=tmpl.rpm,
      fuel=tmpl.sim.fuel,
      wiebe_soc_deg=soc_deg,
      wiebe_duration_deg=tmpl.sim.wiebe_duration_deg,
      wiebe_a=tmpl.sim.wiebe_a,
      wiebe_m=tmpl.sim.wiebe_m,
      ivc_deg=tmpl.th_start,
  )
  return work_j, knock_res.knock_integral


def _select_candidate_spark_btdc(
    sim: FourStrokeSimulator,
    rpm: float,
    m_ivc_kg: float,
    t_ivc_k: float,
    p_ref_pa: float,
    t_ref_k: float,
    cfg: SteadyDynoConfig,
) -> Tuple[float, bool]:
  """Select MBT spark advance via coarse-then-local search under Livengood-Wu ceiling."""
  tmpl = _build_closed_cycle_template(
      sim=sim,
      rpm=rpm,
      m_ivc_kg=m_ivc_kg,
      t_ivc_k=t_ivc_k,
      p_ref_pa=p_ref_pa,
      t_ref_k=t_ref_k,
  )
  eval_cache: Dict[float, Tuple[float, float]] = {}

  def _eval_cached(sp: float) -> Tuple[float, float]:
    sp_key = round(float(np.clip(sp, cfg.min_spark_deg_btdc, cfg.max_spark_deg_btdc)), ROUND_ONE_DECIMAL)
    if sp_key not in eval_cache:
      eval_cache[sp_key] = _evaluate_spark_on_template(tmpl, sp_key)
    return eval_cache[sp_key]

  n_coarse = int(round((cfg.max_spark_deg_btdc - cfg.min_spark_deg_btdc) / COARSE_SPARK_STEP_DEG)) + 1
  for sp_val in np.linspace(cfg.min_spark_deg_btdc, cfg.max_spark_deg_btdc, n_coarse):
    _eval_cached(float(sp_val))

  best_coarse = _pick_best_spark(eval_cache, cfg.max_knock_integral)
  for offset in (-cfg.spark_step_deg, cfg.spark_step_deg, TWO_FACTOR * cfg.spark_step_deg):
    _eval_cached(best_coarse + offset)

  chosen = _pick_best_spark(eval_cache, cfg.max_knock_integral)
  best_safe_exists = any(lw <= cfg.max_knock_integral for _, lw in eval_cache.values())
  next_up = round(min(cfg.max_spark_deg_btdc, chosen + cfg.spark_step_deg), ROUND_ONE_DECIMAL)
  _, next_lw = _eval_cached(next_up) if next_up > chosen else (ZERO_FLOAT, ZERO_FLOAT)
  knock_limited = (not best_safe_exists) or (next_lw > cfg.max_knock_integral)
  return float(chosen), bool(knock_limited)


def _pick_best_spark(eval_cache: Mapping[float, Tuple[float, float]], max_knock: float) -> float:
  """Pick highest-work safe spark from cache, or lowest-knock spark if none safe."""
  best_safe: Optional[float] = None
  best_work = -SENTINEL_LARGE_VALUE
  safest = min(eval_cache.keys())
  lowest_knock = SENTINEL_LARGE_VALUE
  for sp, (work_j, lw_int) in eval_cache.items():
    if lw_int < lowest_knock:
      lowest_knock = lw_int
      safest = sp
    if lw_int <= max_knock and work_j > best_work:
      best_work = work_j
      best_safe = sp
  return best_safe if best_safe is not None else safest


def _solve_mbt_knock_safe_point(
    sim: FourStrokeSimulator,
    rpm: float,
    throttle: float,
    runner: HelmholtzRunnerSpec,
    cfg: SteadyDynoConfig,
    sae_cf: float,
) -> Tuple[DynoOperatingPoint, np.ndarray, np.ndarray]:
  """Solve a single (RPM, throttle) dyno point with MBT knock-safe spark tuning."""
  p_in_pa, p_ex_pa, ram_ratio = evaluate_runner_boundary_pressures(
      rpm=rpm,
      throttle=throttle,
      geom=sim.geom,
      runner=runner,
      room=cfg.room,
  )
  t_in_k = cfg.room.ambient_temp_k

  sim.wiebe_soc_deg = soc_from_spark_advance(BASELINE_SPARK_BTDC)
  seed_res = sim.solve_cycle(
      rpm=rpm,
      p_inlet_pa=p_in_pa,
      t_inlet_k=t_in_k,
      p_exhaust_pa=p_ex_pa,
      t_exhaust_k=DEFAULT_EXHAUST_TEMP_K,
      n_cycles=SEED_CYCLE_COUNT,
      oil_temp_c=DEFAULT_OIL_TEMP_C,
  )
  ivc_idx = int(np.argmin(np.abs(seed_res.theta_deg - sim.timing.ivc_deg)))
  spark_btdc, knock_limited = _select_candidate_spark_btdc(
      sim=sim,
      rpm=rpm,
      m_ivc_kg=seed_res.trapped_mass_kg,
      t_ivc_k=float(seed_res.temperature_k[ivc_idx]),
      p_ref_pa=p_in_pa,
      t_ref_k=t_in_k,
      cfg=cfg,
  )

  # Stage C: Full 4-stroke cycle + coupled sump thermal equilibrium with knock guard
  sim.wiebe_soc_deg = soc_from_spark_advance(spark_btdc)
  full_res = sim.solve_cycle(
      rpm=rpm,
      p_inlet_pa=p_in_pa,
      t_inlet_k=t_in_k,
      p_exhaust_pa=p_ex_pa,
      t_exhaust_k=DEFAULT_EXHAUST_TEMP_K,
      n_cycles=WOT_CYCLE_COUNT,
      oil_temp_c=None,
  )

  step_count = 0
  while full_res.knock_integral > cfg.max_knock_integral and spark_btdc > cfg.min_spark_deg_btdc and step_count < STAGE_C_KNOCK_RETARD_MAX_STEPS:
    spark_btdc = max(cfg.min_spark_deg_btdc, spark_btdc - cfg.spark_step_deg)
    knock_limited = True
    sim.wiebe_soc_deg = soc_from_spark_advance(spark_btdc)
    full_res = sim.solve_cycle(
        rpm=rpm,
        p_inlet_pa=p_in_pa,
        t_inlet_k=t_in_k,
        p_exhaust_pa=p_ex_pa,
        t_exhaust_k=DEFAULT_EXHAUST_TEMP_K,
        n_cycles=WOT_CYCLE_COUNT,
        oil_temp_c=None,
    )
    step_count += 1

  evo_idx = int(np.argmin(np.abs(full_res.theta_deg - sim.timing.evo_deg)))
  obs_kw = float(full_res.brake_power_kw)
  obs_nm = float(full_res.brake_torque_nm)
  sae_kw = obs_kw * sae_cf
  sae_nm = obs_nm * sae_cf
  amb_ve = float(full_res.volumetric_efficiency) * (p_in_pa / max(MIN_DRY_PRESSURE_PA, cfg.room.ambient_pressure_pa))

  pt = DynoOperatingPoint(
      rpm=float(rpm),
      throttle=float(throttle),
      manifold_pressure_pa=p_in_pa,
      exhaust_backpressure_pa=p_ex_pa,
      helmholtz_ram_ratio=ram_ratio,
      spark_advance_deg_btdc=float(spark_btdc),
      is_knock_limited=knock_limited,
      knock_integral=float(full_res.knock_integral),
      knock_margin=float(full_res.knock_margin),
      volumetric_efficiency=amb_ve,
      nimep_bar=float(full_res.nimep_pa * PA_TO_BAR),
      pmep_bar=float(full_res.pmep_pa * PA_TO_BAR),
      fmep_bar=float(full_res.fmep_pa * PA_TO_BAR),
      bmep_bar=float(full_res.bmep_pa * PA_TO_BAR),
      mechanical_efficiency=float(full_res.mechanical_efficiency),
      oil_temp_c=float(full_res.oil_temp_c),
      observed_brake_torque_nm=obs_nm,
      observed_brake_power_kw=obs_kw,
      sae_correction_factor=sae_cf,
      sae_brake_torque_nm=sae_nm,
      sae_brake_torque_lb_ft=sae_nm * NM_TO_LB_FT,
      sae_brake_power_kw=sae_kw,
      sae_brake_power_hp=sae_kw * KW_TO_MECHANICAL_HP,
      bsfc_g_kwh=float(full_res.bsfc_g_kwh),
      p_max_pa=float(full_res.p_max_pa),
      mean_cooling_power_w=float(full_res.mean_cooling_power_w),
      evo_pressure_pa=float(full_res.pressure_pa[evo_idx]),
      evo_temp_k=float(full_res.temperature_k[evo_idx]),
  )
  return pt, full_res.theta_deg, full_res.pressure_pa


def _build_bsfc_map_grid(
    rpms: Tuple[float, ...],
    throttles: Tuple[float, ...],
    point_table: Dict[Tuple[float, float], DynoOperatingPoint],
) -> BsfcMapGrid:
  """Pack solved operating points into a 2D [throttle][rpm] BSFC map and locate optimum."""
  bmep_rows = []
  bsfc_rows = []
  hp_rows = []
  spark_rows = []
  min_bsfc = DEFAULT_BSFC_SENTINEL
  min_rpm = rpms[0]
  min_thr = throttles[-1]
  min_bmep = ZERO_FLOAT

  for thr in throttles:
    bmep_row = []
    bsfc_row = []
    hp_row = []
    spark_row = []
    for rpm in rpms:
      pt = point_table[(rpm, thr)]
      bmep_row.append(round(pt.bmep_bar, ROUND_VALUE_DECIMALS))
      bsfc_val = (
          OVERRUN_BSFC_CAP_G_KWH
          if (pt.bmep_bar <= MIN_POSITIVE_BMEP_BAR or pt.bsfc_g_kwh > OVERRUN_BSFC_CAP_G_KWH)
          else pt.bsfc_g_kwh
      )
      bsfc_row.append(round(bsfc_val, ROUND_TWO_DECIMALS))
      hp_row.append(round(pt.sae_brake_power_hp, ROUND_VALUE_DECIMALS))
      spark_row.append(round(pt.spark_advance_deg_btdc, ROUND_ONE_DECIMAL))
      if ZERO_FLOAT < bsfc_val < min_bsfc and pt.bmep_bar > MIN_POSITIVE_BMEP_BAR:
        min_bsfc = bsfc_val
        min_rpm = rpm
        min_thr = thr
        min_bmep = pt.bmep_bar
    bmep_rows.append(tuple(bmep_row))
    bsfc_rows.append(tuple(bsfc_row))
    hp_rows.append(tuple(hp_row))
    spark_rows.append(tuple(spark_row))

  return BsfcMapGrid(
      rpms=rpms,
      throttles=throttles,
      bmep_bar_grid=tuple(bmep_rows),
      bsfc_g_kwh_grid=tuple(bsfc_rows),
      sae_power_hp_grid=tuple(hp_rows),
      spark_deg_grid=tuple(spark_rows),
      min_bsfc_g_kwh=float(min_bsfc),
      min_bsfc_rpm=float(min_rpm),
      min_bsfc_throttle=float(min_thr),
      min_bsfc_bmep_bar=float(min_bmep),
  )


def run_steady_dyno_sweep(config: Optional[SteadyDynoConfig] = None) -> SteadyDynoSweepResult:
  """Execute steady-state brake dynamometer sweep from 1400 to 7000 RPM across throttles."""
  cfg = config or SteadyDynoConfig()
  runner = cfg.runner or HelmholtzRunnerSpec.from_cad()
  sim = build_live_twin_simulator(config_path=cfg.config_path, use_live_twin=cfg.use_live_twin)
  sae_cf = compute_sae_j1349_factor(cfg.room)
  tuned_in_rpm, scavenge_rpm = compute_runner_tuned_speeds(
      geom=sim.geom,
      runner=runner,
      ambient_temp_k=cfg.room.ambient_temp_k,
  )

  rpms_sorted = tuple(sorted(float(r) for r in cfg.rpms))
  throttles_sorted = tuple(sorted(float(t) for t in cfg.throttles))
  point_table: Dict[Tuple[float, float], DynoOperatingPoint] = {}
  wot_cycle_pressures: Dict[float, Tuple[float, ...]] = {}
  wot_theta_deg: Tuple[float, ...] = ()

  max_thr = throttles_sorted[-1]
  for thr in throttles_sorted:
    for rpm in rpms_sorted:
      pt, th_arr, p_arr = _solve_mbt_knock_safe_point(
          sim=sim,
          rpm=rpm,
          throttle=thr,
          runner=runner,
          cfg=cfg,
          sae_cf=sae_cf,
      )
      point_table[(rpm, thr)] = pt
      if thr == max_thr:
        if not wot_theta_deg:
          wot_theta_deg = tuple(float(v) for v in th_arr)
        wot_cycle_pressures[rpm] = tuple(float(v) for v in p_arr)

  wot_points = tuple(point_table[(rpm, max_thr)] for rpm in rpms_sorted)
  bsfc_map = _build_bsfc_map_grid(rpms_sorted, throttles_sorted, point_table)
  peak_power_pt = max(wot_points, key=lambda p: p.sae_brake_power_hp)
  peak_torque_pt = max(wot_points, key=lambda p: p.sae_brake_torque_nm)

  return SteadyDynoSweepResult(
      sae_j1349_factor=sae_cf,
      helmholtz_tuned_rpm=tuned_in_rpm,
      exhaust_scavenge_rpm=scavenge_rpm,
      wot_points=wot_points,
      bsfc_map=bsfc_map,
      peak_sae_power_hp=peak_power_pt.sae_brake_power_hp,
      peak_sae_power_kw=peak_power_pt.sae_brake_power_kw,
      peak_power_rpm=peak_power_pt.rpm,
      peak_sae_torque_lb_ft=peak_torque_pt.sae_brake_torque_lb_ft,
      peak_sae_torque_nm=peak_torque_pt.sae_brake_torque_nm,
      peak_torque_rpm=peak_torque_pt.rpm,
      wot_cycle_theta_deg=wot_theta_deg,
      wot_cycle_pressures_pa=wot_cycle_pressures,
  )
