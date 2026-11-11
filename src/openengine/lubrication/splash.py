"""Wet-Sump Splash Dipper Kinematics, Mist Throw, and Cam-Tunnel Drain-Back."""

from dataclasses import dataclass, field
import math
from typing import Optional
import numpy as np
from openengine.cad.connecting_rod import DIPPER_WIDTH_X_MM
from openengine.cad.crankshaft import CRANK_SPROCKET_TEETH
from openengine.cad.cylinder_block import (
    CHAIN_SLOT_MAX_Y_MM,
    CHAIN_SLOT_MIN_Y_MM,
    CHAIN_SLOT_WIDTH_X_MM,
)
from openengine.cad.twin_metrics import (
    get_cad_dipper_immersion_mm,
    get_cad_dipper_tip_radius_mm,
    get_cad_sump_oil_volume_l,
)
from openengine.lubrication.rheology import (
    EngineLubricationConfigRoot,
    OilGradeSpec,
    OilState,
)

GRAVITY_M_S2: float = 9.80665
SECONDS_PER_MINUTE: float = 60.0
TWO_PI: float = 2.0 * math.pi
MM_TO_M: float = 1.0e-3
M_TO_MM: float = 1.0e3
LITER_TO_M3: float = 1.0e-3
M3_TO_LITER: float = 1.0e3
KG_S_TO_G_S: float = 1.0e3
M3_S_TO_L_MIN: float = 60.0e3

# Default wet-sump and dipper geometry for 196.1cc billet crankcase
DEFAULT_SUMP_FLOOR_AREA_M2: float = 0.0145
DEFAULT_DIPPER_WIDTH_MM: float = DIPPER_WIDTH_X_MM
DEFAULT_PISTON_UNDERCROWN_BDC_M: float = 0.088
DEFAULT_LINER_SKIRT_HEIGHT_M: float = 0.072

# #25 ANSI roller chain and vertical chain-tunnel geometry
DEFAULT_CHAIN_PITCH_M: float = 0.00635
DEFAULT_CHAIN_WIDTH_M: float = 0.00318
DEFAULT_CRANK_SPROCKET_TEETH: int = CRANK_SPROCKET_TEETH
DEFAULT_TUNNEL_WIDTH_M: float = CHAIN_SLOT_WIDTH_X_MM * MM_TO_M
DEFAULT_TUNNEL_DEPTH_M: float = (CHAIN_SLOT_MAX_Y_MM - CHAIN_SLOT_MIN_Y_MM) * MM_TO_M
MAX_UNFLOODED_FILM_M: float = 0.0016

# Physical correlations and safety bounds
MIN_STATIC_IMMERSION_MM: float = 4.0
MAX_STATIC_IMMERSION_MM: float = 6.0
MIN_DYNAMIC_IMMERSION_MM: float = 1.5
HOLDUP_BASE_FRACTION: float = 0.018
HOLDUP_SPEED_REF_RPM: float = 5500.0
HOLDUP_VISC_REF_CST: float = 14.0
HOLDUP_VISC_EXP: float = 0.22
MAX_HOLDUP_FRACTION: float = 0.085
DROPLET_DRAG_EFFICIENCY: float = 0.65
HALF_FACTOR: float = 0.5
TWO_STROKES_PER_REV_ARC: float = 2.0
FOUR_STROKE_DIPS_PER_CYCLE: float = 2.0
CHAIN_WETTED_SURFACE_FACTOR: float = 2.4
CHAIN_FILM_COEFF: float = 0.038
MAX_CHAIN_FILM_M: float = 1.2e-4
NUSSELT_FILM_DIVISOR: float = 3.0
NUSSELT_FILM_CUBIC_EXP: float = 3.0
MIN_DRAIN_SAFETY_RATIO: float = 2.0
MIN_CAM_LUBE_FLOW_G_S: float = 0.5
MIN_CHAIN_SPEED_M_S: float = 0.1
MIN_FLOW_EPSILON: float = 1.0e-6
RAD_TO_DEG: float = 180.0 / math.pi


@dataclass(frozen=True, slots=True)
class SplashSystemParams:
  """Geometric parameters for the wet-sump splash-dipper and chain tunnel."""

  sump_volume_l: float = field(default_factory=get_cad_sump_oil_volume_l)
  sump_floor_area_m2: float = DEFAULT_SUMP_FLOOR_AREA_M2
  dipper_immersion_mm: float = field(default_factory=get_cad_dipper_immersion_mm)
  dipper_width_mm: float = DEFAULT_DIPPER_WIDTH_MM
  dipper_tip_radius_mm: float = field(default_factory=get_cad_dipper_tip_radius_mm)
  piston_undercrown_bdc_m: float = DEFAULT_PISTON_UNDERCROWN_BDC_M
  liner_skirt_height_m: float = DEFAULT_LINER_SKIRT_HEIGHT_M
  chain_pitch_m: float = DEFAULT_CHAIN_PITCH_M
  chain_width_m: float = DEFAULT_CHAIN_WIDTH_M
  crank_sprocket_teeth: int = DEFAULT_CRANK_SPROCKET_TEETH
  tunnel_width_m: float = DEFAULT_TUNNEL_WIDTH_M
  tunnel_depth_m: float = DEFAULT_TUNNEL_DEPTH_M

  @classmethod
  def from_config(cls, cfg: EngineLubricationConfigRoot) -> "SplashSystemParams":
    """Build SplashSystemParams from a parsed engine_specs.yaml mapping."""
    lub = cfg.get("lubrication")
    if lub is None:
      return cls()
    return cls(
        sump_volume_l=float(lub.get("sump_volume_l", get_cad_sump_oil_volume_l())),
        dipper_immersion_mm=float(
            lub.get("dipper_immersion_mm", get_cad_dipper_immersion_mm())
        ),
        dipper_width_mm=float(lub.get("dipper_width_mm", DEFAULT_DIPPER_WIDTH_MM)),
        dipper_tip_radius_mm=float(
            lub.get("dipper_tip_radius_mm", get_cad_dipper_tip_radius_mm())
        ),
    )


@dataclass(frozen=True, slots=True)
class SplashTransportReport:
  """Steady-state splash submergence, bore throw, and chain return metrics."""

  rpm: float
  oil_temp_c: float
  static_pool_depth_mm: float
  oil_in_flight_l: float
  dynamic_level_drop_mm: float
  static_immersion_mm: float
  dynamic_immersion_mm: float
  submerged_arc_per_rev_deg: float
  submerged_arc_per_cycle_deg: float
  tip_throw_velocity_m_s: float
  droplet_apex_height_m: float
  reaches_piston_undercrown: bool
  chain_speed_m_s: float
  chain_oil_carry_g_s: float
  chain_oil_carry_l_min: float
  tunnel_drain_capacity_g_s: float
  drain_safety_ratio: float
  is_dipper_window_valid: bool
  is_transport_safe: bool


def evaluate_splash_transport(
    rpm: float,
    oil_temp_c: float,
    oil_spec: Optional[OilGradeSpec] = None,
    params: Optional[SplashSystemParams] = None,
) -> SplashTransportReport:
  """Evaluate wet-sump dipper immersion, mist throw, and SOHC drain-back."""
  spec = oil_spec or OilGradeSpec.sae_10w40()
  geom = params or SplashSystemParams()
  oil: OilState = spec.evaluate(oil_temp_c)

  sump_vol_m3 = geom.sump_volume_l * LITER_TO_M3
  static_depth_mm = (sump_vol_m3 / geom.sump_floor_area_m2) * M_TO_MM

  # Suspended oil holdup increases with speed and viscosity
  speed_ratio = max(0.0, rpm) / HOLDUP_SPEED_REF_RPM
  visc_ratio = (oil.kinematic_viscosity_cst / HOLDUP_VISC_REF_CST) ** HOLDUP_VISC_EXP
  holdup_frac = float(
      np.clip(HOLDUP_BASE_FRACTION * speed_ratio * visc_ratio, 0.0, MAX_HOLDUP_FRACTION)
  )
  in_flight_m3 = sump_vol_m3 * holdup_frac
  level_drop_mm = (in_flight_m3 / geom.sump_floor_area_m2) * M_TO_MM
  dyn_immersion_mm = max(0.0, geom.dipper_immersion_mm - level_drop_mm)

  r_dip_m = geom.dipper_tip_radius_mm * MM_TO_M
  h_dyn_m = dyn_immersion_mm * MM_TO_M
  cos_arg = float(np.clip(1.0 - (h_dyn_m / r_dip_m), -1.0, 1.0))
  half_arc_deg = math.acos(cos_arg) * RAD_TO_DEG
  arc_per_rev_deg = TWO_STROKES_PER_REV_ARC * half_arc_deg
  arc_per_cycle_deg = FOUR_STROKE_DIPS_PER_CYCLE * arc_per_rev_deg

  omega = max(0.0, rpm) * (TWO_PI / SECONDS_PER_MINUTE)
  v_throw = omega * r_dip_m
  apex_m = DROPLET_DRAG_EFFICIENCY * (v_throw**2) / (2.0 * GRAVITY_M_S2)
  reaches_piston = apex_m >= geom.piston_undercrown_bdc_m

  # #25 timing chain oil carry up to SOHC head
  v_chain = (max(0.0, rpm) / SECONDS_PER_MINUTE) * (
      geom.crank_sprocket_teeth * geom.chain_pitch_m
  )
  nu_m2_s = oil.dynamic_viscosity_pa_s / oil.density_kg_m3
  capillary_film = CHAIN_FILM_COEFF * math.sqrt(
      (nu_m2_s * max(MIN_CHAIN_SPEED_M_S, v_chain)) / GRAVITY_M_S2
  )
  film_m = min(MAX_CHAIN_FILM_M, capillary_film)
  wetted_width = CHAIN_WETTED_SURFACE_FACTOR * geom.chain_width_m
  vol_carry_m3_s = v_chain * wetted_width * film_m
  mass_carry_kg_s = vol_carry_m3_s * oil.density_kg_m3

  # Nusselt laminar falling-film gravity return down chain tunnel walls
  tunnel_perimeter_m = 2.0 * (geom.tunnel_width_m + geom.tunnel_depth_m)
  vol_drain_m3_s = (
      oil.density_kg_m3
      * GRAVITY_M_S2
      * tunnel_perimeter_m
      * (MAX_UNFLOODED_FILM_M**NUSSELT_FILM_CUBIC_EXP)
  ) / (NUSSELT_FILM_DIVISOR * oil.dynamic_viscosity_pa_s)
  mass_drain_kg_s = vol_drain_m3_s * oil.density_kg_m3

  carry_g_s = mass_carry_kg_s * KG_S_TO_G_S
  drain_g_s = mass_drain_kg_s * KG_S_TO_G_S
  drain_ratio = drain_g_s / max(MIN_FLOW_EPSILON, carry_g_s)

  window_ok = (
      MIN_STATIC_IMMERSION_MM <= geom.dipper_immersion_mm <= MAX_STATIC_IMMERSION_MM
  )
  transport_ok = (
      window_ok
      and (dyn_immersion_mm >= MIN_DYNAMIC_IMMERSION_MM)
      and reaches_piston
      and (carry_g_s >= MIN_CAM_LUBE_FLOW_G_S)
      and (drain_ratio >= MIN_DRAIN_SAFETY_RATIO)
  )

  return SplashTransportReport(
      rpm=float(rpm),
      oil_temp_c=float(oil.temperature_c),
      static_pool_depth_mm=float(static_depth_mm),
      oil_in_flight_l=float(in_flight_m3 * M3_TO_LITER),
      dynamic_level_drop_mm=float(level_drop_mm),
      static_immersion_mm=float(geom.dipper_immersion_mm),
      dynamic_immersion_mm=float(dyn_immersion_mm),
      submerged_arc_per_rev_deg=float(arc_per_rev_deg),
      submerged_arc_per_cycle_deg=float(arc_per_cycle_deg),
      tip_throw_velocity_m_s=float(v_throw),
      droplet_apex_height_m=float(apex_m),
      reaches_piston_undercrown=reaches_piston,
      chain_speed_m_s=float(v_chain),
      chain_oil_carry_g_s=float(carry_g_s),
      chain_oil_carry_l_min=float(vol_carry_m3_s * M3_S_TO_L_MIN),
      tunnel_drain_capacity_g_s=float(drain_g_s),
      drain_safety_ratio=float(drain_ratio),
      is_dipper_window_valid=window_ok,
      is_transport_safe=transport_ok,
  )
