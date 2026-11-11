"""Tests for wet-sump splash lubrication, rheology, 4-zone FMEP, and oil cooling."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from openengine.cad.twin_metrics import get_cad_sump_oil_volume_l
from openengine.cfd.cht_cooling import ChtCoolingConfig
from openengine.dynamics.bearing import BearingGeometry, HydrodynamicBearing
from openengine.dynamics.sohc_cam import SOHCCamshaft
from openengine.fea.piston import PistonFeaConfig
from openengine.lubrication import (
    OilGradeSpec,
    SplashSystemParams,
    SumpThermalParams,
    compute_piston_splash_cooling,
    compute_sump_overall_ua_w_k,
    compute_valvetrain_mechanical_work_j,
    evaluate_engine_friction,
    evaluate_splash_transport,
    solve_sump_thermal_balance,
)
from openengine.thermo.cycle import EngineGeometry
from openengine.thermo.gas_exchange import FourStrokeSimulator

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_PATH = REPO_ROOT / "config" / "engine_specs.yaml"


def _load_specs() -> dict:
  with open(SPECS_PATH, "r") as f:
    return yaml.safe_load(f)


def test_walther_and_vogel_viscosity_correlations() -> None:
  """Verify ASTM D341 Walther and 3-point Vogel rheology from 20 C to 140 C."""
  oil_10w40 = OilGradeSpec.sae_10w40()
  oil_20w50 = OilGradeSpec.sae_20w50()

  assert oil_10w40.kinematic_viscosity_walther_cst(40.0) == pytest.approx(92.0, rel=1e-6)
  assert oil_10w40.kinematic_viscosity_walther_cst(100.0) == pytest.approx(14.2, rel=1e-6)
  assert oil_20w50.kinematic_viscosity_walther_cst(40.0) == pytest.approx(158.0, rel=1e-6)
  assert oil_20w50.kinematic_viscosity_walther_cst(100.0) == pytest.approx(19.0, rel=1e-6)

  temps = [20.0, 40.0, 80.0, 100.0, 120.0, 140.0]
  states_10 = [oil_10w40.evaluate(t) for t in temps]
  states_20 = [oil_20w50.evaluate(t) for t in temps]

  for s10, s20 in zip(states_10, states_20):
    assert s20.kinematic_viscosity_cst > s10.kinematic_viscosity_cst
    assert s20.dynamic_viscosity_pa_s > s10.dynamic_viscosity_pa_s
    assert s20.density_kg_m3 > s10.density_kg_m3
    assert s10.prandtl_number > 50.0
    assert s10.vogel_dynamic_viscosity_pa_s == pytest.approx(
        s10.dynamic_viscosity_pa_s, rel=0.025
    )
    assert s20.vogel_dynamic_viscosity_pa_s == pytest.approx(
        s20.dynamic_viscosity_pa_s, rel=0.025
    )

  for idx in range(len(states_10) - 1):
    cur = states_10[idx]
    nxt = states_10[idx + 1]
    assert cur.kinematic_viscosity_cst > nxt.kinematic_viscosity_cst
    assert cur.dynamic_viscosity_pa_s > nxt.dynamic_viscosity_pa_s
    assert cur.density_kg_m3 > nxt.density_kg_m3
    assert cur.specific_heat_j_kg_k < nxt.specific_heat_j_kg_k
    assert cur.thermal_conductivity_w_m_k > nxt.thermal_conductivity_w_m_k

  assert OilGradeSpec.from_name("10w-40").grade == "SAE 10W-40"
  assert OilGradeSpec.from_name("SAE_20W-50").grade == "SAE 20W-50"
  with pytest.raises(ValueError, match="Unsupported oil grade"):
    OilGradeSpec.from_name("SAE 0W-8")

  brg_oil = oil_10w40.to_bearing_oil_properties(95.0)
  brg = HydrodynamicBearing(
      geom=BearingGeometry(name="Test", journal_diameter_m=0.030, bearing_width_m=0.016),
      oil=brg_oil,
  )
  theta = np.linspace(0.0, 720.0, 73)
  loads = np.full_like(theta, 2500.0)
  rep = brg.analyze_bearing_safety(rpm=5500.0, theta_deg=theta, radial_load_n=loads)
  assert rep.min_film_thickness_um > 1.0


def test_splash_dipper_immersion_throw_and_chain_drainback() -> None:
  """Verify dipper submergence window, dynamic level drop, throw reach, and drain-back."""
  cfg = _load_specs()
  params = SplashSystemParams.from_config(cfg)
  default_params = SplashSystemParams.from_config({})
  assert params.dipper_immersion_mm == pytest.approx(5.0)
  assert default_params.sump_volume_l == pytest.approx(get_cad_sump_oil_volume_l())

  oil_10 = OilGradeSpec.sae_10w40()
  oil_20 = OilGradeSpec.sae_20w50()

  rep_idle = evaluate_splash_transport(1400.0, 60.0, oil_10, params)
  rep_rated = evaluate_splash_transport(5500.0, 95.0, oil_10, params)
  rep_redline = evaluate_splash_transport(7000.0, 115.0, oil_10, params)
  rep_heavy = evaluate_splash_transport(5500.0, 95.0, oil_20, params)

  for rep in (rep_idle, rep_rated, rep_redline, rep_heavy):
    assert 4.0 <= rep.static_immersion_mm <= 6.0
    assert rep.dynamic_immersion_mm == pytest.approx(
        rep.static_immersion_mm - rep.dynamic_level_drop_mm
    )
    assert rep.dynamic_immersion_mm < rep.static_immersion_mm
    assert rep.dynamic_immersion_mm >= 1.5
    assert rep.reaches_piston_undercrown is True
    assert rep.chain_oil_carry_g_s >= 0.5
    assert rep.drain_safety_ratio >= 2.0
    assert rep.is_dipper_window_valid is True
    assert rep.is_transport_safe is True

  assert rep_redline.oil_in_flight_l > rep_idle.oil_in_flight_l
  assert rep_heavy.oil_in_flight_l > rep_rated.oil_in_flight_l
  assert rep_redline.tip_throw_velocity_m_s > rep_rated.tip_throw_velocity_m_s > rep_idle.tip_throw_velocity_m_s
  assert rep_rated.submerged_arc_per_cycle_deg == pytest.approx(
      2.0 * rep_rated.submerged_arc_per_rev_deg
  )

  shallow_params = SplashSystemParams(dipper_immersion_mm=2.5)
  rep_shallow = evaluate_splash_transport(5500.0, 95.0, oil_10, shallow_params)
  assert rep_shallow.is_dipper_window_valid is False
  assert rep_shallow.is_transport_safe is False


def test_four_zone_fmep_temperature_grade_and_speed_scaling() -> None:
  """Verify 4-zone FMEP physical scaling across oil temperature, grade, and RPM."""
  cfg = _load_specs()
  sim = FourStrokeSimulator.from_config(cfg, use_cantera=False)
  geom = sim.geom
  res_5500 = sim.solve_cycle(rpm=5500.0, oil_temp_c=95.0)

  # Motored fallback trace (theta_deg=None) vs fired trace
  f_motored = evaluate_engine_friction(geom=geom, rpm=5500.0, oil_temp_c=95.0)
  f_hot_10w40 = evaluate_engine_friction(
      geom=geom,
      rpm=5500.0,
      oil_temp_c=95.0,
      theta_deg=res_5500.theta_deg,
      pressure_pa=res_5500.pressure_pa,
      oil_spec=OilGradeSpec.sae_10w40(),
  )
  f_cold_10w40 = evaluate_engine_friction(
      geom=geom,
      rpm=5500.0,
      oil_temp_c=25.0,
      theta_deg=res_5500.theta_deg,
      pressure_pa=res_5500.pressure_pa,
      oil_spec=OilGradeSpec.sae_10w40(),
  )
  f_hot_20w50 = evaluate_engine_friction(
      geom=geom,
      rpm=5500.0,
      oil_temp_c=95.0,
      theta_deg=res_5500.theta_deg,
      pressure_pa=res_5500.pressure_pa,
      oil_spec=OilGradeSpec.sae_20w50(),
  )

  assert f_hot_10w40.ring_skirt_fmep_pa > 0.0
  assert f_hot_10w40.rolling_bearings_fmep_pa > 0.0
  assert f_hot_10w40.valvetrain_fmep_pa > 0.0
  assert f_hot_10w40.dipper_windage_fmep_pa > 0.0
  assert f_motored.total_fmep_bar > 0.8

  zone_sum_pa = (
      f_hot_10w40.ring_skirt_fmep_pa
      + f_hot_10w40.rolling_bearings_fmep_pa
      + f_hot_10w40.valvetrain_fmep_pa
      + f_hot_10w40.dipper_windage_fmep_pa
  )
  assert f_hot_10w40.total_fmep_pa == pytest.approx(zone_sum_pa)
  assert f_hot_10w40.total_fmep_bar == pytest.approx(zone_sum_pa * 1.0e-5)

  heat_sum_w = (
      f_hot_10w40.ring_skirt_heat_w
      + f_hot_10w40.rolling_bearings_heat_w
      + f_hot_10w40.valvetrain_heat_w
      + f_hot_10w40.dipper_windage_heat_w
  )
  assert f_hot_10w40.total_friction_power_w == pytest.approx(heat_sum_w)

  # Cold oil and heavier viscosity grade increase viscous drag
  assert f_cold_10w40.total_fmep_pa > f_hot_10w40.total_fmep_pa
  assert f_hot_20w50.total_fmep_pa > f_hot_10w40.total_fmep_pa

  # Speed scaling from idle to redline
  f_idle = evaluate_engine_friction(geom=geom, rpm=1400.0, oil_temp_c=95.0)
  f_redline = evaluate_engine_friction(geom=geom, rpm=7000.0, oil_temp_c=95.0)
  assert f_idle.total_fmep_pa < f_hot_10w40.total_fmep_pa < f_redline.total_fmep_pa
  assert f_idle.dipper_windage_fmep_pa < f_hot_10w40.dipper_windage_fmep_pa < f_redline.dipper_windage_fmep_pa

  # Slider-crank rod-ratio side thrust, gas load sensitivity, seal drag, and top-half asymmetry
  ultra_short_rod = EngineGeometry(0.068, 0.054, 0.028, 9.0)
  th_quad = np.linspace(85.0, 95.0, 21)
  p_quad = np.full_like(th_quad, 40.0e5)
  f_ultra_short = evaluate_engine_friction(ultra_short_rod, 3000.0, 95.0, th_quad, p_quad)
  f_nom_quad = evaluate_engine_friction(geom, 3000.0, 95.0, th_quad, p_quad)
  assert f_ultra_short.ring_skirt_fmep_pa > 2.2 * f_nom_quad.ring_skirt_fmep_pa

  # Motored trace compression ratio scaling and CR=2.0 positive compression check
  geom_cr2 = EngineGeometry(0.068, 0.054, 5.0, 2.0)
  th_full = np.linspace(0.0, 720.0, 721)
  f_mot_cr2 = evaluate_engine_friction(geom_cr2, 3000.0, 95.0)
  f_atm_cr2 = evaluate_engine_friction(geom_cr2, 3000.0, 95.0, th_full, np.full_like(th_full, 101325.0))
  assert f_mot_cr2.ring_skirt_fmep_pa > f_atm_cr2.ring_skirt_fmep_pa + 300.0

  # Sub-atmospheric pressures clip gas-backed ring loading to zero when side thrust is negligible
  long_rod = EngineGeometry(0.068, 0.054, 5.0, 9.0)
  f_vac_50 = evaluate_engine_friction(long_rod, 10.0, 95.0, th_full, np.full_like(th_full, 50000.0))
  f_vac_100 = evaluate_engine_friction(long_rod, 10.0, 95.0, th_full, np.full_like(th_full, 100000.0))
  assert f_vac_50.ring_skirt_fmep_pa >= f_vac_100.ring_skirt_fmep_pa
  assert f_vac_50.ring_skirt_fmep_pa == pytest.approx(f_vac_100.ring_skirt_fmep_pa, rel=3e-3)

  # Zero gage pressure minimum at 1 atm and near-zero speed (vacuum and boost both increase thrust)
  th_top = np.linspace(0.0, 90.0, 91)
  th_bot = np.linspace(90.0, 180.0, 91)
  p_flat_vac = np.full_like(th_top, 40000.0)
  p_flat_low = np.full_like(th_top, 101325.0)
  p_flat_boost = np.full_like(th_top, 250000.0)
  p_flat_high = np.full_like(th_top, 60.0e5)

  f_creep_vac = evaluate_engine_friction(geom, 1.0, 95.0, th_top, p_flat_vac)
  f_creep = evaluate_engine_friction(geom, 1.0, 95.0, th_top, p_flat_low)
  f_creep_boost = evaluate_engine_friction(geom, 1.0, 95.0, th_top, p_flat_boost)
  assert f_creep.ring_skirt_fmep_pa < f_creep_vac.ring_skirt_fmep_pa
  assert f_creep.ring_skirt_fmep_pa < f_creep_boost.ring_skirt_fmep_pa
  assert f_creep.rolling_bearings_fmep_pa > 1400.0

  # Inertia relief near TDC: moderate gas pressure opposes upward piston inertia at 7000 RPM
  th_tdc = np.linspace(0.0, 15.0, 16)
  f_tdc_atm = evaluate_engine_friction(geom, 7000.0, 95.0, th_tdc, np.full_like(th_tdc, 101325.0))
  f_tdc_bal = evaluate_engine_friction(geom, 7000.0, 95.0, th_tdc, np.full_like(th_tdc, 9.0e5))
  assert f_tdc_bal.rolling_bearings_fmep_pa < f_tdc_atm.rolling_bearings_fmep_pa

  f_low_p = evaluate_engine_friction(geom, 5500.0, 95.0, th_top, p_flat_low)
  f_high_p = evaluate_engine_friction(geom, 5500.0, 95.0, th_top, p_flat_high)
  assert f_high_p.ring_skirt_fmep_pa > f_low_p.ring_skirt_fmep_pa
  assert f_high_p.rolling_bearings_fmep_pa > f_low_p.rolling_bearings_fmep_pa
  assert f_hot_10w40.ring_skirt_fmep_pa > 0.80e5
  assert f_hot_10w40.valvetrain_fmep_pa > 0.135e5
  assert compute_valvetrain_mechanical_work_j(
      5500.0,
      camshaft=SOHCCamshaft(),
      splash_params=SplashSystemParams(),
  ) > 2.05

  # Piston moves faster in top quadrant [0..90] than bottom quadrant [90..180]
  f_bot = evaluate_engine_friction(geom, 5500.0, 95.0, th_bot, p_flat_low)
  assert f_low_p.dipper_windage_fmep_pa > f_bot.dipper_windage_fmep_pa


def test_steady_state_sump_thermal_balance_and_bounds() -> None:
  """Verify steady-state sump equilibrium temperature <= 130 C up to redline."""
  cfg = _load_specs()
  sim = FourStrokeSimulator.from_config(cfg, use_cantera=False)
  splash_p = SplashSystemParams.from_config(cfg)
  therm_p = SumpThermalParams.from_config(cfg)
  assert SumpThermalParams.from_config({}).max_oil_temp_c == pytest.approx(130.0)
  assert compute_sump_overall_ua_w_k(therm_p) > 10.0

  prev_temp_c = 27.0
  for rpm in (1400.0, 3500.0, 5500.0, 7000.0):
    cyc = sim.solve_cycle(rpm=rpm, oil_temp_c=95.0)
    tb = solve_sump_thermal_balance(
        geom=sim.geom,
        rpm=rpm,
        theta_deg=cyc.theta_deg,
        pressure_pa=cyc.pressure_pa,
        oil_spec=OilGradeSpec.sae_10w40(),
        splash_params=splash_p,
        thermal_params=therm_p,
    )
    assert prev_temp_c < tb.equilibrium_oil_temp_c <= 130.0
    assert tb.oil_temp_margin_c == pytest.approx(
        tb.max_allowable_oil_temp_c - tb.equilibrium_oil_temp_c
    )
    assert tb.oil_temp_margin_c >= 0.0
    expected_q_fric = (
        tb.friction.rolling_bearings_heat_w
        + tb.friction.valvetrain_heat_w
        + 0.70 * tb.friction.dipper_windage_heat_w
        + 0.35 * tb.friction.ring_skirt_heat_w
    )
    assert tb.friction_heat_to_oil_w == pytest.approx(expected_q_fric)
    assert tb.equilibrium_viscosity_cst >= 6.5
    assert tb.sump_heat_rejected_w == pytest.approx(tb.total_oil_heat_load_w, rel=1e-4)
    assert tb.is_oil_temp_safe is True
    assert tb.is_viscosity_safe is True
    assert tb.is_thermal_safe is True
    prev_temp_c = tb.equilibrium_oil_temp_c

  # Starved fin area and zero airflow must trigger overheat failure (> 130 C)
  starved_therm = SumpThermalParams(sump_fin_area_m2=0.015, cooling_airflow_m_s=0.0)
  tb_overheat = solve_sump_thermal_balance(
      geom=sim.geom,
      rpm=7000.0,
      thermal_params=starved_therm,
  )
  assert tb_overheat.equilibrium_oil_temp_c > 130.0
  assert tb_overheat.is_oil_temp_safe is False
  assert tb_overheat.is_thermal_safe is False

  # Extreme ambient cooling (high dry_crown_temp_c=25C => 0 heat at 0 RPM) hits lower clamp
  zero_heat_therm = SumpThermalParams(
      ambient_temp_c=27.0,
      sump_fin_area_m2=5.0,
      cooling_airflow_m_s=30.0,
      dry_crown_temp_c=25.0,
  )
  h_sp, q_sp = compute_piston_splash_cooling(
      geom=EngineGeometry(0.068, 0.054, 0.092, 9.0),
      oil_state=OilGradeSpec.sae_10w40().evaluate(40.0),
      splash_report=evaluate_splash_transport(0.0, 40.0),
      thermal_params=zero_heat_therm,
  )
  h_sp_fast, q_sp_fast = compute_piston_splash_cooling(
      geom=EngineGeometry(0.068, 0.054, 0.092, 9.0),
      oil_state=OilGradeSpec.sae_10w40().evaluate(95.0),
      splash_report=evaluate_splash_transport(5500.0, 95.0),
      thermal_params=therm_p,
  )
  assert h_sp == pytest.approx(95.0)
  assert q_sp == 0.0
  assert h_sp_fast > h_sp
  assert q_sp_fast > 150.0


def test_coupled_brake_output_and_fea_cht_cooling_integration() -> None:
  """Verify coupled BMEP/BSFC in FourStrokeSimulator and splash relief in FEA/CHT."""
  cfg = _load_specs()
  sim = FourStrokeSimulator.from_config(cfg, use_cantera=False)

  res_eq = sim.solve_cycle(rpm=5500.0)
  res_cold = sim.solve_cycle(rpm=5500.0, oil_temp_c=25.0)

  assert res_eq.oil_grade == "SAE 10W-40"
  assert 70.0 <= res_eq.oil_temp_c <= 130.0
  assert res_eq.fmep_pa > 0.8e5
  assert res_eq.bmep_pa == pytest.approx(res_eq.nimep_pa - res_eq.fmep_pa)
  assert 7.5 <= res_eq.brake_power_kw < res_eq.power_kw
  assert 12.0 <= res_eq.brake_torque_nm < res_eq.torque_nm
  assert 0.80 <= res_eq.mechanical_efficiency <= 0.92
  assert 250.0 <= res_eq.bsfc_g_kwh <= 340.0

  # Cold oil drag lowers brake power and increases BSFC relative to hot equilibrium
  assert res_cold.oil_temp_c == pytest.approx(25.0)
  assert res_cold.fmep_pa > res_eq.fmep_pa
  assert res_cold.brake_power_kw < res_eq.brake_power_kw
  assert res_cold.bsfc_g_kwh > res_eq.bsfc_g_kwh

  # Coupling into PistonFeaConfig and ChtCoolingConfig preserves defaults
  tb = solve_sump_thermal_balance(
      geom=sim.geom,
      rpm=5500.0,
      theta_deg=res_eq.theta_deg,
      pressure_pa=res_eq.pressure_pa,
      oil_spec=sim.oil_spec,
      splash_params=sim.splash_params,
      thermal_params=sim.thermal_params,
  )

  piston_dry = PistonFeaConfig()
  piston_cooled = piston_dry.with_splash_cooling(tb)
  assert piston_dry.crown_temp_c == 250.0
  assert piston_cooled.crown_temp_c < piston_dry.crown_temp_c
  assert piston_cooled.ring_land_temp_c < piston_dry.ring_land_temp_c
  assert piston_cooled.skirt_temp_c < piston_dry.skirt_temp_c
  assert piston_cooled.undercrown_oil_htc_w_m2_k == pytest.approx(tb.undercrown_htc_w_m2_k)
  assert piston_cooled.sump_oil_temp_c == pytest.approx(tb.equilibrium_oil_temp_c)

  cht_dry = ChtCoolingConfig()
  cht_cooled = cht_dry.with_splash_cooling(tb)
  assert cht_dry.heat_budget_w == 1300.0
  assert cht_cooled.heat_budget_w == pytest.approx(tb.adjusted_barrel_heat_budget_w)
  assert cht_cooled.heat_budget_w < cht_dry.heat_budget_w
  assert cht_cooled.piston_splash_relief_w == pytest.approx(
      1300.0 - tb.adjusted_barrel_heat_budget_w
  )
