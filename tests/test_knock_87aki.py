"""Verification suite for US 87-Octane (AKI 87 E10) kinetics and Livengood-Wu knock."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import yaml

from openengine.cfd.combustion import CombustionConfig, setup_combustion_case
from openengine.thermo import (
    CanteraThermochemistry,
    ClosedCycleSimulator,
    EngineGeometry,
    FourStrokeSimulator,
    FuelBlendSpec,
    KnockSweepPoint,
    compute_adiabatic_end_gas_temperature,
    douaud_eyzat_ignition_delay_s,
    evaluate_cycle_knock,
    select_best_knock_safe_map,
)
from openengine.thermo.knock import DEFAULT_KNOCK_SAFETY_CEILING


def test_engine_specs_87aki_e10_config() -> None:
  """Verify config/engine_specs.yaml defines US 87-octane E10 and runs knock-free."""
  cfg_path = Path(__file__).resolve().parent.parent / "config" / "engine_specs.yaml"
  with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

  assert "fuel" in cfg
  fuel_spec = FuelBlendSpec.from_config(cfg)
  assert fuel_spec.aki == pytest.approx(87.0)
  assert fuel_spec.ethanol_pct == pytest.approx(10.0)
  assert fuel_spec.lhv_j_kg == pytest.approx(42.0e6)
  assert fuel_spec.stoich_afr == pytest.approx(14.1)

  sim = FourStrokeSimulator.from_config(cfg)
  res = sim.solve_cycle(rpm=float(cfg["operating_conditions"]["rated_rpm"]))
  assert not res.is_knocking
  assert res.knock_integral < DEFAULT_KNOCK_SAFETY_CEILING
  assert res.knock_margin > 0.12
  assert res.power_kw > 8.5


def test_prf_ethanol_blend_and_cantera_kinetics() -> None:
  """Verify 3-component PRF+E10 surrogate composition, LHV, equilibrium, and fallback."""
  e10 = FuelBlendSpec.us_87_e10()
  iso = FuelBlendSpec.pure_isooctane()

  vols = e10.liquid_volume_fractions()
  assert vols["C2H5OH"] == pytest.approx(0.10)
  assert vols["C8H18"] > 0.70
  assert vols["n-C7H16"] > 0.10
  assert sum(vols.values()) == pytest.approx(1.0)

  moles = e10.mole_fractions()
  assert set(moles.keys()) == {"C8H18", "n-C7H16", "C2H5OH"}
  assert sum(moles.values()) == pytest.approx(1.0)
  assert e10.latent_heat_j_kg > iso.latent_heat_j_kg

  chem = CanteraThermochemistry(fuel_blend=e10)
  assert "C8H18" in chem.gas.species_names
  assert "n-C7H16" in chem.gas.species_names
  assert "C2H5OH" in chem.gas.species_names
  assert len(chem.gas.reactions()) >= 40

  stoich_afr_chem = chem.stoichiometric_air_fuel_ratio()
  assert 14.0 < stoich_afr_chem < 14.6

  lhv_vapor = chem.lower_heating_value_j_kg()
  assert 41.0e6 < lhv_vapor < 44.0e6

  eq = chem.constant_volume_equilibrium(t_unburned_k=650.0, p_unburned_pa=1.6e6)
  assert 2650.0 < eq.adiabatic_temperature_k < 2950.0
  assert eq.major_species_mole_fractions["CO2"] > 0.08
  assert eq.major_species_mole_fractions["H2O"] > 0.11
  assert eq.major_species_mole_fractions["NO"] > 0.002

  # Legacy methane GRI-3.0 backward compatibility
  legacy_chem = CanteraThermochemistry(mechanism="gri30.yaml", fuel_species="CH4")
  legacy_chem.set_unburned_mixture(300.0, 101325.0, 1.0)
  assert 27.0 < legacy_chem.current_state().mean_molecular_weight < 28.5


def test_douaud_eyzat_ignition_delay_monotonicity() -> None:
  """Verify ignition delay decreases with T and P and increases with AKI."""
  tau_base = douaud_eyzat_ignition_delay_s(
      pressure_pa=3.5e6,
      end_gas_temp_k=900.0,
      octane_number=87.0,
  )
  tau_hot = douaud_eyzat_ignition_delay_s(
      pressure_pa=3.5e6,
      end_gas_temp_k=960.0,
      octane_number=87.0,
  )
  tau_high_p = douaud_eyzat_ignition_delay_s(
      pressure_pa=5.5e6,
      end_gas_temp_k=900.0,
      octane_number=87.0,
  )
  tau_100_aki = douaud_eyzat_ignition_delay_s(
      pressure_pa=3.5e6,
      end_gas_temp_k=900.0,
      octane_number=100.0,
  )

  assert isinstance(tau_base, float)
  tau_vec = douaud_eyzat_ignition_delay_s(
      pressure_pa=np.array([3.5e6, 5.5e6]),
      end_gas_temp_k=np.array([900.0, 960.0]),
      octane_number=87.0,
  )
  assert isinstance(tau_vec, np.ndarray)
  assert tau_vec.shape == (2,)
  tau_mixed_p = douaud_eyzat_ignition_delay_s(3.5e6, np.array([900.0, 960.0]), 87.0)
  tau_mixed_t = douaud_eyzat_ignition_delay_s(np.array([3.5e6, 5.5e6]), 900.0, 87.0)
  assert isinstance(tau_mixed_p, np.ndarray) and isinstance(tau_mixed_t, np.ndarray)
  assert tau_hot < tau_base
  assert tau_high_p < tau_base
  assert tau_100_aki > tau_base * 1.5

  # Adiabatic end-gas temperature rises strictly with compression pressure
  p_ramp = np.linspace(1.0e5, 4.0e6, 50)
  tu_ramp = compute_adiabatic_end_gas_temperature(p_ramp, t_initial_k=340.0)
  assert np.all(np.diff(tu_ramp) > 0.0)
  assert 800.0 < tu_ramp[-1] < 1000.0


def test_legacy_9_8_cr_knocks_on_87_aki_but_passes_on_100_aki() -> None:
  """Verify aggressive 9.8:1 CR + 22 deg advance knocks on 87 AKI yet passes on 100 AKI."""
  high_cr_geom = EngineGeometry(
      bore=0.068,
      stroke=0.054,
      con_rod_length=0.108,
      compression_ratio=9.8,
  )

  sim_100 = FourStrokeSimulator(
      high_cr_geom,
      t_wall_k=450.0,
      wiebe_soc_deg=-22.0,
      use_cantera=False,
      fuel=FuelBlendSpec.pure_isooctane(),
  )
  res_100 = sim_100.solve_cycle(rpm=5500.0)
  assert not res_100.is_knocking
  assert res_100.knock_integral < 0.85
  assert res_100.knock_onset_deg is None

  sim_87_cool = FourStrokeSimulator(
      high_cr_geom,
      t_wall_k=430.0,
      wiebe_soc_deg=-22.0,
      use_cantera=False,
      fuel=FuelBlendSpec.us_87_e10(),
  )
  res_87_cool = sim_87_cool.solve_cycle(rpm=5500.0)
  assert res_87_cool.is_knocking
  assert res_87_cool.knock_integral > 1.0
  assert res_87_cool.knock_onset_deg is not None
  assert -10.0 <= res_87_cool.knock_onset_deg <= 30.0

  # Hotter air-cooled cylinder head wall increases end-gas temperature and knock severity
  sim_87_hot = FourStrokeSimulator(
      high_cr_geom,
      t_wall_k=500.0,
      wiebe_soc_deg=-22.0,
      use_cantera=False,
      fuel=FuelBlendSpec.us_87_e10(),
  )
  res_87_hot = sim_87_hot.solve_cycle(rpm=5500.0)
  assert res_87_hot.end_gas_temp_max_k > res_87_cool.end_gas_temp_max_k
  assert res_87_hot.knock_integral > res_87_cool.knock_integral


def test_knock_safe_cr_and_spark_map_selection() -> None:
  """Verify automated CR + spark sweep selects a knock-free map across WOT speeds."""
  base_geom = EngineGeometry(
      bore=0.068,
      stroke=0.054,
      con_rod_length=0.108,
      compression_ratio=9.0,
  )
  sim = FourStrokeSimulator(
      base_geom,
      t_wall_k=450.0,
      use_cantera=False,
      fuel=FuelBlendSpec.us_87_e10(),
  )

  safe_map = sim.sweep_knock_safe_operating_map(
      compression_ratios=(8.6, 9.0, 9.8),
      rpms=(3500.0, 5500.0),
      spark_advances_deg_btdc=(16.0, 18.0, 22.0, 26.0),
      max_knock_integral=DEFAULT_KNOCK_SAFETY_CEILING,
  )

  assert safe_map.selected_compression_ratio == pytest.approx(9.0)
  assert safe_map.worst_selected_knock_integral <= DEFAULT_KNOCK_SAFETY_CEILING
  assert (
      safe_map.spark_schedule_deg_btdc[5500.0]
      >= safe_map.spark_schedule_deg_btdc[3500.0]
  )

  # Verify fallback picks lowest-knock timing when ceiling is stricter than all points
  strict_pts = [
      KnockSweepPoint(8.8, 3500.0, 22.0, -18.0, 1.0e6, 6.0, 16.0, 5.5e6, 920.0, 0.95, 0.05, False),
      KnockSweepPoint(8.8, 3500.0, 14.0, -10.0, 0.95e6, 5.7, 15.2, 4.8e6, 880.0, 0.72, 0.28, False),
  ]
  fallback_map = select_best_knock_safe_map(strict_pts, max_knock_integral=0.50)
  assert fallback_map.spark_schedule_deg_btdc[3500.0] == pytest.approx(14.0)
  assert fallback_map.worst_selected_knock_integral == pytest.approx(0.72)

  # Confirm ClosedCycleSimulator [-180, 180] trace input and exact margin / onset interpolation
  closed_res = ClosedCycleSimulator(
      base_geom,
      q_fuel_j_per_kg=42.0e6,
      air_fuel_ratio=14.1,
  ).solve_cycle(theta_soc_deg=-14.0)
  knock_closed = evaluate_cycle_knock(
      theta_deg=closed_res.theta_deg,
      pressure_pa=closed_res.pressure_pa,
      temperature_k=closed_res.temperature_k,
      rpm=5500.0,
      fuel=FuelBlendSpec.us_87_e10(),
      wiebe_soc_deg=-14.0,
  )
  assert knock_closed.peak_end_gas_temp_k > 800.0
  assert 0.1 < knock_closed.knock_integral < 1.0
  assert knock_closed.knock_margin == pytest.approx(1.0 - knock_closed.knock_integral)

  # Verify exact linear onset interpolation when knock integral crosses 1.0
  knock_severe = evaluate_cycle_knock(
      theta_deg=closed_res.theta_deg,
      pressure_pa=closed_res.pressure_pa * 1.6,
      temperature_k=closed_res.temperature_k,
      rpm=2500.0,
      fuel=FuelBlendSpec.us_87_e10(),
      wiebe_soc_deg=-22.0,
  )
  assert knock_severe.is_knocking
  assert knock_severe.knock_onset_crank_angle_deg is not None
  idx_cross = int(np.flatnonzero(knock_severe.knock_integral_trace >= 1.0)[0])
  i0 = knock_severe.knock_integral_trace[idx_cross - 1]
  i1 = knock_severe.knock_integral_trace[idx_cross]
  frac_expected = (1.0 - i0) / (i1 - i0)
  th_expected = (
      knock_severe.crank_angle_deg[idx_cross - 1]
      + frac_expected
      * (knock_severe.crank_angle_deg[idx_cross] - knock_severe.crank_angle_deg[idx_cross - 1])
  )
  xb_expected = (
      knock_severe.burned_fraction[idx_cross - 1]
      + frac_expected
      * (knock_severe.burned_fraction[idx_cross] - knock_severe.burned_fraction[idx_cross - 1])
  )
  assert knock_severe.knock_onset_crank_angle_deg == pytest.approx(th_expected)
  assert knock_severe.unburned_fraction_at_knock == pytest.approx(1.0 - xb_expected)


def test_openfoam_e10_cfd_fuel_properties() -> None:
  """Verify OpenFOAM combustion templates configure E10 stoichiometry and flame speed."""
  with tempfile.TemporaryDirectory() as td:
    case_path = Path(td) / "e10_case"
    cfg = CombustionConfig(compression_ratio=9.0, spark_crank_angle_deg=-18.0)
    setup_combustion_case(case_path, cfg)

    thermo_txt = (case_path / "constant" / "thermophysicalProperties").read_text()
    assert "stoichiometricAirFuelMassRatio [0 0 0 0 0 0 0] 14.1;" in thermo_txt
    assert "molWeight       96.2959;" in thermo_txt

    comb_txt = (case_path / "constant" / "combustionProperties").read_text()
    assert "fuel            GasolineE10;" in comb_txt
    assert "GasolineE10" in comb_txt
    assert "W               0.4720;" in comb_txt
