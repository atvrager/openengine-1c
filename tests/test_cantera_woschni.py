"""Unit tests for Cantera thermochemistry, chemical equilibrium, and Woschni heat transfer."""

import numpy as np
from openengine.thermo.cantera_kinetics import CanteraThermochemistry
from openengine.thermo.cycle import EngineGeometry
from openengine.thermo.gas_exchange import FourStrokeSimulator
from openengine.thermo.woschni import WoschniHeatTransfer
import pytest


@pytest.fixture
def ref_geom():
  return EngineGeometry(
      bore=0.068,
      stroke=0.054,
      con_rod_length=0.108,
      compression_ratio=9.8,
  )


def test_woschni_heat_transfer_scaling(ref_geom):
  woschni = WoschniHeatTransfer(ref_geom, t_wall_k=430.0)

  # Check heat transfer during compression (theta = 650 deg CA)
  out_comp = woschni.calculate_heat_transfer(
      theta_deg=650.0,
      rpm=5500.0,
      p_pa=1.5e6,
      t_gas_k=650.0,
  )
  assert 200.0 < out_comp.heat_transfer_coeff_w_m2_k < 1200.0
  assert out_comp.heat_flux_w > 0.0

  # Check heat transfer during peak combustion (theta = 10 deg CA)
  out_comb = woschni.calculate_heat_transfer(
      theta_deg=10.0,
      rpm=5500.0,
      p_pa=6.0e6,
      t_gas_k=2400.0,
      p_motored_pa=1.8e6,
  )
  # High pressure and flame turbulence should boost hc significantly
  assert out_comb.heat_transfer_coeff_w_m2_k > out_comp.heat_transfer_coeff_w_m2_k * 2.0
  # Heat flux should be in megawatts per square meter range
  area = out_comb.wall_area_m2
  flux_mw = (out_comb.heat_flux_w / area) / 1.0e6
  assert 2.0 < flux_mw < 15.0


def test_cantera_thermochemistry():
  chem = CanteraThermochemistry()

  # Test unburned mixture properties at room temperature (E10 PRF+air MW ~ 30.2 g/mol)
  chem.set_unburned_mixture(t_k=300.0, p_pa=101325.0, equivalence_ratio=1.0)
  state = chem.current_state()
  assert 1.35 < state.gamma < 1.42
  assert 25.0 < state.mean_molecular_weight < 31.0
  assert 1000.0 < state.cp_j_kg_k < 1200.0

  # Test compressed mixture: gamma should drop due to vibrational activation
  cp_hot, cv_hot, gamma_hot, _ = chem.get_properties_at(t_k=700.0, p_pa=1.6e6)
  assert gamma_hot < state.gamma
  assert cp_hot > state.cp_j_kg_k

  # Test UV chemical equilibrium (constant-volume adiabatic flame)
  eq = chem.constant_volume_equilibrium(t_unburned_k=650.0, p_unburned_pa=1.6e6, equivalence_ratio=1.0)
  # Real dissociation caps adiabatic flame temperature well below 3000 K
  assert 2600.0 < eq.adiabatic_temperature_k < 2950.0
  assert 5.0e6 < eq.peak_pressure_pa < 9.0e6
  assert 1.20 < eq.gamma_burned < 1.28

  # Verify presence of major dissociated species
  species = eq.major_species_mole_fractions
  assert species["CO2"] > 0.05
  assert species["H2O"] > 0.10
  assert species["CO"] > 0.01  # Significant CO dissociation
  assert species["NO"] > 0.002  # Thermal NOx formation


def test_coupled_four_stroke_with_woschni_and_cantera(ref_geom):
  sim = FourStrokeSimulator(ref_geom, use_woschni=True, use_cantera=True)
  res = sim.solve_cycle(rpm=5500.0)

  # With Woschni heat loss and dissociation, peak temperature must be realistic (~2400-2650 K)
  assert 2350.0 < res.t_max_k < 2650.0

  # Peak cylinder pressure should be in the realistic range
  assert 5.0e6 < res.p_max_pa < 7.5e6

  # Wall heat loss accounting
  assert res.q_wall_loss_j > 30.0  # > 30 J lost per cycle
  assert 0.05 < res.heat_loss_fraction < 0.25  # 5% to 25% of fuel energy to walls
  assert 1500.0 < res.mean_cooling_power_w < 5000.0  # 1.5 to 5.0 kW cooling dissipation required

  # Check power & torque
  assert 8.0 < res.power_kw < 15.0  # ~11 - 20 hp
  assert 14.0 < res.torque_nm < 24.0

  # Verify species dictionary is populated
  assert "CO2" in res.equilibrium_species
  assert "H2O" in res.equilibrium_species
