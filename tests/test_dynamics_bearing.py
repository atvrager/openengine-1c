"""Unit tests for engine shaking forces, balance sweeps, flywheel sizing, and hydrodynamic bearing lubrication."""

import numpy as np
from openengine.dynamics.bearing import (
    BearingGeometry,
    HydrodynamicBearing,
    ocvirk_load_capacity,
)
from openengine.dynamics.flywheel import FlywheelDimensions, FlywheelDynamics
from openengine.dynamics.slider_crank import SliderCrankDynamics
from openengine.thermo.cycle import EngineGeometry
from openengine.thermo.gas_exchange import FourStrokeSimulator


def test_shaking_force_and_balance_sweep():
  """Verify 2D polar shaking force calculation and balance factor optimization."""
  geom = EngineGeometry(
      bore=0.068, stroke=0.054, con_rod_length=0.108, compression_ratio=9.8
  )
  sim = FourStrokeSimulator(geom=geom)
  res = sim.solve_cycle(rpm=5500.0)

  dyn = SliderCrankDynamics(geom=geom)
  forces = dyn.solve_forces(
      rpm=5500.0,
      theta_deg=res.theta_deg,
      pressure_pa=res.pressure_pa,
      recip_balance_factor=0.50,
  )

  # Check that shaking forces were computed
  assert forces.shaking_force_x_n is not None
  assert forces.shaking_force_y_n is not None
  assert forces.shaking_force_mag_n is not None
  assert forces.peak_shaking_force_n > 0.0
  assert forces.rms_shaking_force_n > 0.0

  # Check balance factor sweep
  sweep = dyn.sweep_balance_factors(rpm=5500.0, forces=forces)
  assert len(sweep.balance_factors) == 21
  # The optimal balance factor for RMS shaking should be around 35% - 50%
  assert 0.30 <= sweep.optimal_factor_rms <= 0.60
  # Shaking at 50% balance factor is lower than at 0% or 100%
  idx_0 = 0  # 0%
  idx_50 = 10  # 50%
  idx_100 = 20  # 100%
  assert sweep.rms_shaking_forces_n[idx_50] < sweep.rms_shaking_forces_n[idx_0]
  assert sweep.rms_shaking_forces_n[idx_50] < sweep.rms_shaking_forces_n[idx_100]


def test_flywheel_energy_integration_and_sizing():
  """Verify flywheel torque-energy integration, speed fluctuation, and hoop stress."""
  geom = EngineGeometry(
      bore=0.068, stroke=0.054, con_rod_length=0.108, compression_ratio=9.8
  )
  sim = FourStrokeSimulator(geom=geom)
  res = sim.solve_cycle(rpm=5500.0)

  dyn = SliderCrankDynamics(geom=geom)
  forces = dyn.solve_forces(
      rpm=5500.0,
      theta_deg=res.theta_deg,
      pressure_pa=res.pressure_pa,
  )

  fw = FlywheelDynamics()
  rep = fw.size_flywheel(
      rated_rpm=5500.0,
      theta_deg=res.theta_deg,
      torque_nm=forces.crankshaft_torque_nm,
      target_coeff_fluctuation=0.035,
      redline_rpm=7000.0,
  )

  # Fluctuations within 3.5% target
  assert rep.actual_fluctuation_coeff <= 0.036
  assert 5300.0 < rep.min_rpm_at_rated < 5500.0
  assert 5500.0 < rep.max_rpm_at_rated < 5700.0

  # Sized dimensions match physical engine expectations
  assert 0.015 <= rep.flywheel_inertia_kg_m2 <= 0.030
  assert 4.0 <= rep.flywheel_mass_kg <= 8.0
  assert 20.0 <= rep.flywheel_thickness_mm <= 50.0

  # Hoop stress at redline (7000 RPM) must have a burst safety factor > 4.0
  assert rep.burst_safety_factor_redline >= 4.0
  assert rep.hoop_stress_redline_mpa < 60.0
  assert rep.is_safe


def test_hydrodynamic_journal_bearing():
  """Verify Ocvirk short bearing solution, oil film thickness, and full hydrodynamic lubrication."""
  # Test Ocvirk capacity function monotonically decreases with eccentricity eps
  cap_02 = ocvirk_load_capacity(0.2)
  cap_05 = ocvirk_load_capacity(0.5)
  cap_08 = ocvirk_load_capacity(0.8)
  assert cap_02 > cap_05 > cap_08 > 0.0

  geom = EngineGeometry(
      bore=0.068, stroke=0.054, con_rod_length=0.108, compression_ratio=9.8
  )
  sim = FourStrokeSimulator(geom=geom)
  res = sim.solve_cycle(rpm=5500.0)

  dyn = SliderCrankDynamics(geom=geom)
  forces = dyn.solve_forces(
      rpm=5500.0,
      theta_deg=res.theta_deg,
      pressure_pa=res.pressure_pa,
  )

  f_big_end = np.sqrt(forces.net_wrist_pin_force_n**2 + forces.cylinder_side_thrust_n**2)

  bearing = HydrodynamicBearing(
      geom=BearingGeometry(
          name="ConnectingRodBigEnd",
          journal_diameter_m=0.030,
          bearing_width_m=0.016,
          radial_clearance_m=20.0e-6,
          max_allowable_pressure_mpa=75.0,
      )
  )

  rep = bearing.analyze_bearing_safety(
      rpm=5500.0,
      theta_deg=res.theta_deg,
      radial_load_n=f_big_end,
  )

  # Bearing safety checks
  assert rep.is_safe
  assert rep.is_hydrodynamic
  assert rep.peak_pressure_mpa < rep.allowable_pressure_mpa
  assert rep.min_film_thickness_um > 1.5  # Greater than 1.5 um
  assert rep.min_lambda_ratio > 3.0  # Lambda > 3.0 ensures full fluid-film lubrication
  assert rep.peak_eccentricity_ratio < 0.95
