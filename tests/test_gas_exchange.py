"""Unit tests for 1D gas exchange, valve flow, and 720-degree 4-stroke cycle."""

import numpy as np
from openengine.thermo.cycle import EngineGeometry
from openengine.thermo.gas_exchange import (
    FourStrokeSimulator,
    ValveGeometry,
    ValveTiming,
    compressible_orifice_flow,
)
import pytest


@pytest.fixture
def ref_geom():
  return EngineGeometry(
      bore=0.068,
      stroke=0.054,
      con_rod_length=0.108,
      compression_ratio=9.8,
  )


def test_valve_lift_curves():
  valves = ValveGeometry(intake_max_lift_m=0.0085, exhaust_max_lift_m=0.0080)
  timing = ValveTiming(evo_deg=135.0, evc_deg=375.0, ivo_deg=345.0, ivc_deg=585.0)

  # When valves are fully closed
  assert valves.lift(0.0, timing.ivo_deg, timing.ivc_deg, valves.intake_max_lift_m) == 0.0
  assert valves.lift(500.0, timing.evo_deg, timing.evc_deg, valves.exhaust_max_lift_m) == 0.0

  # At midpoint of lift event, lift should equal maximum lift
  mid_intake = (timing.ivo_deg + timing.ivc_deg) / 2.0  # 465 deg
  lift_in_mid = valves.lift(mid_intake, timing.ivo_deg, timing.ivc_deg, valves.intake_max_lift_m)
  assert np.isclose(lift_in_mid, valves.intake_max_lift_m, rtol=1e-3)

  mid_exhaust = (timing.evo_deg + timing.evc_deg) / 2.0  # 255 deg
  lift_ex_mid = valves.lift(mid_exhaust, timing.evo_deg, timing.evc_deg, valves.exhaust_max_lift_m)
  assert np.isclose(lift_ex_mid, valves.exhaust_max_lift_m, rtol=1e-3)

  # Check curtain area calculation
  area = valves.curtain_area(lift_m=0.0085, diameter_m=0.030)
  assert area > 1.0e-4  # > 1 cm^2


def test_compressible_orifice_flow():
  cd = 0.65
  area = 2.0e-4  # 2 cm^2
  t_up = 300.0

  # Zero area -> zero flow
  assert compressible_orifice_flow(100000.0, t_up, 90000.0, cd, 0.0) == 0.0

  # Subsonic flow: pressure ratio 0.90
  flow_subsonic = compressible_orifice_flow(100000.0, t_up, 90000.0, cd, area)
  assert flow_subsonic > 0.0

  # Choked flow: pressure ratio 0.30 (< 0.528 critical ratio)
  flow_choked = compressible_orifice_flow(100000.0, t_up, 30000.0, cd, area)
  assert flow_choked > flow_subsonic

  # Further pressure drop past choked point should NOT increase mass flow rate
  flow_choked_deep = compressible_orifice_flow(100000.0, t_up, 10000.0, cd, area)
  assert np.isclose(flow_choked, flow_choked_deep, rtol=1e-3)


def test_full_four_stroke_simulation(ref_geom):
  sim = FourStrokeSimulator(ref_geom)
  res = sim.solve_cycle(rpm=5500.0)

  # 720 degree cycle consistency
  assert len(res.theta_deg) == 720
  assert np.isclose(res.theta_deg[0], 0.0, atol=1.0)
  assert np.isclose(res.theta_deg[-1], 720.0, atol=1.0)

  # Pumping work must be negative (energy absorbed to breathe)
  assert res.pump_work_j < 0.0
  assert res.pmep_pa > 0.3e5  # > 0.3 bar pumping loss
  assert res.gimep_pa > res.nimep_pa  # Gross > Net

  # Volumetric efficiency should be realistic for high-revving SOHC
  assert 0.85 < res.volumetric_efficiency < 1.05

  # Power & torque
  assert 10.0 < res.power_kw < 18.0  # 13 - 24 hp range
  assert 18.0 < res.torque_nm < 28.0


def test_rpm_breathing_dependence(ref_geom):
  sim = FourStrokeSimulator(ref_geom)
  res_low = sim.solve_cycle(rpm=2500.0)
  res_high = sim.solve_cycle(rpm=7000.0)

  # High RPM should experience higher pumping losses across valve restrictions
  assert res_high.pmep_pa > res_low.pmep_pa

  # High RPM volumetric efficiency should roll off compared to low RPM
  assert res_high.volumetric_efficiency < res_low.volumetric_efficiency

  # Power should increase with RPM despite breathing drop
  assert res_high.power_kw > res_low.power_kw
