"""Unit tests for SOHC 4-5-6-7 polynomial cam profile, valvetrain dynamics, and valve float."""

import numpy as np
from openengine.dynamics.sohc_cam import (
    CamLobeParams,
    SOHCCamshaft,
    ValvetrainMassSpring,
    polynomial_4567,
)


def test_polynomial_4567_boundary_conditions():
  """Verify that 4-5-6-7 polynomial guarantees zero velocity, acceleration, and jerk at boundaries."""
  # Boundary at u = 0 (valve starts lifting from seat)
  y0, dy0, d2y0, d3y0 = polynomial_4567(np.array([0.0]))
  assert np.isclose(y0[0], 0.0, atol=1e-7)
  assert np.isclose(dy0[0], 0.0, atol=1e-7)
  assert np.isclose(d2y0[0], 0.0, atol=1e-7)
  assert np.isclose(d3y0[0], 0.0, atol=1e-7)

  # Boundary at u = 1 (valve at peak lift)
  y1, dy1, d2y1, d3y1 = polynomial_4567(np.array([1.0]))
  assert np.isclose(y1[0], 1.0, atol=1e-7)
  assert np.isclose(dy1[0], 0.0, atol=1e-7)
  assert np.isclose(d2y1[0], 0.0, atol=1e-7)
  assert np.isclose(d3y1[0], 0.0, atol=1e-7)


def test_polynomial_4567_continuity_and_derivatives():
  """Verify numerical differentiation against analytical derivative formulas."""
  u = np.linspace(0.05, 0.95, 100)
  du = 1e-6
  y, dy, d2y, d3y = polynomial_4567(u)
  y_fwd, dy_fwd, d2y_fwd, _ = polynomial_4567(u + du)

  num_dy = (y_fwd - y) / du
  num_d2y = (dy_fwd - dy) / du
  num_d3y = (d2y_fwd - d2y) / du

  assert np.allclose(dy, num_dy, rtol=1e-3, atol=1e-4)
  assert np.allclose(d2y, num_d2y, rtol=1e-3, atol=1e-4)
  assert np.allclose(d3y, num_d3y, rtol=1e-2, atol=1e-3)


def test_sohc_camshaft_kinematics():
  """Verify cam lobe kinematics, peak valve lift, and rocker ratio scaling."""
  cam = SOHCCamshaft()
  intake = cam.intake_params
  kin = cam.evaluate_lobe_cam_angle(intake, rpm=5500.0, num_points=720)

  # Peak valve lift matches target 8.5 mm
  assert np.isclose(np.max(kin.valve_lift_m), intake.max_valve_lift_m, rtol=1e-3)

  # Cam lift is scaled by rocker ratio
  expected_cam_lift = intake.max_valve_lift_m / intake.rocker_ratio
  assert np.isclose(np.max(kin.cam_lift_m), expected_cam_lift, rtol=1e-3)

  # Valve velocity is zero at peak lift (crest)
  crest_idx = np.argmax(kin.valve_lift_m)
  assert np.isclose(kin.valve_velocity_m_s[crest_idx], 0.0, atol=1e-3)


def test_valvetrain_safety_and_valve_float_margin():
  """Verify valve float RPM is > 30% above 1.2x redline (8400 RPM) and contact stress < 1500 MPa."""
  cam = SOHCCamshaft()
  rated_rpm = 5500.0
  redline_rpm = 7000.0
  test_rpm = 1.2 * redline_rpm  # 8400 RPM

  intake_rep = cam.analyze_valvetrain_safety(
      cam.intake_params, rated_rpm=rated_rpm, redline_rpm=redline_rpm
  )
  exhaust_rep = cam.analyze_valvetrain_safety(
      cam.exhaust_params, rated_rpm=rated_rpm, redline_rpm=redline_rpm
  )

  # Both valves must pass valvetrain safety check
  assert intake_rep.is_safe, f"Intake safety failed: {intake_rep}"
  assert exhaust_rep.is_safe, f"Exhaust safety failed: {exhaust_rep}"

  # Float RPM must be > 1.30 * 8400 = 10920 RPM
  assert intake_rep.valve_float_rpm > 10920.0
  assert exhaust_rep.valve_float_rpm > 10920.0

  # Float margins >= 30%
  assert intake_rep.float_margin_percent >= 30.0
  assert exhaust_rep.float_margin_percent >= 30.0

  # Follower contact force must remain positive at 8400 RPM test condition
  assert intake_rep.min_contact_force_at_test_n > 100.0
  assert exhaust_rep.min_contact_force_at_test_n > 100.0

  # Peak Hertzian contact stress must be well below 1500 MPa allowable limit
  assert intake_rep.peak_hertz_stress_mpa < 1400.0
  assert exhaust_rep.peak_hertz_stress_mpa < 1400.0


def test_valve_lift_crank_angle_integration():
  """Verify valve lift evaluated over full 720 crank degrees."""
  cam = SOHCCamshaft()
  theta = np.linspace(0.0, 720.0, 721)
  intake_lifts = cam.evaluate_valve_lift_crank_angle(cam.intake_params, theta)
  exhaust_lifts = cam.evaluate_valve_lift_crank_angle(cam.exhaust_params, theta)

  # Check durations and peaks
  assert np.isclose(np.max(intake_lifts), 0.0085, atol=1e-5)
  assert np.isclose(np.max(exhaust_lifts), 0.0075, atol=1e-5)

  # Valves closed during combustion stroke (e.g. at 30 deg after firing TDC = 30 deg)
  assert intake_lifts[30] == 0.0
  assert exhaust_lifts[30] == 0.0
