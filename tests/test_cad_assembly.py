"""Unit tests for Master Parametric Engine Assembly, Kinematics, and Clearance Verification."""

import numpy as np
from openengine.cad.assembly import (
    EngineAssembly,
    generate_engine_assembly,
    verify_assembly_clearances,
)


def test_assembly_kinematic_positions():
  """Verify analytical kinematic coordinates at TDC, 90 deg, and BDC."""
  asm = EngineAssembly()

  # TDC (crank angle = 0 deg)
  st_tdc = asm.solve_kinematics(0.0)
  assert np.isclose(st_tdc.crankpin_x_mm, 0.0, atol=1e-3)
  assert np.isclose(st_tdc.crankpin_z_mm, -132.0, atol=1e-3)
  assert np.isclose(st_tdc.wrist_pin_z_mm, -24.0, atol=1e-3)
  # Piston crown at TDC must be flush with the deck face (Z = 0)
  assert np.isclose(st_tdc.piston_crown_z_mm, 0.0, atol=1e-3)

  # BDC (crank angle = 180 deg)
  st_bdc = asm.solve_kinematics(180.0)
  assert np.isclose(st_bdc.crankpin_x_mm, 0.0, atol=1e-3)
  assert np.isclose(st_bdc.crankpin_z_mm, -186.0, atol=1e-3)
  assert np.isclose(st_bdc.wrist_pin_z_mm, -78.0, atol=1e-3)
  # Piston crown at BDC must be exactly stroke (54 mm) below deck face
  assert np.isclose(st_bdc.piston_crown_z_mm, -54.0, atol=1e-3)


def test_assembly_clearance_verification():
  """Verify 720-degree kinematic cycle sweep guarantees zero mechanical interference."""
  rep = verify_assembly_clearances()
  assert rep.is_clearance_safe, f"Assembly clearance failed: {rep}"

  # Intake P-V clearance >= 1.5 mm
  assert rep.min_intake_pv_clearance_mm >= 1.50
  assert rep.min_intake_pv_clearance_mm > 5.0  # Actually ~5.95 mm

  # Exhaust P-V clearance >= 1.5 mm
  assert rep.min_exhaust_pv_clearance_mm >= 1.50
  assert rep.min_exhaust_pv_clearance_mm > 5.0  # Actually ~5.85 mm

  # Connecting rod-to-liner clearance >= 5.0 mm (1:1 C500 billet ~10.9 mm)
  assert rep.min_rod_to_liner_clearance_mm >= 5.00
  assert rep.min_rod_to_liner_clearance_mm > 9.5


def test_master_assembly_compound():
  """Verify generation of the complete multi-body engine assembly Compound."""
  comp = generate_engine_assembly(crank_angle_deg=0.0)
  assert comp.is_valid, "Engine assembly compound is not a valid B-Rep"
  vol_cm3 = comp.volume / 1000.0
  # Combined 1:1 billet volume of head, block, liner, crank, rod, piston, pin, valves ~ 1481 cm3
  assert 1400.0 <= vol_cm3 <= 1700.0
  assert len(comp.solids()) == 9


def _overlap_volume_mm3(part_a, part_b) -> float:
  """Compute 3D B-Rep intersection volume in mm^3 between two solids."""
  inter = part_a.intersect(part_b)
  if not inter:
    return 0.0
  return float(sum(s.volume for s in inter))


def test_assembly_720deg_3d_interference():
  """Verify zero 3D B-Rep geometric interference across the full 720-degree kinematic cycle."""
  asm = EngineAssembly()
  collisions = asm.check_3d_interference(step_deg=15.0)
  assert not collisions, (
      f"3D B-Rep interference detected across 720-deg cycle ({len(collisions)} hits); "
      f"first hits: {collisions[:6]}"
  )

  # Also verify build_assembly() itself produces 9 distinct non-overlapping solids at key crank angles
  for angle_deg in (0.0, 90.0, 180.0, 270.0, 360.0):
    comp = asm.build_assembly(angle_deg)
    solids = comp.solids()
    assert len(solids) == 9, (
        f"Expected 9 solid components in master assembly at CA={angle_deg}, got {len(solids)}"
    )
    # Solids order in build_assembly: [head, block, liner, crank, rod, piston, wrist_pin, in_valve, ex_valve]
    _, _, b_liner, b_crank, b_rod, b_piston, _, b_in_v, b_ex_v = solids
    assert _overlap_volume_mm3(b_crank, b_rod) <= 1e-3
    assert _overlap_volume_mm3(b_crank, b_piston) <= 1e-3
    assert _overlap_volume_mm3(b_crank, b_liner) <= 1e-3
    assert _overlap_volume_mm3(b_rod, b_liner) <= 1e-3
    assert _overlap_volume_mm3(b_piston, b_in_v) <= 1e-3
    assert _overlap_volume_mm3(b_piston, b_ex_v) <= 1e-3
