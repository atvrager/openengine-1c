"""Unit tests for SOHC Pent-Roof Cylinder Head CAD generation and CFD fluid extraction."""

from openengine.cad.cylinder_head import (
    CylinderHeadParams,
    calculate_combustion_chamber_volume,
    extract_intake_port_fluid_domain,
    generate_cylinder_head,
    generate_poppet_valve,
)


def test_combustion_chamber_dome_volume():
  """Verify combustion chamber dome volume matches 9.0:1 clearance budget ~20.46 cm3."""
  vol_mm3, vol_cm3 = calculate_combustion_chamber_volume()
  # Target is ~20.46 cm3 for 9.0:1 compression ratio with 196.1 cc displacement
  assert 19.8 <= vol_cm3 <= 21.2, f"Chamber volume out of range: {vol_cm3:.2f} cm3"
  assert vol_mm3 > 19800.0


def test_cylinder_head_solid_generation():
  """Verify cylinder head B-Rep solid model validity, volume, and topology."""
  head = generate_cylinder_head()
  assert head.is_valid, "Cylinder head solid is not a valid B-Rep"
  # Volume of 6061-T6 billet cylinder head should be ~615 cm3 (within 580-680 cm3)
  vol_cm3 = head.volume / 1000.0
  assert 580.0 <= vol_cm3 <= 680.0, f"Head billet volume unexpected: {vol_cm3:.1f} cm3"
  # Expected rich topological detail: ports, guides, fins, stud holes
  assert len(head.faces()) >= 50


def test_poppet_valves_generation():
  """Verify intake and exhaust 4140 steel poppet valve solid models."""
  # Intake valve: 30 mm head, 5 mm stem
  in_valve = generate_poppet_valve(head_diam_m=0.030, stem_diam_m=0.005)
  assert in_valve.is_valid
  in_vol_cm3 = in_valve.volume / 1000.0
  assert 4.0 <= in_vol_cm3 <= 5.5
  # 4140 alloy steel mass (7.85 g/cm3)
  in_mass_g = in_vol_cm3 * 7.85
  assert 34.0 <= in_mass_g <= 42.0

  # Exhaust valve: 26 mm head, 5 mm stem
  ex_valve = generate_poppet_valve(head_diam_m=0.026, stem_diam_m=0.005)
  assert ex_valve.is_valid
  ex_vol_cm3 = ex_valve.volume / 1000.0
  assert 3.2 <= ex_vol_cm3 <= 4.5
  ex_mass_g = ex_vol_cm3 * 7.85
  assert 27.0 <= ex_mass_g <= 34.0


def test_intake_port_cfd_fluid_domain():
  """Verify extraction of the negative CFD fluid domain for simpleFoam flow bench."""
  fluid = extract_intake_port_fluid_domain(cylinder_depth_m=0.050)
  assert fluid.is_valid, "CFD fluid domain solid is not valid"
  fluid_vol_cm3 = fluid.volume / 1000.0
  # Runner + seat + chamber + 50 mm cylinder bore
  assert 200.0 <= fluid_vol_cm3 <= 260.0, f"Fluid domain volume unexpected: {fluid_vol_cm3:.1f} cm3"
