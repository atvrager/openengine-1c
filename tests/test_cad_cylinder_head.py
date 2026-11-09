"""Unit tests for SOHC Pent-Roof Cylinder Head CAD generation and CFD fluid extraction."""

from openengine.cad.cylinder_head import (
    CylinderHeadParams,
    calculate_combustion_chamber_volume,
    extract_intake_port_fluid_domain,
    generate_cylinder_head,
    generate_poppet_valve,
)


def test_combustion_chamber_dome_volume():
  """Verify combustion chamber dome volume matches thermodynamic clearance budget ~17.0 cm3."""
  vol_mm3, vol_cm3 = calculate_combustion_chamber_volume()
  # Target is 17.04 cm3 (to achieve rc = 9.80:1 with gasket and crevice)
  assert 16.5 <= vol_cm3 <= 17.5, f"Chamber volume out of range: {vol_cm3:.2f} cm3"
  assert vol_mm3 > 16500.0


def test_cylinder_head_solid_generation():
  """Verify cylinder head B-Rep solid model validity, volume, and topology."""
  head = generate_cylinder_head()
  assert head.is_valid, "Cylinder head solid is not a valid B-Rep"
  # Volume of aluminum casting should be between 700 and 850 cm3
  vol_cm3 = head.volume / 1000.0
  assert 700.0 <= vol_cm3 <= 850.0, f"Head casting volume unexpected: {vol_cm3:.1f} cm3"
  # Expected rich topological detail: ports, guides, fins, stud holes
  assert len(head.faces()) >= 50


def test_poppet_valves_generation():
  """Verify intake and exhaust poppet valve solid models."""
  # Intake valve: 30 mm head, 5 mm stem
  in_valve = generate_poppet_valve(head_diam_m=0.030, stem_diam_m=0.005)
  assert in_valve.is_valid
  in_vol_cm3 = in_valve.volume / 1000.0
  assert 4.0 <= in_vol_cm3 <= 5.5
  # Titanium mass (4.50 g/cm3) should be ~21.3 g (matching ValvetrainMassSpring spec)
  in_mass_g = in_vol_cm3 * 4.50
  assert 19.0 <= in_mass_g <= 24.0

  # Exhaust valve: 26 mm head, 5 mm stem
  ex_valve = generate_poppet_valve(head_diam_m=0.026, stem_diam_m=0.005)
  assert ex_valve.is_valid
  ex_vol_cm3 = ex_valve.volume / 1000.0
  assert 3.2 <= ex_vol_cm3 <= 4.5
  ex_mass_g = ex_vol_cm3 * 4.50
  assert 15.0 <= ex_mass_g <= 20.0


def test_intake_port_cfd_fluid_domain():
  """Verify extraction of the negative CFD fluid domain for simpleFoam flow bench."""
  fluid = extract_intake_port_fluid_domain(cylinder_depth_m=0.050)
  assert fluid.is_valid, "CFD fluid domain solid is not valid"
  fluid_vol_cm3 = fluid.volume / 1000.0
  # Runner + seat + chamber + 50 mm cylinder bore
  assert 200.0 <= fluid_vol_cm3 <= 260.0, f"Fluid domain volume unexpected: {fluid_vol_cm3:.1f} cm3"
