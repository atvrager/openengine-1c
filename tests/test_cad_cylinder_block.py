"""Unit tests for Cylinder Block, Liner Sleeve, and Radial Cooling Fin Array CAD generation."""

from openengine.cad.cylinder_block import (
    CylinderBlockParams,
    calculate_block_thermal_capacity,
    generate_cylinder_block,
    generate_cylinder_liner,
)


def test_cylinder_liner_solid():
  """Verify honed DOM steel cylinder liner sleeve dimensions, volume, and mass."""
  liner = generate_cylinder_liner()
  assert liner.is_valid, "Cylinder liner solid is not a valid B-Rep"
  vol_cm3 = liner.volume / 1000.0
  # 68 mm ID, 74 mm OD, 105 mm length -> theoretical vol ~ 70.26 cm3
  assert 68.0 <= vol_cm3 <= 73.0
  # DOM steel mass (7.85 g/cm3) ~ 552 g
  mass_g = vol_cm3 * 7.85
  assert 535.0 <= mass_g <= 575.0


def test_cylinder_block_solid():
  """Verify monolithic aluminum cylinder block solid model validity, volume, and topology."""
  block = generate_cylinder_block()
  assert block.is_valid, "Cylinder block solid is not a valid B-Rep"
  vol_cm3 = block.volume / 1000.0
  # Sized for 510 cm3 of A356 aluminum (mass ~ 1.38 kg)
  assert 480.0 <= vol_cm3 <= 550.0
  # Rich topological detail: 8 fins, 4 head stud holes, 4 base stud holes, spigot
  assert len(block.faces()) >= 100


def test_cylinder_block_thermal_capacity():
  """Verify radial fin array surface area and thermal dissipation capacity exceed Phase 1 budget."""
  therm = calculate_block_thermal_capacity()
  assert therm["is_adequate"], "Cooling fin capacity is inadequate for thermal budget"
  # Exterior cooling area must be > 0.17 m2
  assert therm["exterior_area_m2"] >= 0.17
  # Thermal dissipation capacity must exceed the 1300 W barrel budget
  assert therm["thermal_dissipation_w"] >= 1300.0
  # Thermal safety margin >= 20%
  assert therm["thermal_margin_percent"] >= 20.0
