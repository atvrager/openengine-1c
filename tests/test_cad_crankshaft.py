"""Unit tests for Forged 4340 Counterweighted Crankshaft CAD generation and balancing."""

from openengine.cad.crankshaft import (
    CrankshaftParams,
    calculate_crankshaft_balance,
    generate_crankshaft,
)


def test_crankshaft_solid():
  """Verify forged steel crankshaft B-Rep solid model validity, volume, and mass."""
  crank = generate_crankshaft()
  assert crank.is_valid, "Crankshaft solid is not a valid B-Rep"
  vol_cm3 = crank.volume / 1000.0
  # Steel volume ~ 214 cm3 -> mass ~ 1.68 kg
  assert 200.0 <= vol_cm3 <= 230.0
  mass_kg = vol_cm3 * 7.85 / 1000.0
  assert 1.55 <= mass_kg <= 1.80
  # Rich topological detail: journals, webs, lightening holes, oil gallery
  assert len(crank.faces()) >= 30


def test_crankshaft_balance_factor():
  """Verify physical counterweight cheeks produce a 40% - 60% reciprocating balance factor."""
  bal = calculate_crankshaft_balance()
  assert bal["is_balanced"], f"Crankshaft out of balance bounds: {bal}"
  # Center of mass Z must be negative (opposite the crankpin at +Z)
  assert bal["center_of_mass_z_mm"] < -5.0
  # Effective balance factor must be in 40% to 60% sweet spot
  assert 0.40 <= bal["effective_balance_factor"] <= 0.60
  # Total unbalance ~ 10,000 to 11,000 g.mm
  assert 9500.0 <= bal["counterweight_unbalance_g_mm"] <= 11500.0
