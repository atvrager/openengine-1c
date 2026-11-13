"""Unit tests for Forged 4340 Counterweighted Crankshaft CAD generation and balancing."""

from build123d import Box

from openengine.cad.connecting_rod import generate_connecting_rod
from openengine.cad.crankshaft import (
    CrankshaftParams,
    calculate_crankshaft_balance,
    generate_crankshaft,
)


def test_crankshaft_solid():
  """Verify forged steel crankshaft B-Rep solid model validity, volume, and mass."""
  crank = generate_crankshaft()
  assert crank.is_valid, "Crankshaft solid is not a valid B-Rep"
  assert len(crank.solids()) == 1, (
      f"Crankshaft must be a single connected solid, found {len(crank.solids())} disconnected solids"
  )
  vol_cm3 = crank.volume / 1000.0
  # Steel volume ~ 214 cm3 -> mass ~ 1.68 kg
  assert 200.0 <= vol_cm3 <= 230.0
  mass_kg = vol_cm3 * 7.85 / 1000.0
  assert 1.55 <= mass_kg <= 1.80
  # Rich topological detail: journals, webs, lightening holes, oil gallery
  assert len(crank.faces()) >= 30


def test_crankshaft_pin_pocket_and_rod_clearance():
  """Verify crank cheeks leave an open crankpin pocket wide enough for the connecting rod big-end."""
  params = CrankshaftParams()
  crank = generate_crankshaft(params)
  rod = generate_connecting_rod()

  # Slice around Y = 0 (crankpin midplane): only the crankpin cylinder should be present,
  # not extruded counterweight/cheek material.
  r_throw_mm = (params.stroke_m * 1000.0) / 2.0
  r_pin_mm = (params.crankpin_diameter_m * 1000.0) / 2.0
  midplane_slice = crank.intersect(Box(200.0, 1.0, 200.0))
  assert midplane_slice, "Expected crankpin geometry at Y = 0 midplane"
  bbs = [s.bounding_box() for s in midplane_slice]
  z_min = min(bb.min.Z for bb in bbs)
  z_max = max(bb.max.Z for bb in bbs)
  x_min = min(bb.min.X for bb in bbs)
  x_max = max(bb.max.X for bb in bbs)
  assert z_min >= (r_throw_mm - r_pin_mm) - 0.1, (
      f"Non-crankpin material extends into Y=0 midplane (z_min={z_min:.2f} mm)"
  )
  assert z_max <= (r_throw_mm + r_pin_mm) + 0.1
  assert x_min >= -r_pin_mm - 0.1 and x_max <= r_pin_mm + 0.1

  # Connecting rod pin bore axis is along X in generate_connecting_rod() local frame
  rod_bb = rod.bounding_box()
  rod_big_end_width_mm = rod_bb.max.X - rod_bb.min.X
  crankpin_width_mm = params.crankpin_width_m * 1000.0
  assert crankpin_width_mm > rod_big_end_width_mm + 0.2, (
      f"Crankpin width ({crankpin_width_mm:.2f} mm) must exceed con-rod big-end width "
      f"({rod_big_end_width_mm:.2f} mm) with axial thrust clearance"
  )


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
