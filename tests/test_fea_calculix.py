"""Unit tests for CalculiX material models, stress tensor maths, and parser."""

import math
from pathlib import Path
import pytest

from openengine.fea.calculix import (
    ALUMINUM_4032_T6,
    STEEL_4340,
    STEEL_GRADE_10_9,
    FeaNodalResults,
    compute_principal_stresses,
    compute_von_mises,
    parse_frd_results,
)


def test_material_specifications():
  """Verify physical consistency of structural engine materials."""
  # 4032-T6 Aluminum
  assert ALUMINUM_4032_T6.youngs_modulus_mpa == 79000.0
  assert ALUMINUM_4032_T6.yield_strength_mpa == 315.0
  assert ALUMINUM_4032_T6.thermal_expansion_coeff > 18e-6

  # 4340 Steel
  assert STEEL_4340.youngs_modulus_mpa == 210000.0
  assert STEEL_4340.yield_strength_mpa == 850.0
  assert STEEL_4340.ultimate_strength_mpa == 1050.0
  assert STEEL_4340.endurance_limit_mpa == 480.0

  # Grade 10.9 Bolts
  assert STEEL_GRADE_10_9.yield_strength_mpa == 940.0
  assert STEEL_GRADE_10_9.ultimate_strength_mpa == 1040.0


def test_von_mises_stress_calculation():
  """Verify von Mises equivalent stress formula against analytical stress states."""
  # Uniaxial tension: sigma_xx = 100 MPa, others = 0 -> von Mises = 100 MPa
  vm_uni = compute_von_mises(100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  assert pytest.approx(vm_uni, rel=1e-5) == 100.0

  # Pure shear: tau_xy = 57.735 MPa -> von Mises = sqrt(3) * tau = 100 MPa
  vm_shear = compute_von_mises(0.0, 0.0, 0.0, 57.7350269, 0.0, 0.0)
  assert pytest.approx(vm_shear, rel=1e-4) == 100.0

  # Hydrostatic compression: sigma_xx = sigma_yy = sigma_zz = -200 MPa -> von Mises = 0
  vm_hydro = compute_von_mises(-200.0, -200.0, -200.0, 0.0, 0.0, 0.0)
  assert pytest.approx(vm_hydro, abs=1e-6) == 0.0


def test_principal_stresses_calculation():
  """Verify principal stress eigensolver."""
  # Diagonal tensor
  p1, p2, p3 = compute_principal_stresses(100.0, 50.0, -20.0, 0.0, 0.0, 0.0)
  assert pytest.approx(p1, rel=1e-4) == 100.0
  assert pytest.approx(p2, rel=1e-4) == 50.0
  assert pytest.approx(p3, rel=1e-4) == -20.0
