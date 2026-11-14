"""Unit and integration tests for STEP edge healing, material colors, and default transparency."""

import json
from pathlib import Path
import re
import shutil
import subprocess

from build123d import import_step

from openengine.cad import (
    SINGLE_HOUSING_OPACITY,
    export_styled_step,
    generate_cylinder_head,
    generate_dressed_engine_assembly,
    generate_engine_assembly,
    max_solid_edge_surface_gap_mm,
    style_single_part,
)
from openengine.cad.cots_parts import generate_carburetor_vm26
from openengine.cad.crankcase import generate_crankcase_right
from openengine.cad.step_export import HEALED_BSPLINE_GAP_TOL_MM, STEP_PRECISION_TOL_MM

MIN_EXPECTED_COLORS: int = 8
RGB_CHANNEL_COUNT: int = 3
COLOR_ROUND_DIGITS: int = 2
ROUND_TRIP_SAMPLES: int = 51
TRANSPARENCY_TOL: float = 1.0e-3
EXPECTED_ASSEMBLY_TRANSPARENCIES: tuple[float, ...] = (0.82, 0.75, 0.65, 0.60)
EXPECTED_STANDALONE_TRANSPARENCY: float = 0.55
EXPECTED_DRESSED_PRODUCTS: tuple[str, ...] = (
    "crankcase_right",
    "carburetor_vm26",
    "piston",
    "sump_oil_pool",
    "fan_shroud",
)
TRANSPARENCY_PATTERN: re.Pattern[str] = re.compile(r"SURFACE_STYLE_TRANSPARENT\s*\(\s*([0-9.]+)\s*\)")
COLOUR_RGB_PATTERN: re.Pattern[str] = re.compile(
    r"COLOUR_RGB\s*\(\s*'[^']*'\s*,\s*([0-9.Ee+-]+)\s*,\s*([0-9.Ee+-]+)\s*,\s*([0-9.Ee+-]+)\s*\)"
)
PRODUCT_PATTERN: re.Pattern[str] = re.compile(r"PRODUCT\s*\(\s*'([^']+)'")


def test_crankcase_right_and_carburetor_edge_gaps_sub_precision(tmp_path: Path) -> None:
  """Verify healed parts keep B-spline gaps <= 1e-8 mm and all gaps < 1e-7 mm across STEP round-trip."""
  right_case = generate_crankcase_right()
  carburetor = generate_carburetor_vm26()
  assert right_case.is_valid
  assert carburetor.is_valid

  assert max_solid_edge_surface_gap_mm(right_case, bspline_only=True) <= HEALED_BSPLINE_GAP_TOL_MM
  assert max_solid_edge_surface_gap_mm(carburetor, bspline_only=True) <= HEALED_BSPLINE_GAP_TOL_MM
  assert max_solid_edge_surface_gap_mm(right_case) < STEP_PRECISION_TOL_MM
  assert max_solid_edge_surface_gap_mm(carburetor) < STEP_PRECISION_TOL_MM

  right_step = export_styled_step(right_case, tmp_path / "crankcase_right.step")
  carb_step = export_styled_step(carburetor, tmp_path / "carburetor_vm26.step")
  assert max_solid_edge_surface_gap_mm(import_step(str(right_step)), sample_count=ROUND_TRIP_SAMPLES) < STEP_PRECISION_TOL_MM
  assert max_solid_edge_surface_gap_mm(import_step(str(carb_step)), sample_count=ROUND_TRIP_SAMPLES) < STEP_PRECISION_TOL_MM


def test_assemblies_edge_gaps_within_step_precision() -> None:
  """Verify both master and dressed engine assemblies keep every solid edge below 1e-7 mm."""
  master_asm = generate_engine_assembly(0.0)
  dressed_asm = generate_dressed_engine_assembly(0.0)
  assert master_asm.is_valid
  assert dressed_asm.is_valid
  assert max_solid_edge_surface_gap_mm(master_asm, sample_count=ROUND_TRIP_SAMPLES) < STEP_PRECISION_TOL_MM
  assert max_solid_edge_surface_gap_mm(dressed_asm, sample_count=ROUND_TRIP_SAMPLES) < STEP_PRECISION_TOL_MM


def test_step_export_emits_material_colors_transparency_and_product_names(tmp_path: Path) -> None:
  """Verify exported STEP text contains distinct material RGBs, default transparencies, and part names."""
  dressed_step = export_styled_step(generate_dressed_engine_assembly(0.0), tmp_path / "dressed_engine_assembly.step")
  step_text = dressed_step.read_text(encoding="utf-8")

  assert "SURFACE_STYLE_RENDERING_WITH_PROPERTIES" in step_text
  distinct_colors = {
      (round(float(r), 3), round(float(g), 3), round(float(b), 3))
      for r, g, b in COLOUR_RGB_PATTERN.findall(step_text)
  }
  assert len(distinct_colors) >= MIN_EXPECTED_COLORS

  transparencies = [float(val) for val in TRANSPARENCY_PATTERN.findall(step_text)]
  for expected in EXPECTED_ASSEMBLY_TRANSPARENCIES:
    assert any(abs(val - expected) <= TRANSPARENCY_TOL for val in transparencies), (
        f"Missing transparency {expected} in {sorted(set(transparencies))}"
    )

  products = set(PRODUCT_PATTERN.findall(step_text))
  for expected_name in EXPECTED_DRESSED_PRODUCTS:
    assert expected_name in products
  assert "COMPOUND" not in products

  head_step = export_styled_step(
      style_single_part(generate_cylinder_head(), "cylinder_head", "ALUMINUM_6061_T6", SINGLE_HOUSING_OPACITY),
      tmp_path / "cylinder_head.step",
  )
  head_text = head_step.read_text(encoding="utf-8")
  head_trans = [float(val) for val in TRANSPARENCY_PATTERN.findall(head_text)]
  assert any(abs(val - EXPECTED_STANDALONE_TRANSPARENCY) <= TRANSPARENCY_TOL for val in head_trans)
  assert "cylinder_head" in set(PRODUCT_PATTERN.findall(head_text))


def _assert_xcaf_step_roundtrip(step_path: Path, expect_dressed: bool) -> None:
  """Verify STEP XCAF round-trip validity, <1e-7 mm edge gaps, product names, and RGB palette via OCCT."""
  imported = import_step(str(step_path))
  assert imported.is_valid
  assert len(imported.solids()) > 0
  assert max_solid_edge_surface_gap_mm(imported, sample_count=ROUND_TRIP_SAMPLES) < STEP_PRECISION_TOL_MM

  if not expect_dressed:
    return

  labels = {child.label for child in imported.children}
  assert set(EXPECTED_DRESSED_PRODUCTS).issubset(labels)
  distinct_colors = {
      tuple(round(float(channel), COLOR_ROUND_DIGITS) for channel in tuple(child.color)[:RGB_CHANNEL_COUNT])
      for child in imported.children
      if child.color is not None
  }
  assert len(distinct_colors) >= MIN_EXPECTED_COLORS


def _verify_optional_solidcraft_insert(step_path: Path, expect_dressed: bool, cli_path: str | None) -> None:
  """Run optional SolidCraft CLI file.insert_step verification when solidcraft-cli is installed."""
  if cli_path is None:
    return
  raw_out = subprocess.check_output(
      [cli_path, "exec", "file.insert_step", json.dumps({"path": str(step_path)})],
      text=True,
  )
  payload = json.loads(raw_out)
  assert len(payload.get("bodies", [])) > 0
  assert payload.get("warnings") == []
  if expect_dressed:
    bodies = set(payload["bodies"])
    assert "crankcase_right" in bodies
    assert "carburetor_vm26" in bodies


def test_solidcraft_inserts_styled_assemblies_with_zero_warnings(tmp_path: Path) -> None:
  """Verify dressed and master STEP assemblies pass OCCT XCAF checks and optional SolidCraft import."""
  dressed_path = export_styled_step(generate_dressed_engine_assembly(0.0), tmp_path / "dressed_engine_assembly.step")
  master_path = export_styled_step(generate_engine_assembly(0.0), tmp_path / "engine_assembly.step")

  solidcraft_bin = shutil.which("solidcraft-cli")
  for step_path, expect_dressed in ((dressed_path, True), (master_path, False)):
    _assert_xcaf_step_roundtrip(step_path, expect_dressed=expect_dressed)
    _verify_optional_solidcraft_insert(step_path, expect_dressed=expect_dressed, cli_path=None)
    _verify_optional_solidcraft_insert(step_path, expect_dressed=expect_dressed, cli_path=solidcraft_bin)


def perf_heal_carburetor_edges(
    n: int = 4,
    *,
    _scales: tuple[int, ...] = (2, 4, 8),
    _max_complexity: str = "O(N log N)",
):
  """Empirical benchmark verifying edge gap checks scale linearly across shape copies."""
  _ = _scales, _max_complexity
  carb = generate_carburetor_vm26()
  yield
  for _ in range(n):
    assert max_solid_edge_surface_gap_mm(carb, sample_count=ROUND_TRIP_SAMPLES, bspline_only=True) <= HEALED_BSPLINE_GAP_TOL_MM
