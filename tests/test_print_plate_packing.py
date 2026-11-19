"""Tests for 2D build-plate bin packing, merged meta-STLs, and filament mass by type."""

from __future__ import annotations

import json
from pathlib import Path

from build123d import Align, Box, BuildPart, Part
import numpy as np
import pytest
from typer.testing import CliRunner

from openengine.cli import app
from openengine.printing import (
    BuildPlateReport,
    DEFAULT_BRIM_PLATE_GAP_MM,
    DEFAULT_PLATE_GAP_MM,
    DEFAULT_PLATE_GROUP_MODE,
    DEFAULT_PLATE_MARGIN_MM,
    PartPrintReport,
    PlatePackingResult,
    PrintKitConfig,
    VALID_PLATE_GROUP_MODES,
    analyze_part_printability,
    compute_filament_mass_by_type_g,
    pack_print_kit_plates,
    run_print_kit_workflow,
)
from openengine.printing.analysis import _read_binary_stl_arrays


def _make_box_report(
    tmp_path: Path,
    name: str,
    dims_mm: tuple[float, float, float],
    *,
    quantity: int = 1,
    filament: str = "PETG",
    support_override: str | None = None,
    brim_override: bool | None = None,
) -> PartPrintReport:
  """Build a watertight seated box STL and return its ``PartPrintReport``."""
  dx, dy, dz = dims_mm
  with BuildPart() as bp:
    Box(dx, dy, dz, align=(Align.CENTER, Align.CENTER, Align.MIN))
  part: Part = bp.part
  cfg = PrintKitConfig(filament=filament)
  rep = analyze_part_printability(
      part=part,
      part_name=name,
      stl_path=tmp_path / f"{name}.stl",
      cfg=cfg,
      quantity=quantity,
      orient_first=False,
      compensate_shrinkage=False,
  )
  if support_override is not None:
    rep.support_recommendation = support_override
  if brim_override is not None:
    rep.brim_recommended = brim_override
  return rep


def _measure_stl_bounds(stl_path: Path) -> tuple[np.ndarray, np.ndarray, int, bool, float]:
  """Read binary STL and return ``(min_xyz, max_xyz, tri_count, watertight, signed_vol)``."""
  tri_count, watertight, v0, v1, v2, signed_vol = _read_binary_stl_arrays(stl_path)
  verts = np.concatenate((v0, v1, v2), axis=0)
  return np.min(verts, axis=0), np.max(verts, axis=0), tri_count, watertight, signed_vol


def test_print_kit_config_plate_validation() -> None:
  """Verify PrintKitConfig defaults and validation for plate-packing options."""
  cfg = PrintKitConfig()
  assert cfg.pack_plates is True
  assert cfg.plate_group_mode == DEFAULT_PLATE_GROUP_MODE == "fewest"
  assert cfg.plate_margin_mm == DEFAULT_PLATE_MARGIN_MM == 8.0
  assert cfg.plate_gap_mm == DEFAULT_PLATE_GAP_MM == 6.0
  assert VALID_PLATE_GROUP_MODES == ("fewest", "slicer", "category")

  cfg_slicer = PrintKitConfig(plate_group_mode=" SLICER ", plate_margin_mm=10.0, plate_gap_mm=8.0)
  assert cfg_slicer.plate_group_mode == "slicer"

  with pytest.raises(ValueError):
    PrintKitConfig(plate_group_mode="invalid_mode")
  with pytest.raises(ValueError):
    PrintKitConfig(plate_margin_mm=-1.0)
  with pytest.raises(ValueError):
    PrintKitConfig(plate_gap_mm=0.1)


def test_pack_quantity_expansion_z90_rotation_and_centered_watertight_stl(tmp_path: Path) -> None:
  """Verify quantity expansion, 90-deg Z-rotation, exact spacing, centering, and watertightness."""
  parts_dir = tmp_path / "parts"
  # Two 110x40 bars + three 30x25 blocks easily fit on a 160x160 bed (usable 144x144).
  rep_bar = _make_box_report(parts_dir, "long_bar", (110.0, 40.0, 12.0), quantity=2, brim_override=False)
  rep_blk = _make_box_report(parts_dir, "small_block", (30.0, 25.0, 10.0), quantity=3, brim_override=False)

  res = pack_print_kit_plates(
      parts_reports=[rep_bar, rep_blk],
      output_dir=tmp_path / "plates",
      printer_bed_mm=(160.0, 160.0, 160.0),
      part_categories={"long_bar": "rotating", "small_block": "valvetrain"},
      group_mode="fewest",
      margin_mm=8.0,
      gap_mm=6.0,
  )

  assert isinstance(res, PlatePackingResult)
  assert res.all_watertight is True
  assert res.all_fit_bed is True
  assert res.unplaced_parts == ()
  assert res.total_placed_pieces == 5
  assert len(res.plates) == 1

  plate = res.plates[0]
  assert isinstance(plate, BuildPlateReport)
  assert plate.plate_name == "plate_01"
  assert plate.piece_count == 5
  assert plate.unique_part_count == 2
  assert plate.effective_margin_mm >= 8.0 - 1e-2
  assert plate.effective_gap_mm == pytest.approx(6.0)
  assert plate.watertight is True
  assert plate.stl_path.is_file()

  min_xyz, max_xyz, tri_count, watertight, signed_vol = _measure_stl_bounds(plate.stl_path)
  expected_tris = (2 * rep_bar.triangle_count) + (3 * rep_blk.triangle_count)
  expected_vol_mm3 = (2 * 110.0 * 40.0 * 12.0) + (3 * 30.0 * 25.0 * 10.0)
  assert watertight is True
  assert tri_count == expected_tris
  assert signed_vol == pytest.approx(expected_vol_mm3, rel=1e-3)
  # Centered in XY around (0, 0) and seated on Z = 0.
  assert (min_xyz[0] + max_xyz[0]) == pytest.approx(0.0, abs=0.05)
  assert (min_xyz[1] + max_xyz[1]) == pytest.approx(0.0, abs=0.05)
  assert min_xyz[2] == pytest.approx(0.0, abs=0.05)
  assert (max_xyz[0] - min_xyz[0]) == pytest.approx(plate.used_dims_mm[0], abs=0.1)
  assert (max_xyz[1] - min_xyz[1]) == pytest.approx(plate.used_dims_mm[1], abs=0.1)


def test_brim_gap_upgrade_and_grouping_modes_with_multi_filament_mass(tmp_path: Path) -> None:
  """Verify 12 mm brim gap, grouping modes (fewest/slicer/category), and multi-filament totals."""
  parts_dir = tmp_path / "parts"
  reports = [
      _make_box_report(
          parts_dir,
          "case_left",
          (80.0, 60.0, 18.0),
          quantity=1,
          filament="PETG",
          support_override="none",
          brim_override=False,
      ),
      _make_box_report(
          parts_dir,
          "head_dome",
          (75.0, 55.0, 22.0),
          quantity=1,
          filament="PETG",
          support_override="buildplate_only",
          brim_override=False,
      ),
      _make_box_report(
          parts_dir,
          "tall_pin",
          (14.0, 14.0, 45.0),
          quantity=2,
          filament="PLA",
          support_override="none",
          brim_override=True,
      ),
      _make_box_report(
          parts_dir,
          "flex_seal",
          (40.0, 40.0, 6.0),
          quantity=1,
          filament="TPU_95A",
          support_override="none",
          brim_override=False,
      ),
  ]
  categories = {
      "case_left": "housings",
      "head_dome": "housings",
      "tall_pin": "valvetrain",
      "flex_seal": "rotating",
  }

  # Multi-filament mass aggregation helper check
  mass_by_type = compute_filament_mass_by_type_g(reports, primary_filament="PETG")
  assert list(mass_by_type.keys()) == ["PETG", "PLA", "TPU_95A"]
  expected_petg = reports[0].estimated_mass_g + reports[1].estimated_mass_g
  expected_pla = 2 * reports[2].estimated_mass_g
  expected_tpu = reports[3].estimated_mass_g
  assert mass_by_type["PETG"] == pytest.approx(expected_petg, abs=0.01)
  assert mass_by_type["PLA"] == pytest.approx(expected_pla, abs=0.01)
  assert mass_by_type["TPU_95A"] == pytest.approx(expected_tpu, abs=0.01)

  # 1. Fewest mode packs all 5 pieces onto 1 plate and upgrades gap to 12 mm due to tall_pin brim.
  res_fewest = pack_print_kit_plates(
      parts_reports=reports,
      output_dir=tmp_path / "plates_fewest",
      printer_bed_mm=(256.0, 256.0, 256.0),
      part_categories=categories,
      group_mode="fewest",
      margin_mm=8.0,
      gap_mm=6.0,
  )
  assert len(res_fewest.plates) == 1
  assert res_fewest.plates[0].brim_recommended is True
  assert res_fewest.plates[0].effective_gap_mm == pytest.approx(DEFAULT_BRIM_PLATE_GAP_MM)
  assert res_fewest.plates[0].support_recommendation == "buildplate_only"
  assert res_fewest.filament_mass_by_type_g == pytest.approx(mass_by_type, abs=0.01)
  assert any("Mixed support" in w for w in res_fewest.plates[0].warnings)
  assert any("Multiple filament" in w for w in res_fewest.plates[0].warnings)

  # 2. Slicer mode separates filament + support + brim -> 4 distinct uniform plates.
  res_slicer = pack_print_kit_plates(
      parts_reports=reports,
      output_dir=tmp_path / "plates_slicer",
      printer_bed_mm=(256.0, 256.0, 256.0),
      part_categories=categories,
      group_mode="slicer",
      margin_mm=8.0,
      gap_mm=6.0,
  )
  assert len(res_slicer.plates) == 4
  assert res_slicer.total_placed_pieces == 5
  assert res_slicer.all_watertight is True
  for plate in res_slicer.plates:
    assert len({p.filament_type for p in plate.placements}) == 1
    assert len({p.support_recommendation for p in plate.placements}) == 1
    assert len({p.brim_recommended for p in plate.placements}) == 1
    expected_gap = DEFAULT_BRIM_PLATE_GAP_MM if plate.brim_recommended else 6.0
    assert plate.effective_gap_mm == pytest.approx(expected_gap)

  # 3. Category mode separates housings, rotating, valvetrain -> 3 plates.
  res_cat = pack_print_kit_plates(
      parts_reports=reports,
      output_dir=tmp_path / "plates_category",
      printer_bed_mm=(256.0, 256.0, 256.0),
      part_categories=categories,
      group_mode="category",
      margin_mm=8.0,
      gap_mm=6.0,
  )
  assert len(res_cat.plates) == 3
  assert [p.group_label for p in res_cat.plates] == ["housings", "rotating", "valvetrain"]
  for plate in res_cat.plates:
    assert len({p.category for p in plate.placements}) == 1


def test_margin_oversized_and_bed_overflow_handling(tmp_path: Path) -> None:
  """Verify parts exceeding margin pack solo with warning, while bed-overflow parts report unplaced."""
  parts_dir = tmp_path / "parts"
  # On a 100x100x100 bed with 8 mm margin (usable 84x84):
  # - 94x50 fits bed (solo, 3 mm margin) with warning
  # - 120x40 exceeds physical bed -> unplaced
  # - 30x30 fits normally
  rep_margin_over = _make_box_report(parts_dir, "wide_flange", (94.0, 50.0, 10.0), brim_override=False)
  rep_too_big = _make_box_report(parts_dir, "giant_case", (120.0, 40.0, 10.0), brim_override=False)
  rep_normal = _make_box_report(parts_dir, "small_pin", (30.0, 30.0, 10.0), brim_override=False)

  res = pack_print_kit_plates(
      parts_reports=[rep_margin_over, rep_too_big, rep_normal],
      output_dir=tmp_path / "plates",
      printer_bed_mm=(100.0, 100.0, 100.0),
      group_mode="fewest",
      margin_mm=8.0,
      gap_mm=6.0,
  )

  assert res.unplaced_parts == ("giant_case",)
  assert res.all_fit_bed is False
  assert res.all_watertight is True
  assert res.total_placed_pieces == 2
  assert len(res.plates) == 2

  solo_plate = next(p for p in res.plates if p.placements[0].part_name == "wide_flange")
  assert solo_plate.fits_bed is True
  assert solo_plate.effective_margin_mm == pytest.approx(3.0, abs=0.1)
  assert any("margin" in w.lower() for w in solo_plate.warnings)


def test_real_kit_plate_packing_modes_and_cli_flags(tmp_path: Path) -> None:
  """Verify real 42-piece kit packs cleanly across all modes and CLI flags report filament mass."""
  kit_dir = tmp_path / "kit_default"
  cfg = PrintKitConfig(scale=0.5, filament="PETG", include_coupon=True, pack_plates=True)
  summary = run_print_kit_workflow(cfg, output_dir=kit_dir, verify_kinematics=False)

  assert summary.plate_packing is not None
  pp = summary.plate_packing
  assert pp.group_mode == "fewest"
  assert pp.total_placed_pieces == 42
  assert pp.unplaced_parts == ()
  assert pp.all_watertight is True
  assert pp.all_fit_bed is True
  assert 1 <= len(pp.plates) <= 6

  # Coupon stays excluded from packed build plates.
  placed_names = [pl.part_name for plate in pp.plates for pl in plate.placements]
  assert "tolerance_calibration_coupon" not in placed_names
  assert len(placed_names) == 42

  # Total filament mass separated by filament type matches kit total.
  assert list(summary.filament_mass_by_type_g.keys()) == ["PETG"]
  assert summary.filament_mass_by_type_g["PETG"] == pytest.approx(summary.total_mass_g, abs=0.05)
  assert pp.filament_mass_by_type_g["PETG"] == pytest.approx(summary.total_mass_g, abs=0.05)

  # Re-pack the already-exported real STLs in slicer & category modes for fast comparison.
  manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
  assert manifest["totals"]["total_filament_mass_by_type_g"]["PETG"] == pytest.approx(
      summary.total_mass_g, abs=0.05
  )
  assert manifest["plates"]["plate_count"] == len(pp.plates)
  assert manifest["plates"]["total_placed_pieces"] == 42
  assert manifest["artifacts"]["plates_dir"] is not None

  guide_text = summary.guide_path.read_text(encoding="utf-8")
  assert "Total Filament Mass by Filament Type" in guide_text
  assert "Packed Build Plates (Ready-to-Slice Meta-STLs)" in guide_text

  part_categories = {p["part_name"]: p["category"] for p in manifest["parts"]}
  res_slicer = pack_print_kit_plates(
      parts_reports=summary.parts_reports,
      output_dir=tmp_path / "real_slicer_plates",
      printer_bed_mm=cfg.printer_bed_mm,
      part_categories=part_categories,
      group_mode="slicer",
  )
  res_cat = pack_print_kit_plates(
      parts_reports=summary.parts_reports,
      output_dir=tmp_path / "real_category_plates",
      printer_bed_mm=cfg.printer_bed_mm,
      part_categories=part_categories,
      group_mode="category",
  )
  assert res_slicer.total_placed_pieces == 42 and res_slicer.all_watertight is True
  assert res_cat.total_placed_pieces == 42 and res_cat.all_watertight is True
  assert len(pp.plates) <= len(res_slicer.plates)
  assert len(pp.plates) <= len(res_cat.plates)

  # CLI checks: --no-pack-plates skips plates/, invalid --plate-group exits 1.
  runner = CliRunner()
  no_pack_dir = tmp_path / "cli_no_pack"
  res_no_pack = runner.invoke(
      app,
      [
          "print-kit",
          "--scale",
          "0.5",
          "--filament",
          "PETG",
          "--skip-kinematics",
          "--no-coupon",
          "--no-pack-plates",
          "--output-dir",
          str(no_pack_dir),
      ],
  )
  assert res_no_pack.exit_code == 0, res_no_pack.output
  assert not (no_pack_dir / "plates").exists()
  assert "Total Filament Mass by Type" in res_no_pack.output
  no_pack_manifest = json.loads((no_pack_dir / "print_kit_manifest.json").read_text(encoding="utf-8"))
  assert no_pack_manifest["plates"] is None
  assert "PETG" in no_pack_manifest["totals"]["total_filament_mass_by_type_g"]

  res_bad_grp = runner.invoke(app, ["print-kit", "--plate-group", "bogus"])
  assert res_bad_grp.exit_code == 1
  assert "Invalid print-kit configuration" in res_bad_grp.output
