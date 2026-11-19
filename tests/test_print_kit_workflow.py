"""End-to-end integration tests for the OpenEngine 3D-printable model kit workflow."""

import json
from pathlib import Path

from typer.testing import CliRunner

from openengine.cli import app
import openengine.printing as printing
from openengine.printing import (
    FILAMENT_PROFILES,
    FitmentCheckItem,
    KitPart,
    PrintKitConfig,
    PrintKitSummary,
    build_full_kit_assembly_parts,
    build_full_kit_parts,
    run_print_kit_workflow,
    verify_kit_fitment_and_collisions,
)
from openengine.printing.analysis import _read_binary_stl_arrays

REQUIRED_FITMENT_JOINTS: frozenset[str] = frozenset({
    "piston_to_cylinder_block",
    "crankpin_to_conrod",
    "wrist_pin_to_conrod",
    "main_journal_to_main_bushing",
    "valve_stem_to_cylinder_head",
    "cam_core_to_cam_tower",
    "rocker_arm_to_rocker_shaft",
    "idler_gear_to_idler_pin",
    "pto_shaft_to_timing_cover",
})


def test_printing_package_public_exports() -> None:
  """Verify openengine.printing cleanly re-exports public symbols across all 11 submodules."""
  expected_attrs = (
      "FILAMENT_PROFILES",
      "FilamentProfile",
      "get_filament_profile",
      "PrintKitConfig",
      "KitPart",
      "SnapFitDesign",
      "auto_size_cantilever_snap",
      "make_cantilever_snap_hook",
      "make_teardrop_bore_cutter",
      "make_compliant_wave_spring",
      "orient_part_for_printing",
      "apply_shrinkage_compensation",
      "analyze_part_printability",
      "export_part_stl",
      "generate_tolerance_coupon",
      "build_kit_rotating_parts",
      "build_kit_housings_parts",
      "build_kit_valvetrain_parts",
      "build_full_kit_parts",
      "build_full_kit_assembly_parts",
      "verify_kit_fitment_and_collisions",
      "run_print_kit_workflow",
      "write_print_guide_markdown",
      "pack_print_kit_plates",
      "compute_filament_mass_by_type_g",
      "BuildPlateReport",
      "PlatePackingResult",
      "PlatePlacement",
      "FitmentCheckItem",
      "PrintKitSummary",
  )
  for attr in expected_attrs:
    assert hasattr(printing, attr), f"Missing export openengine.printing.{attr}"
  assert "PETG" in FILAMENT_PROFILES


def test_full_kit_inventory_and_assembly_expansion() -> None:
  """Verify combined BOM contains 33 unique KitParts (42 pieces) and 42 posed instances."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  bom = build_full_kit_parts(cfg)
  assert len(bom) == 33
  assert sum(kp.quantity for kp in bom.values()) == 42
  assert all(isinstance(kp, KitPart) and kp.part.is_valid for kp in bom.values())

  asm = build_full_kit_assembly_parts(cfg, crank_angle_deg=90.0)
  assert len(asm) == 42
  assert all(kp.quantity == 1 and kp.part.is_valid for kp in asm.values())


def test_cutaway_cross_module_collisions_and_fitment() -> None:
  """Verify zero full-engine cross-module collisions at 0/90/180/270 deg in cutaway mode."""
  cfg_cut = PrintKitConfig(scale=0.5, filament="PETG", cutaway=True)
  res = verify_kit_fitment_and_collisions(
      cfg_cut,
      crank_angles_deg=(0.0, 90.0, 180.0, 270.0),
  )
  assert res.static_collisions == ()
  assert res.kinematic_collisions == ()
  assert res.all_passed is True

  joint_map = {item.joint_name: item for item in res.fitment_checks}
  assert REQUIRED_FITMENT_JOINTS.issubset(joint_map.keys())
  for joint_name in REQUIRED_FITMENT_JOINTS:
    chk = joint_map[joint_name]
    assert isinstance(chk, FitmentCheckItem)
    assert chk.is_valid is True
    assert chk.measured_clearance_mm > 0.0
    assert abs(chk.measured_clearance_mm - chk.target_clearance_mm) <= 0.03

  for snap_name, design in res.snap_fit_designs.items():
    assert design.is_safe is True, f"Unsafe snap design {snap_name}: SF={design.safety_factor}"
    assert design.safety_factor >= 1.5


def test_end_to_end_workflow_petg_and_pla_comparison(tmp_path: Path) -> None:
  """Verify full PETG workflow (33 STLs + coupon + preview + JSON + MD) and PLA comparison."""
  petg_dir = tmp_path / "petg_kit"
  cfg_petg = PrintKitConfig(scale=0.5, filament="PETG", include_coupon=True)
  summary_petg = run_print_kit_workflow(
      cfg_petg,
      output_dir=petg_dir,
      export_assembled_preview=True,
      verify_kinematics=True,
  )

  assert isinstance(summary_petg, PrintKitSummary)
  assert summary_petg.total_unique_parts == 33
  assert summary_petg.total_printed_pieces == 42
  assert summary_petg.total_mass_g > 0.0
  assert list(summary_petg.filament_mass_by_type_g.keys()) == ["PETG"]
  assert abs(summary_petg.filament_mass_by_type_g["PETG"] - summary_petg.total_mass_g) <= 0.05
  assert summary_petg.plate_packing is not None
  assert summary_petg.plate_packing.total_placed_pieces == 42
  assert summary_petg.plate_packing.all_watertight is True
  assert summary_petg.plate_packing.all_fit_print_volume is True
  assert summary_petg.total_print_time_min > 0.0
  assert summary_petg.all_fit_print_volume is True
  assert summary_petg.all_watertight is True
  assert summary_petg.static_collisions == ()
  assert summary_petg.kinematic_collisions == ()
  assert len(summary_petg.parts_reports) == 33
  assert summary_petg.coupon_report is not None

  # Check all 33 kit STLs + coupon STL + preview STL on disk & watertightness
  stl_files = sorted(summary_petg.stl_dir.glob("*.stl"))
  assert len(stl_files) == 34
  for rep in list(summary_petg.parts_reports) + [summary_petg.coupon_report]:
    assert rep.stl_path.is_file()
    assert rep.watertight is True
    assert rep.fits_print_volume is True
    assert rep.triangle_count > 100
    tri_cnt, wt, *_ = _read_binary_stl_arrays(rep.stl_path)
    assert wt is True and tri_cnt == rep.triangle_count

  assert summary_petg.assembled_preview_stl_path is not None
  assert summary_petg.assembled_preview_stl_path.is_file()
  prev_tris, prev_wt, *_ = _read_binary_stl_arrays(summary_petg.assembled_preview_stl_path)
  assert prev_wt is True and prev_tris > 10000

  # Verify fitment checks & snap safety
  fit_map = {c.joint_name: c for c in summary_petg.fitment_checks}
  assert REQUIRED_FITMENT_JOINTS.issubset(fit_map.keys())
  assert all(c.is_valid for c in summary_petg.fitment_checks)
  assert all(d.is_safe for d in summary_petg.snap_fit_designs.values())

  # Verify manifest JSON
  assert summary_petg.manifest_path.is_file()
  manifest = json.loads(summary_petg.manifest_path.read_text(encoding="utf-8"))
  assert manifest["totals"]["total_unique_parts"] == 33
  assert manifest["totals"]["total_printed_pieces"] == 42
  assert manifest["totals"]["all_fit_print_volume"] is True
  assert manifest["totals"]["all_watertight"] is True
  assert manifest["mesh_tessellation"]["quality"] == "ultra"
  assert manifest["verification"]["collisions_passed"] is True
  assert manifest["verification"]["fitment_all_passed"] is True
  assert manifest["verification"]["snap_all_safe"] is True
  assert len(manifest["parts"]) == 33
  assert manifest["coupon"] is not None

  # Verify Markdown Print & Assembly Guide
  assert summary_petg.guide_path.is_file()
  guide_text = summary_petg.guide_path.read_text(encoding="utf-8")
  for required_phrase in (
      "Filament & Slicer Setup",
      "PEI",
      "Calibration Coupon Workflow",
      "tolerance_calibration_coupon.stl",
      "--clearance-bias",
      "Full Per-Part Print Table",
      "Snap-Fit Mechanics & Clearance Fitment Audit",
      "Sub-Assembly 1:",
      "Sub-Assembly 5:",
      "kit_crank_left",
      "kit_cylinder_block",
      "kit_timing_gear_cam_28t",
  ):
    assert required_phrase in guide_text, f"Missing {required_phrase!r} in print_guide.md"

  # PLA comparison: tighter tolerances, lower mass, thinner brittle-adapted snap root
  pla_dir = tmp_path / "pla_kit"
  cfg_pla = PrintKitConfig(scale=0.5, filament="PLA", include_coupon=True)
  summary_pla = run_print_kit_workflow(
      cfg_pla,
      output_dir=pla_dir,
      export_assembled_preview=False,
      verify_kinematics=False,
  )
  assert summary_pla.all_watertight is True
  assert summary_pla.all_fit_print_volume is True
  assert summary_pla.static_collisions == ()
  assert summary_pla.total_mass_g < summary_petg.total_mass_g
  assert (
      summary_pla.snap_fit_designs["conrod_cap"].root_thickness_mm
      < summary_petg.snap_fit_designs["conrod_cap"].root_thickness_mm
  )
  assert summary_pla.snap_fit_designs["conrod_cap"].is_safe is True
  pla_fit_map = {c.joint_name: c for c in summary_pla.fitment_checks}
  assert (
      pla_fit_map["piston_to_cylinder_block"].measured_clearance_mm
      < fit_map["piston_to_cylinder_block"].measured_clearance_mm
  )


def test_cli_print_kit_command(tmp_path: Path) -> None:
  """Verify openengine print-kit CLI handles valid runs, print-volume overflow warnings, and invalid configs."""
  runner = CliRunner()

  # 1. Valid CLI execution
  ok_dir = tmp_path / "cli_ok"
  res_ok = runner.invoke(
      app,
      [
          "print-kit",
          "--scale",
          "0.5",
          "--filament",
          "PETG",
          "--skip-kinematics",
          "--no-coupon",
          "--output-dir",
          str(ok_dir),
      ],
  )
  assert res_ok.exit_code == 0, res_ok.output
  assert (ok_dir / "print_kit_manifest.json").is_file()
  assert (ok_dir / "print_guide.md").is_file()
  assert len(list((ok_dir / "parts").glob("*.stl"))) == 33
  assert len(list((ok_dir / "plates").glob("plate_*.stl"))) >= 1
  assert "OpenEngine Print Kit Summary" in res_ok.output
  assert "Total Filament Mass by Type" in res_ok.output
  assert "Packed Build Plates" in res_ok.output

  # 2. Tight print-volume overflow prints warning but still exits 0 (not collision/watertight/config failure)
  tight_dir = tmp_path / "cli_tight_volume"
  res_tight = runner.invoke(
      app,
      [
          "print-kit",
          "--scale",
          "0.5",
          "--print-volume",
          "60,60,60",
          "--skip-kinematics",
          "--no-coupon",
          "--output-dir",
          str(tight_dir),
      ],
  )
  assert res_tight.exit_code == 0, res_tight.output
  assert "exceeded" in res_tight.output.lower()

  # 3. Invalid filament / print-volume / mesh-quality exits non-zero cleanly
  res_bad_fil = runner.invoke(app, ["print-kit", "--filament", "UNOBTAINIUM"])
  assert res_bad_fil.exit_code == 1
  assert "Invalid print-kit configuration" in res_bad_fil.output

  res_bad_vol = runner.invoke(app, ["print-kit", "--print-volume", "250,220"])
  assert res_bad_vol.exit_code == 1
  assert "Invalid print-kit configuration" in res_bad_vol.output

  res_bad_mesh = runner.invoke(app, ["print-kit", "--mesh-quality", "blocky"])
  assert res_bad_mesh.exit_code == 1
  assert "Invalid print-kit configuration" in res_bad_mesh.output
