"""Unit tests for FDM orientation, STL export, watertightness, printability, and calibration coupon."""

from pathlib import Path
import struct

from build123d import Align, Axis, Box, Cylinder, Part, chamfer
import numpy as np
import pytest

from openengine.cad.piston import generate_piston
from openengine.printing.analysis import (
    PartPrintReport,
    _read_binary_stl_arrays,
    analyze_part_printability,
    export_part_stl,
)
from openengine.printing.coupon import (
    BORE_X_POSITIONS_MM,
    BORE_Y_MM,
    PEG_TAB_HEIGHT_Z_MM,
    PEG_TAB_MAX_X_MM,
    PEG_TAB_MAX_Y_MM,
    PEG_TAB_MIN_X_MM,
    PEG_TAB_MIN_Y_MM,
    coupon_bore_radial_offsets_mm,
    generate_tolerance_coupon,
)
from openengine.printing.orientation import (
    OrientedPartResult,
    apply_shrinkage_compensation,
    orient_part_for_printing,
)
from openengine.printing.tolerances import (
    PrintKitConfig,
    running_clearance_radial_mm,
    shrinkage_compensation_xyz,
    slip_clearance_radial_mm,
)


def _make_l_bracket() -> Part:
  """Build an asymmetric L-bracket (50x20x6 mm base + 6x20x40 mm vertical arm)."""
  base = Box(50.0, 20.0, 6.0, align=(Align.MIN, Align.CENTER, Align.MIN))
  arm = Box(6.0, 20.0, 40.0, align=(Align.MIN, Align.CENTER, Align.MIN))
  return Part(base.fuse(arm).wrapped)


def _make_c_bracket() -> Part:
  """Build a C-bracket with an upper overhang directly above its lower base plate."""
  bottom = Box(32.0, 20.0, 5.0, align=(Align.MIN, Align.CENTER, Align.MIN))
  spine = Box(6.0, 20.0, 30.0, align=(Align.MIN, Align.CENTER, Align.MIN))
  top = Box(32.0, 20.0, 5.0, align=(Align.MIN, Align.CENTER, Align.MIN)).translate((0.0, 0.0, 25.0))
  return Part(bottom.fuse(spine).fuse(top).wrapped)


def test_orient_l_bracket_picks_flat_face_down(tmp_path: Path):
  """An upside-down L-bracket must flip to put its large flat base on Z=0 with zero overhang."""
  l_upside_down = _make_l_bracket().rotate(Axis.X, 180.0).translate((45.0, -30.0, 120.0))
  res = orient_part_for_printing(l_upside_down, "l_bracket")

  assert isinstance(res, OrientedPartResult)
  assert res.part_name == "l_bracket"
  assert res.overhang_area_mm2 == pytest.approx(0.0, abs=1.0e-3)
  assert res.overhang_pct == pytest.approx(0.0, abs=1.0e-3)
  assert res.contact_area_mm2 == pytest.approx(1000.0, abs=2.0)

  bbox = res.oriented_part.bounding_box()
  assert bbox.min.Z == pytest.approx(0.0, abs=1.0e-5)
  assert bbox.center().X == pytest.approx(0.0, abs=1.0e-5)
  assert bbox.center().Y == pytest.approx(0.0, abs=1.0e-5)

  # Exported STL coordinates must reflect the baked oriented pose directly.
  stl_path, tri_count, watertight = export_part_stl(res.oriented_part, tmp_path / "l_bracket.stl")
  assert watertight is True
  assert tri_count > 0
  _, _, v0, v1, v2, _ = _read_binary_stl_arrays(stl_path)
  verts = np.vstack((v0, v1, v2))
  assert float(verts[:, 2].min()) == pytest.approx(0.0, abs=1.0e-4)
  assert float(0.5 * (verts[:, 0].min() + verts[:, 0].max())) == pytest.approx(0.0, abs=1.0e-4)
  assert float(0.5 * (verts[:, 1].min() + verts[:, 1].max())) == pytest.approx(0.0, abs=1.0e-4)


def test_orient_flanged_cylinder_and_side_cylinder():
  """Flanged hub must orient flange-down and sideways cylinder must stand upright."""
  flange = Cylinder(25.0, 5.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
  shaft = Cylinder(8.0, 30.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).translate((0.0, 0.0, 5.0))
  flanged_upside = Part(flange.fuse(shaft).wrapped).rotate(Axis.X, 180.0)

  res_flange = orient_part_for_printing(flanged_upside, "flanged_hub")
  assert res_flange.overhang_area_mm2 == pytest.approx(0.0, abs=1.0)
  assert res_flange.contact_area_mm2 > 1900.0
  assert res_flange.oriented_dims_xyz_mm[2] == pytest.approx(35.0, abs=0.1)

  cyl_sideways = Part(Cylinder(10.0, 40.0).rotate(Axis.X, 90.0).wrapped)
  res_cyl = orient_part_for_printing(cyl_sideways, "shaft_blank")
  assert res_cyl.overhang_area_mm2 == pytest.approx(0.0, abs=1.0)
  assert res_cyl.contact_area_mm2 > 300.0
  assert res_cyl.oriented_dims_xyz_mm[2] == pytest.approx(40.0, abs=0.1)


def test_orient_preferred_rotation_and_volume_fit_z_rotation():
  """Preferred rotation respects +90 deg Z rotation for narrow print volumes and 45 deg diagonal fallback."""
  long_bar = Part(Box(120.0, 20.0, 10.0).wrapped)
  res_rot90 = orient_part_for_printing(
      long_bar,
      "long_bar",
      print_volume_mm=(50.0, 150.0, 100.0),
      preferred_rotation_xyz_deg=(0.0, 0.0, 0.0),
  )
  assert res_rot90.rotation_xyz_deg == pytest.approx((0.0, 0.0, 90.0))
  assert res_rot90.oriented_dims_xyz_mm[0] <= 50.0 + 1.0e-4
  assert res_rot90.oriented_dims_xyz_mm[1] <= 150.0 + 1.0e-4

  # Preferred orientation that already fits must be preserved even if upside-down.
  flange = Cylinder(25.0, 5.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
  shaft = Cylinder(8.0, 30.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).translate((0.0, 0.0, 5.0))
  flanged = Part(flange.fuse(shaft).wrapped)
  res_pref = orient_part_for_printing(
      flanged,
      "flanged_forced",
      preferred_rotation_xyz_deg=(180.0, 0.0, 0.0),
  )
  assert res_pref.rotation_xyz_deg == pytest.approx((180.0, 0.0, 0.0))
  assert res_pref.overhang_area_mm2 > 1000.0

  # Diagonal 45-deg Z fallback when 280 mm bar exceeds 250x220 orthogonal axes.
  diag_bar = Part(Box(280.0, 30.0, 15.0).wrapped)
  res_diag = orient_part_for_printing(diag_bar, "diag_bar", print_volume_mm=(250.0, 220.0, 270.0))
  assert res_diag.oriented_dims_xyz_mm[0] <= 250.0
  assert res_diag.oriented_dims_xyz_mm[1] <= 220.0


def test_export_part_stl_watertight_and_corrupt_detection(tmp_path: Path):
  """Binary STL export produces valid watertight mesh and detects open boundary edges."""
  piston_half = generate_piston().scale(0.5)
  oriented = orient_part_for_printing(piston_half, "piston_05x")
  stl_file = tmp_path / "nested" / "piston_05x.stl"

  out_path, tri_count, watertight = export_part_stl(oriented.oriented_part, stl_file)
  assert out_path == stl_file
  assert out_path.is_file()
  assert tri_count > 500
  assert watertight is True

  raw = out_path.read_bytes()
  header_count = struct.unpack("<I", raw[80:84])[0]
  assert header_count == tri_count
  assert len(raw) == 84 + (50 * tri_count)

  # Remove one triangle from the binary STL -> must report non-watertight.
  corrupted = bytearray(raw[:-50])
  struct.pack_into("<I", corrupted, 80, tri_count - 1)
  corrupt_path = tmp_path / "corrupt.stl"
  corrupt_path.write_bytes(bytes(corrupted))
  corr_count, corr_wt, *_ = _read_binary_stl_arrays(corrupt_path)
  assert corr_count == tri_count - 1
  assert corr_wt is False


def test_overhang_45deg_chamfer_and_support_classification(tmp_path: Path):
  """Verify 45-deg overhangs pass support-free while buildplate and occluded overhangs classify properly."""
  # 1. Box with 45-deg bottom chamfer (self-supporting at 45 deg).
  block = Box(30.0, 30.0, 20.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
  bottom_edges = block.edges().filter_by(lambda e: e.center().Z < 0.1)
  chamfered = Part(chamfer(bottom_edges, length=3.0).wrapped)
  rep_chamfer = analyze_part_printability(
      chamfered,
      "chamfer45",
      tmp_path / "chamfer45.stl",
      max_overhang_deg=45.0,
  )
  assert rep_chamfer.watertight is True
  assert rep_chamfer.overhang_area_mm2 == pytest.approx(0.0, abs=1.0)
  assert rep_chamfer.support_recommendation == "none"

  # 2. Upside-down L-bracket (exposed horizontal leg over open build plate).
  l_upside = Part(_make_l_bracket().rotate(Axis.X, 180.0).wrapped)
  l_oriented = orient_part_for_printing(
      l_upside,
      "l_upside",
      preferred_rotation_xyz_deg=(0.0, 0.0, 0.0),
  )
  rep_bp = analyze_part_printability(
      l_oriented.oriented_part,
      "l_upside",
      tmp_path / "l_upside.stl",
  )
  assert rep_bp.overhang_area_mm2 > 650.0
  assert rep_bp.support_recommendation == "buildplate_only"

  # 3. C-bracket (top flange occluded below by bottom flange -> tree supports).
  rep_tree = analyze_part_printability(
      _make_c_bracket(),
      "c_bracket",
      tmp_path / "c_bracket.stl",
  )
  assert rep_tree.overhang_area_mm2 > 450.0
  assert rep_tree.support_recommendation == "tree_supports"


def test_volume_fit_brim_mass_and_time_estimates(tmp_path: Path):
  """Verify print-volume fit failure, brim recommendation, mesh smoothness, and slicer tunables."""
  tall_slender = Part(Box(8.0, 8.0, 60.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).wrapped)
  rep_tall = analyze_part_printability(
      tall_slender,
      "tall_pin",
      tmp_path / "tall.stl",
      print_volume_mm=(250.0, 220.0, 50.0),
  )
  assert isinstance(rep_tall, PartPrintReport)
  assert rep_tall.fits_print_volume is False
  assert rep_tall.brim_recommended is True
  assert len(rep_tall.warnings) >= 2

  flat_block = Part(Box(50.0, 35.0, 12.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).wrapped)
  rep_20 = analyze_part_printability(
      flat_block,
      "flat_20",
      tmp_path / "flat20.stl",
      infill_pct=20.0,
      density_g_cm3=1.27,
  )
  rep_100 = analyze_part_printability(
      flat_block,
      "flat_100",
      tmp_path / "flat100.stl",
      infill_pct=100.0,
      density_g_cm3=1.27,
  )
  rep_thick_tb = analyze_part_printability(
      flat_block,
      "flat_tb8",
      tmp_path / "flat_tb8.stl",
      infill_pct=20.0,
      top_bottom_layers=8,
      density_g_cm3=1.27,
  )
  rep_fast_flow = analyze_part_printability(
      flat_block,
      "flat_fast",
      tmp_path / "flat_fast.stl",
      infill_pct=20.0,
      volumetric_flow_mm3_s=16.0,
      density_g_cm3=1.27,
  )
  assert rep_20.fits_print_volume is True
  assert rep_20.brim_recommended is False
  assert rep_20.volume_cm3 == pytest.approx(21.0, abs=0.05)
  assert rep_100.estimated_mass_g == pytest.approx(21.0 * 1.27, abs=0.2)
  assert 0.0 < rep_20.estimated_mass_g < rep_thick_tb.estimated_mass_g < rep_100.estimated_mass_g
  assert rep_20.estimated_print_time_min > rep_fast_flow.estimated_print_time_min > 1.0

  # Verify support mass overhead scales supported part mass cleanly.
  c_part = _make_c_bracket()
  rep_sup_0 = analyze_part_printability(
      c_part,
      "c_sup0",
      tmp_path / "c_sup0.stl",
      support_mass_overhead_pct=0.0,
  )
  rep_sup_25 = analyze_part_printability(
      c_part,
      "c_sup25",
      tmp_path / "c_sup25.stl",
      support_mass_overhead_pct=25.0,
  )
  assert rep_sup_25.estimated_mass_g == pytest.approx(rep_sup_0.estimated_mass_g * 1.25, rel=1e-3)

  # Verify ultra mesh quality preserves substantially finer cylinder smoothness than draft.
  cyl = Part(Cylinder(20.0, 15.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).wrapped)
  rep_draft = analyze_part_printability(
      cyl,
      "cyl_draft",
      tmp_path / "cyl_draft.stl",
      cfg=PrintKitConfig(mesh_quality="draft"),
  )
  rep_ultra = analyze_part_printability(
      cyl,
      "cyl_ultra",
      tmp_path / "cyl_ultra.stl",
      cfg=PrintKitConfig(mesh_quality="ultra"),
  )
  assert rep_draft.watertight is True and rep_ultra.watertight is True
  assert rep_ultra.triangle_count > (3 * rep_draft.triangle_count)


def test_print_kit_config_and_shrinkage_compensation(tmp_path: Path):
  """PrintKitConfig drives density/overhang settings and oriented-space shrinkage compensation."""
  plate = Part(Box(100.0, 50.0, 20.0, align=(Align.CENTER, Align.CENTER, Align.MIN)).wrapped)
  cfg_petg = PrintKitConfig(filament="PETG", infill_pct=20.0)
  cfg_asa = PrintKitConfig(filament="ASA", infill_pct=20.0)

  rep_petg = analyze_part_printability(plate, "plate_petg", tmp_path / "petg.stl", cfg_petg)
  rep_asa = analyze_part_printability(
      plate,
      "plate_asa",
      tmp_path / "asa.stl",
      cfg_asa,
      compensate_shrinkage=True,
  )
  assert rep_petg.estimated_mass_g > rep_asa.estimated_mass_g
  assert rep_asa.watertight is True

  k_xy, _, k_z = shrinkage_compensation_xyz(cfg_asa.filament_profile)
  assert rep_asa.oriented_dims_mm[0] == pytest.approx(100.0 * k_xy, abs=1.0e-3)
  assert rep_asa.oriented_dims_mm[1] == pytest.approx(50.0 * k_xy, abs=1.0e-3)
  assert rep_asa.oriented_dims_mm[2] == pytest.approx(20.0 * k_z, abs=1.0e-3)

  comp = apply_shrinkage_compensation(plate, (k_xy, k_xy, k_z))
  assert comp.bounding_box().min.Z == pytest.approx(0.0, abs=1.0e-5)


def test_tolerance_coupon_geometry_mating_and_watertightness(tmp_path: Path):
  """Calibration coupon is ~50x35x12 mm, support-free, watertight, and functionally mates."""
  cfg_petg = PrintKitConfig(filament="PETG")
  coupon = generate_tolerance_coupon(cfg_petg)
  assert coupon.is_valid
  solids = list(coupon.solids())
  assert len(solids) == 3

  bbox = coupon.bounding_box()
  assert bbox.size.X == pytest.approx(50.0, abs=0.2)
  assert bbox.size.Y == pytest.approx(35.0, abs=0.2)
  assert bbox.size.Z == pytest.approx(12.0, abs=0.2)
  assert bbox.min.Z == pytest.approx(0.0, abs=1.0e-5)

  # All 3 printed solids must be strictly separated on the build plate.
  for i in range(len(solids)):
    for j in range(i + 1, len(solids)):
      inter = solids[i].intersect(solids[j])
      ov = sum(s.volume for s in inter) if inter else 0.0
      assert ov == pytest.approx(0.0, abs=1.0e-5)

  # Verify binary STL export and printability metrics.
  rep = analyze_part_printability(coupon, "coupon_petg", tmp_path / "coupon_petg.stl", cfg_petg)
  assert rep.watertight is True
  assert rep.fits_print_volume is True
  assert rep.support_recommendation == "none"
  assert rep.overhang_area_mm2 == pytest.approx(0.0, abs=1.0e-3)
  assert rep.triangle_count > 1000

  # Verify the 5 bore offsets step cleanly around slip & running clearances.
  offsets = coupon_bore_radial_offsets_mm(cfg_petg)
  assert len(offsets) == 5
  assert offsets == pytest.approx((-0.05, 0.0, 0.10, 0.20, 0.30), abs=1.0e-6)
  assert offsets[3] == pytest.approx(slip_clearance_radial_mm(cfg_petg))
  assert offsets[4] == pytest.approx(running_clearance_radial_mm(cfg_petg))

  # Test peg interference in press bore (#1) vs clean clearance in slip/running bores (#4, #5).
  gauge_solid, latch_solid, peg_solid = solids
  peg_cx = 0.5 * (PEG_TAB_MIN_X_MM + PEG_TAB_MAX_X_MM)
  peg_cy = 0.5 * (PEG_TAB_MIN_Y_MM + PEG_TAB_MAX_Y_MM)
  press_peg = peg_solid.translate((BORE_X_POSITIONS_MM[0] - peg_cx, BORE_Y_MM - peg_cy, -PEG_TAB_HEIGHT_Z_MM))
  press_hit = gauge_solid.intersect(press_peg)
  assert (sum(s.volume for s in press_hit) if press_hit else 0.0) > 1.0

  for bore_idx in (3, 4):
    clear_peg = peg_solid.translate((BORE_X_POSITIONS_MM[bore_idx] - peg_cx, BORE_Y_MM - peg_cy, -PEG_TAB_HEIGHT_Z_MM))
    clear_hit = gauge_solid.intersect(clear_peg)
    assert (sum(s.volume for s in clear_hit) if clear_hit else 0.0) == pytest.approx(0.0, abs=1.0e-5)

  # Test guided cantilever snap-latch: clears seated pocket at (-15, 0, +2), locks laterally
  # and vertically via dovetail rail, and engages catch midway even at worst-case lateral play.
  seated_latch = latch_solid.translate((-15.0, 0.0, 2.0))
  seated_hit = gauge_solid.intersect(seated_latch)
  assert (sum(s.volume for s in seated_hit) if seated_hit else 0.0) == pytest.approx(0.0, abs=1.0e-5)

  midway_latch = latch_solid.translate((-11.5, 0.0, 2.0))
  midway_hit = gauge_solid.intersect(midway_latch)
  assert (sum(s.volume for s in midway_hit) if midway_hit else 0.0) > 1.0

  r_slip = slip_clearance_radial_mm(cfg_petg)
  # Worst-case lateral shift within guide rail clearance must still solidly engage the catch ledge.
  worst_dy_midway = latch_solid.translate((-12.0, -r_slip + 0.01, 2.0))
  worst_dy_hit = gauge_solid.intersect(worst_dy_midway)
  assert (sum(s.volume for s in worst_dy_hit) if worst_dy_hit else 0.0) > 1.5

  # Attempting to shift latch further -Y into deflection relief hits the solid guide channel wall.
  lateral_stop = latch_solid.translate((-15.0, -(r_slip + 0.35), 2.0))
  stop_hit = gauge_solid.intersect(lateral_stop)
  assert (sum(s.volume for s in stop_hit) if stop_hit else 0.0) > 5.0

  # Attempting to lift latch straight up (+Z) hits the 25.1-deg dovetail retention shoulders.
  lift_z = latch_solid.translate((-15.0, 0.0, 3.5))
  lift_hit = gauge_solid.intersect(lift_z)
  assert (sum(s.volume for s in lift_hit) if lift_hit else 0.0) > 5.0

  # Verify filament-specific adaptation (PLA vs TPU_95A).
  coupon_pla = generate_tolerance_coupon(PrintKitConfig(filament="PLA"))
  coupon_tpu = generate_tolerance_coupon(PrintKitConfig(filament="TPU_95A"))
  assert coupon_pla.solids()[1].volume < coupon.solids()[1].volume < coupon_tpu.solids()[1].volume
  rep_pla = analyze_part_printability(coupon_pla, "coupon_pla", tmp_path / "coupon_pla.stl")
  rep_tpu = analyze_part_printability(coupon_tpu, "coupon_tpu", tmp_path / "coupon_tpu.stl")
  assert rep_pla.watertight is True
  assert rep_tpu.watertight is True
