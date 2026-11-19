"""Unit, kinematic collision, clearance, and printability tests for kit_rotating."""

from pathlib import Path

from build123d import Axis, Box, BuildPart, Cylinder, Locations, Mode, Part
import pytest

from openengine.cad.assemblies._common import find_pairwise_collisions, overlap_volume_mm3
from openengine.cad.assembly import EngineAssembly
from openengine.printing.analysis import analyze_part_printability
from openengine.printing.kit_rotating import (
    BUSHING_MAX_Y_MM,
    BUSHING_MIN_Y_MM,
    CRANK_AXIS_Z_MM,
    _conrod_geometry_params,
    _place_conrod_snap_shape,
    build_kit_rotating_assembly_parts,
    build_kit_rotating_parts,
)
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    min_printable_wall_mm,
    running_clearance_radial_mm,
)

EXPECTED_CANONICAL_KEYS: set[str] = {
    "kit_crank_left",
    "kit_crank_right",
    "kit_crankpin",
    "kit_conrod_beam",
    "kit_conrod_cap",
    "kit_piston",
    "kit_wrist_pin",
    "kit_main_bushing",
    "kit_flywheel",
    "kit_hand_crank",
}


def _annular_probe_y(r_in: float, r_out: float, y_min: float, y_max: float, x: float = 0.0, z: float = 0.0) -> Part:
  """Build a cylindrical annulus probe along Y to verify positive radial joint clearance."""
  length = y_max - y_min
  y_mid = 0.5 * (y_min + y_max)
  with BuildPart() as ring:
    with Locations((x, y_mid, z)):
      Cylinder(radius=r_out, height=length, rotation=(90.0, 0.0, 0.0))
      Cylinder(radius=r_in, height=length + 2.0, rotation=(90.0, 0.0, 0.0), mode=Mode.SUBTRACT)
  return ring.part


def test_bom_keys_quantities_and_solid_validity() -> None:
  """Verify all 10 canonical STLs (11 physical pieces) are valid single-solid KitParts."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  parts = build_kit_rotating_parts(cfg, crank_angle_deg=0.0)
  assert set(parts.keys()) == EXPECTED_CANONICAL_KEYS

  total_pieces = 0
  for name, kp in parts.items():
    assert isinstance(kp, KitPart)
    assert kp.name == name
    assert kp.category == "rotating"
    assert kp.preferred_rotation_xyz_deg is not None
    assert kp.part.is_valid
    assert len(kp.part.solids()) == 1
    assert kp.part.volume > 0.0
    total_pieces += kp.quantity

  assert parts["kit_main_bushing"].quantity == 2
  assert total_pieces == 11

  assembled = build_kit_rotating_assembly_parts(cfg, crank_angle_deg=0.0)
  assert len(assembled) == 11
  assert "kit_main_bushing_left" in assembled
  assert "kit_main_bushing_right" in assembled
  assert all(kp.quantity == 1 for kp in assembled.values())


def test_world_coordinates_and_kinematic_posing() -> None:
  """Verify crank datum, Y shaft bounds, piston TDC/BDC, and hand-crank cover clearance."""
  s = 0.5
  cfg = PrintKitConfig(scale=s)
  min_wall = min_printable_wall_mm(cfg, structural=True)

  tdc = build_kit_rotating_parts(cfg, crank_angle_deg=0.0)
  bdc = build_kit_rotating_parts(cfg, crank_angle_deg=180.0)
  ca90 = build_kit_rotating_parts(cfg, crank_angle_deg=90.0)

  # Piston TDC crown sits at Z=0, BDC crown at Z=-54*s
  assert tdc["kit_piston"].part.bounding_box().max.Z == pytest.approx(0.0, abs=1.0e-4)
  assert bdc["kit_piston"].part.bounding_box().max.Z == pytest.approx(-54.0 * s, abs=1.0e-4)

  # Wrist pin centered on crank kinematics
  st_90 = EngineAssembly().solve_kinematics(90.0)
  wp_90_bb = ca90["kit_wrist_pin"].part.bounding_box()
  assert 0.5 * (wp_90_bb.min.Z + wp_90_bb.max.Z) == pytest.approx(st_90.wrist_pin_z_mm * s, abs=1.0e-3)

  # Crankpin orbits around Z=-159*s with throw 27*s
  cp_tdc_bb = tdc["kit_crankpin"].part.bounding_box()
  cp_bdc_bb = bdc["kit_crankpin"].part.bounding_box()
  cp_90_bb = ca90["kit_crankpin"].part.bounding_box()
  assert 0.5 * (cp_tdc_bb.min.Z + cp_tdc_bb.max.Z) > (-159.0 + 27.0) * s - 0.5
  assert 0.5 * (cp_bdc_bb.min.Z + cp_bdc_bb.max.Z) < (-159.0 - 27.0) * s + 0.5
  assert 0.5 * (cp_90_bb.min.X + cp_90_bb.max.X) > 27.0 * s - 0.5

  # Crank Y spans, flywheel, hand crank, and main bushing coordinates
  cl_bb = tdc["kit_crank_left"].part.bounding_box()
  cr_bb = tdc["kit_crank_right"].part.bounding_box()
  fw_bb = tdc["kit_flywheel"].part.bounding_box()
  hc_bb = tdc["kit_hand_crank"].part.bounding_box()
  mb_bb = tdc["kit_main_bushing"].part.bounding_box()

  assert cl_bb.min.Y == pytest.approx(-97.0 * s, abs=1.0e-4)
  assert cl_bb.max.Y == pytest.approx(-10.5 * s, abs=1.0e-4)
  assert cr_bb.min.Y == pytest.approx(10.5 * s, abs=1.0e-4)
  assert cr_bb.max.Y == pytest.approx(96.0 * s, abs=1.0e-4)
  assert fw_bb.min.Y == pytest.approx(-86.75 * s, abs=0.2)
  assert fw_bb.max.Y == pytest.approx(-57.0 * s, abs=1.0e-4)
  assert fw_bb.max.X == pytest.approx(80.0 * s, abs=1.0e-4)
  assert hc_bb.min.Y >= max(68.0 * s + min_wall, 70.0 * s)
  assert mb_bb.min.Y == pytest.approx(BUSHING_MIN_Y_MM * s, abs=1.0e-4)
  assert mb_bb.max.Y == pytest.approx(BUSHING_MAX_Y_MM * s, abs=1.0e-4)
  assert mb_bb.max.Z == pytest.approx((CRANK_AXIS_Z_MM + 33.0) * s, abs=1.0e-4)


@pytest.mark.parametrize("scale", [0.4, 0.5, 1.0])
def test_zero_collisions_across_angles_and_scales(scale: float) -> None:
  """Verify zero unintended solid intersections across 0, 90, 180, 270 deg and scales."""
  cfg = PrintKitConfig(scale=scale, filament="PETG")
  for angle_deg in (0.0, 90.0, 180.0, 270.0):
    assembled = build_kit_rotating_assembly_parts(cfg, crank_angle_deg=angle_deg)
    parts_map = {name: kp.part for name, kp in assembled.items()}
    max_ov, hits = find_pairwise_collisions(parts_map, vol_tol_mm3=1.0e-3)
    assert not hits, f"Collisions at scale={scale}, CA={angle_deg}: {hits}"
    assert max_ov <= 1.0e-3


def test_multi_filament_support_zero_collisions() -> None:
  """Verify brittle and high-shrinkage filament profiles adapt safely without collisions."""
  for filament in ("PLA", "PETG_CF", "ABS"):
    cfg = PrintKitConfig(scale=0.5, filament=filament)
    assembled = build_kit_rotating_assembly_parts(cfg, crank_angle_deg=90.0)
    max_ov, hits = find_pairwise_collisions({k: v.part for k, v in assembled.items()}, vol_tol_mm3=1.0e-3)
    assert not hits, f"Collisions for {filament}: {hits}"
    assert max_ov <= 1.0e-3


def test_running_clearances_sweep_envelope_and_keying_poka_yoke() -> None:
  """Verify positive running clearances, sweep bounds, snap catch shelf, and keyed indexing."""
  s = 0.5
  cfg = PrintKitConfig(scale=s, filament="PETG")
  r_run = running_clearance_radial_mm(cfg)
  assert r_run > 0.0

  assembled = build_kit_rotating_assembly_parts(cfg, crank_angle_deg=0.0)
  z_crank = CRANK_AXIS_Z_MM * s
  z_cp = (-159.0 + 27.0) * s
  z_wp = -24.0 * s

  # 1. Piston radial running clearance inside nominal 68*s cylinder bore
  piston_bb = assembled["kit_piston"].part.bounding_box()
  assert piston_bb.max.X == pytest.approx((34.0 * s) - r_run, abs=1.0e-3)
  assert piston_bb.max.Y == pytest.approx((34.0 * s) - r_run, abs=1.0e-3)

  # 2. Positive annular clearance gaps on crankpin/conrod, wrist-pin/conrod, and main journals/bushings
  cp_gap = _annular_probe_y(14.0 * s + 0.05, 14.0 * s + r_run - 0.05, -3.5, 3.5, x=0.0, z=z_cp)
  assert overlap_volume_mm3(cp_gap, assembled["kit_crankpin"].part) == pytest.approx(0.0, abs=1.0e-5)
  assert overlap_volume_mm3(cp_gap, assembled["kit_conrod_beam"].part) == pytest.approx(0.0, abs=1.0e-5)
  assert overlap_volume_mm3(cp_gap, assembled["kit_conrod_cap"].part) == pytest.approx(0.0, abs=1.0e-5)

  wp_gap = _annular_probe_y(8.0 * s + 0.05, 8.0 * s + r_run - 0.05, -3.0, 3.0, x=0.0, z=z_wp)
  assert overlap_volume_mm3(wp_gap, assembled["kit_wrist_pin"].part) == pytest.approx(0.0, abs=1.0e-5)
  assert overlap_volume_mm3(wp_gap, assembled["kit_conrod_beam"].part) == pytest.approx(0.0, abs=1.0e-5)

  for side, y_lo, y_hi in (
      ("kit_main_bushing_left", -20.5, -14.0),
      ("kit_main_bushing_right", 14.0, 20.5),
  ):
    main_gap = _annular_probe_y(15.0 * s + 0.05, 15.0 * s + r_run - 0.05, y_lo, y_hi, x=0.0, z=z_crank)
    assert overlap_volume_mm3(main_gap, assembled[side].part) == pytest.approx(0.0, abs=1.0e-5)
    crank_half = "kit_crank_left" if "left" in side else "kit_crank_right"
    assert overlap_volume_mm3(main_gap, assembled[crank_half].part) == pytest.approx(0.0, abs=1.0e-5)

  # 3. Conrod cap + snap arm sweep radius clears 67*s crankcase sweep and <= 30*(s/0.5)
  cap_bb = assembled["kit_conrod_cap"].part.bounding_box()
  max_dx = max(abs(cap_bb.min.X), abs(cap_bb.max.X))
  max_dz = max(abs(cap_bb.min.Z - z_cp), abs(cap_bb.max.Z - z_cp))
  sweep_r_est = (max_dx ** 2 + max_dz ** 2) ** 0.5
  assert sweep_r_est <= 30.0 * (s / 0.5)
  assert (27.0 * s) + max_dx < 67.0 * s

  # 4. Conrod beam retains a real mechanical snap catch lip for the cap hook
  g = _conrod_geometry_params(cfg)
  snap_design = g["snap_design"]
  l_mm = float(g["l_mm"])
  ear_z_out = float(g["ear_z_out"])
  y_root_off = max(2.2, 4.5 * s)
  with BuildPart() as probe:
    with Locations((
        snap_design.tip_thickness_mm + (0.75 * snap_design.undercut_mm),  # type: ignore[union-attr]
        0.0,
        0.85 * snap_design.beam_length_mm,  # type: ignore[union-attr]
    )):
      Box(0.2 * snap_design.undercut_mm, 1.0, 0.5)  # type: ignore[union-attr]
  placed_probe = (
      _place_conrod_snap_shape(probe.part, 1.0, l_mm, ear_z_out, y_root_off)
      .rotate(Axis.Z, 90.0)
      .rotate(Axis.Y, -90.0)
      .translate((0.0, 0.0, -24.0 * s))
  )
  assert overlap_volume_mm3(placed_probe, assembled["kit_conrod_beam"].part) > 0.0

  # 5. Poka-yoke D-keyed indexing: 180-deg reversed crankpin/flywheel/hand-crank MUST interfere
  wrong_pin = assembled["kit_crankpin"].part.translate((0.0, 0.0, -z_cp)).rotate(Axis.Y, 180.0).translate((0.0, 0.0, z_cp))
  assert overlap_volume_mm3(wrong_pin, assembled["kit_crank_left"].part) > 1.0

  wrong_fw = assembled["kit_flywheel"].part.translate((0.0, 0.0, -z_crank)).rotate(Axis.Y, 180.0).translate((0.0, 0.0, z_crank))
  assert overlap_volume_mm3(wrong_fw, assembled["kit_crank_left"].part) > 1.0

  wrong_hc = assembled["kit_hand_crank"].part.translate((0.0, 0.0, -z_crank)).rotate(Axis.Y, 180.0).translate((0.0, 0.0, z_crank))
  assert overlap_volume_mm3(wrong_hc, assembled["kit_crank_right"].part) > 1.0


def test_stl_export_watertightness_and_printability(tmp_path: Path) -> None:
  """Verify every rotating kit STL exports watertight, fits bed, and prints cleanly."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  parts = build_kit_rotating_parts(cfg, crank_angle_deg=0.0)

  for name, kp in parts.items():
    rep = analyze_part_printability(
        part=kp.part,
        part_name=name,
        stl_path=tmp_path / f"{name}.stl",
        cfg=cfg,
        quantity=kp.quantity,
        orient_first=True,
        preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
    )
    assert rep.watertight is True, f"{name} STL not watertight"
    assert rep.fits_print_volume is True, f"{name} exceeds print volume: {rep.oriented_dims_mm}"
    assert rep.triangle_count > 100
    assert rep.overhang_pct < 5.0, f"{name} overhang too high: {rep.overhang_pct:.2f}%"
    assert rep.estimated_mass_g > 0.0
