"""Tests for 3D-printable SOHC valvetrain and 5-gear 2:1 timing drive kit."""

import math
from pathlib import Path

from build123d import Axis, Part
import pytest

from openengine.cad.assemblies._common import find_pairwise_collisions, overlap_volume_mm3
from openengine.cad.assembly import EngineAssembly
from openengine.printing.analysis import analyze_part_printability
from openengine.printing.kit_valvetrain import (
    CAM_AXIS_Z_MM,
    CRANK_AXIS_Z_MM,
    IDLER_K_STEPS,
    TIMING_CENTER_DISTANCE_MM,
    TIMING_MODULE_DIVISOR,
    build_kit_valvetrain_assembly_parts,
    build_kit_valvetrain_parts,
)
from openengine.printing.primitives import make_keyed_shaft_solid
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    min_printable_wall_mm,
    running_clearance_radial_mm,
)

EXPECTED_BOM_QUANTITIES: dict[str, int] = {
    "kit_timing_gear_crank_14t": 1,
    "kit_timing_gear_idler_24t": 3,
    "kit_timing_gear_cam_28t": 1,
    "kit_timing_idler_pin": 3,
    "kit_camshaft_core": 1,
    "kit_cam_lobes": 1,
    "kit_rocker_shaft": 2,
    "kit_rocker_arm": 2,
    "kit_valve_intake": 1,
    "kit_valve_exhaust": 1,
    "kit_valve_spring": 2,
    "kit_spring_retainer": 2,
}


def test_bom_and_assembly_part_inventory() -> None:
  """Verify BOM dict keys, quantities, single-solid validity, and 20-instance expansion."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  bom = build_kit_valvetrain_parts(cfg, crank_angle_deg=0.0)
  assert set(bom.keys()) == set(EXPECTED_BOM_QUANTITIES.keys())

  for name, expected_qty in EXPECTED_BOM_QUANTITIES.items():
    kp = bom[name]
    assert isinstance(kp, KitPart)
    assert kp.name == name
    assert kp.quantity == expected_qty
    assert isinstance(kp.part, Part)
    assert kp.part.is_valid
    assert len(kp.part.solids()) == 1
    assert kp.part.volume > 0.0

  asm = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=0.0)
  assert len(asm) == sum(EXPECTED_BOM_QUANTITIES.values()) == 20
  for name, kp in asm.items():
    assert kp.quantity == 1
    assert kp.part.is_valid
    assert len(kp.part.solids()) == 1


@pytest.mark.parametrize("scale", [0.4, 0.5, 1.0])
def test_five_gear_timing_drive_geometry_and_tunnel_clearance(scale: float) -> None:
  """Verify exact 2:1 gear centers, module, ODs, Y-tunnel clearance, and D-keyed coupling."""
  cfg = PrintKitConfig(scale=scale, filament="PETG")
  asm = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=0.0)

  cd = TIMING_CENTER_DISTANCE_MM * scale
  m = cd / TIMING_MODULE_DIVISOR
  expected_face_w = max(4.0, 11.2 * scale)
  expected_y_min = (56.0 * scale) - (0.5 * expected_face_w)
  expected_y_max = (56.0 * scale) + (0.5 * expected_face_w)

  # All 5 gears must fit inside widened [48*s, 64*s] timing tunnel with positive side clearance.
  gear_specs = [
      ("kit_timing_gear_crank_14t", CRANK_AXIS_Z_MM * scale, 14),
      ("kit_timing_gear_idler_24t_1", (CRANK_AXIS_Z_MM * scale) + (IDLER_K_STEPS[0] * m), 24),
      ("kit_timing_gear_idler_24t_2", (CRANK_AXIS_Z_MM * scale) + (IDLER_K_STEPS[1] * m), 24),
      ("kit_timing_gear_idler_24t_3", (CRANK_AXIS_Z_MM * scale) + (IDLER_K_STEPS[2] * m), 24),
      ("kit_timing_gear_cam_28t", CAM_AXIS_Z_MM * scale, 28),
  ]

  for key, z_center, teeth in gear_specs:
    bb = asm[key].part.bounding_box()
    assert bb.min.Y == pytest.approx(expected_y_min, abs=1e-4)
    assert bb.max.Y == pytest.approx(expected_y_max, abs=1e-4)
    assert bb.min.Y > 48.0 * scale + 0.5
    assert bb.max.Y < 64.0 * scale - 0.5

    # Check exact Z center and outer diameter (D_o ~ (N + 2) * m).
    z_mid = 0.5 * (bb.min.Z + bb.max.Z)
    expected_od = (teeth + 2.0) * m
    if teeth % 2 == 0 and (teeth // 2) % 2 == 0:
      assert z_mid == pytest.approx(z_center, abs=0.3 * m)
    else:
      # 14T has tooth at +Z and valley at -Z, so X width reflects true OD.
      assert abs(z_mid - z_center) < 1.5 * m
    x_span = bb.max.X - bb.min.X
    assert (expected_od - 0.8 * m) <= x_span <= (expected_od + 0.05)

  # Verify crank timing journal D-shaft fits inside 14T crank gear bore with slip clearance.
  r_crank_journal = 10.0 * scale
  flat_d = max(1.0, 2.0 * scale)
  crank_journal = (
      make_keyed_shaft_solid(r_crank_journal, expected_face_w + 2.0, flat_d)
      .rotate(Axis.X, -90.0)
      .translate((0.0, (56.0 * scale) - 0.5 * expected_face_w - 1.0, CRANK_AXIS_Z_MM * scale))
  )
  ov = overlap_volume_mm3(crank_journal, asm["kit_timing_gear_crank_14t"].part)
  assert ov == pytest.approx(0.0, abs=1e-5)

  # Verify rotated crank journal at 90 deg also stays keyed without overlapping rotated 14T gear.
  asm_90 = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=90.0)
  crank_journal_90 = crank_journal.rotate(
      Axis((0.0, 0.0, CRANK_AXIS_Z_MM * scale), (0.0, 1.0, 0.0)),
      90.0,
  )
  ov_90 = overlap_volume_mm3(crank_journal_90, asm_90["kit_timing_gear_crank_14t"].part)
  assert ov_90 == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("scale", [0.4, 0.5, 1.0])
def test_scale_clamped_valves_springs_retainers_and_rocker_bosses(scale: float) -> None:
  """Verify scale-clamped wall, stem, margin, spring pocket, retainer, and pin bounds."""
  cfg = PrintKitConfig(scale=scale, filament="PETG")
  parts = build_kit_valvetrain_parts(cfg, crank_angle_deg=0.0)
  r_run = running_clearance_radial_mm(cfg)
  min_wall = min_printable_wall_mm(cfg)

  # Valves: head OD and clamped stem OD.
  v_in_bb = parts["kit_valve_intake"].part.bounding_box()
  v_ex_bb = parts["kit_valve_exhaust"].part.bounding_box()
  assert (v_in_bb.max.Y - v_in_bb.min.Y) == pytest.approx(30.0 * scale, abs=0.05)
  assert (v_ex_bb.max.Y - v_ex_bb.min.Y) == pytest.approx(26.0 * scale, abs=0.05)

  # Wave spring must fit inside head pocket (OD <= 21*s) and clear stem (ID >= stem_d + 2*r_run + 1.6).
  spr_bb = parts["kit_valve_spring"].part.bounding_box()
  spr_od = spr_bb.max.Y - spr_bb.min.Y
  stem_d = max(3.5, 5.0 * scale)
  assert spr_od <= (21.0 * scale) + 0.05
  assert spr_od >= stem_d + (2.0 * r_run) + 1.6 + (2.0 * min_wall) - 0.1

  # Retainer flange OD matches spring and thickness >= max(1.4, 2.0*s).
  ret_bb = parts["kit_spring_retainer"].part.bounding_box()
  ret_od = ret_bb.max.Y - ret_bb.min.Y
  assert ret_od == pytest.approx(spr_od, abs=0.25)

  # Rocker shaft OD = max(2.6, 6.4*s), idler pin shoulder > shaft OD = max(4.0, 10.0*s).
  rshaft_bb = parts["kit_rocker_shaft"].part.bounding_box()
  rshaft_z = rshaft_bb.max.Z - rshaft_bb.min.Z
  assert rshaft_z >= max(2.6, 6.4 * scale) + 1.5  # includes front retaining flange

  pin_bb = parts["kit_timing_idler_pin"].part.bounding_box()
  pin_x = pin_bb.max.X - pin_bb.min.X
  assert pin_x > max(4.0, 10.0 * scale) + 0.8

  # Rocker arm boss around shaft bore must clear shaft cleanly.
  arm_bb = parts["kit_rocker_arm"].part.bounding_box()
  assert arm_bb.max.Z - (100.0 * scale) >= (0.5 * max(2.6, 6.4 * scale)) + r_run + min_wall - 0.1


def test_gear_fine_angle_mesh_zero_overlap() -> None:
  """Verify spur gear teeth mesh cleanly across fine fractional crank steps with zero overlap."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  for angle in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0):
    asm = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=angle)
    gear_parts = {
        k: v.part
        for k, v in asm.items()
        if k.startswith("kit_timing_gear_") or k.startswith("kit_timing_idler_pin_")
    }
    max_ov, hits = find_pairwise_collisions(gear_parts, vol_tol_mm3=1e-3)
    assert not hits, f"Gear/pin collision at CA={angle}: {hits} (max_ov={max_ov:.5f})"


@pytest.mark.parametrize("scale", [0.4, 0.5, 1.0])
@pytest.mark.parametrize("angle_deg", [0.0, 90.0, 180.0, 257.5, 270.0, 467.5])
def test_valvetrain_zero_collisions_across_angles_and_scales(scale: float, angle_deg: float) -> None:
  """Verify zero pairwise solid intersections across all 20 valvetrain parts, angles, and scales."""
  cfg = PrintKitConfig(scale=scale, filament="PETG")
  asm = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=angle_deg)
  solids_map = {name: kp.part for name, kp in asm.items()}

  max_ov, hits = find_pairwise_collisions(solids_map, vol_tol_mm3=1e-3)
  assert not hits, (
      f"Valvetrain collisions at scale={scale}, CA={angle_deg}: {hits} (max_ov={max_ov:.4f} mm^3)"
  )


def test_valve_kinematic_displacements_at_peak_lifts() -> None:
  """Verify intake and exhaust valves, rockers, and retainers track 4-stroke lift laws."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  asm_0 = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=0.0)
  asm_ex = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=257.5)
  asm_in = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=467.5)

  st_ex = EngineAssembly().solve_kinematics(257.5)
  st_in = EngineAssembly().solve_kinematics(467.5)
  cos15 = math.cos(math.radians(15.0))

  dz_ex_actual = asm_0["kit_valve_exhaust"].part.bounding_box().min.Z - asm_ex["kit_valve_exhaust"].part.bounding_box().min.Z
  dz_in_actual = asm_0["kit_valve_intake"].part.bounding_box().min.Z - asm_in["kit_valve_intake"].part.bounding_box().min.Z
  assert dz_ex_actual == pytest.approx(st_ex.exhaust_valve_lift_mm * 0.5 * cos15, abs=0.08)
  assert dz_in_actual == pytest.approx(st_in.intake_valve_lift_mm * 0.5 * cos15, abs=0.08)

  # Retainers move downward with their valves while spring top compresses accordingly.
  ret_dz_in = asm_0["kit_spring_retainer_intake"].part.bounding_box().min.Z - asm_in["kit_spring_retainer_intake"].part.bounding_box().min.Z
  assert ret_dz_in == pytest.approx(dz_in_actual, abs=0.08)


def test_all_valvetrain_parts_stl_watertight_and_bed_printable(tmp_path: Path) -> None:
  """Verify every BOM KitPart exports to a closed-manifold watertight STL fitting the build plate."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  bom = build_kit_valvetrain_parts(cfg, crank_angle_deg=0.0)

  no_support_parts = {
      "kit_timing_gear_crank_14t",
      "kit_timing_gear_idler_24t",
      "kit_timing_gear_cam_28t",
      "kit_timing_idler_pin",
      "kit_camshaft_core",
      "kit_rocker_shaft",
      "kit_valve_intake",
      "kit_valve_exhaust",
      "kit_spring_retainer",
  }

  for name, kp in bom.items():
    rep = analyze_part_printability(
        kp.part,
        part_name=name,
        stl_path=tmp_path / f"{name}.stl",
        cfg=cfg,
        quantity=kp.quantity,
        orient_first=True,
        preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
    )
    assert rep.watertight is True, f"{name} STL not watertight: {rep.warnings}"
    assert rep.fits_bed is True, f"{name} exceeds bed: {rep.oriented_dims_mm}"
    assert rep.triangle_count > 100
    assert rep.contact_area_mm2 > 0.0
    if name in no_support_parts:
      assert rep.support_recommendation == "none", (
          f"{name} expected support_recommendation='none', got {rep.support_recommendation} "
          f"(oh={rep.overhang_area_mm2:.2f} mm^2, {rep.overhang_pct:.2f}%)"
      )
