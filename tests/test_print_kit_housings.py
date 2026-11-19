"""B-Rep geometry, datum, cutaway, collision, and STL printability tests for kit_housings."""

from pathlib import Path

from build123d import Box, Cylinder
import pytest

from openengine.cad.assemblies._common import (
    find_cross_collisions,
    find_pairwise_collisions,
    overlap_volume_mm3,
)
from openengine.printing.analysis import analyze_part_printability
from openengine.printing.kit_housings import (
    CRANK_AXIS_Z_MM,
    IDLER_MODULE_STEPS,
    TIMING_CD_MM,
    TIMING_MODULE_DIVISOR,
    build_kit_housings_assembly_parts,
    build_kit_housings_parts,
)
from openengine.printing.kit_rotating import build_kit_rotating_assembly_parts
from openengine.printing.kit_valvetrain import build_kit_valvetrain_assembly_parts
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    min_fin_thickness_mm,
    min_printable_wall_mm,
    running_clearance_radial_mm,
    slip_clearance_radial_mm,
)

EXPECTED_HOUSING_KEYS: frozenset[str] = frozenset({
    "kit_crankcase_left",
    "kit_crankcase_right",
    "kit_sump_pan",
    "kit_cylinder_block",
    "kit_cylinder_head",
    "kit_cam_tower",
    "kit_cam_cover",
    "kit_right_timing_cover",
    "kit_fan_shroud",
    "kit_intake_carb_unit",
    "kit_spark_plug",
})


def test_housing_catalog_and_datums_05x() -> None:
  """Verify all 11 housing KitParts, exact 0.5x datums, bore clearance, and timing sockets."""
  cfg = PrintKitConfig(scale=0.5, filament="PETG")
  s = cfg.scale
  parts = build_kit_housings_parts(cfg)
  assert set(parts.keys()) == EXPECTED_HOUSING_KEYS

  asm_parts = build_kit_housings_assembly_parts(cfg, crank_angle_deg=90.0)
  assert set(asm_parts.keys()) == EXPECTED_HOUSING_KEYS

  for name, kp in parts.items():
    assert isinstance(kp, KitPart)
    assert kp.name == name
    assert kp.quantity == 1
    assert kp.category == "housings"
    assert kp.preferred_rotation_xyz_deg is not None
    assert kp.part.is_valid
    assert len(kp.part.solids()) == 1, f"{name} must be a single connected solid"
    assert kp.part.volume > 0.0

  # Verify nominal master world coordinates at 0.5x
  bb_left = parts["kit_crankcase_left"].part.bounding_box()
  bb_right = parts["kit_crankcase_right"].part.bounding_box()
  assert bb_left.min.Z == pytest.approx(-241.0 * s, abs=0.05)
  assert bb_left.max.Z == pytest.approx(-85.0 * s, abs=0.05)
  assert bb_left.min.Y == pytest.approx(-52.0 * s, abs=0.05)
  assert bb_right.max.Y == pytest.approx(52.0 * s, abs=0.05)

  bb_sump = parts["kit_sump_pan"].part.bounding_box()
  assert bb_sump.min.Z == pytest.approx(-278.0 * s, abs=0.05)
  assert bb_sump.max.Z > -241.0 * s  # Raised locating lip + upward perimeter snaps

  bb_block = parts["kit_cylinder_block"].part.bounding_box()
  assert bb_block.max.Z == pytest.approx(0.0, abs=0.05)
  assert bb_block.min.Z < -105.0 * s  # Merged liner spigot down to -105*s + base snaps

  # Seamless cylinder bore must cleanly accommodate 34.0*s - r_run piston and reject > 34.0*s
  block_part = parts["kit_cylinder_block"].part
  bore_probe = Cylinder(radius=34.0 * s - 0.05, height=40.0 * s).translate((0.0, 0.0, -40.0 * s))
  assert overlap_volume_mm3(block_part, bore_probe) == 0.0

  # Confirm cylinder wall exists right outside 34.0*s nominal bore
  wall_probe = Cylinder(radius=34.5 * s, height=40.0 * s).translate((0.0, 0.0, -40.0 * s))
  assert overlap_volume_mm3(block_part, wall_probe) > 10.0

  # Confirm cam cover rear roof extends to >= 70*s and timing cover spans [-196*s, +106.5*s]
  bb_cover = parts["kit_cam_cover"].part.bounding_box()
  assert bb_cover.max.Z == pytest.approx(128.5 * s, abs=0.05)
  assert bb_cover.max.Y >= 70.0 * s - 1e-4

  bb_tcov = parts["kit_right_timing_cover"].part.bounding_box()
  assert bb_tcov.min.Z == pytest.approx(-196.0 * s, abs=0.05)
  assert bb_tcov.max.Z == pytest.approx(106.5 * s, abs=0.05)

  # Confirm timing cover rear wall behind idler sockets preserves >= min_printable_wall_mm
  w_min = min_printable_wall_mm(cfg, structural=True)
  assert bb_tcov.max.Y >= 68.0 * s + w_min - 1e-4

  # Confirm PTO clearance bore (20*s + 2*r_run) is open through timing cover back wall
  r_run = running_clearance_radial_mm(cfg)
  pto_probe = Cylinder(
      radius=10.0 * s + r_run - 0.05,
      height=20.0,
      rotation=(90.0, 0.0, 0.0),
  ).translate((0.0, 66.0 * s, CRANK_AXIS_Z_MM * s))
  assert overlap_volume_mm3(parts["kit_right_timing_cover"].part, pto_probe) == 0.0

  # Confirm 3 idler blind sockets are open at Y=48*s in right case / block / head
  r_slip = slip_clearance_radial_mm(cfg)
  idler_hosts = ("kit_crankcase_right", "kit_cylinder_block", "kit_cylinder_head")
  for idx, (host_name, step) in enumerate(zip(idler_hosts, IDLER_MODULE_STEPS)):
    z_idler = (CRANK_AXIS_Z_MM + (step / TIMING_MODULE_DIVISOR) * TIMING_CD_MM) * s
    pin_probe = Cylinder(
        radius=max(2.0, 5.0 * s) + r_slip - 0.05,
        height=3.5 * s,
        rotation=(90.0, 0.0, 0.0),
    ).translate((0.0, 46.2 * s, z_idler))
    assert overlap_volume_mm3(parts[host_name].part, pin_probe) == 0.0, f"Idler-{idx + 1} socket blocked in {host_name}"


@pytest.mark.parametrize("scale", [0.4, 0.5, 1.0])
@pytest.mark.parametrize("cutaway", [False, True])
def test_zero_housing_collisions_across_scales_and_cutaways(scale: float, cutaway: bool) -> None:
  """Verify single-solid validity and zero static intersections across scales & cutaway modes."""
  cfg = PrintKitConfig(scale=scale, cutaway=cutaway)
  parts = build_kit_housings_parts(cfg)
  raw = {name: kp.part for name, kp in parts.items()}

  for name, part in raw.items():
    assert part.is_valid, f"{name} invalid at scale={scale}, cutaway={cutaway}"
    assert len(part.solids()) == 1, f"{name} disconnected at scale={scale}, cutaway={cutaway}"

  max_ov, hits = find_pairwise_collisions(raw)
  assert max_ov <= 1e-2, f"Collision at scale={scale}, cutaway={cutaway}: {hits}"
  assert hits == []


def test_cutaway_viewing_windows_open_and_preserve_structure() -> None:
  """Verify cutaway=True removes window material while preserving structural bridges & snaps."""
  cfg_solid = PrintKitConfig(scale=0.5, cutaway=False)
  cfg_cut = PrintKitConfig(scale=0.5, cutaway=True)
  solid_parts = build_kit_housings_parts(cfg_solid)
  cut_parts = build_kit_housings_parts(cfg_cut)
  s = 0.5

  for key in (
      "kit_crankcase_left",
      "kit_crankcase_right",
      "kit_cylinder_block",
      "kit_cylinder_head",
      "kit_right_timing_cover",
  ):
    assert cut_parts[key].part.volume < solid_parts[key].part.volume, f"{key} unchanged in cutaway"

  # Unmodified parts remain identical in volume
  for key in ("kit_sump_pan", "kit_cam_tower", "kit_cam_cover", "kit_fan_shroud", "kit_intake_carb_unit"):
    assert cut_parts[key].part.volume == pytest.approx(solid_parts[key].part.volume, rel=1e-5)

  # Center spark plug bridge (X in [-10*s, +10*s]) in head remains solid around cutaway Z=[10*s, 44*s]
  bridge_probe = Box(8.0 * s, 10.0 * s, 10.0 * s).translate((0.0, -45.0 * s, 38.0 * s))
  assert overlap_volume_mm3(cut_parts["kit_cylinder_head"].part, bridge_probe) > 10.0

  # Twin -Y head valve windows are open at X = +-27*s, Y = -45*s, Z = 27*s
  for sx in (-27.0 * s, 27.0 * s):
    win_probe = Box(10.0 * s, 10.0 * s, 10.0 * s).translate((sx, -45.0 * s, 27.0 * s))
    assert overlap_volume_mm3(cut_parts["kit_cylinder_head"].part, win_probe) == 0.0

  # Timing cover viewing slot is open at (0, 67*s, -20*s) while PTO boss at Z=-159*s stays intact
  slot_probe = Box(20.0 * s, 6.0 * s, 20.0 * s).translate((0.0, 67.0 * s, -20.0 * s))
  assert overlap_volume_mm3(cut_parts["kit_right_timing_cover"].part, slot_probe) == 0.0


def test_fin_clamping_and_pla_filament_compatibility() -> None:
  """Verify fin thickness/pitch clamping at 0.4x and clean assembly in brittle PLA."""
  cfg_04 = PrintKitConfig(scale=0.4, filament="PETG")
  fin_min = min_fin_thickness_mm(cfg_04)
  assert fin_min >= 1.0
  # Probe fin outer tips along X=+49.5*0.4 (+19.8 mm, outside R_barrel=17.6 mm) in cylinder block
  block_04 = build_kit_housings_parts(cfg_04)["kit_cylinder_block"].part
  fin_probe_z = [
      z_c
      for idx in range(1, 30)
      if (z_c := -4.0 - idx * 0.5) > -30.0
      and overlap_volume_mm3(block_04, Box(2.0, 2.0, 0.2).translate((19.5, 0.0, z_c))) > 0.0
  ]
  assert len(fin_probe_z) >= 4

  cfg_pla = PrintKitConfig(scale=0.5, filament="PLA")
  pla_parts = build_kit_housings_parts(cfg_pla)
  pla_raw = {k: kp.part for k, kp in pla_parts.items()}
  for k, p in pla_raw.items():
    assert len(p.solids()) == 1, f"PLA {k} disconnected"
  _, hits = find_pairwise_collisions(pla_raw)
  assert hits == []


@pytest.mark.parametrize("scale", [0.4, 0.5])
def test_cross_module_clearances_rotating_and_valvetrain(scale: float) -> None:
  """Verify housings clear moving rotating & valvetrain assemblies at TDC (0) and BDC (180)."""
  cfg = PrintKitConfig(scale=scale, cutaway=False)
  housings = {k: kp.part for k, kp in build_kit_housings_parts(cfg).items()}

  for ca in (0.0, 180.0):
    rotating = {k: kp.part for k, kp in build_kit_rotating_assembly_parts(cfg, crank_angle_deg=ca).items()}
    valvetrain = {k: kp.part for k, kp in build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=ca).items()}

    max_hr, hits_hr = find_cross_collisions(housings, rotating)
    assert hits_hr == [], f"Housings vs rotating collision at scale={scale}, CA={ca}: {hits_hr} (max={max_hr})"

    max_hv, hits_hv = find_cross_collisions(housings, valvetrain)
    assert hits_hv == [], f"Housings vs valvetrain collision at scale={scale}, CA={ca}: {hits_hv} (max={max_hv})"


def test_stl_export_watertight_and_bed_fit_05x(tmp_path: Path) -> None:
  """Verify all 11 housing STLs at 0.5x are closed-manifold watertight and fit 256x256x256 bed."""
  cfg = PrintKitConfig(scale=0.5, cutaway=False)
  parts = build_kit_housings_parts(cfg)

  for name, kp in parts.items():
    rep = analyze_part_printability(
        kp.part,
        part_name=name,
        stl_path=tmp_path / f"{name}.stl",
        cfg=cfg,
        quantity=kp.quantity,
        orient_first=True,
        preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
    )
    assert rep.watertight is True, f"{name} STL is not watertight"
    assert rep.fits_bed is True, f"{name} exceeds printer bed: {rep.oriented_dims_mm}"
    assert rep.triangle_count > 100, f"{name} has too few triangles: {rep.triangle_count}"
