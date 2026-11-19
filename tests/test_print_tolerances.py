"""Unit and B-Rep geometry tests for FDM filament profiles, tolerances, and snap-fit primitives."""

from collections import Counter
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import struct

from build123d import Align, Box, BuildPart, Locations, Part, export_stl
import pytest

from openengine.printing import KitPart
from openengine.printing.materials import (
    DEFAULT_FILAMENT_NAME,
    FILAMENT_PROFILES,
    FilamentProfile,
    get_filament_profile,
)
from openengine.printing.primitives import (
    make_alignment_hole_cutter,
    make_alignment_pin,
    make_cantilever_snap_hook,
    make_compliant_wave_spring,
    make_keyed_bore_cutter,
    make_keyed_shaft_solid,
    make_snap_pocket_cutter,
    make_teardrop_bore_cutter,
)
from openengine.printing.tolerances import (
    KitPart as TolerancesKitPart,
    PrintKitConfig,
    SnapFitDesign,
    auto_size_cantilever_snap,
    design_cantilever_snap,
    min_fin_thickness_mm,
    min_printable_wall_mm,
    press_interference_radial_mm,
    running_clearance_radial_mm,
    shrinkage_compensation_xyz,
    slip_clearance_radial_mm,
    snap_clearance_mm,
)


def _shape_volume(shape: object) -> float:
  """Compute total solid volume across None, Solid, Compound, or ShapeList results."""
  if shape is None:
    return 0.0
  if hasattr(shape, "volume"):
    return float(shape.volume)  # type: ignore[attr-defined]
  return float(sum(item.volume for item in shape))  # type: ignore[attr-defined]


def _assert_watertight_part(part: Part, tmp_path: Path, name: str = "part.stl") -> int:
  """Export a Part to binary STL and verify closed-manifold watertightness."""
  assert isinstance(part, Part)
  assert part.is_valid
  assert len(part.solids()) == 1
  assert part.volume > 0.0

  stl_path = tmp_path / name
  export_stl(part, str(stl_path), tolerance=0.05, angular_tolerance=0.2)
  data = stl_path.read_bytes()
  assert len(data) >= 84
  tri_count = struct.unpack("<I", data[80:84])[0]
  assert tri_count > 0
  assert len(data) == 84 + (50 * tri_count)

  edge_counts: Counter[tuple[tuple[float, float, float], tuple[float, float, float]]] = Counter()
  for idx in range(tri_count):
    offset = 84 + (50 * idx) + 12
    coords = struct.unpack("<9f", data[offset : offset + 36])
    verts = (
        (round(coords[0], 4), round(coords[1], 4), round(coords[2], 4)),
        (round(coords[3], 4), round(coords[4], 4), round(coords[5], 4)),
        (round(coords[6], 4), round(coords[7], 4), round(coords[8], 4)),
    )
    for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
      if a != b:
        edge = (a, b) if a < b else (b, a)
        edge_counts[edge] += 1

  open_or_nonmanifold = [count for count in edge_counts.values() if count != 2]
  assert not open_or_nonmanifold, f"Found {len(open_or_nonmanifold)} non-manifold/open STL edges"
  return tri_count


def test_filament_registry_and_aliases() -> None:
  """Verify PETG primary default, all 7 required filaments, immutability, and aliases."""
  keys = list(FILAMENT_PROFILES.keys())
  assert keys[0] == DEFAULT_FILAMENT_NAME == "PETG"
  expected = {"PETG", "PLA", "PLA_PLUS", "PETG_CF", "ASA", "ABS", "TPU_95A"}
  assert set(keys) == expected

  petg = get_filament_profile("petg")
  assert isinstance(petg, FilamentProfile)
  assert petg.name == "PETG"
  assert petg.allowable_snap_strain > get_filament_profile("PLA").allowable_snap_strain
  assert get_filament_profile("PETG_CF").flexural_modulus_mpa > petg.flexural_modulus_mpa
  assert get_filament_profile("ABS").shrinkage_xy > get_filament_profile("PLA").shrinkage_xy
  assert get_filament_profile("TPU").name == "TPU_95A"

  assert get_filament_profile("PLA+").name == "PLA_PLUS"
  assert get_filament_profile("pla-plus").name == "PLA_PLUS"
  assert get_filament_profile("  petg cf  ").name == "PETG_CF"
  assert get_filament_profile("tpu-95a").name == "TPU_95A"

  with pytest.raises(FrozenInstanceError):
    petg.nozzle_temp_c = 250.0  # type: ignore[misc]

  with pytest.raises(KeyError):
    get_filament_profile("UNOBTAINIUM")


def test_print_kit_config_validation_and_scale_invariance() -> None:
  """Verify config bounds and that FDM walls/clearances do not shrink to zero at small scales."""
  cfg_default = PrintKitConfig()
  assert cfg_default.scale == 0.5
  assert cfg_default.filament == "PETG"
  assert cfg_default.nozzle_diameter_mm == 0.4
  assert cfg_default.layer_height_mm == 0.2
  assert cfg_default.print_volume_mm == (250.0, 220.0, 270.0)
  assert cfg_default.wall_loops == 3
  assert cfg_default.top_bottom_layers == 4
  assert cfg_default.volumetric_flow_mm3_s == 8.0
  assert cfg_default.support_mass_overhead_pct == 15.0
  assert cfg_default.brim_width_mm == 5.0
  assert cfg_default.mesh_quality == "ultra"
  assert cfg_default.mesh_deflections == (0.005, 0.03)
  assert cfg_default.compensate_shrinkage is True
  assert cfg_default.include_coupon is True

  with pytest.raises(ValueError):
    PrintKitConfig(scale=0.1)
  with pytest.raises(ValueError):
    PrintKitConfig(scale=2.5)
  with pytest.raises(ValueError):
    PrintKitConfig(layer_height_mm=0.5, nozzle_diameter_mm=0.4)
  with pytest.raises(ValueError):
    PrintKitConfig(wall_loops=0)
  with pytest.raises(ValueError):
    PrintKitConfig(top_bottom_layers=0)
  with pytest.raises(ValueError):
    PrintKitConfig(volumetric_flow_mm3_s=0.0)
  with pytest.raises(ValueError):
    PrintKitConfig(support_mass_overhead_pct=-5.0)
  with pytest.raises(ValueError):
    PrintKitConfig(brim_width_mm=-1.0)
  with pytest.raises(ValueError):
    PrintKitConfig(mesh_quality="potato")
  with pytest.raises(KeyError):
    PrintKitConfig(filament="UNKNOWN")

  cfg_quarter = PrintKitConfig(scale=0.25)
  cfg_full = PrintKitConfig(scale=1.0)
  cfg_double = PrintKitConfig(scale=2.0)

  assert min_printable_wall_mm(cfg_quarter) == pytest.approx(1.2)
  assert min_printable_wall_mm(cfg_quarter) == min_printable_wall_mm(cfg_full)
  assert min_printable_wall_mm(cfg_double, structural=False) == pytest.approx(0.8)
  assert min_fin_thickness_mm(cfg_quarter) == pytest.approx(1.0)

  assert running_clearance_radial_mm(cfg_quarter) == pytest.approx(0.30)
  assert slip_clearance_radial_mm(cfg_quarter) == pytest.approx(0.20)
  assert snap_clearance_mm(cfg_quarter) == pytest.approx(0.25)
  assert press_interference_radial_mm(cfg_quarter) == pytest.approx(0.05)
  assert running_clearance_radial_mm(cfg_quarter) > slip_clearance_radial_mm(cfg_quarter) > 0.0

  cfg_coarse = PrintKitConfig(nozzle_diameter_mm=0.6, layer_height_mm=0.28, wall_loops=4, clearance_bias_mm=0.05)
  assert min_printable_wall_mm(cfg_coarse) == pytest.approx(2.4)
  assert running_clearance_radial_mm(cfg_coarse) > running_clearance_radial_mm(cfg_default)
  assert press_interference_radial_mm(cfg_coarse) < press_interference_radial_mm(cfg_default)

  sx, sy, sz = shrinkage_compensation_xyz(get_filament_profile("ABS"))
  assert sx == pytest.approx(sy)
  assert sx > sz > 1.0
  assert sx == pytest.approx(1.0 / (1.0 - 0.007))


def test_cantilever_snap_mechanics_and_auto_sizing() -> None:
  """Verify closed-form beam equations and filament-adaptive snap-fit auto-sizing."""
  cfg_petg = PrintKitConfig(filament="PETG")
  # Uniform rectangular beam check against textbook Euler-Bernoulli formula
  L, b, h, y = 15.0, 6.0, 1.5, 1.0
  snap_uniform = design_cantilever_snap(
      cfg_petg,
      target_undercut_mm=y,
      beam_length_mm=L,
      root_thickness_mm=h,
      beam_width_mm=b,
      taper_ratio=1.0,
      stress_concentration_factor=1.0,
      lead_angle_deg=30.0,
  )
  assert isinstance(snap_uniform, SnapFitDesign)
  e_mpa = cfg_petg.filament_profile.flexural_modulus_mpa
  mu = cfg_petg.filament_profile.sliding_friction_coeff
  expected_p = (e_mpa * b * (h ** 3) * y) / (4.0 * (L ** 3))
  expected_eps = (3.0 * h * y) / (2.0 * (L ** 2))
  tan30 = math.tan(math.radians(30.0))
  expected_w = expected_p * (mu + tan30) / (1.0 - (mu * tan30))

  assert snap_uniform.deflection_force_n == pytest.approx(expected_p, rel=1e-9)
  assert snap_uniform.nominal_strain == pytest.approx(expected_eps, rel=1e-9)
  assert snap_uniform.max_strain == pytest.approx(expected_eps, rel=1e-9)
  assert snap_uniform.mating_force_n == pytest.approx(expected_w, rel=1e-9)
  assert snap_uniform.safety_factor == pytest.approx(0.045 / expected_eps, rel=1e-9)

  # Auto-sizing across brittle vs ductile filaments
  petg_auto = auto_size_cantilever_snap(cfg_petg, available_length_mm=12.0, width_mm=6.0)
  pla_auto = auto_size_cantilever_snap(PrintKitConfig(filament="PLA"), available_length_mm=12.0, width_mm=6.0)
  cf_auto = auto_size_cantilever_snap(PrintKitConfig(filament="PETG_CF"), available_length_mm=12.0, width_mm=6.0)

  assert petg_auto.is_safe and pla_auto.is_safe and cf_auto.is_safe
  assert petg_auto.safety_factor >= 2.0
  assert pla_auto.safety_factor >= 1.45
  assert cf_auto.safety_factor >= 1.45
  assert petg_auto.deflection_force_n >= 5.0
  assert pla_auto.root_thickness_mm < petg_auto.root_thickness_mm
  assert cf_auto.root_thickness_mm < petg_auto.root_thickness_mm


def test_snap_hook_and_pocket_cutter_geometry(tmp_path: Path) -> None:
  """Verify snap hook local frame, watertightness, and non-binding pocket clearance."""
  cfg = PrintKitConfig(filament="PETG")
  design = auto_size_cantilever_snap(cfg, available_length_mm=12.0, width_mm=6.0)

  hook = make_cantilever_snap_hook(design)
  _assert_watertight_part(hook, tmp_path, "snap_hook.stl")
  hbox = hook.bounding_box()
  rf = min(design.root_fillet_mm, 0.4 * design.beam_length_mm)
  expected_max_x = max(design.root_thickness_mm + rf, design.tip_thickness_mm + design.undercut_mm)
  assert hbox.min.X == pytest.approx(0.0, abs=1e-5)
  assert hbox.min.Z == pytest.approx(0.0, abs=1e-5)
  assert hbox.min.Y == pytest.approx(-3.0, abs=1e-5)
  assert hbox.max.Y == pytest.approx(3.0, abs=1e-5)
  assert hbox.max.X == pytest.approx(expected_max_x, abs=1e-5)

  pocket = make_snap_pocket_cutter(design, design.clearance_mm)
  _assert_watertight_part(pocket, tmp_path, "snap_pocket.stl")

  # Receiver block spanning distal engagement region [0.5*L, z_nose + 3]
  with BuildPart() as blk_bp:
    with Locations((0.5 * (design.root_thickness_mm + design.undercut_mm), 0.0, 0.5 * hbox.max.Z + 2.0)):
      Box(
          design.root_thickness_mm + (3.0 * design.undercut_mm) + 4.0,
          design.beam_width_mm + 6.0,
          hbox.max.Z + 4.0,
      )
  receiver = blk_bp.part.cut(pocket)
  overlap = hook.intersect(receiver)
  assert _shape_volume(overlap) == pytest.approx(0.0, abs=1e-5)

  # Confirm the receiver retains a real latching lip below z = beam_length_mm inside the barb radial span
  with BuildPart() as probe_bp:
    with Locations((design.tip_thickness_mm + 0.8 * design.undercut_mm, 0.0, 0.85 * design.beam_length_mm)):
      Box(0.2 * design.undercut_mm, 1.0, 0.5)
  lip_hit = probe_bp.part.intersect(receiver)
  assert _shape_volume(lip_hit) > 0.0


def test_alignment_teardrop_and_keyed_primitives(tmp_path: Path) -> None:
  """Verify alignment pin/hole, support-free 45-deg teardrop bore, and D-shaft coupling."""
  cfg = PrintKitConfig(filament="PETG")

  pin = make_alignment_pin(radius_mm=3.0, length_mm=10.0, chamfer_mm=0.5)
  _assert_watertight_part(pin, tmp_path, "align_pin.stl")
  pbox = pin.bounding_box()
  assert pbox.min.Z == pytest.approx(0.0, abs=1e-5)
  assert pbox.max.Z == pytest.approx(10.0, abs=1e-5)

  slip = slip_clearance_radial_mm(cfg)
  hole = make_alignment_hole_cutter(radius_mm=3.0, depth_mm=10.5, clearance_mm=slip)
  _assert_watertight_part(hole, tmp_path, "align_hole.stl")
  with BuildPart() as plate_bp:
    Box(20.0, 20.0, 12.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
  plate_with_hole = plate_bp.part.cut(hole)
  pin_ov = pin.intersect(plate_with_hole)
  assert _shape_volume(pin_ov) == pytest.approx(0.0, abs=1e-5)

  # Horizontal teardrop bores along Y and X
  r_td, len_td = 5.0, 20.0
  expected_td_vol = ((0.75 * math.pi) + 1.0) * (r_td ** 2) * len_td
  for axis_name in ("Y", "X"):
    td = make_teardrop_bore_cutter(radius_mm=r_td, length_mm=len_td, axis=axis_name)
    _assert_watertight_part(td, tmp_path, f"teardrop_{axis_name}.stl")
    tbox = td.bounding_box()
    assert tbox.min.Z == pytest.approx(-r_td, abs=1e-4)
    assert tbox.max.Z == pytest.approx(r_td * math.sqrt(2.0), abs=1e-4)
    assert td.volume == pytest.approx(expected_td_vol, rel=1e-4)

  with pytest.raises(ValueError):
    make_teardrop_bore_cutter(radius_mm=4.0, length_mm=10.0, axis="Z")
  with pytest.raises(ValueError):
    make_keyed_shaft_solid(radius_mm=3.0, length_mm=10.0, flat_depth_mm=3.5)

  # D-shaft and mating D-bore clearance
  shaft = make_keyed_shaft_solid(radius_mm=5.0, length_mm=16.0, flat_depth_mm=1.2)
  bore = make_keyed_bore_cutter(radius_mm=5.0, length_mm=16.0, flat_depth_mm=1.2, clearance_mm=slip)
  _assert_watertight_part(shaft, tmp_path, "keyed_shaft.stl")
  _assert_watertight_part(bore, tmp_path, "keyed_bore.stl")
  sbox = shaft.bounding_box()
  bbox_bore = bore.bounding_box()
  assert sbox.max.X == pytest.approx(5.0 - 1.2, abs=1e-5)
  assert bbox_bore.max.X == pytest.approx(5.0 - 1.2 + slip, abs=1e-5)
  rem = shaft.cut(bore)
  assert _shape_volume(rem) == pytest.approx(0.0, abs=1e-5)


def test_compliant_wave_spring_printable_and_watertight(tmp_path: Path) -> None:
  """Verify compliant wave spring is flat-seated, single-solid, and STL-watertight."""
  spring = make_compliant_wave_spring(
      outer_radius_mm=8.9,
      inner_radius_mm=5.8,
      free_height_mm=14.0,
      wire_radial_mm=1.6,
      waves=3,
      turns=2,
  )
  _assert_watertight_part(spring, tmp_path, "wave_spring.stl")
  sbox = spring.bounding_box()
  assert sbox.min.Z == pytest.approx(0.0, abs=1e-4)
  assert sbox.max.Z == pytest.approx(14.0, abs=1e-4)
  assert max(abs(sbox.min.X), abs(sbox.max.X)) <= 8.9 + 1e-4

  # Also test 0.5-scaled compact valve-spring envelope
  compact = make_compliant_wave_spring(
      outer_radius_mm=4.45,
      inner_radius_mm=2.9,
      free_height_mm=12.0,
      wire_radial_mm=1.0,
      waves=3,
      turns=2,
  )
  _assert_watertight_part(compact, tmp_path, "wave_spring_compact.stl")


def test_kit_part_dataclass() -> None:
  """Verify KitPart metadata dataclass exports, defaults, validation, and immutability."""
  assert KitPart is TolerancesKitPart
  pin = make_alignment_pin(radius_mm=2.0, length_mm=8.0)
  kp = KitPart(name="align_pin", part=pin, quantity=2, category="fasteners")
  assert kp.name == "align_pin"
  assert kp.quantity == 2
  assert kp.category == "fasteners"
  assert kp.preferred_rotation_xyz_deg is None
  assert kp.notes == ""

  with pytest.raises(FrozenInstanceError):
    kp.quantity = 3  # type: ignore[misc]
  with pytest.raises(ValueError):
    KitPart(name="bad_qty", part=pin, quantity=0)
  with pytest.raises(ValueError):
    KitPart(name="   ", part=pin)
