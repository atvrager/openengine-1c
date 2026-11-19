"""Full 3D-printable model kit assembly, collision/fitment audit, and export workflow.

Integrates ``kit_rotating``, ``kit_housings``, and ``kit_valvetrain`` into a unified
33-part (42-piece) snap-fit engine kit with multi-angle B-Rep collision verification,
process-aware clearance & cantilever snap-fit audits, oriented-space shrinkage
compensation, watertight binary STL export, JSON manifest, and Markdown guide.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from build123d import GeomType, Part
from OCP.BRepAdaptor import BRepAdaptor_Surface

from openengine.cad.assemblies._common import (
    find_cross_collisions,
    find_pairwise_collisions,
    overlap_volume_mm3,
)
from openengine.printing.analysis import (
    DEFAULT_ANGULAR_DEFLECTION_RAD,
    DEFAULT_LINEAR_DEFLECTION_MM,
    PartPrintReport,
    analyze_part_printability,
    export_part_stl,
)
from openengine.printing.coupon import (
    SNAP_AVAILABLE_LEN_MM,
    SNAP_BEAM_WIDTH_MM,
    generate_tolerance_coupon,
)
from openengine.printing.kit_housings import (
    IDLER_MODULE_STEPS,
    TIMING_CD_MM,
    TIMING_MODULE_DIVISOR,
    build_kit_housings_assembly_parts,
    build_kit_housings_parts,
)
from openengine.printing.kit_rotating import (
    CRANK_AXIS_Z_MM,
    CRANK_THROW_MM,
    build_kit_rotating_assembly_parts,
    build_kit_rotating_parts,
)
from openengine.printing.kit_valvetrain import (
    CAM_AXIS_Z_MM,
    ROCKER_PIVOT_X_1TO1_MM,
    ROCKER_PIVOT_Z_1TO1_MM,
    VALVE_IN_X_1TO1_MM,
    VALVE_SEAT_Z_1TO1_MM,
    VALVE_TILT_DEG,
    build_kit_valvetrain_assembly_parts,
    build_kit_valvetrain_parts,
)
from openengine.printing.materials import FilamentProfile
from openengine.printing.orientation import (
    OrientedPartResult,
    apply_shrinkage_compensation,
    bake_part_transform,
    orient_part_for_printing,
)
from openengine.printing.packing import (
    BuildPlateReport,
    PlatePackingResult,
    pack_print_kit_plates,
)
from openengine.printing.tolerances import (
    KitPart,
    PrintKitConfig,
    SnapFitDesign,
    auto_size_cantilever_snap,
    min_fin_thickness_mm,
    min_printable_wall_mm,
    press_interference_radial_mm,
    running_clearance_radial_mm,
    shrinkage_compensation_xyz,
    slip_clearance_radial_mm,
    snap_clearance_mm,
)

DEFAULT_COLLISION_VOL_TOL_MM3: float = 1.0e-3
DEFAULT_CRANK_ANGLES_DEG: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
CLEARANCE_MATCH_TOL_MM: float = 0.03
MIN_Z_RESEAT_TOL_MM: float = 1.0e-5
STL_FLOAT32_ULP_GUARD_MM: float = 5.0e-6

STATIONARY_ROTATING_KEYS: frozenset[str] = frozenset({
    "kit_main_bushing_left",
    "kit_main_bushing_right",
})
STATIONARY_VALVETRAIN_KEYS: frozenset[str] = frozenset({
    "kit_timing_idler_pin_1",
    "kit_timing_idler_pin_2",
    "kit_timing_idler_pin_3",
    "kit_rocker_shaft_intake",
    "kit_rocker_shaft_exhaust",
})


@dataclass(frozen=True, slots=True)
class FitmentCheckItem:
  """Audit record for a mating clearance joint or cantilever snap-fit connection."""

  joint_name: str
  part_a: str
  part_b: str
  fit_type: str
  target_clearance_mm: float
  measured_clearance_mm: float
  is_valid: bool
  notes: str


@dataclass(frozen=True, slots=True)
class KitFitmentVerificationResult:
  """Combined collision, running/slip clearance, and snap-fit verification output."""

  static_collisions: tuple[str, ...]
  kinematic_collisions: tuple[str, ...]
  fitment_checks: tuple[FitmentCheckItem, ...]
  snap_fit_designs: dict[str, SnapFitDesign]
  all_passed: bool


@dataclass(frozen=True, slots=True)
class PrintKitSummary:
  """Complete artifact, printability, collision, and fitment summary for a print kit run."""

  config: PrintKitConfig
  filament_profile: FilamentProfile
  total_unique_parts: int
  total_printed_pieces: int
  total_mass_g: float
  filament_mass_by_type_g: dict[str, float]
  total_print_time_min: float
  all_fit_print_volume: bool
  all_watertight: bool
  static_collisions: tuple[str, ...]
  kinematic_collisions: tuple[str, ...]
  fitment_checks: tuple[FitmentCheckItem, ...]
  snap_fit_designs: dict[str, SnapFitDesign]
  parts_reports: tuple[PartPrintReport, ...]
  coupon_report: Optional[PartPrintReport]
  stl_dir: Path
  manifest_path: Path
  guide_path: Path
  assembled_preview_stl_path: Optional[Path]
  plate_packing: Optional[PlatePackingResult] = None


def compute_filament_mass_by_type_g(
    parts_reports: Sequence[PartPrintReport],
    *,
    primary_filament: Optional[str] = None,
) -> dict[str, float]:
  """Aggregate total filament mass in grams separated by filament type across kit parts."""
  totals: dict[str, float] = {}
  for rep in parts_reports:
    fil = str(rep.filament_type)
    totals[fil] = totals.get(fil, 0.0) + (float(rep.estimated_mass_g) * int(rep.quantity))

  ordered_keys = sorted(
      totals.keys(),
      key=lambda name: (0 if (primary_filament is not None and name == primary_filament) else 1, name),
  )
  return {name: round(totals[name], 3) for name in ordered_keys}


def _merge_disjoint_parts(
    *groups: Mapping[str, KitPart],
) -> dict[str, KitPart]:
  """Merge part dictionaries while guarding against accidental duplicate keys."""
  merged: dict[str, KitPart] = {}
  for grp in groups:
    for key, kp in grp.items():
      if key in merged:
        raise ValueError(f"Duplicate KitPart key {key!r} encountered while assembling kit")
      merged[key] = kp
  return merged


def build_full_kit_parts(cfg: PrintKitConfig) -> dict[str, KitPart]:
  """Build all 33 canonical unique printable ``KitPart`` BOM items (42 pieces total)."""
  return _merge_disjoint_parts(
      build_kit_rotating_parts(cfg, crank_angle_deg=0.0),
      build_kit_housings_parts(cfg),
      build_kit_valvetrain_parts(cfg, crank_angle_deg=0.0),
  )


def build_full_kit_assembly_parts(
    cfg: PrintKitConfig,
    crank_angle_deg: float = 0.0,
) -> dict[str, KitPart]:
  """Build all 42 posed physical ``KitPart`` instances at ``crank_angle_deg``."""
  return _merge_disjoint_parts(
      build_kit_rotating_assembly_parts(cfg, crank_angle_deg=crank_angle_deg),
      build_kit_housings_assembly_parts(cfg, crank_angle_deg=crank_angle_deg),
      build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=crank_angle_deg),
  )


def _extract_cylindrical_radii(
    part: Part,
    axis_dir: tuple[float, float, float],
    axis_pt: tuple[float, float, float],
    *,
    pt_tol_mm: float = 0.15,
    dir_tol: float = 0.98,
    y_range_mm: Optional[tuple[float, float]] = None,
) -> list[float]:
  """Extract unique cylindrical face radii on ``part`` coaxial with ``(axis_pt, axis_dir)``."""
  ax_x, ax_y, ax_z = axis_dir
  px, py, pz = axis_pt
  radii: list[float] = []

  for face in part.faces().filter_by(GeomType.CYLINDER):
    cyl = BRepAdaptor_Surface(face.wrapped).Cylinder()
    d = cyl.Axis().Direction()
    if abs(d.X() * ax_x + d.Y() * ax_y + d.Z() * ax_z) < dir_tol:
      continue

    loc = cyl.Location()
    dx, dy, dz = loc.X() - px, loc.Y() - py, loc.Z() - pz
    proj = dx * ax_x + dy * ax_y + dz * ax_z
    perp_sq = max(0.0, (dx * dx) + (dy * dy) + (dz * dz) - (proj * proj))
    if math.sqrt(perp_sq) > pt_tol_mm:
      continue

    if y_range_mm is not None:
      fc = face.center()
      if not (y_range_mm[0] <= fc.Y <= y_range_mm[1]):
        continue
    radii.append(float(cyl.Radius()))

  return sorted(set(round(r, 6) for r in radii))


def _compute_kit_snap_designs(cfg: PrintKitConfig) -> dict[str, SnapFitDesign]:
  """Recompute exact ``SnapFitDesign`` parameters for every snap joint in the kit."""
  s = cfg.scale
  min_wall = min_printable_wall_mm(cfg, structural=True)
  conrod_snap_len = max(7.0, 14.0 * s)
  conrod_snap_width = min((19.0 * s) - (2.0 * min_wall), max(4.5, 10.0 * s))

  specs: dict[str, tuple[float, float, float]] = {
      "conrod_cap": (conrod_snap_len, conrod_snap_width, 75.0),
      "crankcase_split": (min(10.0, max(7.0, 16.0 * s)), min(6.0, 12.0 * s), 45.0),
      "sump_pan": (min(10.0, max(7.0, 16.0 * s)), min(6.0, 14.0 * s), 45.0),
      "block_to_case": (min(11.0, max(8.0, 20.0 * s)), min(6.0, 12.0 * s), 45.0),
      "head_to_block": (min(13.0, max(8.5, 22.0 * s)), min(6.0, 12.0 * s), 45.0),
      "tower_to_head": (min(12.0, max(7.5, 18.0 * s)), min(5.5, 11.0 * s), 45.0),
      "cover_to_tower": (min(13.0, max(8.5, 22.0 * s)), min(6.0, 12.0 * s), 45.0),
      "timing_cover": (min(12.0, max(8.0, 20.0 * s)), min(6.0, 12.0 * s), 45.0),
      "fan_shroud": (min(10.0, max(7.0, 16.0 * s)), min(5.5, 11.0 * s), 45.0),
      "intake_carb": (min(11.0, max(8.0, 16.0 * s)), min(4.0, 9.0 * s), 45.0),
      "calibration_coupon": (SNAP_AVAILABLE_LEN_MM, SNAP_BEAM_WIDTH_MM, 75.0),
  }
  return {
      joint: auto_size_cantilever_snap(
          cfg,
          available_length_mm=beam_len,
          width_mm=width,
          retention_angle_deg=ret_deg,
      )
      for joint, (beam_len, width, ret_deg) in specs.items()
  }


def _check_radial_joint(
    joint_name: str,
    part_a: str,
    part_b: str,
    fit_type: str,
    target_mm: float,
    inner_r: Optional[float],
    outer_r: Optional[float],
    overlap_mm3: float,
    notes_detail: str,
) -> FitmentCheckItem:
  """Validate positive B-Rep radial clearance and zero overlap for a cylindrical joint."""
  if inner_r is None or outer_r is None:
    return FitmentCheckItem(
        joint_name=joint_name,
        part_a=part_a,
        part_b=part_b,
        fit_type=fit_type,
        target_clearance_mm=round(target_mm, 4),
        measured_clearance_mm=0.0,
        is_valid=False,
        notes=f"Failed to resolve cylindrical B-Rep faces ({notes_detail})",
    )
  measured = outer_r - inner_r
  is_ok = (
      measured > 0.0
      and abs(measured - target_mm) <= CLEARANCE_MATCH_TOL_MM
      and overlap_mm3 <= DEFAULT_COLLISION_VOL_TOL_MM3
  )
  return FitmentCheckItem(
      joint_name=joint_name,
      part_a=part_a,
      part_b=part_b,
      fit_type=fit_type,
      target_clearance_mm=round(target_mm, 4),
      measured_clearance_mm=round(measured, 4),
      is_valid=is_ok,
      notes=f"r_shaft={inner_r:.3f} mm, r_bore={outer_r:.3f} mm, ov={overlap_mm3:.5f} mm^3; {notes_detail}",
  )


def _audit_kit_clearances_and_snaps(
    cfg: PrintKitConfig,
    asm_0: Mapping[str, KitPart],
) -> tuple[tuple[FitmentCheckItem, ...], dict[str, SnapFitDesign]]:
  """Audit all running/slip B-Rep joint clearances and cantilever snap safety factors."""
  s = cfg.scale
  r_run = running_clearance_radial_mm(cfg)
  r_slip = slip_clearance_radial_mm(cfg)

  def _part(name: str) -> Part:
    return asm_0[name].part

  def _ov(name_a: str, name_b: str) -> float:
    return overlap_volume_mm3(_part(name_a), _part(name_b))

  checks: list[FitmentCheckItem] = []

  # 1. Piston / Cylinder Block
  piston_bb = _part("kit_piston").bounding_box()
  piston_r = 0.5 * (float(piston_bb.max.X) - float(piston_bb.min.X))
  block_radii = _extract_cylindrical_radii(_part("kit_cylinder_block"), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
  checks.append(_check_radial_joint(
      "piston_to_cylinder_block",
      "kit_piston",
      "kit_cylinder_block",
      "running",
      r_run,
      piston_r,
      min(block_radii) if block_radii else None,
      _ov("kit_piston", "kit_cylinder_block"),
      "Piston OD vs cylinder block bore",
  ))

  # 2. Crankpin / Conrod Big End
  z_cp = (CRANK_AXIS_Z_MM + CRANK_THROW_MM) * s
  cp_radii = _extract_cylindrical_radii(
      _part("kit_crankpin"), (0.0, 1.0, 0.0), (0.0, 0.0, z_cp), y_range_mm=(-5.0 * s, 5.0 * s)
  )
  rod_big_radii = _extract_cylindrical_radii(_part("kit_conrod_beam"), (0.0, 1.0, 0.0), (0.0, 0.0, z_cp))
  checks.append(_check_radial_joint(
      "crankpin_to_conrod",
      "kit_crankpin",
      "kit_conrod_beam",
      "running",
      r_run,
      max(cp_radii) if cp_radii else None,
      min(rod_big_radii) if rod_big_radii else None,
      _ov("kit_crankpin", "kit_conrod_beam") + _ov("kit_crankpin", "kit_conrod_cap"),
      "Crankpin journal vs split conrod big-end bore",
  ))

  # 3. Wrist Pin / Conrod Small End
  z_wp = -24.0 * s
  wp_radii = _extract_cylindrical_radii(
      _part("kit_wrist_pin"), (0.0, 1.0, 0.0), (0.0, 0.0, z_wp), y_range_mm=(-5.0 * s, 5.0 * s)
  )
  rod_small_radii = _extract_cylindrical_radii(_part("kit_conrod_beam"), (0.0, 1.0, 0.0), (0.0, 0.0, z_wp))
  checks.append(_check_radial_joint(
      "wrist_pin_to_conrod",
      "kit_wrist_pin",
      "kit_conrod_beam",
      "running",
      r_run,
      max(wp_radii) if wp_radii else None,
      min(rod_small_radii) if rod_small_radii else None,
      _ov("kit_wrist_pin", "kit_conrod_beam"),
      "Wrist pin journal vs conrod small-end eye",
  ))

  # 4. Main Journal / Main Bushing
  z_crank = CRANK_AXIS_Z_MM * s
  mj_radii = [
      r for r in _extract_cylindrical_radii(
          _part("kit_crank_left"), (0.0, 1.0, 0.0), (0.0, 0.0, z_crank), y_range_mm=(-60.0 * s, -22.0 * s)
      )
      if r > 12.0 * s
  ]
  mb_radii = _extract_cylindrical_radii(_part("kit_main_bushing_left"), (0.0, 1.0, 0.0), (0.0, 0.0, z_crank))
  checks.append(_check_radial_joint(
      "main_journal_to_main_bushing",
      "kit_crank_left",
      "kit_main_bushing_left",
      "running",
      r_run,
      min(mj_radii) if mj_radii else None,
      min(mb_radii) if mb_radii else None,
      _ov("kit_crank_left", "kit_main_bushing_left") + _ov("kit_crank_right", "kit_main_bushing_right"),
      "Left/right main journals vs keyed main sleeve bushings",
  ))

  # 5. Valve Stem / Cylinder Head Guide
  sin15 = math.sin(math.radians(VALVE_TILT_DEG))
  cos15 = math.cos(math.radians(VALVE_TILT_DEG))
  v_axis = (sin15, 0.0, cos15)
  v_origin = (VALVE_IN_X_1TO1_MM * s, 0.0, VALVE_SEAT_Z_1TO1_MM * s)
  stem_radii = [
      r for r in _extract_cylindrical_radii(_part("kit_valve_intake"), v_axis, v_origin)
      if r < 0.55 * (15.0 * s)
  ]
  guide_radii = _extract_cylindrical_radii(_part("kit_cylinder_head"), v_axis, v_origin)
  checks.append(_check_radial_joint(
      "valve_stem_to_cylinder_head",
      "kit_valve_intake",
      "kit_cylinder_head",
      "running",
      r_run,
      max(stem_radii) if stem_radii else None,
      min(guide_radii) if guide_radii else None,
      _ov("kit_valve_intake", "kit_cylinder_head") + _ov("kit_valve_exhaust", "kit_cylinder_head"),
      "Intake/exhaust valve stems vs merged head guides",
  ))

  # 6. Cam Core / Cam Tower
  z_cam = CAM_AXIS_Z_MM * s
  cam_core_r_nom = max(3.0, 10.0 * s)
  cam_core_radii = [
      r for r in _extract_cylindrical_radii(
          _part("kit_camshaft_core"), (0.0, 1.0, 0.0), (0.0, 0.0, z_cam), y_range_mm=(-31.0 * s, 20.0 * s)
      )
      if r < cam_core_r_nom + 0.5
  ]
  cam_tower_radii = _extract_cylindrical_radii(_part("kit_cam_tower"), (0.0, 1.0, 0.0), (0.0, 0.0, z_cam))
  checks.append(_check_radial_joint(
      "cam_core_to_cam_tower",
      "kit_camshaft_core",
      "kit_cam_tower",
      "running",
      r_run,
      max(cam_core_radii) if cam_core_radii else None,
      min(cam_tower_radii) if cam_tower_radii else None,
      _ov("kit_camshaft_core", "kit_cam_tower"),
      "Camshaft core journals vs teardrop cam tower bores",
  ))

  # 7. Rocker Arm / Rocker Shaft
  rx = ROCKER_PIVOT_X_1TO1_MM * s
  rz = ROCKER_PIVOT_Z_1TO1_MM * s
  rshaft_radii = _extract_cylindrical_radii(_part("kit_rocker_shaft_intake"), (0.0, 1.0, 0.0), (rx, 0.0, rz))
  rarm_radii = _extract_cylindrical_radii(_part("kit_rocker_arm_intake"), (0.0, 1.0, 0.0), (rx, 0.0, rz))
  checks.append(_check_radial_joint(
      "rocker_arm_to_rocker_shaft",
      "kit_rocker_arm_intake",
      "kit_rocker_shaft_intake",
      "running",
      r_run,
      min(rshaft_radii) if rshaft_radii else None,
      min(rarm_radii) if rarm_radii else None,
      _ov("kit_rocker_arm_intake", "kit_rocker_shaft_intake")
      + _ov("kit_rocker_arm_exhaust", "kit_rocker_shaft_exhaust"),
      "Rocker arm pivot bores vs rocker shafts",
  ))

  # 8. Idler Gear / Idler Pin
  z_i1 = (CRANK_AXIS_Z_MM + (IDLER_MODULE_STEPS[0] / TIMING_MODULE_DIVISOR) * TIMING_CD_MM) * s
  ipin_radii = _extract_cylindrical_radii(
      _part("kit_timing_idler_pin_1"), (0.0, 1.0, 0.0), (0.0, 0.0, z_i1), y_range_mm=(52.0 * s, 67.0 * s)
  )
  igear_radii = _extract_cylindrical_radii(_part("kit_timing_gear_idler_24t_1"), (0.0, 1.0, 0.0), (0.0, 0.0, z_i1))
  checks.append(_check_radial_joint(
      "idler_gear_to_idler_pin",
      "kit_timing_gear_idler_24t_1",
      "kit_timing_idler_pin_1",
      "running",
      r_run,
      min(ipin_radii) if ipin_radii else None,
      min(igear_radii) if igear_radii else None,
      _ov("kit_timing_gear_idler_24t_1", "kit_timing_idler_pin_1"),
      "24T timing idler gears vs stepped idler pins",
  ))

  # 9. PTO Shaft / Timing Cover
  pto_shaft_radii = [
      r for r in _extract_cylindrical_radii(_part("kit_crank_right"), (0.0, 1.0, 0.0), (0.0, 0.0, z_crank))
      if r < 12.0 * s
  ]
  pto_bore_radii = _extract_cylindrical_radii(
      _part("kit_right_timing_cover"), (0.0, 1.0, 0.0), (0.0, 0.0, z_crank)
  )
  checks.append(_check_radial_joint(
      "pto_shaft_to_timing_cover",
      "kit_crank_right",
      "kit_right_timing_cover",
      "running",
      r_run,
      min(pto_shaft_radii) if pto_shaft_radii else None,
      min(pto_bore_radii) if pto_bore_radii else None,
      _ov("kit_crank_right", "kit_right_timing_cover"),
      "Crank PTO journal vs right timing cover clearance bore",
  ))

  # 10. Bonus Slip & D-Keyed Coupling Audits
  case_pkt_radii = [
      r for r in _extract_cylindrical_radii(
          _part("kit_crankcase_left"), (0.0, 1.0, 0.0), (0.0, 0.0, z_crank), y_range_mm=(-44.0 * s, -25.0 * s)
      )
      if r > 20.0 * s
  ]
  checks.append(_check_radial_joint(
      "main_bushing_to_crankcase",
      "kit_main_bushing_left",
      "kit_crankcase_left",
      "slip",
      r_slip,
      max(mb_radii) if mb_radii else None,
      min(case_pkt_radii) if case_pkt_radii else None,
      _ov("kit_main_bushing_left", "kit_crankcase_left"),
      "Keyed main bushing OD vs crankcase pocket",
  ))

  case_sock_radii = _extract_cylindrical_radii(_part("kit_crankcase_right"), (0.0, 1.0, 0.0), (0.0, 0.0, z_i1))
  checks.append(_check_radial_joint(
      "idler_pin_to_crankcase_right",
      "kit_timing_idler_pin_1",
      "kit_crankcase_right",
      "slip",
      r_slip,
      min(ipin_radii) if ipin_radii else None,
      min(case_sock_radii) if case_sock_radii else None,
      _ov("kit_timing_idler_pin_1", "kit_crankcase_right"),
      "Stepped idler pin base vs crankcase_right blind teardrop socket",
  ))

  if not cfg.cutaway:
    cover_sock_radii = _extract_cylindrical_radii(_part("kit_right_timing_cover"), (0.0, 1.0, 0.0), (0.0, 0.0, z_i1))
    checks.append(_check_radial_joint(
        "idler_pin_to_timing_cover",
        "kit_timing_idler_pin_1",
        "kit_right_timing_cover",
        "slip",
        r_slip,
        min(ipin_radii) if ipin_radii else None,
        min(cover_sock_radii) if cover_sock_radii else None,
        _ov("kit_timing_idler_pin_1", "kit_right_timing_cover"),
        "Stepped idler pin tips vs timing cover blind sockets",
    ))

  crank_gear_radii = _extract_cylindrical_radii(_part("kit_timing_gear_crank_14t"), (0.0, 1.0, 0.0), (0.0, 0.0, z_crank))
  checks.append(_check_radial_joint(
      "crank_shaft_to_timing_gear_14t",
      "kit_crank_right",
      "kit_timing_gear_crank_14t",
      "keyed_slip",
      r_slip,
      min(pto_shaft_radii) if pto_shaft_radii else None,
      min(crank_gear_radii) if crank_gear_radii else None,
      _ov("kit_crank_right", "kit_timing_gear_crank_14t"),
      "Crank D-keyed timing journal vs 14T crank gear bore",
  ))

  cam_nose_radii = _extract_cylindrical_radii(
      _part("kit_camshaft_core"), (0.0, 1.0, 0.0), (0.0, 0.0, z_cam), y_range_mm=(41.0 * s, 63.0 * s)
  )
  cam_gear_radii = _extract_cylindrical_radii(_part("kit_timing_gear_cam_28t"), (0.0, 1.0, 0.0), (0.0, 0.0, z_cam))
  checks.append(_check_radial_joint(
      "cam_nose_to_timing_gear_28t",
      "kit_camshaft_core",
      "kit_timing_gear_cam_28t",
      "keyed_slip",
      r_slip,
      min(cam_nose_radii) if cam_nose_radii else None,
      min(cam_gear_radii) if cam_gear_radii else None,
      _ov("kit_camshaft_core", "kit_timing_gear_cam_28t"),
      "Cam D-keyed nose vs 28T cam gear bore",
  ))

  # 11. Cantilever Snap-Fit Mechanics Audits
  snap_designs = _compute_kit_snap_designs(cfg)
  for snap_name, design in snap_designs.items():
    checks.append(
        FitmentCheckItem(
            joint_name=f"snap_{snap_name}",
            part_a=snap_name,
            part_b="mating_pocket",
            fit_type="snap_mechanics",
            target_clearance_mm=round(design.clearance_mm, 4),
            measured_clearance_mm=round(design.clearance_mm, 4),
            is_valid=bool(design.is_safe and design.safety_factor >= 1.0),
            notes=(
                f"SF={design.safety_factor:.2f} (max_strain={design.max_strain * 100.0:.2f}% vs "
                f"allow={design.allowable_strain * 100.0:.2f}%), L={design.beam_length_mm:.1f} mm, "
                f"h_root={design.root_thickness_mm:.2f} mm, undercut={design.undercut_mm:.2f} mm, "
                f"mate={design.mating_force_n:.1f} N"
            ),
        )
    )

  return tuple(checks), snap_designs


def _partition_stationary_and_moving(
    asm_parts: Mapping[str, KitPart],
) -> tuple[dict[str, Part], dict[str, Part]]:
  """Split a posed 42-part assembly into stationary housings/pins and moving parts."""
  stationary: dict[str, Part] = {}
  moving: dict[str, Part] = {}
  for name, kp in asm_parts.items():
    if (
        kp.category == "housings"
        or name in STATIONARY_ROTATING_KEYS
        or name in STATIONARY_VALVETRAIN_KEYS
    ):
      stationary[name] = kp.part
    else:
      moving[name] = kp.part
  return stationary, moving


def _verify_from_precomputed_ca0(
    cfg: PrintKitConfig,
    housings_asm: Mapping[str, KitPart],
    asm_0: Mapping[str, KitPart],
    crank_angles_deg: Sequence[float],
    vol_tol_mm3: float = DEFAULT_COLLISION_VOL_TOL_MM3,
) -> KitFitmentVerificationResult:
  """Run static/kinematic collisions and fitment audits reusing CA=0 and housing B-Reps."""
  stationary_parts, moving_0 = _partition_stationary_and_moving(asm_0)

  # Static stationary check once, plus per-angle moving checks (avoiding duplicate CA=0 work).
  _, stationary_hits = find_pairwise_collisions(stationary_parts, vol_tol_mm3=vol_tol_mm3)
  ca0_moving_checked = False
  ca0_moving_hits: list[str] = []

  kinematic_hits: list[str] = []
  for angle_deg in crank_angles_deg:
    is_zero = math.isclose(float(angle_deg), 0.0, abs_tol=1.0e-6)
    if is_zero:
      moving_now = moving_0
    else:
      asm_angle = _merge_disjoint_parts(
          build_kit_rotating_assembly_parts(cfg, crank_angle_deg=float(angle_deg)),
          housings_asm,
          build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=float(angle_deg)),
      )
      _, moving_now = _partition_stationary_and_moving(asm_angle)

    _, inner_hits = find_pairwise_collisions(moving_now, vol_tol_mm3=vol_tol_mm3)
    _, cross_hits = find_cross_collisions(moving_now, stationary_parts, vol_tol_mm3=vol_tol_mm3)
    combined_angle_hits = inner_hits + cross_hits
    if is_zero:
      ca0_moving_checked = True
      ca0_moving_hits = combined_angle_hits
    for label in combined_angle_hits:
      kinematic_hits.append(f"CA={float(angle_deg):.1f}: {label}")

  if not ca0_moving_checked:
    _, inner_0 = find_pairwise_collisions(moving_0, vol_tol_mm3=vol_tol_mm3)
    _, cross_0 = find_cross_collisions(moving_0, stationary_parts, vol_tol_mm3=vol_tol_mm3)
    ca0_moving_hits = inner_0 + cross_0

  static_hits = stationary_hits + ca0_moving_hits

  fitment_checks, snap_designs = _audit_kit_clearances_and_snaps(cfg, asm_0)
  all_passed = (
      not static_hits
      and not kinematic_hits
      and all(item.is_valid for item in fitment_checks)
  )
  return KitFitmentVerificationResult(
      static_collisions=tuple(static_hits),
      kinematic_collisions=tuple(kinematic_hits),
      fitment_checks=fitment_checks,
      snap_fit_designs=snap_designs,
      all_passed=all_passed,
  )


def verify_kit_fitment_and_collisions(
    cfg: PrintKitConfig,
    crank_angles_deg: tuple[float, ...] = DEFAULT_CRANK_ANGLES_DEG,
) -> KitFitmentVerificationResult:
  """Verify pairwise B-Rep collisions across crank angles and audit clearances/snaps."""
  housings_asm = build_kit_housings_assembly_parts(cfg, crank_angle_deg=0.0)
  asm_0 = _merge_disjoint_parts(
      build_kit_rotating_assembly_parts(cfg, crank_angle_deg=0.0),
      housings_asm,
      build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=0.0),
  )
  return _verify_from_precomputed_ca0(
      cfg=cfg,
      housings_asm=housings_asm,
      asm_0=asm_0,
      crank_angles_deg=crank_angles_deg,
  )


def _reseat_min_z_zero(part: Part) -> Part:
  """Re-seat a shrunk Part onto Z=0 and guard against float32 ULP splits near origin.

  Non-uniform ``BRepBuilderAPI_GTransform`` shrinkage scaling can leave ~1e-15 mm
  discrepancies on faces near X=0/Y=0/Z=0 where IEEE-754 float32 ULP spacing is
  sub-femtometer; a 5 nm (5e-6 mm) shift moves faces out of that ULP regime while
  remaining well within 1e-5 mm of Z=0.
  """
  min_z = float(part.bounding_box().min.Z)
  dz = (-min_z if abs(min_z) > MIN_Z_RESEAT_TOL_MM else 0.0) + STL_FLOAT32_ULP_GUARD_MM
  return bake_part_transform(
      part,
      (0.0, 0.0, 0.0),
      (STL_FLOAT32_ULP_GUARD_MM, STL_FLOAT32_ULP_GUARD_MM, dz),
  )


def _serialize_part_report(
    rep: PartPrintReport,
    meta: Optional[KitPart],
    orient_res: Optional[OrientedPartResult],
) -> dict[str, Any]:
  """Convert a ``PartPrintReport`` and optional orientation metadata to JSON dict."""
  payload = asdict(rep)
  payload["stl_path"] = str(rep.stl_path)
  if meta is not None:
    payload["category"] = meta.category
    payload["notes"] = meta.notes
    payload["preferred_rotation_xyz_deg"] = (
        list(meta.preferred_rotation_xyz_deg) if meta.preferred_rotation_xyz_deg is not None else None
    )
  if orient_res is not None:
    payload["applied_rotation_xyz_deg"] = list(orient_res.rotation_xyz_deg)
    payload["orientation_rationale"] = orient_res.rationale
  return payload


def _serialize_plate_report(plate: BuildPlateReport) -> dict[str, Any]:
  """Convert a ``BuildPlateReport`` to a JSON-serializable dictionary."""
  payload = asdict(plate)
  payload["stl_path"] = str(plate.stl_path)
  payload["used_dims_mm"] = list(plate.used_dims_mm)
  payload["print_volume_mm"] = list(plate.print_volume_mm)
  payload["warnings"] = list(plate.warnings)
  return payload


def _serialize_plate_packing(packing: PlatePackingResult) -> dict[str, Any]:
  """Convert a ``PlatePackingResult`` to a JSON-serializable manifest section."""
  return {
      "group_mode": packing.group_mode,
      "margin_mm": packing.margin_mm,
      "gap_mm": packing.gap_mm,
      "brim_gap_mm": packing.brim_gap_mm,
      "plates_dir": str(packing.plates_dir),
      "plate_count": len(packing.plates),
      "total_placed_pieces": packing.total_placed_pieces,
      "unplaced_parts": list(packing.unplaced_parts),
      "filament_mass_by_type_g": dict(packing.filament_mass_by_type_g),
      "all_watertight": packing.all_watertight,
      "all_fit_print_volume": packing.all_fit_print_volume,
      "plates": [_serialize_plate_report(plate) for plate in packing.plates],
  }


def _write_kit_manifest(
    summary: PrintKitSummary,
    bom_parts: Mapping[str, KitPart],
    orient_results: Mapping[str, OrientedPartResult],
) -> Path:
  """Write structured ``print_kit_manifest.json`` summarizing config, BOM, and checks."""
  profile = summary.filament_profile
  cfg = summary.config
  sx, sy, sz = (
      shrinkage_compensation_xyz(profile) if cfg.compensate_shrinkage else (1.0, 1.0, 1.0)
  )
  lin_defl, ang_defl = cfg.mesh_deflections

  manifest: dict[str, Any] = {
      "config": asdict(cfg),
      "filament_profile": asdict(profile),
      "shrinkage_compensation_applied": cfg.compensate_shrinkage,
      "shrinkage_compensation_xyz": [sx, sy, sz],
      "mesh_tessellation": {
          "quality": cfg.mesh_quality,
          "linear_deflection_mm": lin_defl,
          "angular_deflection_rad": ang_defl,
      },
      "tolerances_mm": {
          "running_clearance_radial_mm": running_clearance_radial_mm(cfg),
          "slip_clearance_radial_mm": slip_clearance_radial_mm(cfg),
          "snap_clearance_mm": snap_clearance_mm(cfg),
          "press_interference_radial_mm": press_interference_radial_mm(cfg),
          "min_structural_wall_mm": min_printable_wall_mm(cfg, structural=True),
          "min_non_structural_wall_mm": min_printable_wall_mm(cfg, structural=False),
          "min_fin_thickness_mm": min_fin_thickness_mm(cfg),
      },
      "totals": {
          "total_unique_parts": summary.total_unique_parts,
          "total_printed_pieces": summary.total_printed_pieces,
          "total_mass_g": round(summary.total_mass_g, 3),
          "total_filament_mass_by_type_g": dict(summary.filament_mass_by_type_g),
          "coupon_mass_g": (
              round(summary.coupon_report.estimated_mass_g, 3)
              if summary.coupon_report is not None
              else 0.0
          ),
          "total_print_time_min": round(summary.total_print_time_min, 2),
          "packed_plate_count": (
              len(summary.plate_packing.plates) if summary.plate_packing is not None else 0
          ),
          "all_fit_print_volume": summary.all_fit_print_volume,
          "all_watertight": summary.all_watertight,
          "coupon_included": summary.coupon_report is not None,
      },
      "verification": {
          "static_collisions": list(summary.static_collisions),
          "kinematic_collisions": list(summary.kinematic_collisions),
          "collisions_passed": not summary.static_collisions and not summary.kinematic_collisions,
          "fitment_all_passed": all(item.is_valid for item in summary.fitment_checks),
          "snap_all_safe": all(d.is_safe for d in summary.snap_fit_designs.values()),
      },
      "fitment_checks": [asdict(item) for item in summary.fitment_checks],
      "snap_fit_designs": {k: asdict(v) for k, v in summary.snap_fit_designs.items()},
      "parts": [
          _serialize_part_report(
              rep,
              bom_parts.get(rep.part_name),
              orient_results.get(rep.part_name),
          )
          for rep in summary.parts_reports
      ],
      "plates": (
          _serialize_plate_packing(summary.plate_packing)
          if summary.plate_packing is not None
          else None
      ),
      "coupon": (
          _serialize_part_report(summary.coupon_report, None, None)
          if summary.coupon_report is not None
          else None
      ),
      "artifacts": {
          "stl_dir": str(summary.stl_dir),
          "plates_dir": (
              str(summary.plate_packing.plates_dir)
              if summary.plate_packing is not None
              else None
          ),
          "manifest_path": str(summary.manifest_path),
          "guide_path": str(summary.guide_path),
          "assembled_preview_stl_path": (
              str(summary.assembled_preview_stl_path)
              if summary.assembled_preview_stl_path is not None
              else None
          ),
      },
  }
  summary.manifest_path.parent.mkdir(parents=True, exist_ok=True)
  summary.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return summary.manifest_path


def run_print_kit_workflow(
    cfg: PrintKitConfig,
    output_dir: Path | str,
    *,
    export_assembled_preview: bool = False,
    verify_kinematics: bool = True,
    linear_deflection: Optional[float] = None,
    angular_deflection: Optional[float] = None,
) -> PrintKitSummary:
  """Execute the end-to-end OpenEngine 3D-printable model kit build, audit, and export."""
  from openengine.printing.report import write_print_guide_markdown

  out_dir = Path(output_dir)
  stl_dir = out_dir / "parts"
  stl_dir.mkdir(parents=True, exist_ok=True)

  profile = cfg.filament_profile
  shrink_xyz = (
      shrinkage_compensation_xyz(profile) if cfg.compensate_shrinkage else (1.0, 1.0, 1.0)
  )
  cfg_lin, cfg_ang = cfg.mesh_deflections
  eff_lin_deflection = float(linear_deflection) if linear_deflection is not None else cfg_lin
  eff_ang_deflection = float(angular_deflection) if angular_deflection is not None else cfg_ang

  # Build housings once and reuse across canonical BOM, CA=0 check, kinematics, and preview.
  housings_parts = build_kit_housings_parts(cfg)
  rot_0_asm = build_kit_rotating_assembly_parts(cfg, crank_angle_deg=0.0)
  vt_parts = build_kit_valvetrain_parts(cfg, crank_angle_deg=0.0)
  vt_0_asm = build_kit_valvetrain_assembly_parts(cfg, crank_angle_deg=0.0)

  rot_canonical = dict(rot_0_asm)
  bushing_left = rot_canonical.pop("kit_main_bushing_left")
  rot_canonical.pop("kit_main_bushing_right")
  rot_canonical["kit_main_bushing"] = KitPart(
      name="kit_main_bushing",
      part=bushing_left.part,
      quantity=2,
      category=bushing_left.category,
      preferred_rotation_xyz_deg=bushing_left.preferred_rotation_xyz_deg,
      notes=bushing_left.notes,
  )
  bom_parts = _merge_disjoint_parts(rot_canonical, housings_parts, vt_parts)
  asm_0 = _merge_disjoint_parts(rot_0_asm, housings_parts, vt_0_asm)

  crank_angles = DEFAULT_CRANK_ANGLES_DEG if verify_kinematics else (0.0,)
  verification = _verify_from_precomputed_ca0(
      cfg=cfg,
      housings_asm=housings_parts,
      asm_0=asm_0,
      crank_angles_deg=crank_angles,
  )
  kinematic_collisions = verification.kinematic_collisions if verify_kinematics else ()

  parts_reports: list[PartPrintReport] = []
  orient_results: dict[str, OrientedPartResult] = {}

  for name, kp in bom_parts.items():
    oriented = orient_part_for_printing(
        part=kp.part,
        part_name=name,
        max_overhang_deg=profile.max_overhang_deg,
        print_volume_mm=cfg.print_volume_mm,
        preferred_rotation_xyz_deg=kp.preferred_rotation_xyz_deg,
    )
    orient_results[name] = oriented
    compensated = _reseat_min_z_zero(
        apply_shrinkage_compensation(oriented.oriented_part, shrink_xyz)
    )
    stl_path = stl_dir / f"{name}.stl"
    rep = analyze_part_printability(
        part=compensated,
        part_name=name,
        stl_path=stl_path,
        cfg=cfg,
        quantity=kp.quantity,
        linear_deflection=eff_lin_deflection,
        angular_deflection=eff_ang_deflection,
        orient_first=False,
        compensate_shrinkage=False,
    )
    parts_reports.append(rep)

  coupon_report: Optional[PartPrintReport] = None
  if cfg.include_coupon:
    coupon_part = _reseat_min_z_zero(
        apply_shrinkage_compensation(generate_tolerance_coupon(cfg), shrink_xyz)
    )
    coupon_stl = stl_dir / "tolerance_calibration_coupon.stl"
    coupon_report = analyze_part_printability(
        part=coupon_part,
        part_name="tolerance_calibration_coupon",
        stl_path=coupon_stl,
        cfg=cfg,
        quantity=1,
        linear_deflection=eff_lin_deflection,
        angular_deflection=eff_ang_deflection,
        orient_first=False,
        compensate_shrinkage=False,
    )

  assembled_preview_stl_path: Optional[Path] = None
  if export_assembled_preview:
    preview_solids = [sol for kp in asm_0.values() for sol in kp.part.solids()]
    preview_part = Part(children=preview_solids)
    preview_stl_path = out_dir / "assembled_engine_preview.stl"
    export_part_stl(
        preview_part,
        preview_stl_path,
        linear_deflection=eff_lin_deflection,
        angular_deflection=eff_ang_deflection,
    )
    assembled_preview_stl_path = preview_stl_path

  plate_packing: Optional[PlatePackingResult] = None
  if cfg.pack_plates:
    brim_gap = max(cfg.plate_gap_mm, (2.0 * cfg.brim_width_mm) + 2.0)
    plate_packing = pack_print_kit_plates(
        parts_reports=parts_reports,
        output_dir=out_dir / "plates",
        print_volume_mm=cfg.print_volume_mm,
        part_categories={name: kp.category for name, kp in bom_parts.items()},
        group_mode=cfg.plate_group_mode,
        margin_mm=cfg.plate_margin_mm,
        gap_mm=cfg.plate_gap_mm,
        brim_gap_mm=brim_gap,
    )

  all_reports = list(parts_reports) + ([coupon_report] if coupon_report is not None else [])
  total_unique_parts = len(parts_reports)
  total_printed_pieces = sum(r.quantity for r in parts_reports)
  total_mass_g = sum(r.estimated_mass_g * r.quantity for r in parts_reports)
  filament_mass_by_type_g = compute_filament_mass_by_type_g(
      parts_reports,
      primary_filament=profile.name,
  )
  total_print_time_min = sum(r.estimated_print_time_min * r.quantity for r in parts_reports)
  all_fit_print_volume = all(r.fits_print_volume for r in all_reports) and (
      plate_packing.all_fit_print_volume if plate_packing is not None else True
  )
  all_watertight = all(r.watertight for r in all_reports) and (
      plate_packing.all_watertight if plate_packing is not None else True
  )

  manifest_path = out_dir / "print_kit_manifest.json"
  guide_path = out_dir / "print_guide.md"

  summary = PrintKitSummary(
      config=cfg,
      filament_profile=profile,
      total_unique_parts=total_unique_parts,
      total_printed_pieces=total_printed_pieces,
      total_mass_g=total_mass_g,
      filament_mass_by_type_g=filament_mass_by_type_g,
      total_print_time_min=total_print_time_min,
      all_fit_print_volume=all_fit_print_volume,
      all_watertight=all_watertight,
      static_collisions=verification.static_collisions,
      kinematic_collisions=kinematic_collisions,
      fitment_checks=verification.fitment_checks,
      snap_fit_designs=verification.snap_fit_designs,
      parts_reports=tuple(parts_reports),
      coupon_report=coupon_report,
      stl_dir=stl_dir,
      manifest_path=manifest_path,
      guide_path=guide_path,
      assembled_preview_stl_path=assembled_preview_stl_path,
      plate_packing=plate_packing,
  )

  _write_kit_manifest(summary, bom_parts=bom_parts, orient_results=orient_results)
  write_print_guide_markdown(summary, guide_path)
  return summary
