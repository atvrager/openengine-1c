"""Markdown Print & Assembly Guide generator for OpenEngine 3D-printable model kits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from openengine.printing.analysis import PartPrintReport
from openengine.printing.coupon import coupon_bore_radial_offsets_mm
from openengine.printing.tolerances import (
    min_fin_thickness_mm,
    min_printable_wall_mm,
    press_interference_radial_mm,
    running_clearance_radial_mm,
    shrinkage_compensation_xyz,
    slip_clearance_radial_mm,
    snap_clearance_mm,
)

if TYPE_CHECKING:
  from openengine.printing.kit_assembly import PrintKitSummary

MINUTES_PER_HOUR: float = 60.0
PCT_SCALE: float = 100.0
STANDARD_SPOOL_MASS_G: float = 1000.0


def _format_filament_mass_inline(mass_by_type: Mapping[str, float]) -> str:
  """Format a filament-type mass map as ``TYPE: X.X g`` pairs."""
  if not mass_by_type:
    return "0.0 g"
  return ", ".join(f"{fil}: {mass:.1f} g" for fil, mass in mass_by_type.items())


def _format_duration(minutes: float) -> str:
  """Format print duration as ``Xm (Y.Yh)`` for readability."""
  return f"{minutes:.1f} min ({minutes / MINUTES_PER_HOUR:.2f} h)"


def _format_rel_stl_path(stl_path: Path, output_dir: Path) -> str:
  """Format STL path relative to the guide directory when possible."""
  try:
    return stl_path.relative_to(output_dir).as_posix()
  except ValueError:
    return stl_path.as_posix()


def _format_header_section(summary: PrintKitSummary) -> list[str]:
  """Build top-level kit build summary and verification gate overview."""
  cfg = summary.config
  profile = summary.filament_profile
  vx, vy, vz = cfg.print_volume_mm
  lin_defl, ang_defl = cfg.mesh_deflections
  collisions_ok = not summary.static_collisions and not summary.kinematic_collisions
  fitment_ok = all(c.is_valid for c in summary.fitment_checks)
  snaps_ok = all(d.is_safe for d in summary.snap_fit_designs.values())

  filament_breakdown = _format_filament_mass_inline(summary.filament_mass_by_type_g)
  lines = [
      "# OpenEngine-1C 3D-Printable Snap-Fit Model Kit Guide",
      "",
      f"- **Scale & Mode:** `{cfg.scale:.2f}x` | Cutaway display: `{'ON' if cfg.cutaway else 'OFF'}`",
      f"- **Selected Filament:** `{profile.name}` (Density `{profile.density_g_cm3:.2f} g/cm^3`, HDT `{profile.hdt_c:.0f} C`)",
      f"- **Print Volume:** `{vx:.0f} x {vy:.0f} x {vz:.0f} mm` (`{cfg.nozzle_diameter_mm:.2f} mm` nozzle, `{cfg.layer_height_mm:.2f} mm` layer height, mesh=`{cfg.mesh_quality}` `{lin_defl:.3f} mm / {ang_defl:.2f} rad`)",
      f"- **Kit BOM Totals:** `{summary.total_unique_parts}` unique STLs / `{summary.total_printed_pieces}` printed pieces",
      f"- **Total Filament Mass by Type:** `{filament_breakdown}` (Kit Total: `{summary.total_mass_g:.1f} g`)",
      f"- **Estimated Kit Filament & Print Time:** `{summary.total_mass_g:.1f} g` | `{_format_duration(summary.total_print_time_min)}`",
      f"- **Verification Gates:** Print Volume Fit `{'PASS' if summary.all_fit_print_volume else 'FAIL'}` | Watertight `{'PASS' if summary.all_watertight else 'FAIL'}` | Collisions `{'PASS (0 hits)' if collisions_ok else 'FAIL'}` | Fitment & Snaps `{'PASS' if (fitment_ok and snaps_ok) else 'FAIL'}`",
  ]
  if summary.plate_packing is not None:
    pp = summary.plate_packing
    lines.append(
        f"- **Packed Build Plates:** `{len(pp.plates)}` meta-STLs in `plates/` (`{pp.total_placed_pieces}/{summary.total_printed_pieces}` pieces, mode=`{pp.group_mode}`, margin=`{pp.margin_mm:.1f} mm`, gap=`{pp.gap_mm:.1f} mm` / brim=`{pp.brim_gap_mm:.1f} mm`)"
    )
  if summary.assembled_preview_stl_path is not None:
    lines.append(
        f"- **Assembled Preview STL (Reference Only, Not For Printing):** `{_format_rel_stl_path(summary.assembled_preview_stl_path, summary.guide_path.parent)}`"
    )
  return lines


def _format_filament_and_slicer_section(summary: PrintKitSummary) -> list[str]:
  """Build Section 1 covering filament temperatures, PEI warning, slicer walls, and shrinkage."""
  cfg = summary.config
  profile = summary.filament_profile
  sx, sy, sz = (
      shrinkage_compensation_xyz(profile) if cfg.compensate_shrinkage else (1.0, 1.0, 1.0)
  )
  xy_pct = (sx - 1.0) * PCT_SCALE
  z_pct = (sz - 1.0) * PCT_SCALE
  lin_defl, ang_defl = cfg.mesh_deflections

  w_struct = min_printable_wall_mm(cfg, structural=True)
  w_cosmetic = min_printable_wall_mm(cfg, structural=False)
  fin_min = min_fin_thickness_mm(cfg)
  shrink_note = (
      "**Already baked into exported STLs** -- keep slicer XYZ scale at `100%`"
      if cfg.compensate_shrinkage
      else "Disabled (`--no-shrinkage-compensation`) -- apply shrinkage scaling in slicer if needed"
  )

  return [
      "## 1. Filament & Slicer Setup",
      "",
      "> [!IMPORTANT]",
      f"> **Build-Plate Preparation & Release Warning ({profile.name}):** {profile.adhesion_note}",
      "",
      "| Slicer / Material Parameter | Recommended Value | Notes |",
      "| :--- | :--- | :--- |",
      f"| **Filament Profile** | `{profile.name}` | {profile.print_notes} |",
      f"| **Nozzle Temperature** | `{profile.nozzle_temp_c:.0f} C` | Adjust +/-5 C based on hotend thermistor calibration |",
      f"| **Bed Temperature** | `{profile.bed_temp_c:.0f} C` | Keep first layer slow (`20-30 mm/s`) for sharp corners |",
      f"| **Wall Loops / Top-Bottom Layers** | `{cfg.wall_loops}` loops / `{cfg.top_bottom_layers}` solid layers | Structural min wall `{w_struct:.2f} mm`, cosmetic `{w_cosmetic:.2f} mm`, fin `{fin_min:.2f} mm` |",
      f"| **Infill Density** | `{cfg.infill_pct:.0f}%` (Gyroid / Grid) | Use gyroid or 3D honeycomb for uniform stiffness |",
      f"| **Layer Height / Nozzle** | `{cfg.layer_height_mm:.2f} mm` / `{cfg.nozzle_diameter_mm:.2f} mm` | All horizontal bores use 45-deg teardrop crests |",
      f"| **Volumetric Flow / Support Overhead / Brim** | `{cfg.volumetric_flow_mm3_s:.1f} mm^3/s` / `{cfg.support_mass_overhead_pct:.0f}%` tree (`{0.5 * cfg.support_mass_overhead_pct:.1f}%` buildplate) / `{cfg.brim_width_mm:.1f} mm` brim | Used for mass and print-time estimates |",
      f"| **STL Mesh Smoothness** | `{cfg.mesh_quality}` (`{lin_defl:.3f} mm` linear, `{ang_defl:.2f} rad` angular) | Configure with `--mesh-quality draft|standard|fine|ultra` |",
      f"| **Pre-Applied Shrinkage Factors** | `X={sx:.5f} (+{xy_pct:.2f}%)`, `Y={sy:.5f} (+{xy_pct:.2f}%)`, `Z={sz:.5f} (+{z_pct:.2f}%)` | {shrink_note} |",
      f"| **Active Radial Clearances** | Run `{running_clearance_radial_mm(cfg):.3f} mm`, Slip `{slip_clearance_radial_mm(cfg):.3f} mm`, Snap `{snap_clearance_mm(cfg):.3f} mm`, Press `{press_interference_radial_mm(cfg):.3f} mm` | User `--clearance-bias`: `{cfg.clearance_bias_mm:+.3f} mm` |",
      "",
      "### Total Filament Mass by Filament Type",
      "",
      "| Filament Type | Kit Pieces | Kit Mass (g) | Coupon Mass (g) | Total Mass (g) | 1 kg Spool Share |",
      "| :--- | :---: | :---: | :---: | :---: | :---: |",
      *_format_filament_mass_rows(summary),
  ]


def _format_filament_mass_rows(summary: PrintKitSummary) -> list[str]:
  """Build table rows breaking down total filament mass by filament type."""
  piece_counts: dict[str, int] = {}
  for rep in summary.parts_reports:
    piece_counts[rep.filament_type] = piece_counts.get(rep.filament_type, 0) + rep.quantity

  coupon_fil = summary.coupon_report.filament_type if summary.coupon_report is not None else None
  coupon_mass = summary.coupon_report.estimated_mass_g if summary.coupon_report is not None else 0.0

  rows: list[str] = []
  total_kit_g = 0.0
  total_coupon_g = 0.0
  for fil_name, kit_mass_g in summary.filament_mass_by_type_g.items():
    c_mass = coupon_mass if coupon_fil == fil_name else 0.0
    combined = kit_mass_g + c_mass
    spool_pct = (combined / STANDARD_SPOOL_MASS_G) * PCT_SCALE
    total_kit_g += kit_mass_g
    total_coupon_g += c_mass
    rows.append(
        f"| `{fil_name}` | {piece_counts.get(fil_name, 0)} | {kit_mass_g:.1f} | "
        f"{c_mass:.1f} | {combined:.1f} | {spool_pct:.1f}% |"
    )

  grand_total = total_kit_g + total_coupon_g
  rows.append(
      f"| **All Filaments** | **{summary.total_printed_pieces}** | **{total_kit_g:.1f}** | "
      f"**{total_coupon_g:.1f}** | **{grand_total:.1f}** | **{(grand_total / STANDARD_SPOOL_MASS_G) * PCT_SCALE:.1f}%** |"
  )
  return rows


def _format_coupon_section(summary: PrintKitSummary) -> list[str]:
  """Build Section 2 detailing the 15-minute calibration coupon -> --clearance-bias workflow."""
  cfg = summary.config
  offsets = coupon_bore_radial_offsets_mm(cfg)
  coupon_rep = summary.coupon_report
  guide_dir = summary.guide_path.parent

  lines = [
      "## 2. 15-Minute Calibration Coupon Workflow (`tolerance_calibration_coupon.stl` -> `--clearance-bias`)",
      "",
      "Print the compact calibration coupon **first** (~15-20 minutes) using the exact filament spool, nozzle, and slicer profile intended for the full engine kit:",
      "",
  ]
  if coupon_rep is not None:
    dx, dy, dz = coupon_rep.oriented_dims_mm
    rel_stl = _format_rel_stl_path(coupon_rep.stl_path, guide_dir)
    lines.extend([
        f"- **Coupon STL:** `{rel_stl}` (`{dx:.1f} x {dy:.1f} x {dz:.1f} mm`, `{coupon_rep.estimated_mass_g:.1f} g`, `~{coupon_rep.estimated_print_time_min:.0f} min`, watertight=`{coupon_rep.watertight}`, supports=`{coupon_rep.support_recommendation}`)",
        "",
    ])
  else:
    lines.extend([
        "- *Coupon export disabled (`--no-coupon`); re-run with `--coupon` to generate `parts/tolerance_calibration_coupon.stl`.*",
        "",
    ])

  lines.extend([
      "Test the printed `5.0 mm` nominal peg in the 5 numbered bores (tally dimples 1..5 along front edge) and slide the dovetail-guided cantilever latch horizontally into its receiver until it clicks shut (nudge the visible barb tip inward to release):",
      "",
      "| Bore # (Dimples) | Radial Offset (mm) | Expected Physical Fit | Action if Off-Target |",
      "| :---: | :---: | :--- | :--- |",
      f"| **1** | `{offsets[0]:+.3f}` | Light press interference (requires firm push) | If loose, your printer under-extrudes or shrinks less -> `--clearance-bias -0.05` |",
      f"| **2** | `{offsets[1]:+.3f}` | Zero-clearance line-to-line reference | Peg should bind or require heavy force |",
      f"| **3** | `{offsets[2]:+.3f}` | Tight half-slip reference | Should feel snug with slight drag; if already smooth slip, tighten with `--clearance-bias -0.08` to `-0.10` |",
      f"| **4** | `{offsets[3]:+.3f}` | **Target Slip Fit** (pins, D-keys, bushings) | Ideal smooth hand-push fit with zero wobble |",
      f"| **5** | `{offsets[4]:+.3f}` | **Target Running Fit** (bearings, piston, valves) | Free spin/slide; if tight, re-run with `--clearance-bias +0.05` |",
      "",
      "If Bore #5 feels like a snug slip fit instead of free-running, re-export the kit with `--clearance-bias 0.05` (or `+0.08` for over-extruding spools). If Bore #3 already spins freely, tighten with `--clearance-bias -0.05`.",
  ])
  return lines


def _format_part_row(rep: PartPrintReport, guide_dir: Path) -> str:
  """Render one Markdown table row for a ``PartPrintReport``."""
  dx, dy, dz = rep.oriented_dims_mm
  vol_str = "PASS" if rep.fits_print_volume else "FAIL"
  brim_str = "Yes" if rep.brim_recommended else "No"
  wt_str = "PASS" if rep.watertight else "FAIL"
  rel_path = _format_rel_stl_path(rep.stl_path, guide_dir)
  return (
      f"| `{rep.part_name}` | {rep.quantity} | `{dx:.1f} x {dy:.1f} x {dz:.1f}` | "
      f"{vol_str} | {rep.overhang_pct:.2f}% | `{rep.support_recommendation}` | "
      f"{brim_str} | {wt_str} | {rep.estimated_mass_g:.1f} | "
      f"{rep.estimated_print_time_min:.1f} | `{rel_path}` |"
  )


def _format_parts_table_section(summary: PrintKitSummary) -> list[str]:
  """Build Section 3 containing the complete per-part printability & STL manifest table."""
  guide_dir = summary.guide_path.parent
  lines = [
      "## 3. Full Per-Part Print Table",
      "",
      f"All `{summary.total_unique_parts}` unique engine kit STLs (`{summary.total_printed_pieces}` total pieces) are pre-oriented onto `Z = 0`:",
      "",
      "| Part | Qty | Oriented Dims (mm) | Volume Fit | Overhang % | Support Rec | Brim | Watertight | Est. Mass (g) | Est. Time (min) | STL Path |",
      "| :--- | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
  ]
  for rep in summary.parts_reports:
    lines.append(_format_part_row(rep, guide_dir))

  lines.append(
      f"| **Kit Total ({summary.total_unique_parts} unique)** | **{summary.total_printed_pieces}** | — | "
      f"**{'PASS' if summary.all_fit_print_volume else 'FAIL'}** | — | — | — | "
      f"**{'PASS' if summary.all_watertight else 'FAIL'}** | **{summary.total_mass_g:.1f}** | "
      f"**{summary.total_print_time_min:.1f}** | `parts/` |"
  )
  if summary.coupon_report is not None:
    lines.extend([
        "",
        "Optional Pre-Print Calibration Coupon:",
        "",
        "| Part | Qty | Oriented Dims (mm) | Volume Fit | Overhang % | Support Rec | Brim | Watertight | Est. Mass (g) | Est. Time (min) | STL Path |",
        "| :--- | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
        _format_part_row(summary.coupon_report, guide_dir),
    ])

  warnings_lines: list[str] = []
  all_reports: Sequence[PartPrintReport] = list(summary.parts_reports) + (
      [summary.coupon_report] if summary.coupon_report is not None else []
  )
  for rep in all_reports:
    for msg in rep.warnings:
      warnings_lines.append(f"- `{rep.part_name}`: {msg}")
  if warnings_lines:
    lines.extend(["", "### Printability Notes & Warnings", "", *warnings_lines])

  return lines


def _format_fitment_and_snap_section(summary: PrintKitSummary) -> list[str]:
  """Build Section 4 covering clearance measurements, collision gates, and snap mechanics."""
  lines = [
      "## 4. Snap-Fit Mechanics & Clearance Fitment Audit",
      "",
      f"- **Static Collisions (CA=0 deg, 42 posed instances):** `{len(summary.static_collisions)}` hits",
      f"- **Kinematic Collisions (0/90/180/270 deg sweep):** `{len(summary.kinematic_collisions)}` hits",
      "",
      "### Running, Slip & Keyed Clearance Audit",
      "",
      "| Joint | Part A | Part B | Fit Type | Target (mm) | Measured (mm) | Status | Notes |",
      "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
  ]
  for item in summary.fitment_checks:
    if item.fit_type == "snap_mechanics":
      continue
    status = "PASS" if item.is_valid else "FAIL"
    lines.append(
        f"| `{item.joint_name}` | `{item.part_a}` | `{item.part_b}` | `{item.fit_type}` | "
        f"{item.target_clearance_mm:.3f} | {item.measured_clearance_mm:.3f} | {status} | {item.notes} |"
    )

  lines.extend([
      "",
      "### Cantilever Snap-Fit Beam Mechanics (`auto_size_cantilever_snap`)",
      "",
      "| Snap Joint | Beam L (mm) | Root / Tip (mm) | Undercut (mm) | Peak Strain | Allow Strain | Safety Factor | Deflect / Mate Force (N) | Status |",
      "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])
  for name, design in summary.snap_fit_designs.items():
    status = "SAFE" if design.is_safe else "UNSAFE"
    lines.append(
        f"| `{name}` | {design.beam_length_mm:.1f} | {design.root_thickness_mm:.2f} / {design.tip_thickness_mm:.2f} | "
        f"{design.undercut_mm:.2f} | {design.max_strain * PCT_SCALE:.2f}% | "
        f"{design.allowable_strain * PCT_SCALE:.2f}% | {design.safety_factor:.2f}x | "
        f"{design.deflection_force_n:.1f} / {design.mating_force_n:.1f} | {status} |"
    )
  return lines


def _format_assembly_order_section() -> list[str]:
  """Build Section 5 with the 5-stage tool-free snap-fit assembly instructions."""
  return [
      "## 5. Step-by-Step Snap-Fit Assembly Order",
      "",
      "Dry-fit all sliding joints after removing any brim remnants. A light dab of silicone or PTFE dry lube on main bushings, wrist pin, crankpin, and rocker shafts makes hand-cranking silky smooth.",
      "",
      "### Sub-Assembly 1: Crankshaft & Connecting Rod Rotating Core",
      "1. Wrap `kit_conrod_beam` and `kit_conrod_cap` around the central journal of `kit_crankpin`, aligning the two diagonal pins on the cap and pressing home until both side cantilever snaps click shut (verify the splash dipper points away from the rod shank and big-end spins freely).",
      "2. Align `kit_crankpin` with its `+Z` D-flat facing the crank throw (`0 deg` TDC) and press both ends into `kit_crank_left` and `kit_crank_right` counterweight sockets until the web detent tongues lock.",
      "3. Slide the two `kit_main_bushing` sleeves onto the left and right main journals (`+Z` anti-rotation key tabs facing up toward cylinder centerline), then slide `kit_flywheel` (18-1 trigger teeth facing outward `-Y`) onto the left crank D-keyed hub until the hub detent engages.",
      "",
      "### Sub-Assembly 2: Piston, Wrist Pin & Crankcase Core",
      "1. Slip `kit_piston` (ring grooves facing up, crown away from con-rod) over the `kit_conrod_beam` small-end eye and push `kit_wrist_pin` from the `-Y` side until its split `+Y` cantilever barb locks outboard of the pin boss.",
      "2. Seat the rotating crank + bushings into `kit_crankcase_left` (aligning the bushing `+Z` anti-rotation key tabs into the pockets and guiding the piston up the cylinder throat), then align `kit_crankcase_right` onto the 3 split-line alignment pins and press the 4 perimeter snaps closed.",
      "3. Press `kit_sump_pan` (raised locating lip seated inside the crankcase bay, drain boss on `-X`) up onto the crankcase base until all 6 perimeter snap hooks engage.",
      "",
      "### Sub-Assembly 3: Cylinder Block, Pent-Roof Head & Valvetrain Sub-Assembly",
      "1. Slide `kit_cylinder_block` down over `kit_piston` (oriented with the `+Y` timing tunnel on the right crankcase side), seat onto the 2 base alignment pins, and click the 4 corner snap hooks into the crankcase pockets. Rotate the flywheel to verify smooth piston travel.",
      "2. Populate `kit_cylinder_head` from the combustion dome: insert `kit_valve_intake` (larger head on `+X` port side) and `kit_valve_exhaust` (smaller head on `-X` port side), drop one `kit_valve_spring` wave spring into each upper spring pocket, compress lightly, and snap each `kit_spring_retainer` side-slot C-collar onto its valve stem neck groove.",
      "3. Align `kit_cylinder_head` onto the block top-deck diagonal pins (`+Y` timing bay flush), press the 4 exterior corner latch tabs onto the top deck flange, push `kit_spark_plug` into its `-Y` 24-deg well until its retention rib clicks, and snap `kit_intake_carb_unit` onto the `+X` intake flange pins/clips.",
      "",
      "### Sub-Assembly 4: SOHC Cam Tower & 5-Gear 2:1 Timing Train",
      "1. Slide `kit_camshaft_core` through the `kit_cam_tower` front teardrop bore (`-Y` -> `+Y`) while threading `kit_cam_lobes` onto the central keyed section inside the valvetrain bay.",
      "2. Position both `kit_rocker_arm` pivots inside the cam tower bay and press the 2 `kit_rocker_shaft` pins in from the `-Y` front face until their retaining flanges sit flush.",
      "3. Snap `kit_cam_tower` onto the cylinder head's 2 top alignment pins and 4 snaps. Set the crank to TDC (`0 deg` with crankpin straight up and cam lobes on base circle / valves closed), then populate the `+Y` timing tunnel: slide `kit_timing_gear_crank_14t` onto the crank D-shaft, mount the 3 `kit_timing_gear_idler_24t` gears on their 3 `kit_timing_idler_pin` shafts seated in the blind tunnel sockets (`crankcase_right`, `cylinder_block`, `cylinder_head`), and slide `kit_timing_gear_cam_28t` onto the D-keyed cam nose.",
      "",
      "### Sub-Assembly 5: Covers, Fan Shroud & Hand-Crank Finish",
      "1. Snap `kit_cam_cover` down onto the cam tower's 2 top pins and 4 exterior side snap tabs to enclose the valvetrain and timing tunnel roof.",
      "2. Guide `kit_right_timing_cover` over the 3 idler pin tips, PTO crank shaft, and 2 tunnel alignment pins, pressing home until all 6 `-Y` perimeter snap hooks latch onto `crankcase_right`, `cylinder_head`, and `cam_tower`.",
      "3. Snap `kit_fan_shroud` (4 standoff arms clearing the flywheel) onto the `-Y` crankcase face, slide `kit_hand_crank` onto the `+Y` PTO D-shaft until its detent clicks, and turn clockwise to cycle the full 720-deg 4-stroke engine!",
  ]


def _format_packed_plates_section(summary: PrintKitSummary) -> list[str]:
  """Build the ready-to-slice packed build plates section when plate packing is active."""
  packing = summary.plate_packing
  if packing is None:
    return []

  guide_dir = summary.guide_path.parent
  lines = [
      "## Packed Build Plates (Ready-to-Slice Meta-STLs)",
      "",
      (
          f"Grouped with `--plate-group {packing.group_mode}` "
          f"(`{len(packing.plates)}` plates, `{packing.total_placed_pieces}` pieces placed, "
          f"edge margin `{packing.margin_mm:.1f} mm`, gap `{packing.gap_mm:.1f} mm` / "
          f"brim gap `{packing.brim_gap_mm:.1f} mm`). Individual part STLs remain in `parts/` "
          "for single-part reprints; the calibration coupon stays separate."
      ),
      "",
      "| Plate | Group | Pieces | Used XYZ (mm) | Plate % | Supports | Brim | Filament Mass | Est. Time (min) | Watertight | STL Path |",
      "| :--- | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :--- |",
  ]

  for plate in packing.plates:
    ux, uy, uz = plate.used_dims_mm
    rel_path = _format_rel_stl_path(plate.stl_path, guide_dir)
    fil_str = _format_filament_mass_inline(plate.filament_mass_by_type_g)
    lines.append(
        f"| `{plate.plate_name}` | {plate.group_label} | {plate.piece_count} | "
        f"`{ux:.1f} x {uy:.1f} x {uz:.1f}` | {plate.plate_utilization_pct:.1f}% | "
        f"`{plate.support_recommendation}` | {'Yes' if plate.brim_recommended else 'No'} | "
        f"{fil_str} | {plate.estimated_print_time_min:.1f} | "
        f"{'PASS' if plate.watertight else 'FAIL'} | `{rel_path}` |"
    )

  lines.extend(["", "### Plate Contents", ""])
  for plate in packing.plates:
    counts = Counter(p.part_name for p in plate.placements)
    part_list = ", ".join(
        f"`{name}` (x{qty})" if qty > 1 else f"`{name}`"
        for name, qty in sorted(counts.items())
    )
    lines.append(
        f"- **`{plate.plate_name}`** (`{plate.piece_count}` pcs, supports=`{plate.support_recommendation}`, brim=`{'Yes' if plate.brim_recommended else 'No'}`): {part_list}"
    )

  if packing.unplaced_parts:
    unplaced_str = ", ".join(f"`{name}`" for name in packing.unplaced_parts)
    lines.extend(["", f"> [!WARNING]", f"> **Unplaced Oversized Parts (Exceed Print Volume):** {unplaced_str}"])

  plate_warnings = [
      f"- `{plate.plate_name}`: {msg}"
      for plate in packing.plates
      for msg in plate.warnings
  ]
  if plate_warnings:
    lines.extend(["", "### Plate Packing Notes", "", *plate_warnings])

  return lines


def write_print_guide_markdown(summary: PrintKitSummary, output_path: Path) -> Path:
  """Generate the comprehensive Markdown Print & Assembly Guide at ``output_path``."""
  target_path = Path(output_path)
  target_path.parent.mkdir(parents=True, exist_ok=True)

  sections = [
      _format_header_section(summary),
      ["", "---", ""],
      _format_filament_and_slicer_section(summary),
      ["", "---", ""],
      _format_coupon_section(summary),
  ]
  packed_plates_section = _format_packed_plates_section(summary)
  if packed_plates_section:
    sections.extend([["", "---", ""], packed_plates_section])
  sections.extend([
      ["", "---", ""],
      _format_parts_table_section(summary),
      ["", "---", ""],
      _format_fitment_and_snap_section(summary),
      ["", "---", ""],
      _format_assembly_order_section(),
  ])
  lines: list[str] = []
  for block in sections:
    lines.extend(block)

  target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return target_path
