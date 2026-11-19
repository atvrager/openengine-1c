"""2D build-plate bin packing and merged binary meta-STL export for OpenEngine print kits.

Packs pre-oriented, shrinkage-compensated binary part STLs onto rectangular FDM
printer build plates using MaxRects Best-Short-Side-Fit (with 90-deg Z-rotations),
brim-aware inter-part spacing, and direct numpy triangle merging so exported
plate STLs remain watertight without re-tessellating B-Rep geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Mapping, Optional, Sequence

import numpy as np

from openengine.printing.analysis import (
    BINARY_STL_DTYPE,
    PartPrintReport,
    STL_HEADER_BYTES,
    STL_PREFIX_BYTES,
    STL_TRIANGLE_BYTES,
    _read_binary_stl_arrays,
)
from openengine.printing.orientation import BED_FIT_EPS_MM
from openengine.printing.tolerances import (
    DEFAULT_BRIM_PLATE_GAP_MM,
    DEFAULT_PLATE_GAP_MM,
    DEFAULT_PLATE_GROUP_MODE,
    DEFAULT_PLATE_MARGIN_MM,
    MIN_PLATE_GAP_MM,
    MIN_PLATE_MARGIN_MM,
    VALID_PLATE_GROUP_MODES,
)

RECT_FIT_EPS_MM: float = 1.0e-6
RECT_SPLIT_EPS_MM: float = 1.0e-6
MIN_FREE_RECT_MM: float = 1.0e-3
SQUARE_ROTATION_EPS_MM: float = 1.0e-4
LAYOUT_VERIFY_TOL_MM: float = 1.0e-2
HALF_FACTOR: float = 0.5
PCT_SCALE: float = 100.0

SUPPORT_PRIORITY_ORDER: dict[str, int] = {
    "none": 0,
    "buildplate_only": 1,
    "tree_supports": 2,
}
SUPPORT_FILE_SLUGS: dict[str, str] = {
    "none": "nosupport",
    "buildplate_only": "buildplate_sup",
    "tree_supports": "tree_sup",
}
CATEGORY_DISPLAY_ORDER: dict[str, int] = {
    "housings": 0,
    "rotating": 1,
    "valvetrain": 2,
    "timing_drive": 3,
}


@dataclass(frozen=True, slots=True)
class PlatePlacement:
  """Placement of one physical part instance on a packed printer build plate."""

  part_name: str
  instance_index: int
  category: str
  rotated_z_90: bool
  bed_min_x_mm: float
  bed_min_y_mm: float
  placed_dims_mm: tuple[float, float, float]
  support_recommendation: str
  brim_recommended: bool
  filament_type: str
  estimated_mass_g: float
  estimated_print_time_min: float


@dataclass(frozen=True, slots=True)
class BuildPlateReport:
  """Exported meta-STL build plate report and slicer summary for an operator."""

  plate_index: int
  plate_name: str
  group_label: str
  stl_path: Path
  placements: tuple[PlatePlacement, ...]
  piece_count: int
  unique_part_count: int
  used_dims_mm: tuple[float, float, float]
  bed_dims_mm: tuple[float, float, float]
  effective_margin_mm: float
  effective_gap_mm: float
  bed_utilization_pct: float
  fits_bed: bool
  support_recommendation: str
  brim_recommended: bool
  estimated_mass_g: float
  filament_mass_by_type_g: dict[str, float]
  estimated_print_time_min: float
  triangle_count: int
  watertight: bool
  warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlatePackingResult:
  """Complete multi-plate packing summary for a print kit export."""

  group_mode: str
  margin_mm: float
  gap_mm: float
  brim_gap_mm: float
  plates_dir: Path
  plates: tuple[BuildPlateReport, ...]
  unplaced_parts: tuple[str, ...]
  total_placed_pieces: int
  filament_mass_by_type_g: dict[str, float]
  all_watertight: bool
  all_fit_bed: bool


@dataclass(frozen=True, slots=True)
class _Rect:
  """Axis-aligned 2D rectangle in bin coordinates."""

  x: float
  y: float
  w: float
  h: float


@dataclass(frozen=True, slots=True)
class _PackItem:
  """Expanded single printable instance ready for 2D bin packing."""

  part_name: str
  instance_index: int
  category: str
  stl_path: Path
  dx_mm: float
  dy_mm: float
  dz_mm: float
  support_recommendation: str
  brim_recommended: bool
  filament_type: str
  estimated_mass_g: float
  estimated_print_time_min: float


@dataclass(frozen=True, slots=True)
class _PlacedEntry:
  """Intermediate placement in usable-area coordinates before bed centering."""

  item: _PackItem
  rect_x: float
  rect_y: float
  placed_dx: float
  placed_dy: float
  rotated_z_90: bool


@dataclass(frozen=True, slots=True)
class _LoadedStlMesh:
  """Cached binary STL records and measured vertex bounding box."""

  records: np.ndarray
  dims_xyz_mm: tuple[float, float, float]


class _MaxRectsBin:
  """2D MaxRects Best-Short-Side-Fit packer with optional 90-deg Z-rotations."""

  def __init__(self, usable_x: float, usable_y: float, gap_mm: float) -> None:
    self.usable_x = float(usable_x)
    self.usable_y = float(usable_y)
    self.gap_mm = float(gap_mm)
    self.free_rects: list[_Rect] = [
        _Rect(0.0, 0.0, self.usable_x + self.gap_mm, self.usable_y + self.gap_mm)
    ]
    self.entries: list[_PlacedEntry] = []

  def try_place(self, item: _PackItem) -> bool:
    """Place ``item`` into the best free rectangle if it fits, returning True on success."""
    best_score: Optional[tuple[float, float, float, float, int]] = None
    best_choice: Optional[tuple[_Rect, float, float, bool]] = None

    orientations: list[tuple[float, float, bool]] = [(item.dx_mm, item.dy_mm, False)]
    if abs(item.dx_mm - item.dy_mm) > SQUARE_ROTATION_EPS_MM:
      orientations.append((item.dy_mm, item.dx_mm, True))

    for free in self.free_rects:
      for placed_dx, placed_dy, rotated in orientations:
        padded_w = placed_dx + self.gap_mm
        padded_h = placed_dy + self.gap_mm
        if padded_w > free.w + RECT_FIT_EPS_MM or padded_h > free.h + RECT_FIT_EPS_MM:
          continue
        if (
            free.x + placed_dx > self.usable_x + BED_FIT_EPS_MM
            or free.y + placed_dy > self.usable_y + BED_FIT_EPS_MM
        ):
          continue

        leftover_x = max(0.0, free.w - padded_w)
        leftover_y = max(0.0, free.h - padded_h)
        short_side = min(leftover_x, leftover_y)
        long_side = max(leftover_x, leftover_y)
        score = (short_side, long_side, free.y, free.x, 1 if rotated else 0)
        if best_score is None or score < best_score:
          best_score = score
          best_choice = (_Rect(free.x, free.y, padded_w, padded_h), placed_dx, placed_dy, rotated)

    if best_choice is None:
      return False

    used_rect, placed_dx, placed_dy, rotated = best_choice
    self._split_free_rects(used_rect)
    self.entries.append(
        _PlacedEntry(
            item=item,
            rect_x=used_rect.x,
            rect_y=used_rect.y,
            placed_dx=placed_dx,
            placed_dy=placed_dy,
            rotated_z_90=rotated,
        )
    )
    return True

  def _split_free_rects(self, used: _Rect) -> None:
    """Split all free rectangles intersecting ``used`` and prune contained rectangles."""
    next_free: list[_Rect] = []
    used_x2 = used.x + used.w
    used_y2 = used.y + used.h

    for free in self.free_rects:
      free_x2 = free.x + free.w
      free_y2 = free.y + free.h
      disjoint = (
          used.x >= free_x2 - RECT_SPLIT_EPS_MM
          or used_x2 <= free.x + RECT_SPLIT_EPS_MM
          or used.y >= free_y2 - RECT_SPLIT_EPS_MM
          or used_y2 <= free.y + RECT_SPLIT_EPS_MM
      )
      if disjoint:
        next_free.append(free)
        continue

      if used.x > free.x + MIN_FREE_RECT_MM:
        next_free.append(_Rect(free.x, free.y, used.x - free.x, free.h))
      if used_x2 < free_x2 - MIN_FREE_RECT_MM:
        next_free.append(_Rect(used_x2, free.y, free_x2 - used_x2, free.h))
      if used.y > free.y + MIN_FREE_RECT_MM:
        next_free.append(_Rect(free.x, free.y, free.w, used.y - free.y))
      if used_y2 < free_y2 - MIN_FREE_RECT_MM:
        next_free.append(_Rect(free.x, used_y2, free.w, free_y2 - used_y2))

    self.free_rects = _prune_contained_rects(next_free)


def _prune_contained_rects(rects: list[_Rect]) -> list[_Rect]:
  """Drop any free rectangle wholly enclosed by another free rectangle."""
  pruned: list[_Rect] = []
  for idx, rect in enumerate(rects):
    contained = False
    r_x2 = rect.x + rect.w
    r_y2 = rect.y + rect.h
    for other_idx, other in enumerate(rects):
      if idx == other_idx:
        continue
      if (
          rect.x >= other.x - RECT_SPLIT_EPS_MM
          and rect.y >= other.y - RECT_SPLIT_EPS_MM
          and r_x2 <= other.x + other.w + RECT_SPLIT_EPS_MM
          and r_y2 <= other.y + other.h + RECT_SPLIT_EPS_MM
      ):
        contained = True
        break
    if not contained:
      pruned.append(rect)
  return pruned


@dataclass
class _OpenPlate:
  """Mutable packing state for one build plate under construction."""

  bin_packer: _MaxRectsBin
  has_brim: bool
  effective_gap_mm: float
  reduced_margin_warning: Optional[str] = None


def _sanitize_slug(text: str) -> str:
  """Convert a group label into a clean ASCII filename slug."""
  slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
  return slug or "general"


def _load_binary_stl_mesh(stl_path: Path) -> _LoadedStlMesh:
  """Read binary STL triangle records and compute exact vertex bounding dimensions."""
  data = stl_path.read_bytes()
  if len(data) < STL_PREFIX_BYTES:
    raise ValueError(f"Binary STL {stl_path} is truncated (< {STL_PREFIX_BYTES} bytes)")
  tri_count = int(struct.unpack("<I", data[STL_HEADER_BYTES:STL_PREFIX_BYTES])[0])
  expected_size = STL_PREFIX_BYTES + (STL_TRIANGLE_BYTES * tri_count)
  if tri_count <= 0 or len(data) != expected_size:
    raise ValueError(
        f"Binary STL {stl_path} has invalid byte length {len(data)} for {tri_count} triangles"
    )
  records = np.frombuffer(
      data,
      dtype=BINARY_STL_DTYPE,
      offset=STL_PREFIX_BYTES,
      count=tri_count,
  ).copy()

  verts = np.concatenate((records["v0"], records["v1"], records["v2"]), axis=0).astype(np.float64)
  min_xyz = np.min(verts, axis=0)
  max_xyz = np.max(verts, axis=0)
  dims = (
      float(max_xyz[0] - min_xyz[0]),
      float(max_xyz[1] - min_xyz[1]),
      float(max_xyz[2] - min_xyz[2]),
  )
  return _LoadedStlMesh(records=records, dims_xyz_mm=dims)


def _fits_physical_bed(
    dx: float,
    dy: float,
    dz: float,
    bed_dims: tuple[float, float, float],
) -> tuple[bool, bool]:
  """Return ``(fits_bed, requires_z90_rotation)`` against physical bed dimensions."""
  bx, by, bz = bed_dims
  if dz > bz + BED_FIT_EPS_MM:
    return False, False
  if dx <= bx + BED_FIT_EPS_MM and dy <= by + BED_FIT_EPS_MM:
    return True, False
  if dy <= bx + BED_FIT_EPS_MM and dx <= by + BED_FIT_EPS_MM:
    return True, True
  return False, False


def _fits_usable_xy(dx: float, dy: float, usable_x: float, usable_y: float) -> bool:
  """Return True if ``(dx, dy)`` fits inside ``(usable_x, usable_y)`` with 0 or 90-deg Z rotation."""
  if usable_x <= 0.0 or usable_y <= 0.0:
    return False
  return (
      (dx <= usable_x + BED_FIT_EPS_MM and dy <= usable_y + BED_FIT_EPS_MM)
      or (dy <= usable_x + BED_FIT_EPS_MM and dx <= usable_y + BED_FIT_EPS_MM)
  )


def _item_sort_key(item: _PackItem) -> tuple[float, float, str, int]:
  """Deterministic largest-first sort key for 2D bin packing."""
  return (-max(item.dx_mm, item.dy_mm), -(item.dx_mm * item.dy_mm), item.part_name, item.instance_index)


def _try_repack_all(
    items: Sequence[_PackItem],
    usable_x: float,
    usable_y: float,
    gap_mm: float,
) -> Optional[_MaxRectsBin]:
  """Attempt to pack all ``items`` onto a single plate with ``gap_mm`` spacing."""
  candidate_bin = _MaxRectsBin(usable_x=usable_x, usable_y=usable_y, gap_mm=gap_mm)
  for item in sorted(items, key=_item_sort_key):
    if not candidate_bin.try_place(item):
      return None
  return candidate_bin


def _pack_group_items(
    items: Sequence[_PackItem],
    bed_dims: tuple[float, float, float],
    margin_mm: float,
    gap_mm: float,
    brim_gap_mm: float,
) -> tuple[list[_OpenPlate], list[str]]:
  """Pack one logical group of items into one or more build plates."""
  bx, by, _ = bed_dims
  usable_x = bx - (2.0 * margin_mm)
  usable_y = by - (2.0 * margin_mm)

  open_plates: list[_OpenPlate] = []
  oversized_single_plates: list[_OpenPlate] = []
  unplaced: list[str] = []
  packable: list[_PackItem] = []

  for item in sorted(items, key=_item_sort_key):
    fits_phys, needs_rot90 = _fits_physical_bed(item.dx_mm, item.dy_mm, item.dz_mm, bed_dims)
    if not fits_phys:
      unplaced.append(item.part_name)
      continue

    if not _fits_usable_xy(item.dx_mm, item.dy_mm, usable_x, usable_y):
      placed_dx = item.dy_mm if needs_rot90 else item.dx_mm
      placed_dy = item.dx_mm if needs_rot90 else item.dy_mm
      solo_bin = _MaxRectsBin(usable_x=bx, usable_y=by, gap_mm=0.0)
      solo_bin.entries.append(
          _PlacedEntry(
              item=item,
              rect_x=0.0,
              rect_y=0.0,
              placed_dx=placed_dx,
              placed_dy=placed_dy,
              rotated_z_90=needs_rot90,
          )
      )
      actual_margin = min(HALF_FACTOR * (bx - placed_dx), HALF_FACTOR * (by - placed_dy))
      eff_gap = brim_gap_mm if item.brim_recommended else gap_mm
      oversized_single_plates.append(
          _OpenPlate(
              bin_packer=solo_bin,
              has_brim=item.brim_recommended,
              effective_gap_mm=eff_gap,
              reduced_margin_warning=(
                  f"Part '{item.part_name}' exceeds configured {margin_mm:.1f} mm edge margin; "
                  f"placed solo centered with {max(0.0, actual_margin):.1f} mm margin."
              ),
          )
      )
      continue

    packable.append(item)

  for item in packable:
    placed = False
    for plate in open_plates:
      if plate.has_brim or not item.brim_recommended:
        if plate.bin_packer.try_place(item):
          placed = True
          break
      else:
        existing_items = [entry.item for entry in plate.bin_packer.entries]
        repacked = _try_repack_all(
            items=[*existing_items, item],
            usable_x=usable_x,
            usable_y=usable_y,
            gap_mm=brim_gap_mm,
        )
        if repacked is not None:
          plate.bin_packer = repacked
          plate.has_brim = True
          plate.effective_gap_mm = brim_gap_mm
          placed = True
          break

    if not placed:
      plate_gap = brim_gap_mm if item.brim_recommended else gap_mm
      new_bin = _MaxRectsBin(usable_x=usable_x, usable_y=usable_y, gap_mm=plate_gap)
      if not new_bin.try_place(item):
        unplaced.append(item.part_name)
        continue
      open_plates.append(
          _OpenPlate(
              bin_packer=new_bin,
              has_brim=item.brim_recommended,
              effective_gap_mm=plate_gap,
          )
      )

  return oversized_single_plates + open_plates, unplaced


def _verify_plate_placements(
    placements: Sequence[PlatePlacement],
    bed_dims: tuple[float, float, float],
    effective_gap_mm: float,
) -> None:
  """Assert zero XY overlap, minimum gap separation, and bed containment for a plate."""
  bx, by, bz = bed_dims
  for pl in placements:
    px, py, pz = pl.placed_dims_mm
    if (
        pl.bed_min_x_mm < -BED_FIT_EPS_MM
        or pl.bed_min_y_mm < -BED_FIT_EPS_MM
        or pl.bed_min_x_mm + px > bx + BED_FIT_EPS_MM
        or pl.bed_min_y_mm + py > by + BED_FIT_EPS_MM
        or pz > bz + BED_FIT_EPS_MM
    ):
      raise RuntimeError(
          f"Packed part '{pl.part_name}' exceeds bed bounds {bed_dims} at "
          f"({pl.bed_min_x_mm:.2f}, {pl.bed_min_y_mm:.2f}) with dims {pl.placed_dims_mm}"
      )

  count = len(placements)
  for i in range(count):
    a = placements[i]
    ax1, ay1 = a.bed_min_x_mm, a.bed_min_y_mm
    ax2, ay2 = ax1 + a.placed_dims_mm[0], ay1 + a.placed_dims_mm[1]
    for j in range(i + 1, count):
      b = placements[j]
      bx1, by1 = b.bed_min_x_mm, b.bed_min_y_mm
      bx2, by2 = bx1 + b.placed_dims_mm[0], by1 + b.placed_dims_mm[1]
      sep_x = max(ax1 - bx2, bx1 - ax2)
      sep_y = max(ay1 - by2, by1 - ay2)
      if max(sep_x, sep_y) < effective_gap_mm - LAYOUT_VERIFY_TOL_MM:
        raise RuntimeError(
            f"Plate packing separation violation between '{a.part_name}' and '{b.part_name}': "
            f"sep_x={sep_x:.3f} mm, sep_y={sep_y:.3f} mm < gap={effective_gap_mm:.3f} mm"
        )


def _transform_stl_records(
    records: np.ndarray,
    *,
    rotated_z_90: bool,
    target_min_x: float,
    target_min_y: float,
) -> np.ndarray:
  """Copy binary STL records, optionally rotate +90 deg around Z, and translate min XY."""
  out = records.copy()
  if rotated_z_90:
    for field in ("normal", "v0", "v1", "v2"):
      old_x = out[field][:, 0].copy()
      old_y = out[field][:, 1].copy()
      out[field][:, 0] = -old_y
      out[field][:, 1] = old_x

  min_x = float(
      min(np.min(out["v0"][:, 0]), np.min(out["v1"][:, 0]), np.min(out["v2"][:, 0]))
  )
  min_y = float(
      min(np.min(out["v0"][:, 1]), np.min(out["v1"][:, 1]), np.min(out["v2"][:, 1]))
  )
  dx = float(target_min_x) - min_x
  dy = float(target_min_y) - min_y

  for field in ("v0", "v1", "v2"):
    out[field][:, 0] = (out[field][:, 0].astype(np.float64) + dx).astype(np.float32)
    out[field][:, 1] = (out[field][:, 1].astype(np.float64) + dy).astype(np.float32)

  return out


def _write_merged_plate_stl(
    plate_stl_path: Path,
    header_text: str,
    entries: Sequence[_PlacedEntry],
    mesh_cache: Mapping[Path, _LoadedStlMesh],
    stl_shift_x: float,
    stl_shift_y: float,
) -> tuple[int, bool]:
  """Merge all placed part meshes into one binary STL centered around XY=(0, 0) on Z=0."""
  chunks: list[np.ndarray] = []
  for entry in entries:
    mesh = mesh_cache[entry.item.stl_path]
    transformed = _transform_stl_records(
        mesh.records,
        rotated_z_90=entry.rotated_z_90,
        target_min_x=entry.rect_x + stl_shift_x,
        target_min_y=entry.rect_y + stl_shift_y,
    )
    chunks.append(transformed)

  merged = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
  header = header_text.encode("ascii", errors="ignore")[:STL_HEADER_BYTES].ljust(
      STL_HEADER_BYTES, b" "
  )
  tri_count = int(merged.shape[0])

  plate_stl_path.parent.mkdir(parents=True, exist_ok=True)
  with plate_stl_path.open("wb") as handle:
    handle.write(header)
    handle.write(struct.pack("<I", tri_count))
    handle.write(np.ascontiguousarray(merged, dtype=BINARY_STL_DTYPE).tobytes())

  verified_tris, watertight, *_ = _read_binary_stl_arrays(plate_stl_path)
  return verified_tris, bool(watertight and verified_tris == tri_count)


def _aggregate_filament_mass_by_type(items: Sequence[_PackItem]) -> dict[str, float]:
  """Sum estimated filament mass (g) grouped by filament type in deterministic order."""
  totals: dict[str, float] = {}
  for item in items:
    totals[item.filament_type] = totals.get(item.filament_type, 0.0) + item.estimated_mass_g
  return {fil: round(totals[fil], 3) for fil in sorted(totals.keys())}


def _build_item_groups(
    items: Sequence[_PackItem],
    group_mode: str,
) -> list[tuple[str, str, list[_PackItem]]]:
  """Partition expanded items into ordered ``(group_label, file_slug, group_items)`` tuples."""
  if group_mode == "fewest":
    return [("all", "", list(items))]

  if group_mode == "slicer":
    distinct_filaments = {item.filament_type for item in items}
    multi_filament = len(distinct_filaments) > 1
    buckets: dict[tuple[str, str, bool], list[_PackItem]] = {}
    for item in items:
      key = (item.filament_type, item.support_recommendation, item.brim_recommended)
      buckets.setdefault(key, []).append(item)

    ordered_keys = sorted(
        buckets.keys(),
        key=lambda k: (k[0], SUPPORT_PRIORITY_ORDER.get(k[1], 99), int(k[2]), k[1]),
    )
    groups: list[tuple[str, str, list[_PackItem]]] = []
    for filament_type, support_rec, brim_rec in ordered_keys:
      sup_slug = SUPPORT_FILE_SLUGS.get(support_rec, _sanitize_slug(support_rec))
      brim_slug = "brim" if brim_rec else "nobrim"
      if multi_filament:
        fil_slug = _sanitize_slug(filament_type)
        slug = f"{fil_slug}_{sup_slug}_{brim_slug}"
        label = f"filament={filament_type}, supports={support_rec}, brim={'yes' if brim_rec else 'no'}"
      else:
        slug = f"{sup_slug}_{brim_slug}"
        label = f"supports={support_rec}, brim={'yes' if brim_rec else 'no'}"
      groups.append((label, slug, buckets[(filament_type, support_rec, brim_rec)]))
    return groups

  if group_mode == "category":
    cat_buckets: dict[str, list[_PackItem]] = {}
    for item in items:
      cat_buckets.setdefault(item.category, []).append(item)

    ordered_cats = sorted(
        cat_buckets.keys(),
        key=lambda c: (CATEGORY_DISPLAY_ORDER.get(c, 99), c),
    )
    return [(cat, _sanitize_slug(cat), cat_buckets[cat]) for cat in ordered_cats]

  raise ValueError(
      f"plate_group_mode must be one of {VALID_PLATE_GROUP_MODES}, got {group_mode!r}"
  )


def pack_print_kit_plates(
    parts_reports: Sequence[PartPrintReport],
    output_dir: Path | str,
    printer_bed_mm: tuple[float, float, float],
    *,
    part_categories: Optional[Mapping[str, str]] = None,
    group_mode: str = DEFAULT_PLATE_GROUP_MODE,
    margin_mm: float = DEFAULT_PLATE_MARGIN_MM,
    gap_mm: float = DEFAULT_PLATE_GAP_MM,
    brim_gap_mm: Optional[float] = None,
) -> PlatePackingResult:
  """Pack kit part STLs onto minimal printer build plates and export merged binary meta-STLs.

  Args:
    parts_reports: Per-part printability reports with already-exported binary STL paths.
    output_dir: Directory where merged ``plate_*.stl`` files will be written.
    printer_bed_mm: Printer build volume ``(X, Y, Z)`` in mm.
    part_categories: Optional mapping from ``part_name`` to subsystem category.
    group_mode: Grouping strategy: ``"fewest"``, ``"slicer"``, or ``"category"``.
    margin_mm: Minimum clearance from printer bed edges in mm.
    gap_mm: Minimum XY gap between neighboring parts on standard plates in mm.
    brim_gap_mm: Minimum XY gap on plates containing brim-recommended parts
      (defaults to ``max(gap_mm, 12.0)`` mm).

  Returns:
    A ``PlatePackingResult`` containing per-plate reports, placements, and watertightness gates.
  """
  if group_mode not in VALID_PLATE_GROUP_MODES:
    raise ValueError(
        f"group_mode must be one of {VALID_PLATE_GROUP_MODES}, got {group_mode!r}"
    )
  if len(printer_bed_mm) != 3 or any(float(d) <= 0.0 for d in printer_bed_mm):
    raise ValueError(f"printer_bed_mm must contain 3 positive dimensions, got {printer_bed_mm}")
  margin_val = float(margin_mm)
  gap_val = float(gap_mm)
  if margin_val < MIN_PLATE_MARGIN_MM:
    raise ValueError(f"margin_mm must be >= {MIN_PLATE_MARGIN_MM}, got {margin_val}")
  if gap_val < MIN_PLATE_GAP_MM:
    raise ValueError(f"gap_mm must be >= {MIN_PLATE_GAP_MM}, got {gap_val}")

  effective_brim_gap = (
      max(gap_val, DEFAULT_BRIM_PLATE_GAP_MM)
      if brim_gap_mm is None
      else max(gap_val, float(brim_gap_mm))
  )

  bed_dims = (float(printer_bed_mm[0]), float(printer_bed_mm[1]), float(printer_bed_mm[2]))
  bx, by, _ = bed_dims
  bed_area_mm2 = bx * by
  plates_dir = Path(output_dir)
  plates_dir.mkdir(parents=True, exist_ok=True)
  categories = part_categories or {}

  mesh_cache: dict[Path, _LoadedStlMesh] = {}
  expanded_items: list[_PackItem] = []

  for rep in parts_reports:
    stl_path = Path(rep.stl_path)
    if stl_path not in mesh_cache:
      mesh_cache[stl_path] = _load_binary_stl_mesh(stl_path)
    mesh = mesh_cache[stl_path]

    # Use conservative envelope of report bbox and actual tessellated STL bbox.
    dx_mm = max(float(rep.oriented_dims_mm[0]), mesh.dims_xyz_mm[0])
    dy_mm = max(float(rep.oriented_dims_mm[1]), mesh.dims_xyz_mm[1])
    dz_mm = max(float(rep.oriented_dims_mm[2]), mesh.dims_xyz_mm[2])
    category = categories.get(rep.part_name, "general")

    for inst_idx in range(1, int(rep.quantity) + 1):
      expanded_items.append(
          _PackItem(
              part_name=rep.part_name,
              instance_index=inst_idx,
              category=category,
              stl_path=stl_path,
              dx_mm=dx_mm,
              dy_mm=dy_mm,
              dz_mm=dz_mm,
              support_recommendation=rep.support_recommendation,
              brim_recommended=bool(rep.brim_recommended),
              filament_type=str(rep.filament_type),
              estimated_mass_g=float(rep.estimated_mass_g),
              estimated_print_time_min=float(rep.estimated_print_time_min),
          )
      )

  groups = _build_item_groups(expanded_items, group_mode=group_mode)
  plate_reports: list[BuildPlateReport] = []
  all_unplaced: list[str] = []
  plate_counter = 0

  for group_label, group_slug, group_items in groups:
    packed_plates, group_unplaced = _pack_group_items(
        items=group_items,
        bed_dims=bed_dims,
        margin_mm=margin_val,
        gap_mm=gap_val,
        brim_gap_mm=effective_brim_gap,
    )
    all_unplaced.extend(group_unplaced)

    for open_plate in packed_plates:
      entries = open_plate.bin_packer.entries
      if not entries:
        continue

      plate_counter += 1
      plate_name = (
          f"plate_{plate_counter:02d}_{group_slug}"
          if group_slug
          else f"plate_{plate_counter:02d}"
      )
      plate_stl_path = plates_dir / f"{plate_name}.stl"

      occ_min_x = min(e.rect_x for e in entries)
      occ_max_x = max(e.rect_x + e.placed_dx for e in entries)
      occ_min_y = min(e.rect_y for e in entries)
      occ_max_y = max(e.rect_y + e.placed_dy for e in entries)
      used_x = occ_max_x - occ_min_x
      used_y = occ_max_y - occ_min_y
      used_z = max(e.item.dz_mm for e in entries)

      stl_shift_x = -(occ_min_x + (HALF_FACTOR * used_x))
      stl_shift_y = -(occ_min_y + (HALF_FACTOR * used_y))
      bed_origin_x = (HALF_FACTOR * bx) + stl_shift_x
      bed_origin_y = (HALF_FACTOR * by) + stl_shift_y

      placements = tuple(
          PlatePlacement(
              part_name=e.item.part_name,
              instance_index=e.item.instance_index,
              category=e.item.category,
              rotated_z_90=e.rotated_z_90,
              bed_min_x_mm=round(e.rect_x + bed_origin_x, 3),
              bed_min_y_mm=round(e.rect_y + bed_origin_y, 3),
              placed_dims_mm=(
                  round(e.placed_dx, 3),
                  round(e.placed_dy, 3),
                  round(e.item.dz_mm, 3),
              ),
              support_recommendation=e.item.support_recommendation,
              brim_recommended=e.item.brim_recommended,
              filament_type=e.item.filament_type,
              estimated_mass_g=round(e.item.estimated_mass_g, 3),
              estimated_print_time_min=round(e.item.estimated_print_time_min, 2),
          )
          for e in entries
      )

      _verify_plate_placements(
          placements=placements,
          bed_dims=bed_dims,
          effective_gap_mm=open_plate.effective_gap_mm if len(placements) > 1 else 0.0,
      )

      tri_count, watertight = _write_merged_plate_stl(
          plate_stl_path=plate_stl_path,
          header_text=f"OpenEngine-1C Print Kit {plate_name} ({group_label})",
          entries=entries,
          mesh_cache=mesh_cache,
          stl_shift_x=stl_shift_x,
          stl_shift_y=stl_shift_y,
      )

      part_footprint_area = sum(e.placed_dx * e.placed_dy for e in entries)
      bed_util_pct = (part_footprint_area / bed_area_mm2) * PCT_SCALE if bed_area_mm2 > 0.0 else 0.0
      actual_margin_mm = max(
          0.0,
          min(
              HALF_FACTOR * (bx - used_x),
              HALF_FACTOR * (by - used_y),
          ),
      )

      support_levels = {e.item.support_recommendation for e in entries}
      plate_support = max(
          support_levels,
          key=lambda s: SUPPORT_PRIORITY_ORDER.get(s, 99),
      )
      plate_brim = any(e.item.brim_recommended for e in entries)
      has_non_brim = any(not e.item.brim_recommended for e in entries)

      plate_filament_mass = _aggregate_filament_mass_by_type([e.item for e in entries])
      warnings: list[str] = []
      if open_plate.reduced_margin_warning is not None:
        warnings.append(open_plate.reduced_margin_warning)
      if len(plate_filament_mass) > 1:
        fil_list = ", ".join(plate_filament_mass.keys())
        warnings.append(
            f"Multiple filament types on plate ({fil_list}); use --plate-group slicer "
            "to separate filaments onto distinct plates."
        )
      if len(support_levels) > 1:
        warnings.append(
            f"Mixed support requirements on plate; enable '{plate_support}' in slicer "
            "or use --plate-group slicer to separate."
        )
      if plate_brim and has_non_brim:
        warnings.append(
            f"Mixed brim parts on plate ({open_plate.effective_gap_mm:.1f} mm gap applied); "
            "enable brim in slicer or use --plate-group slicer to separate."
        )

      plate_reports.append(
          BuildPlateReport(
              plate_index=plate_counter,
              plate_name=plate_name,
              group_label=group_label,
              stl_path=plate_stl_path,
              placements=placements,
              piece_count=len(placements),
              unique_part_count=len({p.part_name for p in placements}),
              used_dims_mm=(round(used_x, 3), round(used_y, 3), round(used_z, 3)),
              bed_dims_mm=bed_dims,
              effective_margin_mm=round(actual_margin_mm, 3),
              effective_gap_mm=round(open_plate.effective_gap_mm, 3),
              bed_utilization_pct=round(bed_util_pct, 2),
              fits_bed=(
                  used_x <= bx + BED_FIT_EPS_MM
                  and used_y <= by + BED_FIT_EPS_MM
                  and used_z <= bed_dims[2] + BED_FIT_EPS_MM
              ),
              support_recommendation=plate_support,
              brim_recommended=plate_brim,
              estimated_mass_g=round(sum(e.item.estimated_mass_g for e in entries), 3),
              filament_mass_by_type_g=plate_filament_mass,
              estimated_print_time_min=round(sum(e.item.estimated_print_time_min for e in entries), 2),
              triangle_count=tri_count,
              watertight=watertight,
              warnings=tuple(warnings),
          )
      )

  unique_unplaced = tuple(dict.fromkeys(all_unplaced))
  total_placed = sum(p.piece_count for p in plate_reports)
  placed_filament_totals: dict[str, float] = {}
  for plate in plate_reports:
    for fil_name, mass_g in plate.filament_mass_by_type_g.items():
      placed_filament_totals[fil_name] = placed_filament_totals.get(fil_name, 0.0) + mass_g
  total_filament_mass_by_type = {
      fil: round(placed_filament_totals[fil], 3) for fil in sorted(placed_filament_totals.keys())
  }
  all_watertight = all(p.watertight for p in plate_reports)
  all_fit_bed = (not unique_unplaced) and all(p.fits_bed for p in plate_reports)

  return PlatePackingResult(
      group_mode=group_mode,
      margin_mm=margin_val,
      gap_mm=gap_val,
      brim_gap_mm=effective_brim_gap,
      plates_dir=plates_dir,
      plates=tuple(plate_reports),
      unplaced_parts=unique_unplaced,
      total_placed_pieces=total_placed,
      filament_mass_by_type_g=total_filament_mass_by_type,
      all_watertight=all_watertight,
      all_fit_bed=all_fit_bed,
  )
