"""Binary STL export, watertightness verification, and FDM printability analysis."""

from dataclasses import dataclass
import math
from pathlib import Path
import struct
import tempfile
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from build123d import Part, export_stl
import numpy as np

from openengine.printing.materials import DEFAULT_FILAMENT_NAME
from openengine.printing.orientation import (
    BED_FIT_EPS_MM,
    DEFAULT_MAX_OVERHANG_DEG,
    DEFAULT_PRINTER_BED_MM,
    apply_shrinkage_compensation,
    compute_mesh_print_metrics,
    orient_part_for_printing,
)

if TYPE_CHECKING:
  from openengine.printing.tolerances import PrintKitConfig

STL_HEADER_BYTES: int = 80
STL_COUNT_BYTES: int = 4
STL_PREFIX_BYTES: int = STL_HEADER_BYTES + STL_COUNT_BYTES
STL_TRIANGLE_BYTES: int = 50
MM3_PER_CM3: float = 1000.0
PCT_DIVISOR: float = 100.0
HALF_FACTOR: float = 0.5
THIRD_FACTOR: float = 1.0 / 3.0
TET_VOLUME_DIVISOR: float = 6.0

DEFAULT_LINEAR_DEFLECTION_MM: float = 0.05
DEFAULT_ANGULAR_DEFLECTION_RAD: float = 0.2
DEFAULT_DENSITY_G_CM3: float = 1.27
DEFAULT_INFILL_PCT: float = 20.0
DEFAULT_WALL_LOOPS: int = 3
DEFAULT_NOZZLE_DIAM_MM: float = 0.4
DEFAULT_LAYER_HEIGHT_MM: float = 0.2

MIN_OVERHANG_AREA_MM2: float = 2.0
MIN_OVERHANG_PCT: float = 0.5
OCCLUDED_SUPPORT_FRAC_THRESHOLD: float = 0.15
MAX_BUILDPLATE_ONLY_OVERHANG_PCT: float = 35.0
MAX_OCCLUSION_SAMPLES: int = 400
RAY_BELOW_MIN_Z_MM: float = 0.25
RAY_SELF_CLEARANCE_Z_MM: float = 0.30
BARYCENTRIC_DET_EPS: float = 1.0e-12
MIN_TRI_TWICE_AREA_MM2: float = 1.0e-12

SLENDER_HEIGHT_BASE_RATIO: float = 2.5
LOW_CONTACT_AREA_BRIM_MM2: float = 100.0
LOW_CONTACT_MIN_HEIGHT_MM: float = 15.0
SPARSE_CONTACT_FOOTPRINT_FRAC: float = 0.10
SPARSE_CONTACT_MIN_HEIGHT_MM: float = 20.0
MIN_BASE_DIM_MM: float = 0.1

BASE_VOLUMETRIC_FLOW_CM3_MIN: float = 0.18
MIN_FLOW_CM3_MIN: float = 0.03
LAYER_OVERHEAD_MIN: float = 0.05
PRINT_SETUP_OVERHEAD_MIN: float = 1.0
BUILDPLATE_SUPPORT_TIME_MULT: float = 1.15
TREE_SUPPORT_TIME_MULT: float = 1.25

BINARY_STL_DTYPE: np.dtype = np.dtype([
    ("normal", "<f4", (3,)),
    ("v0", "<f4", (3,)),
    ("v1", "<f4", (3,)),
    ("v2", "<f4", (3,)),
    ("attr", "<u2"),
])


@dataclass
class PartPrintReport:
  """Printability, STL mesh, mass, and print-time report for a single printable Part."""

  part_name: str
  stl_path: Path
  quantity: int
  oriented_dims_mm: Tuple[float, float, float]
  fits_bed: bool
  contact_area_mm2: float
  overhang_area_mm2: float
  overhang_pct: float
  support_recommendation: str
  brim_recommended: bool
  volume_cm3: float
  estimated_mass_g: float
  estimated_print_time_min: float
  triangle_count: int
  watertight: bool
  warnings: List[str]
  filament_type: str = DEFAULT_FILAMENT_NAME


def _read_binary_stl_arrays(
    stl_path: Path,
) -> Tuple[int, bool, np.ndarray, np.ndarray, np.ndarray, float]:
  """Parse binary STL, check closed-manifold watertightness, and return triangle arrays + signed volume."""
  data = stl_path.read_bytes()
  if len(data) < STL_PREFIX_BYTES:
    empty = np.zeros((0, 3), dtype=np.float64)
    return 0, False, empty, empty, empty, 0.0

  tri_count = int(struct.unpack("<I", data[STL_HEADER_BYTES:STL_PREFIX_BYTES])[0])
  expected_size = STL_PREFIX_BYTES + (STL_TRIANGLE_BYTES * tri_count)
  if tri_count <= 0 or len(data) != expected_size:
    empty = np.zeros((0, 3), dtype=np.float64)
    return tri_count, False, empty, empty, empty, 0.0

  records = np.frombuffer(data, dtype=BINARY_STL_DTYPE, offset=STL_PREFIX_BYTES, count=tri_count)
  v0_f32 = records["v0"].copy()
  v1_f32 = records["v1"].copy()
  v2_f32 = records["v2"].copy()

  v0 = v0_f32.astype(np.float64)
  v1 = v1_f32.astype(np.float64)
  v2 = v2_f32.astype(np.float64)
  if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
    return tri_count, False, v0, v1, v2, 0.0

  cross = np.cross(v1 - v0, v2 - v0)
  signed_vol_mm3 = float(np.sum(np.einsum("ij,ij->i", v0, cross))) / TET_VOLUME_DIVISOR

  # Canonicalize -0.0 to +0.0 so identical float32 coordinates share identical byte keys.
  v0_f32[v0_f32 == 0.0] = 0.0
  v1_f32[v1_f32 == 0.0] = 0.0
  v2_f32[v2_f32 == 0.0] = 0.0

  edge_a = np.concatenate((v0_f32, v1_f32, v2_f32), axis=0)
  edge_b = np.concatenate((v1_f32, v2_f32, v0_f32), axis=0)
  non_degenerate = np.any(edge_a != edge_b, axis=1)
  edge_a = edge_a[non_degenerate]
  edge_b = edge_b[non_degenerate]

  swap = (
      (edge_a[:, 0] > edge_b[:, 0])
      | (
          (edge_a[:, 0] == edge_b[:, 0])
          & (
              (edge_a[:, 1] > edge_b[:, 1])
              | ((edge_a[:, 1] == edge_b[:, 1]) & (edge_a[:, 2] > edge_b[:, 2]))
          )
      )
  )
  e_min = np.where(swap[:, None], edge_b, edge_a)
  e_max = np.where(swap[:, None], edge_a, edge_b)
  edge_keys = np.ascontiguousarray(np.concatenate((e_min, e_max), axis=1)).view("V24").ravel()
  _, counts = np.unique(edge_keys, return_counts=True)

  watertight = (counts.size > 0) and bool(np.all(counts >= 2)) and (signed_vol_mm3 > 0.0)
  return tri_count, watertight, v0, v1, v2, signed_vol_mm3


def export_part_stl(
    part: Part,
    stl_path: Path,
    linear_deflection: float = DEFAULT_LINEAR_DEFLECTION_MM,
    angular_deflection: float = DEFAULT_ANGULAR_DEFLECTION_RAD,
) -> Tuple[Path, int, bool]:
  """Export a build123d Part to binary STL (mm) and verify triangle count & watertightness."""
  target_path = Path(stl_path)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  exported = export_stl(
      part,
      str(target_path),
      tolerance=linear_deflection,
      angular_tolerance=angular_deflection,
      ascii_format=False,
  )
  if not exported or not target_path.is_file():
    return target_path, 0, False

  tri_count, watertight, _, _, _, _ = _read_binary_stl_arrays(target_path)
  return target_path, tri_count, watertight


def _occluded_overhang_fraction(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    max_overhang_deg: float,
) -> float:
  """Estimate fraction of overhanging area blocked below by lower part geometry."""
  cross = np.cross(v1 - v0, v2 - v0)
  twice_area = np.linalg.norm(cross, axis=1)
  valid = twice_area > MIN_TRI_TWICE_AREA_MM2
  if not np.any(valid):
    return 0.0

  v0_v = v0[valid]
  v1_v = v1[valid]
  v2_v = v2[valid]
  areas = HALF_FACTOR * twice_area[valid]
  nz = cross[valid, 2] / twice_area[valid]

  min_z = float(min(np.min(v0_v[:, 2]), np.min(v1_v[:, 2]), np.min(v2_v[:, 2])))
  z_max = np.maximum(np.maximum(v0_v[:, 2], v1_v[:, 2]), v2_v[:, 2]) - min_z
  z_cen = THIRD_FACTOR * (v0_v[:, 2] + v1_v[:, 2] + v2_v[:, 2]) - min_z

  clamped_deg = max(0.0, min(89.99, float(max_overhang_deg)))
  overhang_nz_limit = -math.sin(math.radians(clamped_deg)) - 1.0e-3
  oh_mask = (z_max > 0.2) & (nz < overhang_nz_limit)
  oh_indices = np.where(oh_mask)[0]
  if oh_indices.size == 0:
    return 0.0

  if oh_indices.size > MAX_OCCLUSION_SAMPLES:
    top_order = np.argsort(-areas[oh_indices])[:MAX_OCCLUSION_SAMPLES]
    oh_indices = oh_indices[top_order]

  tri_xmin = np.minimum(np.minimum(v0_v[:, 0], v1_v[:, 0]), v2_v[:, 0])
  tri_xmax = np.maximum(np.maximum(v0_v[:, 0], v1_v[:, 0]), v2_v[:, 0])
  tri_ymin = np.minimum(np.minimum(v0_v[:, 1], v1_v[:, 1]), v2_v[:, 1])
  tri_ymax = np.maximum(np.maximum(v0_v[:, 1], v1_v[:, 1]), v2_v[:, 1])

  blocker_mask = z_cen > RAY_BELOW_MIN_Z_MM
  blocked_area = 0.0
  sampled_area = float(np.sum(areas[oh_indices]))

  for idx in oh_indices:
    cx = THIRD_FACTOR * (v0_v[idx, 0] + v1_v[idx, 0] + v2_v[idx, 0])
    cy = THIRD_FACTOR * (v0_v[idx, 1] + v1_v[idx, 1] + v2_v[idx, 1])
    cz = z_cen[idx]
    cand = (
        blocker_mask
        & (z_cen < cz - RAY_SELF_CLEARANCE_Z_MM)
        & (tri_xmin <= cx)
        & (cx <= tri_xmax)
        & (tri_ymin <= cy)
        & (cy <= tri_ymax)
    )
    if not np.any(cand):
      continue

    ax, ay = v0_v[cand, 0], v0_v[cand, 1]
    bx, by = v1_v[cand, 0], v1_v[cand, 1]
    cx_tri, cy_tri = v2_v[cand, 0], v2_v[cand, 1]
    vec0_x, vec0_y = cx_tri - ax, cy_tri - ay
    vec1_x, vec1_y = bx - ax, by - ay
    vec2_x, vec2_y = cx - ax, cy - ay

    dot00 = (vec0_x * vec0_x) + (vec0_y * vec0_y)
    dot01 = (vec0_x * vec1_x) + (vec0_y * vec1_y)
    dot02 = (vec0_x * vec2_x) + (vec0_y * vec2_y)
    dot11 = (vec1_x * vec1_x) + (vec1_y * vec1_y)
    dot12 = (vec1_x * vec2_x) + (vec1_y * vec2_y)
    denom = (dot00 * dot11) - (dot01 * dot01)
    valid_tri = np.abs(denom) > BARYCENTRIC_DET_EPS
    if not np.any(valid_tri):
      continue

    u = np.zeros_like(denom)
    v = np.zeros_like(denom)
    u[valid_tri] = ((dot11[valid_tri] * dot02[valid_tri]) - (dot01[valid_tri] * dot12[valid_tri])) / denom[valid_tri]
    v[valid_tri] = ((dot00[valid_tri] * dot12[valid_tri]) - (dot01[valid_tri] * dot02[valid_tri])) / denom[valid_tri]
    if np.any(valid_tri & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0)):
      blocked_area += float(areas[idx])

  return blocked_area / max(sampled_area, 1.0e-12)


def _recommend_supports(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    overhang_area_mm2: float,
    overhang_pct: float,
    max_overhang_deg: float,
) -> str:
  """Select support strategy: 'none', 'buildplate_only', or 'tree_supports'."""
  if overhang_area_mm2 <= MIN_OVERHANG_AREA_MM2 and overhang_pct <= MIN_OVERHANG_PCT:
    return "none"
  occluded_frac = _occluded_overhang_fraction(v0, v1, v2, max_overhang_deg=max_overhang_deg)
  if occluded_frac > OCCLUDED_SUPPORT_FRAC_THRESHOLD or overhang_pct > MAX_BUILDPLATE_ONLY_OVERHANG_PCT:
    return "tree_supports"
  return "buildplate_only"


def _recommend_brim(
    dims_xyz: Tuple[float, float, float],
    contact_area_mm2: float,
) -> bool:
  """Return True when slender aspect ratio or small bed contact warrants a brim."""
  dx, dy, dz = dims_xyz
  min_base = max(min(dx, dy), MIN_BASE_DIM_MM)
  eq_contact_width = math.sqrt(max(contact_area_mm2, 0.0)) if contact_area_mm2 > 0.0 else min_base
  effective_base = max(min(min_base, eq_contact_width), MIN_BASE_DIM_MM)

  if (dz / effective_base) > SLENDER_HEIGHT_BASE_RATIO:
    return True
  if contact_area_mm2 < LOW_CONTACT_AREA_BRIM_MM2 and dz > LOW_CONTACT_MIN_HEIGHT_MM:
    return True
  footprint = max(dx * dy, MIN_BASE_DIM_MM)
  if (contact_area_mm2 < SPARSE_CONTACT_FOOTPRINT_FRAC * footprint) and dz > SPARSE_CONTACT_MIN_HEIGHT_MM:
    return True
  return False


def _resolve_process_settings(
    cfg: Optional["PrintKitConfig"],
    max_overhang_deg: Optional[float],
    printer_bed_mm: Optional[Tuple[float, float, float]],
    density_g_cm3: Optional[float],
    infill_pct: Optional[float],
    wall_loops: Optional[int],
    nozzle_diameter_mm: Optional[float],
    layer_height_mm: Optional[float],
) -> Tuple[float, Tuple[float, float, float], float, float, int, float, float]:
  """Resolve filament and printer parameters from optional PrintKitConfig and overrides."""
  resolved_oh = DEFAULT_MAX_OVERHANG_DEG
  resolved_bed = DEFAULT_PRINTER_BED_MM
  resolved_density = DEFAULT_DENSITY_G_CM3
  resolved_infill = DEFAULT_INFILL_PCT
  resolved_walls = DEFAULT_WALL_LOOPS
  resolved_nozzle = DEFAULT_NOZZLE_DIAM_MM
  resolved_layer = DEFAULT_LAYER_HEIGHT_MM

  if cfg is not None:
    from openengine.printing.materials import get_filament_profile

    profile = get_filament_profile(cfg.filament)
    resolved_oh = float(profile.max_overhang_deg)
    resolved_density = float(profile.density_g_cm3)
    resolved_bed = tuple(float(v) for v in cfg.printer_bed_mm)  # type: ignore[assignment]
    resolved_infill = float(cfg.infill_pct)
    resolved_walls = int(cfg.wall_loops)
    resolved_nozzle = float(cfg.nozzle_diameter_mm)
    resolved_layer = float(cfg.layer_height_mm)

  return (
      float(max_overhang_deg) if max_overhang_deg is not None else resolved_oh,
      tuple(float(v) for v in printer_bed_mm) if printer_bed_mm is not None else resolved_bed,  # type: ignore[return-value]
      float(density_g_cm3) if density_g_cm3 is not None else resolved_density,
      float(infill_pct) if infill_pct is not None else resolved_infill,
      int(wall_loops) if wall_loops is not None else resolved_walls,
      float(nozzle_diameter_mm) if nozzle_diameter_mm is not None else resolved_nozzle,
      float(layer_height_mm) if layer_height_mm is not None else resolved_layer,
  )


def analyze_part_printability(
    part: Part,
    part_name: str,
    stl_path: Optional[Union[Path, str]] = None,
    cfg: Optional["PrintKitConfig"] = None,
    quantity: int = 1,
    *,
    max_overhang_deg: Optional[float] = None,
    printer_bed_mm: Optional[Tuple[float, float, float]] = None,
    density_g_cm3: Optional[float] = None,
    infill_pct: Optional[float] = None,
    wall_loops: Optional[int] = None,
    nozzle_diameter_mm: Optional[float] = None,
    layer_height_mm: Optional[float] = None,
    linear_deflection: float = DEFAULT_LINEAR_DEFLECTION_MM,
    angular_deflection: float = DEFAULT_ANGULAR_DEFLECTION_RAD,
    orient_first: bool = False,
    preferred_rotation_xyz_deg: Optional[Tuple[float, float, float]] = None,
    compensate_shrinkage: bool = False,
    filament_type: Optional[str] = None,
) -> PartPrintReport:
  """Export a Part to binary STL and evaluate printability, supports, mass, and print time."""
  resolved_filament_type = (
      str(filament_type)
      if filament_type is not None
      else (cfg.filament_profile.name if cfg is not None else DEFAULT_FILAMENT_NAME)
  )
  (
      eff_oh_deg,
      eff_bed_mm,
      eff_density,
      eff_infill_pct,
      eff_wall_loops,
      eff_nozzle_mm,
      eff_layer_mm,
  ) = _resolve_process_settings(
      cfg=cfg,
      max_overhang_deg=max_overhang_deg,
      printer_bed_mm=printer_bed_mm,
      density_g_cm3=density_g_cm3,
      infill_pct=infill_pct,
      wall_loops=wall_loops,
      nozzle_diameter_mm=nozzle_diameter_mm,
      layer_height_mm=layer_height_mm,
  )

  target_part = part
  if orient_first:
    target_part = orient_part_for_printing(
        part=part,
        part_name=part_name,
        max_overhang_deg=eff_oh_deg,
        printer_bed_mm=eff_bed_mm,
        preferred_rotation_xyz_deg=preferred_rotation_xyz_deg,
    ).oriented_part

  nominal_brep_vol_mm3 = float(target_part.volume)
  if compensate_shrinkage and cfg is not None:
    from openengine.printing.tolerances import shrinkage_compensation_xyz

    target_part = apply_shrinkage_compensation(
        target_part,
        shrinkage_compensation_xyz(cfg.filament_profile),
    )

  if stl_path is None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_stl = Path(tmp_dir) / f"{part_name}.stl"
      _, tri_count, watertight = export_part_stl(
          target_part,
          tmp_stl,
          linear_deflection=linear_deflection,
          angular_deflection=angular_deflection,
      )
      _, _, v0, v1, v2, signed_vol_mm3 = _read_binary_stl_arrays(tmp_stl)
    report_stl_path = Path(f"{part_name}.stl")
  else:
    report_stl_path, tri_count, watertight = export_part_stl(
        target_part,
        Path(stl_path),
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    _, _, v0, v1, v2, signed_vol_mm3 = _read_binary_stl_arrays(report_stl_path)

  bbox = target_part.bounding_box()
  oriented_dims_mm = (float(bbox.size.X), float(bbox.size.Y), float(bbox.size.Z))
  fits_bed = (
      oriented_dims_mm[0] <= eff_bed_mm[0] + BED_FIT_EPS_MM
      and oriented_dims_mm[1] <= eff_bed_mm[1] + BED_FIT_EPS_MM
      and oriented_dims_mm[2] <= eff_bed_mm[2] + BED_FIT_EPS_MM
  )

  metrics = compute_mesh_print_metrics(v0, v1, v2, max_overhang_deg=eff_oh_deg)
  support_rec = _recommend_supports(
      v0=v0,
      v1=v1,
      v2=v2,
      overhang_area_mm2=metrics.overhang_area_mm2,
      overhang_pct=metrics.overhang_pct,
      max_overhang_deg=eff_oh_deg,
  )
  brim_rec = _recommend_brim(oriented_dims_mm, metrics.contact_area_mm2)

  vol_cm3 = (nominal_brep_vol_mm3 if nominal_brep_vol_mm3 > 0.0 else max(0.0, signed_vol_mm3)) / MM3_PER_CM3

  shell_thickness_mm = max(eff_wall_loops, 1) * max(eff_nozzle_mm, 0.1)
  shell_vol_cm3 = min(vol_cm3, (metrics.total_area_mm2 * shell_thickness_mm) / MM3_PER_CM3)
  core_vol_cm3 = max(0.0, vol_cm3 - shell_vol_cm3)
  infill_frac = max(0.0, min(1.0, eff_infill_pct / PCT_DIVISOR))
  effective_filament_cm3 = shell_vol_cm3 + (core_vol_cm3 * infill_frac)
  estimated_mass_g = effective_filament_cm3 * eff_density

  flow_scale = (eff_layer_mm / DEFAULT_LAYER_HEIGHT_MM) * (eff_nozzle_mm / DEFAULT_NOZZLE_DIAM_MM)
  eff_flow_cm3_min = max(MIN_FLOW_CM3_MIN, BASE_VOLUMETRIC_FLOW_CM3_MIN * flow_scale)
  num_layers = oriented_dims_mm[2] / max(eff_layer_mm, 0.05)
  base_time_min = PRINT_SETUP_OVERHEAD_MIN + (effective_filament_cm3 / eff_flow_cm3_min) + (num_layers * LAYER_OVERHEAD_MIN)
  if support_rec == "tree_supports":
    base_time_min *= TREE_SUPPORT_TIME_MULT
  elif support_rec == "buildplate_only":
    base_time_min *= BUILDPLATE_SUPPORT_TIME_MULT

  warnings_list: List[str] = []
  if tri_count == 0:
    warnings_list.append("STL export produced zero triangles")
  if not watertight:
    warnings_list.append("STL mesh is not closed-manifold watertight")
  if not fits_bed:
    warnings_list.append(
        f"Oriented dimensions ({oriented_dims_mm[0]:.1f}x{oriented_dims_mm[1]:.1f}x{oriented_dims_mm[2]:.1f} mm) "
        f"exceed printer bed ({eff_bed_mm[0]:.1f}x{eff_bed_mm[1]:.1f}x{eff_bed_mm[2]:.1f} mm)"
    )
  if brim_rec:
    warnings_list.append(
        f"Slender height-to-base ratio or low contact area ({metrics.contact_area_mm2:.1f} mm^2); brim recommended"
    )
  if support_rec == "tree_supports":
    warnings_list.append(
        f"Occluded/extensive overhangs ({metrics.overhang_area_mm2:.1f} mm^2, {metrics.overhang_pct:.1f}%) "
        f"above {eff_oh_deg:.1f} deg require tree supports"
    )
  elif support_rec == "buildplate_only":
    warnings_list.append(
        f"Overhangs ({metrics.overhang_area_mm2:.1f} mm^2, {metrics.overhang_pct:.1f}%) "
        f"above {eff_oh_deg:.1f} deg need buildplate-only supports"
    )

  return PartPrintReport(
      part_name=part_name,
      stl_path=report_stl_path,
      quantity=int(quantity),
      oriented_dims_mm=oriented_dims_mm,
      fits_bed=fits_bed,
      contact_area_mm2=metrics.contact_area_mm2,
      overhang_area_mm2=metrics.overhang_area_mm2,
      overhang_pct=metrics.overhang_pct,
      support_recommendation=support_rec,
      brim_recommended=brim_rec,
      volume_cm3=vol_cm3,
      estimated_mass_g=estimated_mass_g,
      estimated_print_time_min=base_time_min,
      triangle_count=tri_count,
      watertight=watertight,
      warnings=warnings_list,
      filament_type=resolved_filament_type,
  )
