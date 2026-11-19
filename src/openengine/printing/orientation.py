"""Automatic build-plate orientation and baked B-Rep placement for FDM printing."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

from build123d import BoundBox, Part, Solid
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Trsf
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

DEFAULT_MAX_OVERHANG_DEG: float = 45.0
DEFAULT_PRINT_VOLUME_MM: Tuple[float, float, float] = (250.0, 220.0, 270.0)

PLATE_CONTACT_Z_TOL_MM: float = 0.05
BUILDPLATE_IGNORE_Z_TOL_MM: float = 0.2
CONTACT_NORMAL_Z_THRESHOLD: float = -0.95
OVERHANG_SIN_MARGIN: float = 1.0e-3
MIN_TRIANGLE_TWICE_AREA_MM2: float = 1.0e-12
MIN_VOLUME_MM3: float = 1.0e-6
VOLUME_FIT_EPS_MM: float = 1.0e-5
TET_CENTROID_FACTOR: float = 0.25
PCT_SCALE: float = 100.0
HALF_FACTOR: float = 0.5
THIRD_FACTOR: float = 1.0 / 3.0
RIGHT_ANGLE_DEG: float = 90.0
DIAGONAL_Z_ROT_DEG: float = 45.0

EVAL_LINEAR_DEFLECTION_MM: float = 0.05
EVAL_ANGULAR_DEFLECTION_RAD: float = 0.2

VOLUME_OVERFLOW_PENALTY: float = 1.0e9
OVERHANG_AREA_WEIGHT: float = 3.0
CONTACT_AREA_WEIGHT: float = 1.5
LOW_CONTACT_FRACTION_THRESHOLD: float = 1.0e-3
NO_CONTACT_PENALTY_FACTOR: float = 1.0
CG_Z_WEIGHT: float = 0.15
HEIGHT_WEIGHT: float = 0.10
ROTATION_TIEBREAK_WEIGHT: float = 1.0e-4
Z_ROTATION_PENALTY: float = 1.0
DIAGONAL_ROTATION_PENALTY: float = 2.0

ORTHOGONAL_BASE_ROTATIONS_DEG: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (90.0, 0.0, 0.0),
    (-90.0, 0.0, 0.0),
    (180.0, 0.0, 0.0),
    (0.0, 90.0, 0.0),
    (0.0, -90.0, 0.0),
)


@dataclass
class OrientedPartResult:
  """Result of orienting and placing a B-Rep Part onto the printer build plate."""

  part_name: str
  oriented_part: Part
  rotation_xyz_deg: Tuple[float, float, float]
  translation_xyz_mm: Tuple[float, float, float]
  oriented_dims_xyz_mm: Tuple[float, float, float]
  contact_area_mm2: float
  overhang_area_mm2: float
  overhang_pct: float
  rationale: str


@dataclass(frozen=True)
class MeshPrintMetrics:
  """Surface and volumetric printability metrics computed from a triangle mesh."""

  contact_area_mm2: float
  overhang_area_mm2: float
  overhang_pct: float
  total_area_mm2: float
  cg_z_mm: float


def _snap_cos_sin_deg(angle_deg: float) -> Tuple[float, float]:
  """Return exact (cos, sin) for 90-degree multiples to avoid trigonometric skew."""
  mod_90 = abs(angle_deg % RIGHT_ANGLE_DEG)
  if mod_90 < 1.0e-9 or abs(mod_90 - RIGHT_ANGLE_DEG) < 1.0e-9:
    quarter = int(round(angle_deg / RIGHT_ANGLE_DEG)) % 4
    return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quarter]
  rad = math.radians(angle_deg)
  return math.cos(rad), math.sin(rad)


def _rotation_matrix_extrinsic_xyz(rotation_xyz_deg: Tuple[float, float, float]) -> np.ndarray:
  """Return 3x3 rotation matrix R = Rz * Ry * Rx for extrinsic XYZ degrees."""
  cx, sx = _snap_cos_sin_deg(rotation_xyz_deg[0])
  cy, sy = _snap_cos_sin_deg(rotation_xyz_deg[1])
  cz, sz = _snap_cos_sin_deg(rotation_xyz_deg[2])

  mat_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
  mat_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
  mat_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
  return mat_z @ mat_y @ mat_x


def _build_extrinsic_xyz_trsf(
    rotation_xyz_deg: Tuple[float, float, float],
    translation_xyz_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> gp_Trsf:
  """Build an OCCT gp_Trsf applying extrinsic XYZ rotations followed by translation."""
  rot = _rotation_matrix_extrinsic_xyz(rotation_xyz_deg)
  tx, ty, tz = (float(v) for v in translation_xyz_mm)
  trsf = gp_Trsf()
  trsf.SetValues(
      float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2]), tx,
      float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2]), ty,
      float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2]), tz,
  )
  return trsf


def bake_part_transform(
    part: Part,
    rotation_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation_xyz_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Part:
  """Bake rigid rotation and translation directly into B-Rep solid geometry."""
  trsf = _build_extrinsic_xyz_trsf(rotation_xyz_deg, translation_xyz_mm)
  solids = list(part.solids())
  if not solids:
    raw = part.wrapped.Located(TopLoc_Location())
    combined = trsf * part.wrapped.Location().Transformation()
    transformed = BRepBuilderAPI_Transform(raw, combined, True).Shape()
    out = Part(transformed)
  else:
    baked_solids = []
    for sol in solids:
      combined = trsf * sol.wrapped.Location().Transformation()
      raw = sol.wrapped.Located(TopLoc_Location())
      transformed = BRepBuilderAPI_Transform(raw, combined, True).Shape()
      baked_solids.append(Solid(TopoDS.Solid(transformed)))
    out = Part(children=baked_solids)

  if part.label:
    out.label = part.label
  if part.color is not None:
    out.color = part.color
  return out


def extract_part_triangles(
    part: Part,
    linear_deflection: float = EVAL_LINEAR_DEFLECTION_MM,
    angular_deflection: float = EVAL_ANGULAR_DEFLECTION_RAD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Tessellate a Part in memory and return (v0, v1, v2) arrays of shape (N, 3)."""
  mesh_copy = BRepBuilderAPI_Transform(part.wrapped, gp_Trsf(), True).Shape()
  mesher = BRepMesh_IncrementalMesh(
      mesh_copy,
      linear_deflection,
      True,
      angular_deflection,
      True,
  )
  mesher.Perform()

  vertices: list[Tuple[float, float, float]] = []
  tri_indices: list[Tuple[int, int, int]] = []
  offset = 0

  for face in Part(mesh_copy).faces():
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face.wrapped, loc)
    if triangulation is None:
      continue
    trsf = loc.Transformation()
    node_count = triangulation.NbNodes()
    for node_idx in range(1, node_count + 1):
      pt = triangulation.Node(node_idx).Transformed(trsf)
      vertices.append((pt.X(), pt.Y(), pt.Z()))

    is_reversed = face.wrapped.Orientation() == TopAbs_REVERSED
    for tri in triangulation.Triangles():
      i1, i2, i3 = tri.Get()
      if is_reversed:
        tri_indices.append((offset + i1 - 1, offset + i3 - 1, offset + i2 - 1))
      else:
        tri_indices.append((offset + i1 - 1, offset + i2 - 1, offset + i3 - 1))
    offset += node_count

  if not tri_indices:
    empty = np.zeros((0, 3), dtype=np.float64)
    return empty, empty, empty

  pts = np.asarray(vertices, dtype=np.float64)
  idx = np.asarray(tri_indices, dtype=np.int64)
  return pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]


def compute_mesh_print_metrics(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG,
) -> MeshPrintMetrics:
  """Evaluate build-plate contact area, overhang area, and CG Z relative to mesh min_z."""
  if len(v0) == 0:
    return MeshPrintMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

  cross = np.cross(v1 - v0, v2 - v0)
  twice_area = np.linalg.norm(cross, axis=1)
  valid = twice_area > MIN_TRIANGLE_TWICE_AREA_MM2
  if not np.any(valid):
    return MeshPrintMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

  v0_v = v0[valid]
  v1_v = v1[valid]
  v2_v = v2[valid]
  areas = HALF_FACTOR * twice_area[valid]
  normals = cross[valid] / twice_area[valid, None]
  nz = normals[:, 2]

  min_z = float(min(np.min(v0_v[:, 2]), np.min(v1_v[:, 2]), np.min(v2_v[:, 2])))
  tri_max_z = np.maximum(np.maximum(v0_v[:, 2], v1_v[:, 2]), v2_v[:, 2]) - min_z

  contact_mask = (tri_max_z <= PLATE_CONTACT_Z_TOL_MM) & (nz <= CONTACT_NORMAL_Z_THRESHOLD)
  contact_area = float(np.sum(areas[contact_mask]))

  clamped_deg = max(0.0, min(89.99, float(max_overhang_deg)))
  overhang_nz_limit = -math.sin(math.radians(clamped_deg)) - OVERHANG_SIN_MARGIN
  on_buildplate = tri_max_z <= BUILDPLATE_IGNORE_Z_TOL_MM
  overhang_mask = (~on_buildplate) & (nz < overhang_nz_limit)
  overhang_area = float(np.sum(areas[overhang_mask]))

  total_area = float(np.sum(areas))
  overhang_pct = (PCT_SCALE * overhang_area / total_area) if total_area > 0.0 else 0.0

  z0 = v0_v[:, 2] - min_z
  z1 = v1_v[:, 2] - min_z
  z2 = v2_v[:, 2] - min_z
  shifted_cross = np.cross(np.column_stack((v1_v[:, 0], v1_v[:, 1], z1)), np.column_stack((v2_v[:, 0], v2_v[:, 1], z2)))
  shifted_v0 = np.column_stack((v0_v[:, 0], v0_v[:, 1], z0))
  tet_vol6 = np.einsum("ij,ij->i", shifted_v0, shifted_cross)
  total_vol6 = float(np.sum(tet_vol6))
  if total_vol6 > MIN_VOLUME_MM3:
    tet_cg_z = TET_CENTROID_FACTOR * (z0 + z1 + z2)
    cg_z = float(np.sum(tet_vol6 * tet_cg_z) / total_vol6)
  else:
    tri_cg_z = THIRD_FACTOR * (z0 + z1 + z2)
    cg_z = float(np.sum(areas * tri_cg_z) / max(total_area, 1.0e-12))

  return MeshPrintMetrics(
      contact_area_mm2=contact_area,
      overhang_area_mm2=overhang_area,
      overhang_pct=overhang_pct,
      total_area_mm2=total_area,
      cg_z_mm=max(0.0, cg_z),
  )


def _bbox_dims_tuple(bbox: BoundBox) -> Tuple[float, float, float]:
  """Convert a build123d BoundBox size to a plain (dx, dy, dz) float tuple."""
  size = bbox.size
  return (float(size.X), float(size.Y), float(size.Z))


def _fits_print_volume_dims(
    dims_xyz: Tuple[float, float, float],
    print_volume_mm: Tuple[float, float, float],
) -> bool:
  """Return True if (dx, dy, dz) fits within print_volume_mm with numeric epsilon."""
  return (
      dims_xyz[0] <= print_volume_mm[0] + VOLUME_FIT_EPS_MM
      and dims_xyz[1] <= print_volume_mm[1] + VOLUME_FIT_EPS_MM
      and dims_xyz[2] <= print_volume_mm[2] + VOLUME_FIT_EPS_MM
  )


def _score_orientation_candidate(
    metrics: MeshPrintMetrics,
    dims_xyz: Tuple[float, float, float],
    print_volume_mm: Tuple[float, float, float],
    rotation_xyz_deg: Tuple[float, float, float],
    extra_z_penalty: float = 0.0,
) -> float:
  """Compute minimization cost balancing print-volume fit, overhangs, contact area, and stability."""
  dx, dy, dz = dims_xyz
  vx, vy, vz = print_volume_mm
  overflow = max(0.0, dx - vx) + max(0.0, dy - vy) + max(0.0, dz - vz)
  volume_penalty = VOLUME_OVERFLOW_PENALTY * (overflow + 1.0) if overflow > VOLUME_FIT_EPS_MM else 0.0

  total_area = max(metrics.total_area_mm2, 1.0e-6)
  no_contact_penalty = (
      NO_CONTACT_PENALTY_FACTOR * total_area
      if metrics.contact_area_mm2 < (LOW_CONTACT_FRACTION_THRESHOLD * total_area)
      else 0.0
  )
  max_dim = max(dx, dy, dz, 1.0)
  sum_dims = max(dx + dy + dz, 1.0)
  stability_penalty = (
      CG_Z_WEIGHT * total_area * (metrics.cg_z_mm / max_dim)
      + HEIGHT_WEIGHT * total_area * (dz / sum_dims)
  )
  rot_tiebreak = ROTATION_TIEBREAK_WEIGHT * sum(abs(a) for a in rotation_xyz_deg)

  return (
      volume_penalty
      + (OVERHANG_AREA_WEIGHT * metrics.overhang_area_mm2)
      - (CONTACT_AREA_WEIGHT * metrics.contact_area_mm2)
      + no_contact_penalty
      + stability_penalty
      + rot_tiebreak
      + extra_z_penalty
  )


def _format_rationale(
    part_name: str,
    rotation_xyz_deg: Tuple[float, float, float],
    dims_xyz: Tuple[float, float, float],
    metrics: MeshPrintMetrics,
    fits_print_volume: bool,
    is_preferred: bool,
    rotated_z_for_volume: bool,
) -> str:
  """Generate a concise human-readable explanation of the chosen orientation."""
  prefix = "Preferred orientation" if is_preferred else "Selected orientation"
  vol_note = "fits print volume" if fits_print_volume else "EXCEEDS print volume"
  z_note = " (rotated +90 deg Z for XY print-volume fit)" if rotated_z_for_volume else ""
  return (
      f"{prefix} for {part_name}: rot={rotation_xyz_deg} deg{z_note}, "
      f"dims=({dims_xyz[0]:.1f}x{dims_xyz[1]:.1f}x{dims_xyz[2]:.1f}) mm ({vol_note}), "
      f"contact={metrics.contact_area_mm2:.1f} mm^2, "
      f"overhang={metrics.overhang_area_mm2:.1f} mm^2 ({metrics.overhang_pct:.1f}%)"
  )


def _evaluate_candidate_list(
    part: Part,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    candidates: Sequence[Tuple[Tuple[float, float, float], float]],
    max_overhang_deg: float,
    print_volume_mm: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], MeshPrintMetrics, bool, float]:
  """Evaluate rotation candidates and return (best_rot, best_dims, best_metrics, fits_vol, best_score)."""
  best_score = math.inf
  best_rot = candidates[0][0]
  best_dims = (0.0, 0.0, 0.0)
  best_metrics = MeshPrintMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
  best_fits = False

  # Cache exact B-Rep XY/Z sizes for each unique (rx, ry) + rz%180 pair
  bbox_cache: dict[Tuple[float, float, float], Tuple[float, float, float]] = {}

  for rot_deg, extra_penalty in candidates:
    rx, ry, rz = rot_deg
    base_key = (rx, ry, rz % 180.0)
    if base_key not in bbox_cache:
      if math.isclose(rz % 180.0, RIGHT_ANGLE_DEG, abs_tol=1.0e-6) and (rx, ry, 0.0) in bbox_cache:
        dx0, dy0, dz0 = bbox_cache[(rx, ry, 0.0)]
        bbox_cache[base_key] = (dy0, dx0, dz0)
      else:
        rotated_part = bake_part_transform(part, rot_deg)
        bbox_cache[base_key] = _bbox_dims_tuple(rotated_part.bounding_box())

    dims_xyz = bbox_cache[base_key]
    fits_vol = _fits_print_volume_dims(dims_xyz, print_volume_mm)

    rot_mat = _rotation_matrix_extrinsic_xyz(rot_deg)
    v0_r = v0 @ rot_mat.T
    v1_r = v1 @ rot_mat.T
    v2_r = v2 @ rot_mat.T
    metrics = compute_mesh_print_metrics(v0_r, v1_r, v2_r, max_overhang_deg=max_overhang_deg)

    score = _score_orientation_candidate(
        metrics=metrics,
        dims_xyz=dims_xyz,
        print_volume_mm=print_volume_mm,
        rotation_xyz_deg=rot_deg,
        extra_z_penalty=extra_penalty,
    )
    if score < best_score:
      best_score = score
      best_rot = rot_deg
      best_dims = dims_xyz
      best_metrics = metrics
      best_fits = fits_vol

  return best_rot, best_dims, best_metrics, best_fits, best_score


def orient_part_for_printing(
    part: Part,
    part_name: str,
    max_overhang_deg: float = DEFAULT_MAX_OVERHANG_DEG,
    print_volume_mm: Tuple[float, float, float] = DEFAULT_PRINT_VOLUME_MM,
    preferred_rotation_xyz_deg: Optional[Tuple[float, float, float]] = None,
) -> OrientedPartResult:
  """Orient a Part onto the Z=0 build plate, centered in XY, with baked B-Rep geometry."""
  if not part.faces():
    raise ValueError(f"Part {part_name!r} has no faces to orient for printing")

  v0, v1, v2 = extract_part_triangles(part)

  if preferred_rotation_xyz_deg is not None:
    rx, ry, rz = (float(a) for a in preferred_rotation_xyz_deg)
    pref_rot = (rx, ry, rz)
    pref_z90 = (rx, ry, rz + RIGHT_ANGLE_DEG)
    best_rot, _, _, fits_vol, _ = _evaluate_candidate_list(
        part=part,
        v0=v0,
        v1=v1,
        v2=v2,
        candidates=((pref_rot, 0.0), (pref_z90, Z_ROTATION_PENALTY)),
        max_overhang_deg=max_overhang_deg,
        print_volume_mm=print_volume_mm,
    )
    is_preferred = True
    rotated_z_for_volume = best_rot == pref_z90
  else:
    base_candidates = [(rot, 0.0) for rot in ORTHOGONAL_BASE_ROTATIONS_DEG]
    z90_candidates = [((rx, ry, RIGHT_ANGLE_DEG), Z_ROTATION_PENALTY) for rx, ry, _ in ORTHOGONAL_BASE_ROTATIONS_DEG]
    best_rot, _, _, fits_vol, _ = _evaluate_candidate_list(
        part=part,
        v0=v0,
        v1=v1,
        v2=v2,
        candidates=base_candidates + z90_candidates,
        max_overhang_deg=max_overhang_deg,
        print_volume_mm=print_volume_mm,
    )
    if not fits_vol:
      diag_candidates = [
          ((rx, ry, DIAGONAL_Z_ROT_DEG), DIAGONAL_ROTATION_PENALTY)
          for rx, ry, _ in ORTHOGONAL_BASE_ROTATIONS_DEG
      ]
      best_rot, _, _, fits_vol, _ = _evaluate_candidate_list(
          part=part,
          v0=v0,
          v1=v1,
          v2=v2,
          candidates=base_candidates + z90_candidates + diag_candidates,
          max_overhang_deg=max_overhang_deg,
          print_volume_mm=print_volume_mm,
      )
    is_preferred = False
    rotated_z_for_volume = math.isclose(best_rot[2], RIGHT_ANGLE_DEG, abs_tol=1.0e-6)

  rotated_only = bake_part_transform(part, best_rot)
  bb_rot = rotated_only.bounding_box()
  dx = -HALF_FACTOR * (float(bb_rot.min.X) + float(bb_rot.max.X))
  dy = -HALF_FACTOR * (float(bb_rot.min.Y) + float(bb_rot.max.Y))
  dz = -float(bb_rot.min.Z)
  translation_xyz_mm = (dx, dy, dz)

  oriented_part = bake_part_transform(part, best_rot, translation_xyz_mm)
  bb_final = oriented_part.bounding_box()
  oriented_dims_xyz_mm = _bbox_dims_tuple(bb_final)
  fits_final = _fits_print_volume_dims(oriented_dims_xyz_mm, print_volume_mm)

  vf0, vf1, vf2 = extract_part_triangles(oriented_part)
  final_metrics = compute_mesh_print_metrics(vf0, vf1, vf2, max_overhang_deg=max_overhang_deg)

  rationale = _format_rationale(
      part_name=part_name,
      rotation_xyz_deg=best_rot,
      dims_xyz=oriented_dims_xyz_mm,
      metrics=final_metrics,
      fits_print_volume=fits_final,
      is_preferred=is_preferred,
      rotated_z_for_volume=rotated_z_for_volume,
  )

  return OrientedPartResult(
      part_name=part_name,
      oriented_part=oriented_part,
      rotation_xyz_deg=best_rot,
      translation_xyz_mm=translation_xyz_mm,
      oriented_dims_xyz_mm=oriented_dims_xyz_mm,
      contact_area_mm2=final_metrics.contact_area_mm2,
      overhang_area_mm2=final_metrics.overhang_area_mm2,
      overhang_pct=final_metrics.overhang_pct,
      rationale=rationale,
  )


def apply_shrinkage_compensation(
    oriented_part: Part,
    shrinkage_xyz: Tuple[float, float, float],
) -> Part:
  """Apply non-uniform (sx, sy, sz) thermal shrinkage compensation in build-plate space.

  Should be applied only after ``orient_part_for_printing`` places the part at
  ``min_z == 0.0`` and XY-centered at ``(0, 0)``, so XY vs Z shrinkage matches
  actual print layer orientation without distorting nominal CAD assembly checks.
  """
  sx, sy, sz = (float(v) for v in shrinkage_xyz)
  if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
    raise ValueError(f"shrinkage_xyz factors must be positive, got {shrinkage_xyz}")
  if math.isclose(sx, 1.0, abs_tol=1.0e-9) and math.isclose(sy, 1.0, abs_tol=1.0e-9) and math.isclose(sz, 1.0, abs_tol=1.0e-9):
    return oriented_part

  compensated = oriented_part.scale((sx, sy, sz), about=(0.0, 0.0, 0.0))
  out = Part(children=list(compensated.solids())) if compensated.solids() else Part(compensated.wrapped)
  if oriented_part.label:
    out.label = oriented_part.label
  if oriented_part.color is not None:
    out.color = oriented_part.color
  return out
