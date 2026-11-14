"""Shared geometry, collision-checking, and material-mass helpers for CAD assemblies."""

from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from build123d import BoundBox, Part
from openengine.fea.calculix import MATERIAL_CATALOG

AABB_MARGIN_MM: float = 0.02
COLLISION_VOL_TOL_MM3: float = 1.0e-2
TONNE_TO_KG: float = 1.0e3

EXTRA_DENSITY_KG_MM3: Dict[str, float] = {
    "VITON_FLUOROELASTOMER": 1.80e-6,
    "SAE_10W40_OIL": 0.86e-6,
    "ALUMINA_STEEL_PLUG": 3.80e-6,
    "DIECAST_CARB_ALLOY": 2.90e-6,
    "PHENOLIC_ELECTRONICS": 1.65e-6,
}


def overlap_volume_mm3(
    part_a: Part,
    part_b: Part,
    bbox_a: Optional[BoundBox] = None,
    bbox_b: Optional[BoundBox] = None,
    margin_mm: float = AABB_MARGIN_MM,
) -> float:
  """Return exact B-Rep intersection volume in mm^3 using a fast AABB pre-filter."""
  ba = bbox_a if bbox_a is not None else part_a.bounding_box()
  bb = bbox_b if bbox_b is not None else part_b.bounding_box()
  if (
      ba.max.X + margin_mm < bb.min.X
      or bb.max.X + margin_mm < ba.min.X
      or ba.max.Y + margin_mm < bb.min.Y
      or bb.max.Y + margin_mm < ba.min.Y
      or ba.max.Z + margin_mm < bb.min.Z
      or bb.max.Z + margin_mm < ba.min.Z
  ):
    return 0.0

  inter = part_a.intersect(part_b)
  if not inter:
    return 0.0
  if hasattr(inter, "volume"):
    return float(inter.volume)
  return float(sum(shape.volume for shape in inter))


def find_pairwise_collisions(
    parts: Mapping[str, Part],
    vol_tol_mm3: float = COLLISION_VOL_TOL_MM3,
    skip_pairs: Optional[frozenset[frozenset[str]]] = None,
) -> Tuple[float, List[str]]:
  """Check all unique solid pairs inside parts and return (max_overlap_mm3, hit_labels)."""
  names = list(parts.keys())
  bboxes = {name: parts[name].bounding_box() for name in names}
  ignored = skip_pairs or frozenset()
  max_ov = 0.0
  hits: List[str] = []

  for idx, name_a in enumerate(names):
    for name_b in names[idx + 1 :]:
      if frozenset((name_a, name_b)) in ignored:
        continue
      ov = overlap_volume_mm3(parts[name_a], parts[name_b], bboxes[name_a], bboxes[name_b])
      max_ov = max(max_ov, ov)
      if ov > vol_tol_mm3:
        hits.append(f"{name_a}-{name_b}: {ov:.3f} mm^3")
  return max_ov, hits


def find_cross_collisions(
    group_a: Mapping[str, Part],
    group_b: Mapping[str, Part],
    vol_tol_mm3: float = COLLISION_VOL_TOL_MM3,
) -> Tuple[float, List[str]]:
  """Check all pairs between two disjoint part dicts and return (max_overlap_mm3, hit_labels)."""
  bboxes_a = {name: part.bounding_box() for name, part in group_a.items()}
  bboxes_b = {name: part.bounding_box() for name, part in group_b.items()}
  max_ov = 0.0
  hits: List[str] = []

  for name_a, part_a in group_a.items():
    for name_b, part_b in group_b.items():
      if name_a == name_b:
        continue
      ov = overlap_volume_mm3(part_a, part_b, bboxes_a[name_a], bboxes_b[name_b])
      max_ov = max(max_ov, ov)
      if ov > vol_tol_mm3:
        hits.append(f"{name_a}-{name_b}: {ov:.3f} mm^3")
  return max_ov, hits


def part_mass_kg(part: Part, material_key: str) -> float:
  """Compute B-Rep solid mass in kg from FEA MATERIAL_CATALOG or auxiliary density table."""
  vol_mm3 = float(part.volume)
  if material_key in MATERIAL_CATALOG:
    rho_kg_mm3 = MATERIAL_CATALOG[material_key].density_tonne_mm3 * TONNE_TO_KG
    return vol_mm3 * rho_kg_mm3
  if material_key in EXTRA_DENSITY_KG_MM3:
    return vol_mm3 * EXTRA_DENSITY_KG_MM3[material_key]
  raise KeyError(f"Unknown material key for mass rollup: {material_key}")


def sum_group_mass_kg(parts: Mapping[str, Part], material_map: Mapping[str, str], keys: Iterable[str]) -> float:
  """Sum B-Rep mass in kg over selected keys present in parts."""
  return float(sum(part_mass_kg(parts[k], material_map[k]) for k in keys if k in parts))
