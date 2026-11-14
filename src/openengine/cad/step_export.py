"""B-Rep edge-to-surface healing, material appearance styling, and STEP export utilities."""

import math
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

from build123d import Color, Compound, Part, Shape, Solid, export_step
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.Geom import Geom_BSplineCurve, Geom_Curve, Geom_Surface
from OCP.Geom2d import Geom2d_BSplineCurve, Geom2d_Curve
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec
from OCP.OCP.collections import (
    Array1_double,
    Array1_gp_Pnt,
    Array1_gp_Pnt2d,
    Array1_int,
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
)
from OCP.ShapeFix import ShapeFix_ShapeTolerance
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face

STEP_PRECISION_TOL_MM: float = 1.0e-7
EDGE_TOL_CHECK_MM: float = 1.0e-7
TARGET_EDGE_GAP_TOL_MM: float = 1.0e-8
HEALED_BSPLINE_GAP_TOL_MM: float = 2.0e-8
CHECK_SAMPLE_COUNT: int = 33
DEFAULT_GAP_SAMPLE_COUNT: int = 101
MIN_HERMITE_INTERVALS: int = 128
MAX_HERMITE_INTERVALS: int = 625
MAX_HERMITE_CHORD_MM: float = 0.15
HERMITE_DEGREE: int = 3
HERMITE_END_MULT: int = 4
HERMITE_INNER_MULT: int = 2
HERMITE_TANGENT_DIVISOR: float = 3.0
POLE_PAIR_STRIDE: int = 2
POLE_COUNT_EXTRA: int = 2
VERTEX_FIT_ITERS: int = 4
INTERIOR_COUPLED_ITERS: int = 4
FINAL_UV_ITERS: int = 2
SINGLE_STEP_ITERS: int = 1
UV_JACOBIAN_TOL: float = 1.0e-18
SINGULAR_DET_TOL: float = 1.0e-12
NORMAL_LEN_TOL: float = 1.0e-12
TANGENT_CROSS_TOL: float = 1.0e-9
TWO_FACES: int = 2
FIRST_INDEX: int = 1

SHROUD_OPACITY: float = 0.18
HOUSING_OPACITY: float = 0.25
LINER_OPACITY: float = 0.35
FLUID_OPACITY: float = 0.40
SINGLE_HOUSING_OPACITY: float = 0.45
OPAQUE_OPACITY: float = 1.0

SHROUD_PART_NAMES: frozenset[str] = frozenset({"fan_shroud"})
HOUSING_PART_NAMES: frozenset[str] = frozenset({
    "cylinder_head",
    "cylinder_block",
    "crankcase_left",
    "crankcase_right",
    "sump_pan",
    "cam_tower",
    "cam_cover",
    "right_timing_cover",
    "intake_manifold_adapter",
    "carburetor_vm26",
})
LINER_PART_NAMES: frozenset[str] = frozenset({"cylinder_liner"})
FLUID_PART_NAMES: frozenset[str] = frozenset({
    "sump_oil_pool",
    "fuel_charge",
    "intake_fluid_domain",
})

DEFAULT_PART_RGB: Tuple[float, float, float] = (0.450, 0.520, 0.580)
MATERIAL_RGB_PALETTE: Mapping[str, Tuple[float, float, float]] = {
    "ALUMINUM_6061_T6": (0.565, 0.643, 0.682),
    "ALUMINUM_7075_T6": (0.855, 0.647, 0.259),
    "ALUMINUM_4032_T6": (0.502, 0.871, 0.918),
    "STEEL_4140_PREHARD": (0.180, 0.420, 0.720),
    "STEEL_4340": (0.902, 0.318, 0.000),
    "STEEL_1144_STRESSPROOF": (0.365, 0.427, 0.494),
    "STEEL_GRADE_10_9": (0.216, 0.278, 0.310),
    "DOM_STEEL_A513_T5": (0.455, 0.518, 0.549),
    "BRONZE_SAE_660": (0.804, 0.498, 0.196),
    "VITON_FLUOROELASTOMER": (0.149, 0.651, 0.604),
    "SAE_10W40_OIL": (0.898, 0.620, 0.090),
    "ALUMINA_STEEL_PLUG": (0.925, 0.937, 0.945),
    "DIECAST_CARB_ALLOY": (0.620, 0.659, 0.643),
    "PHENOLIC_ELECTRONICS": (0.776, 0.157, 0.157),
    "FUEL_AIR_CHARGE": (0.161, 0.714, 0.965),
}


def _project_uv_newton(
    surface: Geom_Surface,
    target: gp_Pnt,
    u_coord: float,
    v_coord: float,
    n_iters: int,
) -> Tuple[float, float, gp_Pnt, gp_Vec, gp_Vec]:
  """Project a 3D point locally onto surface UV space using Gauss-Newton D1 steps."""
  p_s = gp_Pnt()
  d1u = gp_Vec()
  d1v = gp_Vec()
  for _ in range(n_iters):
    surface.D1(u_coord, v_coord, p_s, d1u, d1v)
    residual = gp_Vec(p_s, target)
    a11 = d1u.Dot(d1u)
    a12 = d1u.Dot(d1v)
    a22 = d1v.Dot(d1v)
    det = (a11 * a22) - (a12 * a12)
    if det <= UV_JACOBIAN_TOL:
      break
    b1 = residual.Dot(d1u)
    b2 = residual.Dot(d1v)
    u_coord += ((b1 * a22) - (b2 * a12)) / det
    v_coord += ((a11 * b2) - (a12 * b1)) / det
  surface.D1(u_coord, v_coord, p_s, d1u, d1v)
  return u_coord, v_coord, p_s, d1u, d1v


def _solve_surface_pair_point(
    surf1: Geom_Surface,
    surf2: Geom_Surface,
    p_init: gp_Pnt,
    uv1_init: gp_Pnt2d,
    uv2_init: gp_Pnt2d,
) -> Tuple[gp_Pnt, gp_Pnt2d, gp_Pnt2d]:
  """Converge an interior intersection sample onto both surfaces simultaneously."""
  px, py, pz = p_init.X(), p_init.Y(), p_init.Z()
  u1, v1 = uv1_init.X(), uv1_init.Y()
  u2, v2 = uv2_init.X(), uv2_init.Y()

  for _ in range(INTERIOR_COUPLED_ITERS):
    p_curr = gp_Pnt(px, py, pz)
    u1, v1, q1, du1, dv1 = _project_uv_newton(surf1, p_curr, u1, v1, SINGLE_STEP_ITERS)
    u2, v2, q2, du2, dv2 = _project_uv_newton(surf2, p_curr, u2, v2, SINGLE_STEP_ITERS)
    n1 = du1.Crossed(dv1)
    n2 = du2.Crossed(dv2)
    if n1.Magnitude() <= NORMAL_LEN_TOL or n2.Magnitude() <= NORMAL_LEN_TOL:
      break
    n1.Normalize()
    n2.Normalize()
    cos_ang = n1.Dot(n2)
    det = 1.0 - (cos_ang * cos_ang)
    if det <= SINGULAR_DET_TOL:
      break
    rhs1 = gp_Vec(p_curr, q1).Dot(n1)
    rhs2 = gp_Vec(p_curr, q2).Dot(n2)
    alpha1 = (rhs1 - (cos_ang * rhs2)) / det
    alpha2 = (rhs2 - (cos_ang * rhs1)) / det
    delta = (n1 * alpha1) + (n2 * alpha2)
    px += delta.X()
    py += delta.Y()
    pz += delta.Z()

  p_final = gp_Pnt(px, py, pz)
  u1, v1, _, _, _ = _project_uv_newton(surf1, p_final, u1, v1, FINAL_UV_ITERS)
  u2, v2, _, _, _ = _project_uv_newton(surf2, p_final, u2, v2, FINAL_UV_ITERS)
  return p_final, gp_Pnt2d(u1, v1), gp_Pnt2d(u2, v2)


def _pair_gap_at_param(
    curve3d: Geom_Curve,
    surf1: Geom_Surface,
    pc1: Geom2d_Curve,
    surf2: Optional[Geom_Surface],
    pc2: Optional[Geom2d_Curve],
    param: float,
) -> float:
  """Evaluate worst-case orthogonal distance from 3D curve to up to two bound surfaces."""
  pt_3d = curve3d.Value(param)
  uv1 = pc1.Value(param)
  _, _, pt1, _, _ = _project_uv_newton(surf1, pt_3d, uv1.X(), uv1.Y(), FINAL_UV_ITERS)
  gap = float(pt_3d.Distance(pt1))
  if surf2 is not None and pc2 is not None:
    uv2 = pc2.Value(param)
    _, _, pt2, _, _ = _project_uv_newton(surf2, pt_3d, uv2.X(), uv2.Y(), FINAL_UV_ITERS)
    gap = max(gap, float(pt_3d.Distance(pt2)))
  return gap


def _edge_faces_max_gap_mm(
    edge: TopoDS_Edge,
    faces: List[TopoDS_Face],
    sample_count: int,
    bspline_only: bool = False,
) -> float:
  """Measure worst-case distance among 3D curve and face pcurve evaluations along edge."""
  if BRep_Tool.Degenerated_s(edge) or not faces:
    return 0.0
  first, last = BRep_Tool.Range_s(edge)
  curve3d = BRep_Tool.Curve_s(edge, first, last)
  if curve3d is None or last <= first:
    return 0.0
  if bspline_only and not isinstance(curve3d, Geom_BSplineCurve):
    return 0.0

  face_bindings: List[Tuple[Geom_Surface, Geom2d_Curve]] = []
  for face in faces:
    pcurve = BRep_Tool.CurveOnSurface_s(edge, face, first, last)
    if pcurve is not None:
      face_bindings.append((BRep_Tool.Surface_s(face), pcurve))
  if not face_bindings:
    return 0.0

  surf1, pc1 = face_bindings[0]
  surf2, pc2 = face_bindings[1] if len(face_bindings) >= TWO_FACES else (None, None)
  max_gap = 0.0
  denom = float(max(sample_count - 1, 1))
  span = last - first
  for step_idx in range(sample_count):
    param = first + (span * (float(step_idx) / denom))
    max_gap = max(max_gap, _pair_gap_at_param(curve3d, surf1, pc1, surf2, pc2, param))
  return max_gap


def _edge_endpoint_positions(
    edge: TopoDS_Edge,
    curve3d: Geom_Curve,
    first: float,
    last: float,
) -> Tuple[gp_Pnt, gp_Pnt]:
  """Return topological vertex positions matched to curve3d start and end parameters."""
  vertex_points: List[gp_Pnt] = []
  exp = TopExp_Explorer(edge, TopAbs_VERTEX)
  while exp.More():
    vertex_points.append(BRep_Tool.Pnt_s(TopoDS.Vertex(exp.Current())))
    exp.Next()

  c_start = curve3d.Value(first)
  c_end = curve3d.Value(last)
  if not vertex_points:
    return c_start, c_end

  p_start = min(vertex_points, key=lambda vp: vp.Distance(c_start))
  p_end = min(vertex_points, key=lambda vp: vp.Distance(c_end))
  return p_start, p_end


def _edge_vertex_min_tolerance(edge: TopoDS_Edge) -> float:
  """Return STEP precision tolerance capped by tightest topological vertex tolerance on edge."""
  min_tol = STEP_PRECISION_TOL_MM
  exp = TopExp_Explorer(edge, TopAbs_VERTEX)
  while exp.More():
    v_tol = float(BRep_Tool.Tolerance_s(TopoDS.Vertex(exp.Current())))
    min_tol = min(min_tol, v_tol)
    exp.Next()
  return min_tol


def _solve_uv_tangent(du: gp_Vec, dv: gp_Vec, vel_3d: gp_Vec) -> Tuple[float, float]:
  """Project a 3D velocity vector onto local surface UV derivative coordinates."""
  a11 = du.Dot(du)
  a12 = du.Dot(dv)
  a22 = dv.Dot(dv)
  det = (a11 * a22) - (a12 * a12)
  if det <= UV_JACOBIAN_TOL:
    return 0.0, 0.0
  b1 = vel_3d.Dot(du)
  b2 = vel_3d.Dot(dv)
  return ((b1 * a22) - (b2 * a12)) / det, ((a11 * b2) - (a12 * b1)) / det


def _estimate_edge_chord_length_mm(curve3d: Geom_Curve, first: float, last: float) -> float:
  """Estimate curve3d arc length in mm across CHECK_SAMPLE_COUNT uniform samples."""
  denom = float(max(CHECK_SAMPLE_COUNT - 1, 1))
  span = last - first
  prev_pt = curve3d.Value(first)
  arc_len = 0.0
  for idx in range(FIRST_INDEX, CHECK_SAMPLE_COUNT):
    param = first + (span * (float(idx) / denom))
    curr_pt = curve3d.Value(param)
    arc_len += float(prev_pt.Distance(curr_pt))
    prev_pt = curr_pt
  return arc_len


def _intersect_tangent_velocity(du1: gp_Vec, dv1: gp_Vec, du2: gp_Vec, dv2: gp_Vec, d_orig: gp_Vec) -> gp_Vec:
  """Project original curve velocity onto exact surface-pair normal cross-product direction."""
  n1 = du1.Crossed(dv1)
  n2 = du2.Crossed(dv2)
  if n1.Magnitude() <= NORMAL_LEN_TOL or n2.Magnitude() <= NORMAL_LEN_TOL:
    return d_orig
  cross = n1.Normalized().Crossed(n2.Normalized())
  if cross.Magnitude() <= TANGENT_CROSS_TOL:
    return d_orig
  tan_dir = cross.Normalized()
  return tan_dir * d_orig.Dot(tan_dir)


def _sample_hermite_nodes(
    edge: TopoDS_Edge,
    surf1: Geom_Surface,
    surf2: Geom_Surface,
    pc1: Geom2d_Curve,
    pc2: Geom2d_Curve,
    curve3d: Geom_Curve,
    first: float,
    last: float,
    interval_count: int,
) -> Tuple[List[gp_Pnt], List[gp_Vec], List[gp_Pnt2d], List[Tuple[float, float]], List[gp_Pnt2d], List[Tuple[float, float]]]:
  """Evaluate exact surface-pair points, tangents, and UV derivatives at uniform Hermite knots."""
  p_start, p_end = _edge_endpoint_positions(edge, curve3d, first, last)
  dt = (last - first) / float(interval_count)
  p_tmp = gp_Pnt()
  du1, dv1 = gp_Vec(), gp_Vec()
  du2, dv2 = gp_Vec(), gp_Vec()

  pts_3d: List[gp_Pnt] = []
  vels_3d: List[gp_Vec] = []
  uvs_1: List[gp_Pnt2d] = []
  w_uv1: List[Tuple[float, float]] = []
  uvs_2: List[gp_Pnt2d] = []
  w_uv2: List[Tuple[float, float]] = []

  for k in range(interval_count + 1):
    if k == 0:
      param = first
      p_node = p_start
    elif k == interval_count:
      param = last
      p_node = p_end
    else:
      param = first + (dt * float(k))

    uv1_raw = pc1.Value(param)
    uv2_raw = pc2.Value(param)
    if k in (0, interval_count):
      u1, v1, _, du1, dv1 = _project_uv_newton(surf1, p_node, uv1_raw.X(), uv1_raw.Y(), VERTEX_FIT_ITERS)
      u2, v2, _, du2, dv2 = _project_uv_newton(surf2, p_node, uv2_raw.X(), uv2_raw.Y(), VERTEX_FIT_ITERS)
      uv1_node = gp_Pnt2d(u1, v1)
      uv2_node = gp_Pnt2d(u2, v2)
    else:
      p_node, uv1_node, uv2_node = _solve_surface_pair_point(surf1, surf2, curve3d.Value(param), uv1_raw, uv2_raw)
      surf1.D1(uv1_node.X(), uv1_node.Y(), p_tmp, du1, dv1)
      surf2.D1(uv2_node.X(), uv2_node.Y(), p_tmp, du2, dv2)

    vel_3d = _intersect_tangent_velocity(du1, dv1, du2, dv2, curve3d.DN(param, FIRST_INDEX))
    pts_3d.append(p_node)
    vels_3d.append(vel_3d)
    uvs_1.append(uv1_node)
    w_uv1.append(_solve_uv_tangent(du1, dv1, vel_3d))
    uvs_2.append(uv2_node)
    w_uv2.append(_solve_uv_tangent(du2, dv2, vel_3d))

  return pts_3d, vels_3d, uvs_1, w_uv1, uvs_2, w_uv2


def _build_hermite_bsplines(
    first: float,
    last: float,
    interval_count: int,
    pts_3d: List[gp_Pnt],
    vels_3d: List[gp_Vec],
    uvs_1: List[gp_Pnt2d],
    w_uv1: List[Tuple[float, float]],
    uvs_2: List[gp_Pnt2d],
    w_uv2: List[Tuple[float, float]],
) -> Tuple[Geom_BSplineCurve, Geom2d_BSplineCurve, Geom2d_BSplineCurve]:
  """Assemble synchronized clamped C1 cubic Hermite 3D and 2D B-spline curves."""
  knot_count = interval_count + 1
  pole_count = (POLE_PAIR_STRIDE * interval_count) + POLE_COUNT_EXTRA
  dt = (last - first) / float(interval_count)
  step_scale = dt / HERMITE_TANGENT_DIVISOR

  knots = Array1_double(FIRST_INDEX, knot_count)
  mults = Array1_int(FIRST_INDEX, knot_count)
  for k in range(knot_count):
    if k == 0:
      knot_val = first
    elif k == interval_count:
      knot_val = last
    else:
      knot_val = first + (dt * float(k))
    knots.SetValue(k + 1, knot_val)
    mults.SetValue(k + 1, HERMITE_END_MULT if k in (0, interval_count) else HERMITE_INNER_MULT)

  poles_3d = Array1_gp_Pnt(FIRST_INDEX, pole_count)
  poles_2d_1 = Array1_gp_Pnt2d(FIRST_INDEX, pole_count)
  poles_2d_2 = Array1_gp_Pnt2d(FIRST_INDEX, pole_count)

  poles_3d.SetValue(FIRST_INDEX, pts_3d[0])
  poles_3d.SetValue(POLE_PAIR_STRIDE, pts_3d[0].Translated(vels_3d[0] * step_scale))
  poles_2d_1.SetValue(FIRST_INDEX, uvs_1[0])
  poles_2d_1.SetValue(POLE_PAIR_STRIDE, gp_Pnt2d(uvs_1[0].X() + (w_uv1[0][0] * step_scale), uvs_1[0].Y() + (w_uv1[0][1] * step_scale)))
  poles_2d_2.SetValue(FIRST_INDEX, uvs_2[0])
  poles_2d_2.SetValue(POLE_PAIR_STRIDE, gp_Pnt2d(uvs_2[0].X() + (w_uv2[0][0] * step_scale), uvs_2[0].Y() + (w_uv2[0][1] * step_scale)))

  for k in range(1, interval_count):
    pole_idx = (POLE_PAIR_STRIDE * k) + 1
    poles_3d.SetValue(pole_idx, pts_3d[k].Translated(vels_3d[k] * (-step_scale)))
    poles_3d.SetValue(pole_idx + 1, pts_3d[k].Translated(vels_3d[k] * step_scale))
    poles_2d_1.SetValue(pole_idx, gp_Pnt2d(uvs_1[k].X() - (w_uv1[k][0] * step_scale), uvs_1[k].Y() - (w_uv1[k][1] * step_scale)))
    poles_2d_1.SetValue(pole_idx + 1, gp_Pnt2d(uvs_1[k].X() + (w_uv1[k][0] * step_scale), uvs_1[k].Y() + (w_uv1[k][1] * step_scale)))
    poles_2d_2.SetValue(pole_idx, gp_Pnt2d(uvs_2[k].X() - (w_uv2[k][0] * step_scale), uvs_2[k].Y() - (w_uv2[k][1] * step_scale)))
    poles_2d_2.SetValue(pole_idx + 1, gp_Pnt2d(uvs_2[k].X() + (w_uv2[k][0] * step_scale), uvs_2[k].Y() + (w_uv2[k][1] * step_scale)))

  poles_3d.SetValue(pole_count - 1, pts_3d[interval_count].Translated(vels_3d[interval_count] * (-step_scale)))
  poles_3d.SetValue(pole_count, pts_3d[interval_count])
  poles_2d_1.SetValue(
      pole_count - 1,
      gp_Pnt2d(uvs_1[interval_count].X() - (w_uv1[interval_count][0] * step_scale), uvs_1[interval_count].Y() - (w_uv1[interval_count][1] * step_scale)),
  )
  poles_2d_1.SetValue(pole_count, uvs_1[interval_count])
  poles_2d_2.SetValue(
      pole_count - 1,
      gp_Pnt2d(uvs_2[interval_count].X() - (w_uv2[interval_count][0] * step_scale), uvs_2[interval_count].Y() - (w_uv2[interval_count][1] * step_scale)),
  )
  poles_2d_2.SetValue(pole_count, uvs_2[interval_count])

  return (
      Geom_BSplineCurve(poles_3d, knots, mults, HERMITE_DEGREE, False),
      Geom2d_BSplineCurve(poles_2d_1, knots, mults, HERMITE_DEGREE, False),
      Geom2d_BSplineCurve(poles_2d_2, knots, mults, HERMITE_DEGREE, False),
  )


def _refine_two_face_bspline_edge(
    edge: TopoDS_Edge,
    face1: TopoDS_Face,
    face2: TopoDS_Face,
    curve3d: Geom_Curve,
    first: float,
    last: float,
) -> None:
  """Replace a two-face edge with synchronized cubic Hermite B-splines within <= 1e-8 mm."""
  if face1.IsSame(face2) or BRep_Tool.IsClosed_s(edge, face1) or BRep_Tool.IsClosed_s(edge, face2):
    return

  pc1 = BRep_Tool.CurveOnSurface_s(edge, face1, first, last)
  pc2 = BRep_Tool.CurveOnSurface_s(edge, face2, first, last)
  if pc1 is None or pc2 is None:
    return

  surf1 = BRep_Tool.Surface_s(face1)
  surf2 = BRep_Tool.Surface_s(face2)
  arc_len_mm = _estimate_edge_chord_length_mm(curve3d, first, last)
  interval_count = max(MIN_HERMITE_INTERVALS, min(MAX_HERMITE_INTERVALS, int(math.ceil(arc_len_mm / MAX_HERMITE_CHORD_MM))))

  pts_3d, vels_3d, uvs_1, w_uv1, uvs_2, w_uv2 = _sample_hermite_nodes(
      edge, surf1, surf2, pc1, pc2, curve3d, first, last, interval_count
  )
  bs_3d, bs_2d_1, bs_2d_2 = _build_hermite_bsplines(
      first, last, interval_count, pts_3d, vels_3d, uvs_1, w_uv1, uvs_2, w_uv2
  )

  edge_tol = _edge_vertex_min_tolerance(edge)
  builder = BRep_Builder()
  builder.UpdateEdge(edge, bs_3d, edge_tol)
  builder.UpdateEdge(edge, bs_2d_1, face1, edge_tol)
  builder.UpdateEdge(edge, bs_2d_2, face2, edge_tol)
  builder.Range(edge, first, last)
  builder.SameRange(edge, True)
  builder.SameParameter(edge, True)
  ShapeFix_ShapeTolerance().LimitTolerance(edge, 0.0, edge_tol, TopAbs_EDGE)


def _try_heal_indexed_edge(
    edge_face_map: IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
    idx: int,
) -> None:
  """Refine an indexed edge if it binds two faces with B-spline discrepancy > 1e-8 mm."""
  edge = TopoDS.Edge(edge_face_map.FindKey(idx))
  if BRep_Tool.Degenerated_s(edge):
    return
  first, last = BRep_Tool.Range_s(edge)
  curve3d = BRep_Tool.Curve_s(edge, first, last)
  if curve3d is None or last <= first:
    return
  if not isinstance(curve3d, Geom_BSplineCurve) and BRep_Tool.Tolerance_s(edge) <= EDGE_TOL_CHECK_MM:
    return

  faces = [TopoDS.Face(face_shape) for face_shape in edge_face_map.FindFromIndex(idx)]
  if len(faces) != TWO_FACES:
    return
  if _edge_faces_max_gap_mm(edge, faces, CHECK_SAMPLE_COUNT) > TARGET_EDGE_GAP_TOL_MM:
    _refine_two_face_bspline_edge(edge, faces[0], faces[1], curve3d, first, last)


def heal_solid_edges_to_surfaces(solid: Solid) -> Solid:
  """Return a deep copy of solid whose B-spline intersection edges match adjacent faces within <= 1e-8 mm."""
  copied_shape = BRepBuilderAPI_Transform(solid.wrapped, gp_Trsf(), True).Shape()
  edge_face_map = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
  TopExp.MapShapesAndAncestors_s(copied_shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

  for idx in range(FIRST_INDEX, edge_face_map.Extent() + 1):
    _try_heal_indexed_edge(edge_face_map, idx)

  if not BRepCheck_Analyzer(copied_shape).IsValid():
    raise RuntimeError("B-Rep topology validation failed after edge-to-surface healing")
  return Solid(TopoDS.Solid(copied_shape))


def heal_part_edges(part: Part) -> Part:
  """Refine B-spline intersection edges across all solids in a build123d Part."""
  healed_solids = [heal_solid_edges_to_surfaces(solid) for solid in part.solids()]
  result = Part(children=healed_solids)
  if part.label:
    result.label = part.label
  if part.color is not None:
    result.color = part.color
  return result


def max_solid_edge_surface_gap_mm(
    shape: Shape,
    sample_count: int = DEFAULT_GAP_SAMPLE_COUNT,
    bspline_only: bool = False,
) -> float:
  """Compute the worst-case 3D-curve to surface/pcurve discrepancy in mm over all solids in shape."""
  max_gap = 0.0
  for solid in shape.solids():
    edge_face_map = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapesAndAncestors_s(solid.wrapped, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    for idx in range(FIRST_INDEX, edge_face_map.Extent() + 1):
      edge = TopoDS.Edge(edge_face_map.FindKey(idx))
      faces = [TopoDS.Face(face_shape) for face_shape in edge_face_map.FindFromIndex(idx)]
      max_gap = max(max_gap, _edge_faces_max_gap_mm(edge, faces, sample_count, bspline_only=bspline_only))
  return max_gap


def default_assembly_opacity_for_part(part_name: str) -> float:
  """Return the default assembly alpha opacity for a named engine component."""
  if part_name in SHROUD_PART_NAMES:
    return SHROUD_OPACITY
  if part_name in HOUSING_PART_NAMES:
    return HOUSING_OPACITY
  if part_name in LINER_PART_NAMES:
    return LINER_OPACITY
  if part_name in FLUID_PART_NAMES:
    return FLUID_OPACITY
  return OPAQUE_OPACITY


def material_color(material_key: str, opacity: float = OPAQUE_OPACITY) -> Color:
  """Build a build123d Color with normalized RGB and alpha in [0.0, 1.0]."""
  red, green, blue = MATERIAL_RGB_PALETTE.get(material_key, DEFAULT_PART_RGB)
  return Color(red, green, blue, opacity)


def style_single_part(
    part: Part,
    part_name: str,
    material_key: str,
    opacity: Optional[float] = None,
) -> Part:
  """Return a cloned Part annotated with STEP product name, material RGB, and opacity."""
  resolved_opacity = OPAQUE_OPACITY if opacity is None else opacity
  styled = Part(children=list(part.solids()))
  styled.label = part_name
  styled.color = material_color(material_key, resolved_opacity)
  return styled


def style_named_parts_as_compound(
    assembly_label: str,
    parts: Mapping[str, Part],
    material_resolver: Callable[[str], str],
    opacity_resolver: Callable[[str], float] = default_assembly_opacity_for_part,
) -> Compound:
  """Wrap named Parts into an XCAF-styled Compound preserving part names, colors, and transparency."""
  children: List[Solid] = []
  for part_name, part in parts.items():
    material_key = material_resolver(part_name)
    opacity = opacity_resolver(part_name)
    step_color = material_color(material_key, opacity)
    leaves = list(part.solids())
    for idx, leaf in enumerate(leaves, start=FIRST_INDEX):
      styled_leaf = Solid(leaf.wrapped)
      styled_leaf.label = part_name if len(leaves) == 1 else f"{part_name}_{idx}"
      styled_leaf.color = step_color
      children.append(styled_leaf)
  return Compound(label=assembly_label, children=children)


def export_styled_step(shape: Shape, file_path: Union[str, Path]) -> Path:
  """Export a styled build123d Shape or Compound to a STEP AP214 file and ensure parent dir exists."""
  target = Path(file_path)
  target.parent.mkdir(parents=True, exist_ok=True)
  export_step(shape, str(target))
  return target
