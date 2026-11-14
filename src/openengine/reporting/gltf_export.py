"""CAD binary glTF (.glb) and FEA WebGL stress mesh exporters for interactive browser rendering."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from build123d import Compound, Location, Part, Shape, export_gltf
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.TopLoc import TopLoc_Location

from openengine.cad.assemblies import (
    build_dressed_engine_layer_groups,
    build_dressed_engine_parts,
    generate_dressed_engine_assembly,
    material_key_for_dressed_part,
)
from openengine.cad.assemblies.head_assembly import (
    CAM_TO_CRANK_RATIO,
    ROCKER_CONTACT_SCALE,
    ROCKER_RATIO,
)
from openengine.cad.assembly import EngineAssembly, generate_engine_assembly
from openengine.cad.step_export import style_named_parts_as_compound
from openengine.cad.valvetrain_hardware import ROCKER_PIVOT_X_MM
from openengine.fea.calculix import (
    BoundaryMeshData,
    boundary_triangles_from_tets as _boundary_triangles_from_tets,
    parse_calculix_inp_mesh as _parse_calculix_inp_mesh,
    parse_frd_results,
)

ZERO_VAL: float = 0.0
UNIT_ONE: float = 1.0
UPRIGHT_ROT_DEG: float = 90.0
COLOR_Q1: float = 0.25
COLOR_Q2: float = 0.50
COLOR_Q3: float = 0.75
COLOR_RED_DECAY: float = 1.2
COLOR_ROUND_DIGITS: int = 3
DEFLECTION_LINEAR_MM: float = 0.05
DEFLECTION_ANGULAR_RAD: float = 0.2
CYCLE_TOTAL_DEG: int = 720
TRAJECTORY_STEP_DEG: int = 2
TRAJECTORY_ROUND_DIGITS: int = 3
COORD_ROUND_DIGITS: int = 3
STRESS_ROUND_DIGITS: int = 1
DISP_ROUND_DIGITS: int = 4
SF_ROUND_DIGITS: int = 2
MIN_STRESS_EPS_MPA: float = 0.1
MIN_RANGE_EPS_MPA: float = 1.0
P95_PERCENTILE: int = 95
DEFAULT_YIELD_MPA: float = 315.0
DEFAULT_DISP_SCALE: float = 1.0

STATIC_SINGLE_PART_NAMES: Tuple[str, ...] = (
    "cylinder_head",
    "cylinder_block",
    "cylinder_liner",
    "wrist_pin",
    "valve_intake",
    "valve_exhaust",
    "retainer_intake",
    "retainer_exhaust",
    "piston",
)

COUPLED_SUBGROUPS: Mapping[str, Tuple[str, ...]] = {
    "crankshaft": ("crankshaft",),
    "crank_covers_spin": ("flywheel", "trigger_wheel"),
    "crank_vt_spin": ("sprocket_crank_18t",),
    "connecting_rod": ("conrod_beam", "conrod_cap"),
    "conrod_cots": (
        "needle_bearing_nks28",
        "conrod_bolt_1",
        "conrod_bolt_2",
        "conrod_dowel_1",
        "conrod_dowel_2",
    ),
    "piston_cots": ("top_ring", "second_ring", "oil_ring", "circlip_left", "circlip_right"),
    "cam_vt_spin": ("camshaft", "ground_cam_core_shaft", "intake_lobe_2_5d", "exhaust_lobe_2_5d", "sprocket_cam_36t"),
    "rocker_intake": ("rocker_intake",),
    "rocker_exhaust": ("rocker_exhaust",),
}


def turbo_colormap(val: float) -> Tuple[float, float, float]:
  """Polynomial approximation of the Turbo/Rainbow colormap for normalized val in [0, 1]."""
  x = max(ZERO_VAL, min(UNIT_ONE, val))
  if x < COLOR_Q1:
    r, g, b = ZERO_VAL, x / COLOR_Q1, UNIT_ONE
  elif x < COLOR_Q2:
    r, g, b = ZERO_VAL, UNIT_ONE, UNIT_ONE - ((x - COLOR_Q1) / COLOR_Q1)
  elif x < COLOR_Q3:
    r, g, b = (x - COLOR_Q2) / COLOR_Q1, UNIT_ONE, ZERO_VAL
  else:
    frac = (x - COLOR_Q3) / COLOR_Q1
    r, g, b = UNIT_ONE, max(ZERO_VAL, UNIT_ONE - (frac * COLOR_RED_DECAY)), ZERO_VAL
  return (
      round(r, COLOR_ROUND_DIGITS),
      round(g, COLOR_ROUND_DIGITS),
      round(b, COLOR_ROUND_DIGITS),
  )


def bake_shape(shape: Shape) -> Part:
  """Bake OpenCASCADE Location into vertex coordinates in assembly space."""
  if isinstance(shape, Compound) and not isinstance(shape, Part):
    return Part(children=[bake_shape(sol) for sol in shape.solids()])
  solids = list(shape.solids())
  loc = shape.wrapped.Location()
  if loc.IsIdentity():
    return Part(children=solids)
  trsf = loc.Transformation()
  baked_solids = [
      Part(BRepBuilderAPI_Transform(sol.wrapped.Located(TopLoc_Location()), trsf, True).Shape())
      for sol in solids
  ]
  return Part(children=baked_solids)


def _export_glb_with_upright_frame(
    shape: Shape,
    glb_file: Path,
    linear_deflection: float,
    angular_deflection: float,
) -> str:
  """Export styled CAD shape to binary glTF while preserving +Z up CAD assembly coordinates."""
  # build123d.export_gltf applies -90 deg X internally; counteract on root node to keep +Z up.
  original_loc = shape.location
  shape.location = Location((ZERO_VAL, ZERO_VAL, ZERO_VAL), (UNIT_ONE, ZERO_VAL, ZERO_VAL), UPRIGHT_ROT_DEG) * original_loc
  export_gltf(
      shape,
      str(glb_file),
      binary=True,
      linear_deflection=linear_deflection,
      angular_deflection=angular_deflection,
  )
  shape.location = original_loc
  return str(glb_file)


def _export_named_group(
    group_name: str,
    selected: Dict[str, Part],
    out_path: Path,
    linear_deflection: float,
    angular_deflection: float,
) -> str:
  """Build a styled Compound for a named subset of parts and export it as binary glTF."""
  styled = style_named_parts_as_compound(
      group_name,
      selected,
      material_key_for_dressed_part,
  )
  return _export_glb_with_upright_frame(
      styled,
      out_path / f"{group_name}.glb",
      linear_deflection,
      angular_deflection,
  )


def _build_trajectory_frames(assembly: EngineAssembly) -> List[Dict[str, object]]:
  """Sample 720-deg analytical kinematics including cam and rocker arm angles."""
  frames: List[Dict[str, object]] = []
  for th in range(0, CYCLE_TOTAL_DEG, TRAJECTORY_STEP_DEG):
    st = assembly.solve_kinematics(float(th))
    in_ratio = max(-UNIT_ONE, min(UNIT_ONE, (float(st.intake_valve_lift_mm) / ROCKER_RATIO) / ROCKER_PIVOT_X_MM))
    ex_ratio = max(-UNIT_ONE, min(UNIT_ONE, (float(st.exhaust_valve_lift_mm) / ROCKER_RATIO) / ROCKER_PIVOT_X_MM))
    alpha_in = math.degrees(math.asin(in_ratio)) * ROCKER_CONTACT_SCALE
    alpha_ex = -math.degrees(math.asin(ex_ratio)) * ROCKER_CONTACT_SCALE
    frames.append({
        "crank_angle_deg": float(st.crank_angle_deg),
        "cam_angle_deg": float(st.crank_angle_deg) * CAM_TO_CRANK_RATIO,
        "crankpin_x_mm": round(float(st.crankpin_x_mm), TRAJECTORY_ROUND_DIGITS),
        "crankpin_z_mm": round(float(st.crankpin_z_mm), TRAJECTORY_ROUND_DIGITS),
        "con_rod_angle_deg": round(float(st.con_rod_angle_deg), TRAJECTORY_ROUND_DIGITS),
        "wrist_pin_z_mm": round(float(st.wrist_pin_z_mm), TRAJECTORY_ROUND_DIGITS),
        "piston_crown_z_mm": round(float(st.piston_crown_z_mm), TRAJECTORY_ROUND_DIGITS),
        "intake_valve_lift_mm": round(float(st.intake_valve_lift_mm), TRAJECTORY_ROUND_DIGITS),
        "exhaust_valve_lift_mm": round(float(st.exhaust_valve_lift_mm), TRAJECTORY_ROUND_DIGITS),
        "rocker_intake_deg": round(alpha_in, TRAJECTORY_ROUND_DIGITS),
        "rocker_exhaust_deg": round(alpha_ex, TRAJECTORY_ROUND_DIGITS),
        "intake_pv_clearance_mm": round(float(st.intake_pv_clearance_mm), TRAJECTORY_ROUND_DIGITS),
        "exhaust_pv_clearance_mm": round(float(st.exhaust_pv_clearance_mm), TRAJECTORY_ROUND_DIGITS),
        "rod_to_liner_clearance_mm": round(float(st.rod_to_liner_clearance_mm), TRAJECTORY_ROUND_DIGITS),
        "is_safe": bool(st.is_safe),
    })
  return frames


def export_cad_models_to_gltf(
    output_dir: Union[str, Path] = "outputs/reports/models",
    linear_deflection: float = DEFLECTION_LINEAR_MM,
    angular_deflection: float = DEFLECTION_ANGULAR_RAD,
) -> Dict[str, str]:
  """Export styled engine CAD B-Rep solids, layers, and kinematic trajectory to .glb/JSON."""
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  parts = build_dressed_engine_parts(crank_angle_deg=ZERO_VAL, has_fourth_axis=True, include_fluids=True)
  glb_paths: Dict[str, str] = {}

  for part_name in STATIC_SINGLE_PART_NAMES:
    if part_name in parts:
      glb_paths[part_name] = _export_named_group(
          part_name,
          {part_name: parts[part_name]},
          out_path,
          linear_deflection,
          angular_deflection,
      )

  for group_name, member_keys in COUPLED_SUBGROUPS.items():
    selected = {key: parts[key] for key in member_keys if key in parts}
    if selected:
      glb_paths[group_name] = _export_named_group(
          group_name,
          selected,
          out_path,
          linear_deflection,
          angular_deflection,
      )

  layer_groups = build_dressed_engine_layer_groups(crank_angle_deg=ZERO_VAL, has_fourth_axis=True, parts=parts)
  for layer_name, compound in layer_groups.items():
    glb_paths[layer_name] = _export_glb_with_upright_frame(
        compound,
        out_path / f"{layer_name}.glb",
        linear_deflection,
        angular_deflection,
    )

  trajectory = _build_trajectory_frames(EngineAssembly())
  traj_file = out_path / "kinematic_trajectory.json"
  traj_file.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
  glb_paths["kinematic_trajectory"] = str(traj_file)

  glb_paths["engine_assembly"] = _export_glb_with_upright_frame(
      generate_engine_assembly(),
      out_path / "engine_assembly.glb",
      linear_deflection,
      angular_deflection,
  )
  glb_paths["dressed_engine_assembly"] = _export_glb_with_upright_frame(
      generate_dressed_engine_assembly(),
      out_path / "dressed_engine_assembly.glb",
      linear_deflection,
      angular_deflection,
  )
  return glb_paths


def _build_stress_vertex_buffers(
    mesh: BoundaryMeshData,
    displacements: Dict[int, Tuple[float, float, float]],
    von_mises: Dict[int, float],
    disp_scale: float,
) -> Tuple[List[float], List[float], List[int], List[float], float, float, float]:
  """Build compact Three.js vertex position, RGB Turbo color, and triangle index buffers."""
  stresses = [von_mises.get(nid, ZERO_VAL) for nid in mesh.unique_nodes]
  max_stress = max(stresses) if stresses else UNIT_ONE
  min_stress = min(stresses) if stresses else ZERO_VAL
  p95_stress = float(np.percentile(stresses, P95_PERCENTILE)) if stresses else max_stress
  stress_range = max(MIN_RANGE_EPS_MPA, max_stress - min_stress)

  node_to_idx = {nid: idx for idx, nid in enumerate(mesh.unique_nodes)}
  positions: List[float] = []
  colors: List[float] = []
  nodal_stresses: List[float] = []

  for nid in mesh.unique_nodes:
    x0, y0, z0 = mesh.node_coords.get(nid, (ZERO_VAL, ZERO_VAL, ZERO_VAL))
    ux, uy, uz = displacements.get(nid, (ZERO_VAL, ZERO_VAL, ZERO_VAL))
    positions.extend([
        round(x0 + (disp_scale * ux), COORD_ROUND_DIGITS),
        round(y0 + (disp_scale * uy), COORD_ROUND_DIGITS),
        round(z0 + (disp_scale * uz), COORD_ROUND_DIGITS),
    ])
    vm = von_mises.get(nid, ZERO_VAL)
    nodal_stresses.append(round(vm, STRESS_ROUND_DIGITS))
    r, g, b = turbo_colormap((vm - min_stress) / stress_range)
    colors.extend([r, g, b])

  indices: List[int] = []
  for a, b, c in mesh.boundary_triangles:
    indices.extend([node_to_idx[a], node_to_idx[b], node_to_idx[c]])

  return positions, colors, indices, nodal_stresses, min_stress, max_stress, p95_stress


def export_fea_stress_to_webgl(
    frd_path: Union[str, Path],
    inp_path: Union[str, Path],
    output_json_path: Union[str, Path],
    title: str = "FEA Stress Field",
    component_name: str = "Piston",
    yield_strength_mpa: float = DEFAULT_YIELD_MPA,
    disp_scale: float = DEFAULT_DISP_SCALE,
) -> Dict[str, Any]:
  """Extract boundary surface mesh with nodal von Mises stresses and color maps for Three.js."""
  out_json = Path(output_json_path)
  out_json.parent.mkdir(parents=True, exist_ok=True)

  fea_res = parse_frd_results(frd_path)
  mesh = _parse_calculix_inp_mesh(inp_path)
  positions, colors, indices, nodal_stresses, min_s, max_s, p95_s = _build_stress_vertex_buffers(
      mesh=mesh,
      displacements=fea_res.displacements,
      von_mises=fea_res.von_mises_stresses,
      disp_scale=disp_scale,
  )

  sf_yield = yield_strength_mpa / max(MIN_STRESS_EPS_MPA, p95_s)
  metadata: Dict[str, Union[str, int, float]] = {
      "title": title,
      "component": component_name,
      "num_vertices": len(mesh.unique_nodes),
      "num_triangles": len(mesh.boundary_triangles),
      "min_stress_mpa": round(min_s, STRESS_ROUND_DIGITS),
      "max_stress_mpa": round(max_s, STRESS_ROUND_DIGITS),
      "p95_stress_mpa": round(p95_s, STRESS_ROUND_DIGITS),
      "yield_strength_mpa": round(yield_strength_mpa, STRESS_ROUND_DIGITS),
      "yield_safety_factor": round(sf_yield, SF_ROUND_DIGITS),
      "max_displacement_mm": round(fea_res.max_disp_mm, DISP_ROUND_DIGITS),
      "disp_scale": disp_scale,
  }
  payload = {
      "metadata": metadata,
      "geometry": {
          "positions": positions,
          "colors": colors,
          "indices": indices,
          "stresses": nodal_stresses,
      },
  }
  out_json.write_text(json.dumps(payload), encoding="utf-8")
  return dict(metadata)
