"""CAD binary glTF (.glb) and FEA WebGL stress mesh exporters for interactive browser rendering."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from build123d import Axis, Compound, Location, Part, export_gltf
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.TopLoc import TopLoc_Location

from openengine.cad.assemblies import (
    build_dressed_engine_layer_groups,
    generate_dressed_engine_assembly,
)
from openengine.cad.assembly import EngineAssembly, generate_engine_assembly
from openengine.cad.connecting_rod import generate_connecting_rod
from openengine.cad.cots_parts import generate_wrist_pin_16x45
from openengine.cad.crankshaft import generate_crankshaft
from openengine.cad.cylinder_block import (
    generate_cylinder_block,
    generate_cylinder_liner,
)
from openengine.cad.cylinder_head import (
    generate_cylinder_head,
    generate_poppet_valve,
)
from openengine.cad.piston import generate_piston
from openengine.fea.calculix import parse_frd_results


def turbo_colormap(val: float) -> Tuple[float, float, float]:
  """Polynomial approximation of the Turbo/Rainbow colormap for normalized val in [0, 1].

  Smoothly transitions: Deep Blue -> Cyan -> Green -> Yellow -> Orange -> Deep Red.
  """
  x = max(0.0, min(1.0, val))
  if x < 0.25:
    f = x / 0.25
    r, g, b = 0.0, f, 1.0
  elif x < 0.50:
    f = (x - 0.25) / 0.25
    r, g, b = 0.0, 1.0, 1.0 - f
  elif x < 0.75:
    f = (x - 0.50) / 0.25
    r, g, b = f, 1.0, 0.0
  else:
    f = (x - 0.75) / 0.25
    r, g, b = 1.0, max(0.0, 1.0 - f * 1.2), 0.0
  return round(r, 3), round(g, 3), round(b, 3)


def bake_shape(shape: Any) -> Part:
  """Bake OpenCASCADE Location into vertex coordinates so exported glTF is strictly in assembly space."""
  loc = shape.wrapped.Location()
  if loc.IsIdentity():
    return Part(shape.wrapped)
  trsf = loc.Transformation()
  transformer = BRepBuilderAPI_Transform(shape.wrapped.Located(TopLoc_Location()), trsf, True)
  return Part(transformer.Shape())


def _bake_compound(shape: Any) -> Any:
  """Bake Compound or Part vertices into world coordinates."""
  if isinstance(shape, Compound):
    return Compound(children=[bake_shape(sol) for sol in shape.solids()])
  return bake_shape(shape)


def _export_glb_with_upright_frame(
    shape: Any,
    glb_file: Path,
    linear_deflection: float,
    angular_deflection: float,
) -> str:
  """Export CAD shape to binary glTF while preserving +Z up CAD assembly coordinates."""
  # build123d.export_gltf internally applies -90 deg X rotation; counteract to keep +Z up.
  shape.location = Location((0, 0, 0), (1, 0, 0), 90.0)
  export_gltf(
      shape,
      str(glb_file),
      binary=True,
      linear_deflection=linear_deflection,
      angular_deflection=angular_deflection,
  )
  return str(glb_file)


def export_cad_models_to_gltf(
    output_dir: Union[str, Path] = "outputs/reports/models",
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.2,
) -> Dict[str, str]:
  """Export all engine CAD B-Rep solid models directly from the master assembly to binary glTF (.glb).

  Args:
    output_dir: Target directory for .glb files.
    linear_deflection: Chordal deflection tolerance in mm for facet tessellation.
    angular_deflection: Angular deflection tolerance in radians.

  Returns:
    Dictionary mapping component names to absolute .glb file paths.
  """
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  assembly = EngineAssembly()
  st0 = assembly.solve_kinematics(0.0)

  head = generate_cylinder_head()
  block = generate_cylinder_block()
  liner = generate_cylinder_liner()
  crank = generate_crankshaft().translate((0.0, 0.0, assembly.z_crank_main_mm))
  rod = generate_connecting_rod().rotate(Axis.Z, 90.0).rotate(Axis.Y, -90.0).translate((0.0, 0.0, st0.wrist_pin_z_mm))
  piston = generate_piston().translate((0.0, 0.0, st0.wrist_pin_z_mm + 3.0))
  wrist_pin = generate_wrist_pin_16x45().translate((0.0, 0.0, st0.wrist_pin_z_mm))
  in_valve = generate_poppet_valve(head_diam_m=0.030).rotate(Axis.Y, 15.0).translate((16.0, 0.0, 6.0))
  ex_valve = generate_poppet_valve(head_diam_m=0.026).rotate(Axis.Y, -15.0).translate((-16.0, 0.0, 6.0))

  parts_to_bake = {
      "cylinder_head": head,
      "cylinder_block": block,
      "cylinder_liner": liner,
      "crankshaft": crank,
      "connecting_rod": rod,
      "piston": piston,
      "wrist_pin": wrist_pin,
      "valve_intake": in_valve,
      "valve_exhaust": ex_valve,
      **build_dressed_engine_layer_groups(0.0),
  }

  glb_paths: Dict[str, str] = {}
  for name, shape in parts_to_bake.items():
    glb_paths[name] = _export_glb_with_upright_frame(
        _bake_compound(shape),
        out_path / f"{name}.glb",
        linear_deflection,
        angular_deflection,
    )

  trajectory: List[Dict[str, Any]] = []
  for th in range(0, 720, 2):
    st = assembly.solve_kinematics(float(th))
    trajectory.append({
        "crank_angle_deg": float(st.crank_angle_deg),
        "crankpin_x_mm": round(float(st.crankpin_x_mm), 3),
        "crankpin_z_mm": round(float(st.crankpin_z_mm), 3),
        "con_rod_angle_deg": round(float(st.con_rod_angle_deg), 3),
        "wrist_pin_z_mm": round(float(st.wrist_pin_z_mm), 3),
        "piston_crown_z_mm": round(float(st.piston_crown_z_mm), 3),
        "intake_valve_lift_mm": round(float(st.intake_valve_lift_mm), 3),
        "exhaust_valve_lift_mm": round(float(st.exhaust_valve_lift_mm), 3),
        "intake_pv_clearance_mm": round(float(st.intake_pv_clearance_mm), 3),
        "exhaust_pv_clearance_mm": round(float(st.exhaust_pv_clearance_mm), 3),
        "rod_to_liner_clearance_mm": round(float(st.rod_to_liner_clearance_mm), 3),
        "is_safe": bool(st.is_safe),
    })

  traj_file = out_path / "kinematic_trajectory.json"
  traj_file.write_text(json.dumps(trajectory, indent=2))
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


def export_fea_stress_to_webgl(
    frd_path: Union[str, Path],
    inp_path: Union[str, Path],
    output_json_path: Union[str, Path],
    title: str = "FEA Stress Field",
    component_name: str = "Piston",
    yield_strength_mpa: float = 315.0,
    disp_scale: float = 1.0,
) -> Dict[str, Any]:
  """Extract surface boundary mesh with nodal von Mises stresses and color maps for Three.js.

  1. Parses CalculiX .frd file for nodal displacements and von Mises stresses.
  2. Parses CalculiX / Gmsh .inp file for 3D coordinates and element connectivity.
  3. Extracts outward-facing boundary surface triangles from the 3D tetrahedral mesh.
  4. Maps von Mises stress to Turbo colormap RGB vertex colors.
  5. Serializes to a compact Three.js BufferGeometry JSON structure.

  Args:
    frd_path: Path to CalculiX .frd results file.
    inp_path: Path to mesh .inp deck.
    output_json_path: Target JSON filepath for Three.js viewer.
    title: Human-readable load case title.
    component_name: Name of part (e.g. 'Piston', 'Connecting Rod').
    yield_strength_mpa: Material yield strength in MPa for safety factor calculation.
    disp_scale: Magnification scale factor for displacement deformation.

  Returns:
    Dictionary containing mesh summary and serialized payload.
  """
  out_json = Path(output_json_path)
  out_json.parent.mkdir(parents=True, exist_ok=True)

  # 1. Parse FEA results
  fea_res = parse_frd_results(frd_path)

  # 2. Parse nodes and element tets from .inp
  node_coords: Dict[int, Tuple[float, float, float]] = {}
  tets: List[List[int]] = []

  in_nodes = False
  in_elements = False
  with open(inp_path, "r") as f:
    for line in f:
      line_strip = line.strip()
      if line_strip.startswith("*NODE"):
        in_nodes = True
        in_elements = False
        continue
      elif line_strip.startswith("*ELEMENT"):
        in_nodes = False
        in_elements = True
        continue
      elif line_strip.startswith("*") and not line_strip.startswith("*ELEMENT"):
        in_nodes = False
        in_elements = False
        continue

      if in_nodes:
        parts = [p.strip() for p in line_strip.split(",") if p.strip()]
        if len(parts) >= 4 and parts[0].isdigit():
          node_coords[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
      elif in_elements:
        parts = [p.strip() for p in line_strip.split(",") if p.strip()]
        if len(parts) >= 5 and parts[0].isdigit():
          # First 4 vertex corner nodes of tetrahedral element
          tets.append([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])

  # 3. Extract boundary surface triangular faces
  face_counts = Counter()
  face_orientations: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

  for tet in tets:
    # 4 faces of tetrahedron
    tet_faces = [
        (tet[0], tet[1], tet[2]),
        (tet[0], tet[2], tet[3]),
        (tet[0], tet[3], tet[1]),
        (tet[1], tet[3], tet[2]),
    ]
    for face in tet_faces:
      key = tuple(sorted(face))
      face_counts[key] += 1
      face_orientations[key] = face

  boundary_triangles = [face_orientations[k] for k, count in face_counts.items() if count == 1]

  # 4. Map unique boundary nodes to compact 0-based indices
  unique_boundary_nodes = sorted(list({nid for tri in boundary_triangles for nid in tri}))
  node_id_to_idx = {nid: idx for idx, nid in enumerate(unique_boundary_nodes)}

  # Stress normalization range
  stresses = [fea_res.von_mises_stresses.get(nid, 0.0) for nid in unique_boundary_nodes]
  max_stress = max(stresses) if stresses else 1.0
  min_stress = min(stresses) if stresses else 0.0
  p95_stress = float(np.percentile(stresses, 95)) if stresses else max_stress
  stress_range = max(1.0, max_stress - min_stress)

  # Build vertex arrays for Three.js BufferGeometry
  # positions: [x0, y0, z0, x1, y1, z1, ...]
  # colors: [r0, g0, b0, r1, g1, b1, ...]
  # indices: [f0_a, f0_b, f0_c, f1_a, f1_b, f1_c, ...]
  positions: List[float] = []
  colors: List[float] = []
  nodal_stress_vals: List[float] = []

  for nid in unique_boundary_nodes:
    x0, y0, z0 = node_coords.get(nid, (0.0, 0.0, 0.0))
    ux, uy, uz = fea_res.displacements.get(nid, (0.0, 0.0, 0.0))
    # Apply scaled displacement
    x_def = x0 + disp_scale * ux
    y_def = y0 + disp_scale * uy
    z_def = z0 + disp_scale * uz
    positions.extend([round(x_def, 3), round(y_def, 3), round(z_def, 3)])

    vm = fea_res.von_mises_stresses.get(nid, 0.0)
    nodal_stress_vals.append(round(vm, 1))
    norm_val = (vm - min_stress) / stress_range
    r, g, b = turbo_colormap(norm_val)
    colors.extend([r, g, b])

  indices: List[int] = []
  for tri in boundary_triangles:
    indices.extend([node_id_to_idx[tri[0]], node_id_to_idx[tri[1]], node_id_to_idx[tri[2]]])

  sf_yield = yield_strength_mpa / max(0.1, p95_stress)

  payload = {
      "metadata": {
          "title": title,
          "component": component_name,
          "num_vertices": len(unique_boundary_nodes),
          "num_triangles": len(boundary_triangles),
          "min_stress_mpa": round(min_stress, 1),
          "max_stress_mpa": round(max_stress, 1),
          "p95_stress_mpa": round(p95_stress, 1),
          "yield_strength_mpa": round(yield_strength_mpa, 1),
          "yield_safety_factor": round(sf_yield, 2),
          "max_displacement_mm": round(fea_res.max_disp_mm, 4),
          "disp_scale": disp_scale,
      },
      "geometry": {
          "positions": positions,
          "colors": colors,
          "indices": indices,
          "stresses": nodal_stress_vals,
      },
  }

  with open(out_json, "w") as f:
    json.dump(payload, f)

  return payload["metadata"]
