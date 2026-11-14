"""Gmsh CFD mesh generation for OpenFOAM boundary-conforming geometries."""

import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
from build123d import Part, export_step
import gmsh


def mesh_port_fluid_domain(
    geometry: Union[Part, str, Path],
    output_msh: Union[str, Path],
    mesh_size_min: float = 2.0,
    mesh_size_max: float = 4.5,
    cylinder_depth_mm: float = 40.0,
    inlet_x_mm: float = 60.0,
    inlet_z_mm: float = 34.0,
) -> Dict[str, Union[int, float]]:
  """Mesh the intake port fluid domain using Gmsh and export to MSH 2.2 format.

  Identifies the inlet, outlet, and wall boundary patches based on geometry centroids
  and creates tagged Physical Groups ready for conversion via gmshToFoam.

  Args:
    geometry: build123d Part solid or path to STEP file.
    output_msh: Destination path for the .msh file.
    mesh_size_min: Minimum element characteristic length in mm.
    mesh_size_max: Maximum element characteristic length in mm.
    cylinder_depth_mm: Bottom Z datum for outlet face identification.
    inlet_x_mm: Nominal X coordinate of runner inlet flange.
    inlet_z_mm: Nominal Z coordinate of runner inlet flange.

  Returns:
    Dictionary with mesh statistics (num_nodes, num_elements, num_tets, wall_time_s).
  """
  t0 = time.perf_counter()
  output_msh_path = Path(output_msh)
  output_msh_path.parent.mkdir(parents=True, exist_ok=True)

  # Handle Part/Solid/Shape input by writing a temporary STEP file if needed
  step_path: Path
  temp_step: Optional[Path] = None
  if isinstance(geometry, (str, Path)):
    step_path = Path(geometry)
  else:
    temp_step = output_msh_path.parent / "_temp_port_domain.step"
    export_step(geometry, str(temp_step))
    step_path = temp_step

  if not step_path.exists():
    raise FileNotFoundError(f"Input STEP geometry not found: {step_path}")

  # Initialize Gmsh
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 0)
  gmsh.model.add("port_fluid")

  try:
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    volumes = gmsh.model.getEntities(dim=3)

    if not volumes:
      raise ValueError("No 3D solid volumes found in the provided geometry.")

    inlet_tags: List[int] = []
    outlet_tags: List[int] = []
    wall_tags: List[int] = []

    for dim, tag in surfaces:
      com = gmsh.model.occ.getCenterOfMass(dim, tag)
      # Check runner inlet face: near (inlet_x_mm, 0, inlet_z_mm)
      if abs(com[0] - inlet_x_mm) < 1.5 and abs(com[2] - inlet_z_mm) < 1.5:
        inlet_tags.append(tag)
      # Check cylinder bottom outlet face: near z = -cylinder_depth_mm
      elif abs(com[2] - (-cylinder_depth_mm)) < 0.8:
        outlet_tags.append(tag)
      else:
        wall_tags.append(tag)

    if not inlet_tags:
      raise ValueError(f"Could not identify inlet patch near X={inlet_x_mm}, Z={inlet_z_mm}.")
    if not outlet_tags:
      raise ValueError(f"Could not identify outlet patch near Z={-cylinder_depth_mm}.")

    gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
    gmsh.model.addPhysicalGroup(2, wall_tags, name="walls")
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], name="internal")

    # Set mesh sizing parameters
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    # OpenFOAM gmshToFoam requires MSH version 2.2
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(3)

    nodes = gmsh.model.mesh.getNodes()
    elements_3d = gmsh.model.mesh.getElements(dim=3)
    num_nodes = len(nodes[0]) if nodes and len(nodes) > 0 else 0
    num_tets = len(elements_3d[1][0]) if elements_3d and elements_3d[1] else 0

    gmsh.write(str(output_msh_path))
  finally:
    gmsh.finalize()
    if temp_step and temp_step.exists():
      try:
        temp_step.unlink()
      except OSError:
        pass

  elapsed = time.perf_counter() - t0
  return {
      "num_nodes": num_nodes,
      "num_tets": num_tets,
      "wall_time_s": elapsed,
      "inlet_surfaces": len(inlet_tags),
      "outlet_surfaces": len(outlet_tags),
      "wall_surfaces": len(wall_tags),
  }
