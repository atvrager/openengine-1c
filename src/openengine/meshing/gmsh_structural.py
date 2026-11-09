"""Gmsh structural quadratic tetrahedral meshing (C3D10) for CalculiX FEA."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from build123d import Part, export_step
import gmsh
import numpy as np


@dataclass
class SurfaceFilter:
  """Criteria to identify and group boundary nodes on a geometric surface."""

  name: str
  predicate: Callable[[Tuple[float, float, float], Tuple[float, float, float, float, float, float]], bool]


@dataclass
class StructuralMeshResult:
  """Statistics and metadata for generated quadratic structural mesh."""

  num_nodes: int
  num_elements: int
  element_type: str  # "C3D10"
  node_sets: Dict[str, List[int]] = field(default_factory=dict)
  element_sets: Dict[str, List[int]] = field(default_factory=dict)
  inp_path: str = ""
  wall_time_s: float = 0.0


def format_nset_cards(name: str, node_ids: List[int], chunk_size: int = 10) -> str:
  """Format an Abaqus / CalculiX *NSET card with strict <= 16 items per line.

  Args:
    name: Name of the node set.
    node_ids: Sorted list of 1-based integer node IDs.
    chunk_size: Number of node entries per line (default 10).

  Returns:
    Formatted string with *NSET card and comma-separated node IDs.
  """
  if not node_ids:
    return ""
  lines = [f"*NSET, NSET={name}"]
  for i in range(0, len(node_ids), chunk_size):
    chunk = node_ids[i : i + chunk_size]
    lines.append(", ".join(str(n) for n in chunk))
  return "\n".join(lines) + "\n"


def mesh_structural_part_c3d10(
    geometry: Union[Part, str, Path],
    output_inp_path: Union[str, Path],
    mesh_size_min: float = 2.0,
    mesh_size_max: float = 4.5,
    surface_filters: Optional[List[SurfaceFilter]] = None,
    volume_name: str = "EALL",
) -> StructuralMeshResult:
  """Mesh a solid part into quadratic 10-node tetrahedra (C3D10) and export to CalculiX .inp.

  Args:
    geometry: build123d Part solid or path to existing STEP file.
    output_inp_path: Target path for the output CalculiX .inp deck.
    mesh_size_min: Minimum element characteristic length in mm.
    mesh_size_max: Maximum element characteristic length in mm.
    surface_filters: Optional list of SurfaceFilter predicates to group boundary nodes.
    volume_name: Name of the primary volume element set (ELSET).

  Returns:
    StructuralMeshResult containing node/element counts and tagged node sets.
  """
  t0 = time.perf_counter()
  out_path = Path(output_inp_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  # Resolve geometry input to STEP file
  temp_step: Optional[Path] = None
  if isinstance(geometry, (str, Path)):
    step_file = Path(geometry)
  else:
    temp_dir = Path(tempfile.gettempdir())
    temp_step = temp_dir / f"_temp_part_{os.getpid()}_{int(time.time()*1000)}.step"
    export_step(geometry, str(temp_step))
    step_file = temp_step

  gmsh.initialize()
  try:
    gmsh.option.setNumber("General.Verbosity", 1)  # Quiet output
    gmsh.model.add("structural_part")
    gmsh.model.occ.importShapes(str(step_file))
    gmsh.model.occ.synchronize()

    # Get all 3D solid volumes
    volumes = gmsh.model.getEntities(dim=3)
    if not volumes:
      raise ValueError(f"No 3D solid volumes found in geometry {step_file}")

    vol_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name=volume_name)

    # Configure meshing options
    gmsh.option.setNumber("Mesh.SaveAll", 0)  # Only export physical groups
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    # Generate 3D volume mesh and upgrade to quadratic order 2 (C3D10)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)

    # Export base mesh to temporary file
    temp_raw_inp = out_path.parent / f"_raw_{out_path.name}"
    gmsh.write(str(temp_raw_inp))

    # Identify boundary nodes matching surface filters
    node_sets: Dict[str, List[int]] = {}
    if surface_filters:
      surfaces = gmsh.model.getEntities(dim=2)
      for s_filter in surface_filters:
        matched_nodes: List[int] = []
        for dim, tag in surfaces:
          com = gmsh.model.occ.getCenterOfMass(dim, tag)
          bbox = gmsh.model.occ.getBoundingBox(dim, tag)
          if s_filter.predicate(com, bbox):
            nodes, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=tag, includeBoundary=True)
            matched_nodes.extend(nodes)
        node_sets[s_filter.name] = sorted(list(set(matched_nodes)))

    # Query global mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    num_nodes = len(node_tags)
    num_elements = sum(len(t) for t in elem_tags)

  finally:
    gmsh.finalize()
    if temp_step and temp_step.exists():
      try:
        temp_step.unlink()
      except OSError:
        pass

  # Post-process the raw .inp file from Gmsh:
  # 1. Clean up element card header to ensure TYPE=C3D10 and ELSET=volume_name
  # 2. Append tagged *NSET cards formatted for CalculiX (< 16 items per line)
  with open(temp_raw_inp, "r") as f:
    raw_lines = f.readlines()

  clean_lines = []
  in_element_block = False
  skip_next = False
  for line in raw_lines:
    if skip_next:
      skip_next = False
      continue
    if line.strip().lower().startswith("*heading"):
      skip_next = True
      continue
    if line.startswith("*ELEMENT"):
      # Gmsh may output ELSET=Volume1; ensure consistent ELSET=volume_name
      clean_lines.append(f"*ELEMENT, TYPE=C3D10, ELSET={volume_name}\n")
      in_element_block = True
    elif in_element_block and line.startswith("*"):
      in_element_block = False
      clean_lines.append(line)
    else:
      clean_lines.append(line)

  # Append surface node sets
  for nset_name, nset_nodes in node_sets.items():
    clean_lines.append(format_nset_cards(nset_name, nset_nodes))

  with open(out_path, "w") as f:
    f.writelines(clean_lines)

  if temp_raw_inp.exists():
    try:
      temp_raw_inp.unlink()
    except OSError:
      pass

  wall_time = time.perf_counter() - t0
  return StructuralMeshResult(
      num_nodes=num_nodes,
      num_elements=num_elements,
      element_type="C3D10",
      node_sets=node_sets,
      inp_path=str(out_path),
      wall_time_s=wall_time,
  )
