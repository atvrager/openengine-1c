"""Parametric Air-Cooled Cylinder Block and Radial Fin Array Solid Model via build123d.

Generates a watertight B-Rep CAD model of a high-performance single-cylinder 4-stroke engine block:
- Thin-wall centrifugally cast ductile iron cylinder sleeve (68.0 mm inner bore, 74.0 mm OD)
- Cast aluminum barrel with perimeter radial cooling fins sized to shed 1.3 - 1.6 kW heat flux
- Precision top deck mating flush to cylinder head gasket at Z = 0
- 4x M8 head stud clearance holes matching the cylinder head 80 mm x 80 mm pattern
- 10.0 mm heavy base mounting flange with 4x M8 crankcase stud holes (104 mm x 104 mm pattern)
- Lower spigot (78.0 mm OD x 25.0 mm length) piloting into the crankcase bore
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from build123d import (
    Box,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    RectangleRounded,
    extrude,
)


@dataclass
class CylinderBlockParams:
  """Geometric and thermal design parameters for the cylinder block."""

  bore_m: float = 0.068  # 68 mm cylinder bore
  liner_thickness_m: float = 0.003  # 3.0 mm liner wall thickness (74 mm OD)
  liner_length_m: float = 0.115  # 115 mm total liner sleeve length
  block_height_m: float = 0.085  # 85 mm deck-to-deck height
  spigot_length_m: float = 0.025  # 25 mm spigot piloting into crankcase
  spigot_diameter_m: float = 0.078  # 78 mm outer diameter of crankcase spigot
  barrel_diameter_m: float = 0.086  # 86 mm root diameter of aluminum barrel
  top_deck_width_m: float = 0.110  # 110 mm square head gasket deck
  top_deck_thickness_m: float = 0.008  # 8.0 mm thick head deck flange
  base_flange_width_m: float = 0.130  # 130 mm square crankcase mounting flange
  base_flange_thickness_m: float = 0.010  # 10.0 mm thick base flange
  fin_width_m: float = 0.120  # 120 mm outer fin envelope width
  fin_corner_radius_m: float = 0.015  # 15 mm rounded corner radius
  fin_thickness_m: float = 0.0022  # 2.2 mm fin thickness
  fin_pitch_m: float = 0.0075  # 7.5 mm spacing between fin centers
  num_fins: int = 8  # 8 radial cooling fins
  head_bolt_pitch_m: float = 0.080  # 80 mm square bolt circle for head studs
  head_bolt_hole_diam_m: float = 0.009  # 9.0 mm clearance for M8 head studs
  crankcase_bolt_pitch_m: float = 0.104  # 104 mm square bolt circle for base studs
  crankcase_bolt_hole_diam_m: float = 0.009  # 9.0 mm clearance for M8 base studs


def generate_cylinder_liner(params: Optional[CylinderBlockParams] = None) -> Part:
  """Generate a standalone cast iron cylinder liner sleeve solid model."""
  p = params or CylinderBlockParams()
  r_inner = (p.bore_m * 1000.0) / 2.0
  r_outer = r_inner + (p.liner_thickness_m * 1000.0)
  total_length = p.liner_length_m * 1000.0

  with BuildPart() as liner:
    with Locations((0.0, 0.0, -total_length / 2.0)):
      Cylinder(radius=r_outer, height=total_length)
      Cylinder(radius=r_inner, height=total_length + 2.0, mode=Mode.SUBTRACT)

  return liner.part


def generate_cylinder_block(params: Optional[CylinderBlockParams] = None) -> Part:
  """Generate the complete monolithic aluminum cylinder block B-Rep solid model."""
  p = params or CylinderBlockParams()

  bore_r = (p.bore_m * 1000.0) / 2.0
  barrel_r = (p.barrel_diameter_m * 1000.0) / 2.0
  spigot_r = (p.spigot_diameter_m * 1000.0) / 2.0

  block_h = p.block_height_m * 1000.0
  spigot_len = p.spigot_length_m * 1000.0
  deck_t = p.top_deck_thickness_m * 1000.0
  flange_t = p.base_flange_thickness_m * 1000.0

  deck_w = p.top_deck_width_m * 1000.0
  base_w = p.base_flange_width_m * 1000.0
  fin_w = p.fin_width_m * 1000.0
  fin_cr = p.fin_corner_radius_m * 1000.0
  fin_t = p.fin_thickness_m * 1000.0
  fin_pitch = p.fin_pitch_m * 1000.0

  head_stud_pitch = (p.head_bolt_pitch_m * 1000.0) / 2.0
  r_head_hole = (p.head_bolt_hole_diam_m * 1000.0) / 2.0

  base_stud_pitch = (p.crankcase_bolt_pitch_m * 1000.0) / 2.0
  r_base_hole = (p.crankcase_bolt_hole_diam_m * 1000.0) / 2.0

  with BuildPart() as block:
    # 1. Central cylindrical aluminum barrel
    with Locations((0.0, 0.0, -block_h / 2.0)):
      Cylinder(radius=barrel_r, height=block_h)

    # 2. Top head deck flange (mating flush at Z = 0)
    with Locations((0.0, 0.0, -deck_t / 2.0)):
      Box(deck_w, deck_w, deck_t)

    # 3. Base mounting flange (mating to crankcase deck at Z = -block_h)
    with Locations((0.0, 0.0, -block_h + flange_t / 2.0)):
      Box(base_w, base_w, flange_t)

    # 4. Spigot piloting into crankcase bore below base flange
    with Locations((0.0, 0.0, -block_h - spigot_len / 2.0)):
      Cylinder(radius=spigot_r, height=spigot_len)

    # 5. Radial Cooling Fins
    fin_start_z = -deck_t - 2.0
    for i in range(p.num_fins):
      z_fin = fin_start_z - i * fin_pitch
      with Locations((0.0, 0.0, z_fin)):
        with BuildSketch(Plane.XY) as fin_sk:
          RectangleRounded(fin_w, fin_w, fin_cr)
        extrude(fin_sk.sketch, amount=fin_t)

    # 6. Center cylinder bore through entire block and spigot
    total_length = block_h + spigot_len + 6.0
    with Locations((0.0, 0.0, -(block_h + spigot_len) / 2.0)):
      Cylinder(radius=bore_r, height=total_length, mode=Mode.SUBTRACT)

    # 7. Four Cylinder Head Stud Clearance Holes (M8, through entire block)
    for sx in [-head_stud_pitch, head_stud_pitch]:
      for sy in [-head_stud_pitch, head_stud_pitch]:
        with Locations((sx, sy, -block_h / 2.0)):
          Cylinder(radius=r_head_hole, height=block_h + 4.0, mode=Mode.SUBTRACT)

    # 8. Four Crankcase Base Stud Holes (M8, through base flange)
    for bx in [-base_stud_pitch, base_stud_pitch]:
      for by in [-base_stud_pitch, base_stud_pitch]:
        with Locations((bx, by, -block_h + flange_t / 2.0)):
          Cylinder(radius=r_base_hole, height=flange_t + 4.0, mode=Mode.SUBTRACT)

  return block.part


def calculate_block_thermal_capacity(
    params: Optional[CylinderBlockParams] = None,
    air_heat_transfer_coeff: float = 75.0,  # W/(m2.K) forced convection
    temp_diff_k: float = 130.0,  # (T_wall - T_ambient) = 160C - 30C
    fin_efficiency: float = 0.85,
) -> Dict[str, float]:
  """Evaluate exterior cooling fin surface area and continuous heat dissipation capacity."""
  p = params or CylinderBlockParams()
  block = generate_cylinder_block(p)

  total_area_mm2 = float(block.area)
  bore_r = (p.bore_m * 1000.0) / 2.0
  total_length_mm = (p.block_height_m + p.spigot_length_m) * 1000.0
  bore_inner_area_mm2 = 2.0 * np.pi * bore_r * total_length_mm

  # Net exterior cooling area (total area minus inner cylinder bore)
  exterior_area_mm2 = total_area_mm2 - bore_inner_area_mm2
  exterior_area_m2 = exterior_area_mm2 / 1.0e6

  # Heat dissipation: Q = h * A * eta * dT
  q_dissipation_w = air_heat_transfer_coeff * exterior_area_m2 * fin_efficiency * temp_diff_k

  return {
      "total_area_mm2": total_area_mm2,
      "exterior_area_m2": exterior_area_m2,
      "thermal_dissipation_w": q_dissipation_w,
      "barrel_heat_budget_w": 1300.0,
      "thermal_margin_percent": ((q_dissipation_w - 1300.0) / 1300.0) * 100.0,
      "is_adequate": q_dissipation_w >= 1300.0,
  }
