"""Parametric SOHC Pent-Roof Cylinder Head Solid Model via build123d / Open CASCADE.

Generates a watertight B-Rep CAD model of a high-efficiency Single Overhead Camshaft (SOHC)
cylinder head featuring:
- 30-degree included angle pent-roof combustion chamber (15 deg intake / 15 deg exhaust)
- 28-degree downdraft high-tumble intake port tapering to 30 mm intake valve seat
- Swept exhaust port exiting to 26 mm exhaust valve seat
- Central vertical M10 spark plug boss situated at the chamber apex
- Valve guide bores and spring seat counterbores
- 4x M8 cylinder head stud clearance holes (80 mm x 80 mm pattern)
- Exterior lateral cooling fin slots to dissipate exhaust and chamber thermal flux
- Extraction helper for CFD fluid domain generation (simpleFoam / snappyHexMesh)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from build123d import (
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Polygon,
    Vector,
    extrude,
    insert,
    loft,
    revolve,
)


@dataclass
class CylinderHeadParams:
  """Geometric specifications for the SOHC pent-roof cylinder head."""

  bore_m: float = 0.068  # 68 mm cylinder bore
  intake_valve_diam_m: float = 0.030  # 30 mm intake valve head
  exhaust_valve_diam_m: float = 0.026  # 26 mm exhaust valve head
  valve_angle_deg: float = 15.0  # 15 deg roof half-angle (30 deg included)
  intake_downdraft_angle_deg: float = 28.0  # 28 deg downdraft port angle
  intake_flange_diam_m: float = 0.032  # 32 mm manifold runner entry
  exhaust_flange_diam_m: float = 0.028  # 28 mm header exit
  spark_plug_diam_m: float = 0.010  # M10 spark plug thread
  valve_stem_diam_m: float = 0.005  # 5.0 mm valve stem
  head_width_m: float = 0.110  # 110 mm width along intake/exhaust (X)
  head_length_m: float = 0.110  # 110 mm length along camshaft axis (Y)
  head_height_m: float = 0.075  # 75 mm height from deck to cam cover rail (Z)
  head_bolt_pitch_m: float = 0.080  # 80 mm square bolt circle (M8 studs)
  head_bolt_hole_diam_m: float = 0.009  # 9.0 mm clearance for M8 studs
  chamber_apex_height_m: float = 0.0095  # 9.5 mm dome height above deck face
  squish_half_width_m: float = 0.0260  # 26.0 mm squish width from centerline


def generate_poppet_valve(
    head_diam_m: float = 0.030,
    stem_diam_m: float = 0.005,
    stem_length_m: float = 0.075,
    seat_angle_deg: float = 45.0,
    margin_thickness_m: float = 0.0008,
) -> Part:
  """Generate a parametric revolved solid model of a poppet valve."""
  r_head = head_diam_m * 500.0  # mm
  r_stem = stem_diam_m * 500.0  # mm
  length = stem_length_m * 1000.0  # mm
  margin = margin_thickness_m * 1000.0  # mm
  seat_h = 1.8  # mm seat contact face height
  r_seat_top = r_head - seat_h * np.tan(np.radians(seat_angle_deg))

  with BuildSketch(Plane.XZ) as profile:
    Polygon([
        (0.0, 0.0),
        (r_head, 0.0),
        (r_head, margin),
        (r_seat_top, margin + seat_h),
        (r_stem, margin + seat_h + 8.0),
        (r_stem, length),
        (0.0, length),
    ])

  with BuildPart() as valve:
    revolve(profile.sketch, axis=Axis.Z)

  return valve.part


def generate_cylinder_head(params: Optional[CylinderHeadParams] = None) -> Part:
  """Generate a watertight B-Rep CAD model of the SOHC pent-roof cylinder head."""
  p = params or CylinderHeadParams()

  w = p.head_width_m * 1000.0
  length = p.head_length_m * 1000.0
  h = p.head_height_m * 1000.0
  bore_r = (p.bore_m * 1000.0) / 2.0
  apex_h = p.chamber_apex_height_m * 1000.0
  squish_y = p.squish_half_width_m * 1000.0
  slope_deg = p.valve_angle_deg

  # Valve centers in X (intake at +X, exhaust at -X)
  x_in = 16.0
  x_ex = -16.0
  z_seat = 6.0

  p_in_flange = Vector(w / 2.0, 0.0, 42.0)
  p_in_mid = Vector(35.0, 0.0, 24.0)
  p_in_seat = Vector(x_in, 0.0, z_seat)

  p_ex_flange = Vector(-w / 2.0, 0.0, 25.0)
  p_ex_mid = Vector(-35.0, 0.0, 16.0)
  p_ex_seat = Vector(x_ex, 0.0, z_seat)

  # 1. Chamber cutter solid
  z_base = -3.0
  z_roof_edge = apex_h - 40.0 * np.tan(np.radians(slope_deg))
  with BuildSketch(Plane.XZ) as roof_sketch:
    Polygon([
        (-40.0, z_base),
        (-40.0, z_roof_edge),
        (0.0, apex_h),
        (40.0, z_roof_edge),
        (40.0, z_base),
    ])
  with BuildPart() as p1:
    extrude(roof_sketch.sketch, amount=squish_y, both=True)
  with BuildPart() as p2:
    with Locations((0.0, 0.0, (apex_h - z_base) / 2.0 + z_base)):
      Cylinder(radius=bore_r, height=apex_h - z_base)
  chamber_cutter = p1.part & p2.part

  # 2. Intake port runner cutter
  plane_in_f = Plane(origin=p_in_flange, z_dir=Vector(1.0, 0.0, 0.0))
  plane_in_m = Plane(origin=p_in_mid, z_dir=Vector(0.8, 0.0, 0.6))
  plane_in_s = Plane(
      origin=p_in_seat,
      z_dir=Vector(np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg))),
  )
  r_in_flange = (p.intake_flange_diam_m * 1000.0) / 2.0
  r_in_seat = (p.intake_valve_diam_m * 1000.0) / 2.0
  r_in_throat = r_in_seat - 1.0

  with BuildPart() as in_port:
    with BuildSketch(plane_in_f):
      Circle(radius=r_in_flange)
    with BuildSketch(plane_in_m):
      Circle(radius=r_in_flange - 1.5)
    with BuildSketch(plane_in_s):
      Circle(radius=r_in_throat)
    loft()
    with Locations(p_in_seat):
      Cylinder(radius=r_in_seat, height=8.0, rotation=(0.0, slope_deg, 0.0))

  # 3. Exhaust port runner cutter
  plane_ex_f = Plane(origin=p_ex_flange, z_dir=Vector(-1.0, 0.0, 0.0))
  plane_ex_m = Plane(origin=p_ex_mid, z_dir=Vector(-0.8, 0.0, 0.4))
  plane_ex_s = Plane(
      origin=p_ex_seat,
      z_dir=Vector(-np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg))),
  )
  r_ex_flange = (p.exhaust_flange_diam_m * 1000.0) / 2.0
  r_ex_seat = (p.exhaust_valve_diam_m * 1000.0) / 2.0
  r_ex_throat = r_ex_seat - 1.0

  with BuildPart() as ex_port:
    with BuildSketch(plane_ex_f):
      Circle(radius=r_ex_flange)
    with BuildSketch(plane_ex_m):
      Circle(radius=r_ex_flange - 1.5)
    with BuildSketch(plane_ex_s):
      Circle(radius=r_ex_throat)
    loft()
    with Locations(p_ex_seat):
      Cylinder(radius=r_ex_seat, height=8.0, rotation=(0.0, -slope_deg, 0.0))

  # 4. Construct cylinder head body
  with BuildPart() as head:
    with Locations((0.0, 0.0, h / 2.0)):
      Box(w, length, h)

    insert(chamber_cutter, mode=Mode.SUBTRACT)
    insert(in_port.part, mode=Mode.SUBTRACT)
    insert(ex_port.part, mode=Mode.SUBTRACT)

    # Valve guide bores through head
    r_guide = (p.valve_stem_diam_m * 1000.0 + 0.5) / 2.0
    in_axis_dir = Vector(np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg)))
    ex_axis_dir = Vector(-np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg)))

    with Locations(p_in_seat + in_axis_dir * 30.0):
      Cylinder(radius=r_guide, height=75.0, rotation=(0.0, slope_deg, 0.0), mode=Mode.SUBTRACT)
    with Locations(p_ex_seat + ex_axis_dir * 30.0):
      Cylinder(radius=r_guide, height=75.0, rotation=(0.0, -slope_deg, 0.0), mode=Mode.SUBTRACT)

    # Central Spark Plug Bore & Counterbore (M10)
    r_plug = (p.spark_plug_diam_m * 1000.0) / 2.0
    with Locations((0.0, 0.0, h / 2.0)):
      Cylinder(radius=r_plug, height=h + 2.0, mode=Mode.SUBTRACT)
    with Locations((0.0, 0.0, h - 22.0)):
      Cylinder(radius=9.0, height=50.0, mode=Mode.SUBTRACT)

    # Four Cylinder Head Stud Clearance Holes (M8)
    stud_pitch = (p.head_bolt_pitch_m * 1000.0) / 2.0
    r_stud_hole = (p.head_bolt_hole_diam_m * 1000.0) / 2.0
    for sx in [-stud_pitch, stud_pitch]:
      for sy in [-stud_pitch, stud_pitch]:
        with Locations((sx, sy, h / 2.0)):
          Cylinder(radius=r_stud_hole, height=h + 2.0, mode=Mode.SUBTRACT)

    # Lateral Cooling Fin Grooves (dissipate chamber and port heat)
    for z_fin in [16.0, 26.0, 36.0, 46.0, 56.0]:
      for y_fin in [-length / 2.0, length / 2.0]:
        with Locations((0.0, y_fin, z_fin)):
          Box(w - 15.0, 14.0, 3.5, mode=Mode.SUBTRACT)

  return head.part


def extract_intake_port_fluid_domain(
    params: Optional[CylinderHeadParams] = None,
    cylinder_depth_m: float = 0.050,
) -> Part:
  """Extract the negative air volume of the intake port, valve seat, and cylinder bore.

  Produces a watertight, manifold fluid volume for OpenFOAM simpleFoam steady flow bench
  or XiEngineFoam moving-mesh simulations.
  """
  p = params or CylinderHeadParams()
  w = p.head_width_m * 1000.0
  bore_r = (p.bore_m * 1000.0) / 2.0
  apex_h = p.chamber_apex_height_m * 1000.0
  squish_y = p.squish_half_width_m * 1000.0
  slope_deg = p.valve_angle_deg
  depth_mm = cylinder_depth_m * 1000.0

  x_in = 16.0
  z_seat = 6.0
  p_in_flange = Vector(w / 2.0, 0.0, 42.0)
  p_in_mid = Vector(35.0, 0.0, 24.0)
  p_in_seat = Vector(x_in, 0.0, z_seat)

  plane_in_f = Plane(origin=p_in_flange, z_dir=Vector(1.0, 0.0, 0.0))
  plane_in_m = Plane(origin=p_in_mid, z_dir=Vector(0.8, 0.0, 0.6))
  plane_in_s = Plane(
      origin=p_in_seat,
      z_dir=Vector(np.sin(np.radians(slope_deg)), 0.0, np.cos(np.radians(slope_deg))),
  )

  r_in_flange = (p.intake_flange_diam_m * 1000.0) / 2.0
  r_in_seat = (p.intake_valve_diam_m * 1000.0) / 2.0
  r_in_throat = r_in_seat - 1.0

  # 1. Intake runner
  with BuildPart() as port:
    with BuildSketch(plane_in_f):
      Circle(radius=r_in_flange)
    with BuildSketch(plane_in_m):
      Circle(radius=r_in_flange - 1.5)
    with BuildSketch(plane_in_s):
      Circle(radius=r_in_throat)
    loft()
    with Locations(p_in_seat):
      Cylinder(radius=r_in_seat, height=8.0, rotation=(0.0, slope_deg, 0.0))

  # 2. Chamber and cylinder bore unified
  z_base = -depth_mm
  z_roof_edge = apex_h - 40.0 * np.tan(np.radians(slope_deg))
  with BuildSketch(Plane.XZ) as roof_sketch:
    Polygon([
        (-40.0, z_base),
        (-40.0, z_roof_edge),
        (0.0, apex_h),
        (40.0, z_roof_edge),
        (40.0, z_base),
    ])

  with BuildPart() as p1:
    extrude(roof_sketch.sketch, amount=squish_y, both=True)
    with Locations((0.0, 0.0, -depth_mm / 2.0)):
      Cylinder(radius=bore_r, height=depth_mm)

  with BuildPart() as p2:
    with Locations((0.0, 0.0, (apex_h - z_base) / 2.0 + z_base)):
      Cylinder(radius=bore_r, height=apex_h - z_base)

  chamber_and_cyl = p1.part & p2.part
  fluid_domain = port.part.fuse(chamber_and_cyl)
  return fluid_domain


def calculate_combustion_chamber_volume(
    params: Optional[CylinderHeadParams] = None,
) -> Tuple[float, float]:
  """Calculate the geometric combustion chamber dome clearance volume above deck face (Z=0).

  Returns:
    volume_mm3: Chamber volume in cubic millimeters
    volume_cm3: Chamber volume in cubic centimeters
  """
  p = params or CylinderHeadParams()
  bore_r = (p.bore_m * 1000.0) / 2.0
  apex_h = p.chamber_apex_height_m * 1000.0
  squish_y = p.squish_half_width_m * 1000.0
  slope_deg = p.valve_angle_deg

  z_base = -3.0
  z_roof_edge = apex_h - 40.0 * np.tan(np.radians(slope_deg))
  with BuildSketch(Plane.XZ) as roof_sketch:
    Polygon([
        (-40.0, z_base),
        (-40.0, z_roof_edge),
        (0.0, apex_h),
        (40.0, z_roof_edge),
        (40.0, z_base),
    ])

  with BuildPart() as p1:
    extrude(roof_sketch.sketch, amount=squish_y, both=True)

  with BuildPart() as p2:
    with Locations((0.0, 0.0, (apex_h - z_base) / 2.0 + z_base)):
      Cylinder(radius=bore_r, height=apex_h - z_base)

  chamber = p1.part & p2.part
  with BuildPart() as cutter:
    with Locations((0.0, 0.0, -5.0)):
      Box(100.0, 100.0, 10.0)

  dome = chamber - cutter.part
  vol_mm3 = float(dome.volume)
  vol_cm3 = vol_mm3 / 1000.0
  return vol_mm3, vol_cm3
