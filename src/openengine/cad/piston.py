"""Parametric Slipper Piston Solid Model via build123d / Open CASCADE."""

from build123d import (
    BuildPart,
    Cylinder,
    Locations,
    Mode,
    Part,
)


def generate_piston(
    bore_m: float = 0.068,
    wrist_pin_diam_m: float = 0.016,
    compression_height_m: float = 0.024,
    total_height_m: float = 0.042,
    crown_thickness_m: float = 0.0055,
    skirt_wall_thickness_m: float = 0.0025,
) -> Part:
  """Generate a parametric solid B-Rep piston using build123d."""
  b_mm = bore_m * 1000.0
  r_outer = b_mm / 2.0
  h_total = total_height_m * 1000.0
  t_crown = crown_thickness_m * 1000.0
  d_pin = wrist_pin_diam_m * 1000.0
  pin_z = (h_total / 2.0) - (compression_height_m * 1000.0)

  with BuildPart() as piston:
    # 1. Main outer cylindrical body
    Cylinder(
        radius=r_outer,
        height=h_total,
    )

    # 2. Ring Grooves (Top compression ring groove)
    # Top land: 4 mm from crown top, 1.2 mm groove width, 2.5 mm radial depth
    ring1_z = (h_total / 2.0) - 4.0 - 0.6
    with Locations((0, 0, ring1_z)):
      Cylinder(
          radius=r_outer + 1.0,
          height=1.2,
          mode=Mode.SUBTRACT,
      )
      Cylinder(
          radius=r_outer - 2.5,
          height=1.2,
          mode=Mode.ADD,
      )

    # 3. Hollow out underside skirt cavity (leaving crown and skirt wall)
    r_inner = r_outer - (skirt_wall_thickness_m * 1000.0)
    h_cavity = h_total - t_crown
    with Locations((0, 0, -t_crown / 2.0)):
      Cylinder(
          radius=r_inner,
          height=h_cavity + 1.0,
          mode=Mode.SUBTRACT,
      )

    # 4. Cut wrist pin bore through skirt
    with Locations((0, 0, pin_z)):
      Cylinder(
          radius=d_pin / 2.0,
          height=b_mm + 4.0,
          rotation=(90, 0, 0),
          mode=Mode.SUBTRACT,
      )

  return piston.part
