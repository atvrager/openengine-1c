"""Parametric Connecting Rod Solid Model via build123d / Open CASCADE."""

from build123d import (
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Part,
    Plane,
    Rectangle,
    extrude,
)


def generate_connecting_rod(
    center_to_center_m: float = 0.108,
    wrist_pin_diam_m: float = 0.016,
    crankpin_diam_m: float = 0.028,
    small_end_width_m: float = 0.018,
    big_end_width_m: float = 0.020,
    shank_flange_w_m: float = 0.012,
    shank_flange_t_m: float = 0.0035,
    shank_web_t_m: float = 0.0030,
    shank_height_m: float = 0.016,
) -> Part:
  """Generate a parametric solid B-Rep connecting rod using build123d."""
  # Convert meters to millimeters for standard CAD unit consistency
  l_mm = center_to_center_m * 1000.0
  d_pin_mm = wrist_pin_diam_m * 1000.0
  d_crank_mm = crankpin_diam_m * 1000.0
  w_small_mm = small_end_width_m * 1000.0
  w_big_mm = big_end_width_m * 1000.0

  flange_w = shank_flange_w_m * 1000.0
  flange_t = shank_flange_t_m * 1000.0
  web_t = shank_web_t_m * 1000.0
  height = shank_height_m * 1000.0

  r_small_outer = (d_pin_mm / 2.0) + 3.5
  r_big_outer = (d_crank_mm / 2.0) + 5.5

  with BuildPart() as rod:
    # 1. Small End (at origin (0, 0, 0))
    Cylinder(
        radius=r_small_outer,
        height=w_small_mm,
        rotation=(0, 90, 0),
    )
    Cylinder(
        radius=d_pin_mm / 2.0,
        height=w_small_mm + 2.0,
        rotation=(0, 90, 0),
        mode=Mode.SUBTRACT,
    )

    # 2. Big End (at (0, l_mm, 0))
    with Locations((0, l_mm, 0)):
      Cylinder(
          radius=r_big_outer,
          height=w_big_mm,
          rotation=(0, 90, 0),
      )
      Cylinder(
          radius=d_crank_mm / 2.0,
          height=w_big_mm + 2.0,
          rotation=(0, 90, 0),
          mode=Mode.SUBTRACT,
      )

    # 3. I-Beam Shank (swept between the two journals)
    shank_length = l_mm - r_small_outer - r_big_outer
    shank_start_y = r_small_outer

    with BuildSketch(Plane.XZ.offset(shank_start_y)) as shank_profile:
      # Outer envelope
      Rectangle(flange_w, height)
      # Pockets for the I-beam webs
      pocket_w = (flange_w - web_t) / 2.0
      pocket_h = height - (2.0 * flange_t)
      with Locations(
          (-flange_w / 2.0 + pocket_w / 2.0, 0),
          (flange_w / 2.0 - pocket_w / 2.0, 0),
      ):
        Rectangle(pocket_w, pocket_h, mode=Mode.SUBTRACT)

    # Extrude shank along Y axis to connect to big end
    extrude(amount=shank_length)

  return rod.part
