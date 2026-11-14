"""Strictly Offline McMaster-Carr & Powersports COTS Solid Models + Wrench Zones.

Reads verified SKUs from config/mcmaster_bom.yaml, optionally imports user-supplied
local STEP files from assets/cots_step/<SKU>.step, and synthesizes parametric
watertight build123d solids with clearance fits for Boolean assembly verification.
Contains zero network, HTTP, or socket imports.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import yaml
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
    extrude,
    import_step,
    make_hull,
)

DEFAULT_BOM_YAML_PATH: Path = Path("config/mcmaster_bom.yaml")
DEFAULT_COTS_STEP_DIR: Path = Path("assets/cots_step")

SHCS_HEAD_DIMS_MM: Dict[int, tuple[float, float, float]] = {
    # nominal_d_mm: (head_radius_mm, head_height_mm, modeled_tap_minor_r_mm)
    6: (4.95, 5.8, 2.40),
    8: (6.45, 7.8, 3.25),
}
WASHER_DEFAULT_THICK_MM: float = 1.6

PLUG_THREAD_MAJOR_R_MM: float = 4.90
PLUG_REACH_LEN_MM: float = 15.0
PLUG_HEX_RADIUS_MM: float = 8.6
PLUG_HEX_LEN_MM: float = 11.0
PLUG_CERAMIC_RADIUS_MM: float = 5.2
PLUG_CERAMIC_LEN_MM: float = 24.0
PLUG_TERMINAL_RADIUS_MM: float = 2.8
PLUG_TERMINAL_LEN_MM: float = 8.0

SPROCKET_25_PITCH_MM: float = 6.35
SPROCKET_25_PLATE_W_MM: float = 2.8
SPROCKET_25_HUB_W_MM: float = 7.2
CHAIN_25_CENTER_Y_MM: float = 53.0
CHAIN_25_WIDTH_Y_MM: float = 5.6
CHAIN_25_RADIAL_CLEAR_MM: float = 0.25
CHAIN_25_LINK_THICK_MM: float = 1.9

SPRING_OUTER_R_MM: float = 8.9
SPRING_INNER_R_MM: float = 5.8
RING_AXIAL_CLEAR_MM: float = 0.08
RING_RADIAL_THICK_MM: float = 2.2
RING_OD_RADIUS_MM: float = 33.92

CARB_MOUNT_X_MM: float = 90.3
CARB_BODY_LEN_X_MM: float = 54.0
CARB_VENTURI_INNER_R_MM: float = 13.0
CARB_FLANGE_OUTER_R_MM: float = 17.5
CARB_SLIDE_TOP_Z_MM: float = 88.0
CARB_BOWL_BOT_Z_MM: float = -4.0

HALL_SENSOR_X_MM: float = 39.2
HALL_SENSOR_LEN_X_MM: float = 17.0
HALL_SENSOR_Y_MM: float = -84.0
HALL_SENSOR_RADIUS_MM: float = 4.0
CDI_BOX_CENTER_XYZ_MM: tuple[float, float, float] = (-98.0, -22.0, -165.0)
CDI_BOX_SIZE_MM: tuple[float, float, float] = (18.0, 44.0, 38.0)

ROT_X_90_DEG: float = 90.0
ROT_Y_90_DEG: float = 90.0
CUT_MARGIN_MM: float = 4.0
HALF_FACTOR: float = 0.5

DEFAULT_SHCS_DIAM_MM: float = 6.0
DEFAULT_SHCS_LEN_MM: float = 20.0
SHCS_FALLBACK_HEAD_R_RATIO: float = 0.8
SHCS_FALLBACK_HEAD_H_RATIO: float = 0.95
SHCS_THREAD_OFFSET_R_MM: float = 0.55

DEFAULT_WASHER_ID_MM: float = 6.4
DEFAULT_WASHER_OD_MM: float = 11.8
DEFAULT_DOWEL_DIAM_MM: float = 6.0
DEFAULT_DOWEL_LEN_MM: float = 16.0
DOWEL_RADIAL_CLEAR_MM: float = 0.05

CIRCLIP_START_DEG: float = 20.0
CIRCLIP_END_DEG: float = 340.0
CIRCLIP_ARC_SAMPLES: int = 48
CIRCLIP_OUTER_R_MM: float = 8.5
CIRCLIP_INNER_R_MM: float = 7.5
CIRCLIP_HALF_THICK_MM: float = 0.5

DEFAULT_ORING_ID_MM: float = 20.0
DEFAULT_ORING_CS_MM: float = 2.5
ORING_RADIAL_CLEAR_MM: float = 0.05
ORING_MIN_THICK_MM: float = 0.8
ORING_RADIAL_TOL_MM: float = 0.25
ORING_AXIAL_TOL_MM: float = 0.3

SUMP_ORING_CENTER_X_MM: float = 73.0
SUMP_ORING_CENTER_Y_MM: float = 43.5
SUMP_ORING_THICK_MM: float = 2.2
SUMP_ORING_HEIGHT_MM: float = 1.6
SUMP_ORING_Z_MID_MM: float = -242.0

BRG_6206_BORE_MM: float = 30.2
BRG_6206_OD_MM: float = 61.9
BRG_6206_WIDTH_MM: float = 15.6
SEAL_30X47X7_BORE_MM: float = 30.2
SEAL_30X47X7_OD_MM: float = 46.9
SEAL_30X47X7_WIDTH_MM: float = 6.6
BRG_NKS28_BORE_MM: float = 28.1
BRG_NKS28_OD_MM: float = 41.9
BRG_NKS28_WIDTH_MM: float = 19.6

SIGHT_SHANK_X_MM: float = -4.0
SIGHT_SHANK_R_MM: float = 7.1
SIGHT_SHANK_LEN_MM: float = 8.0
SIGHT_HEAD_X_MM: float = 1.5
SIGHT_HEAD_R_MM: float = 10.5
SIGHT_HEAD_LEN_MM: float = 5.0

FILL_SHANK_Z_MM: float = -5.0
FILL_SHANK_R_MM: float = 9.0
FILL_SHANK_LEN_MM: float = 10.0
FILL_HEAD_Z_MM: float = 3.0
FILL_HEAD_R_MM: float = 12.4
FILL_HEAD_LEN_MM: float = 6.0

DRAIN_SHANK_Z_MM: float = -5.0
DRAIN_SHANK_R_MM: float = 5.1
DRAIN_SHANK_LEN_MM: float = 10.0
DRAIN_HEAD_Z_MM: float = 2.5
DRAIN_HEAD_R_MM: float = 8.5
DRAIN_HEAD_LEN_MM: float = 5.0

BREATHER_SHANK_Z_MM: float = -3.5
BREATHER_SHANK_R_MM: float = 4.1
BREATHER_SHANK_LEN_MM: float = 7.0
BREATHER_CAP_Z_MM: float = 4.0
BREATHER_CAP_R_MM: float = 6.2
BREATHER_CAP_LEN_MM: float = 8.0

DEFAULT_SPROCKET_TEETH: int = 18
DEFAULT_SPROCKET_BORE_MM: float = 20.15
SPROCKET_ADDENDUM_MM: float = 1.15
SPROCKET_HUB_MIN_PAD_MM: float = 3.0
SPROCKET_HUB_MAX_INSET_MM: float = 4.0
SPROCKET_HUB_PREF_PAD_MM: float = 4.5

DEFAULT_CRANK_AXIS_Z_MM: float = -159.0
DEFAULT_CAM_AXIS_Z_MM: float = 76.5
CRANK_SPROCKET_TEETH: int = 18
CAM_SPROCKET_TEETH: int = 36

DEFAULT_SPRING_HEIGHT_MM: float = 24.0
PISTON_RING_SPECS_MM: tuple[tuple[str, float, float], ...] = (
    ("top_ring", 17.5, 1.0),
    ("second_ring", 14.5, 1.0),
    ("oil_ring", 11.2, 2.0),
)
WRIST_PIN_OUTER_R_MM: float = 7.96
WRIST_PIN_INNER_R_MM: float = 5.0
WRIST_PIN_LEN_MM: float = 45.0

CARB_MOUNT_Z_MM: float = 34.0
CARB_SLIDE_TOWER_R_MM: float = 14.0
CARB_BOWL_WIDTH_XY_MM: float = 38.0
HALL_SENSOR_Z_MM: float = -159.0

ZONE_PLUG_TILT_DEG: float = 24.0
ZONE_PLUG_ROT_X_DEG: float = 66.0
ZONE_PLUG_ORIGIN_MM: tuple[float, float, float] = (0.0, -13.5, 8.5)
ZONE_PLUG_SOCK_DIST_MM: float = 75.0
ZONE_PLUG_SOCK_R_MM: float = 9.5
ZONE_PLUG_SOCK_LEN_MM: float = 90.0

ZONE_HEAD_BOLT_XY_OFF_MM: float = 40.0
ZONE_HEAD_SOCKET_Z_MM: float = 96.0
ZONE_HEAD_SOCKET_R_MM: float = 7.0
ZONE_HEAD_SOCKET_LEN_MM: float = 88.0

ZONE_BASE_BOLT_X_MM: float = 51.5
ZONE_BASE_BOLT_Y_MM: float = 26.0
ZONE_BASE_HEX_Z_MM: float = -36.0
ZONE_BASE_HEX_R_MM: float = 6.0
ZONE_BASE_HEX_LEN_MM: float = 68.0

ZONE_DRAIN_POS_MM: tuple[float, float, float] = (-118.0, 0.0, -267.5)
ZONE_DRAIN_R_MM: float = 9.5
ZONE_DRAIN_LEN_MM: float = 60.0

ZONE_SIGHT_POS_MM: tuple[float, float, float] = (116.0, 30.0, -211.0)
ZONE_SIGHT_R_MM: float = 11.0
ZONE_SIGHT_LEN_MM: float = 56.0

ZONE_FILL_TILT_DEG: float = 45.0
ZONE_FILL_BASE_MM: tuple[float, float, float] = (86.0, 30.0, -114.0)
ZONE_FILL_OFFSET_MM: float = 32.0
ZONE_FILL_R_MM: float = 13.0
ZONE_FILL_LEN_MM: float = 55.0


def load_mcmaster_bom_metadata(bom_path: Path = DEFAULT_BOM_YAML_PATH) -> Dict[str, object]:
  """Load config/mcmaster_bom.yaml strictly from the local filesystem."""
  return yaml.safe_load(bom_path.read_text(encoding="utf-8"))


def resolve_cots_sku_catalog(bom_path: Path = DEFAULT_BOM_YAML_PATH) -> Dict[str, str]:
  """Return mapping from item_id to offline SKU string from config/mcmaster_bom.yaml."""
  raw = load_mcmaster_bom_metadata(bom_path)
  catalog: Dict[str, str] = {}
  series_map = raw.get("sku_series", {})
  for item in raw.get("items", []):
    item_id = str(item.get("item_id", ""))
    sku = str(item.get("sku", ""))
    if not sku and item.get("series"):
      fam = series_map.get(str(item["series"]), {})
      sku = str(fam.get("anchor_sku", item_id))
    catalog[item_id] = sku
  return catalog


def load_or_generate_cots_part(
    sku: str,
    generator_fn: Callable[[], Part],
    step_dir: Path = DEFAULT_COTS_STEP_DIR,
) -> Part:
  """Import a local COTS STEP model if cached under assets/cots_step/<SKU>.step, else generate."""
  if sku:
    candidate = step_dir / f"{sku}.step"
    if candidate.exists():
      imported = import_step(str(candidate))
      solids = imported.solids()
      if solids:
        return Part(children=solids)
      return Part(children=[imported])
  return generator_fn()


def generate_shcs(
    thread_diam_mm: float = DEFAULT_SHCS_DIAM_MM,
    length_mm: float = DEFAULT_SHCS_LEN_MM,
    sku: str = "",
) -> Part:
  """Generate an ISO 4762 SHCS with head underside at Z = 0 and shank along -Z."""
  _ = sku
  nominal = round(thread_diam_mm)
  head_r, head_h, shank_r = SHCS_HEAD_DIMS_MM.get(
      nominal,
      (
          thread_diam_mm * SHCS_FALLBACK_HEAD_R_RATIO,
          thread_diam_mm * SHCS_FALLBACK_HEAD_H_RATIO,
          (thread_diam_mm * HALF_FACTOR) - SHCS_THREAD_OFFSET_R_MM,
      ),
  )
  with BuildPart() as screw:
    with Locations((0.0, 0.0, head_h * HALF_FACTOR)):
      Cylinder(radius=head_r, height=head_h)
    with Locations((0.0, 0.0, -length_mm * HALF_FACTOR)):
      Cylinder(radius=shank_r, height=length_mm)
  return screw.part


def generate_flat_washer(
    inner_diam_mm: float = DEFAULT_WASHER_ID_MM,
    outer_diam_mm: float = DEFAULT_WASHER_OD_MM,
    thickness_mm: float = WASHER_DEFAULT_THICK_MM,
    sku: str = "",
) -> Part:
  """Generate a DIN 125 flat washer sitting on Z in [0, thickness_mm]."""
  _ = sku
  with BuildPart() as washer:
    with Locations((0.0, 0.0, thickness_mm * HALF_FACTOR)):
      Cylinder(radius=outer_diam_mm * HALF_FACTOR, height=thickness_mm)
      Cylinder(radius=inner_diam_mm * HALF_FACTOR, height=thickness_mm + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
  return washer.part


def generate_dowel_pin(diam_mm: float = DEFAULT_DOWEL_DIAM_MM, length_mm: float = DEFAULT_DOWEL_LEN_MM, sku: str = "") -> Part:
  """Generate a precision ISO 2338 316SS alignment dowel centered at the origin along Z."""
  _ = sku
  with BuildPart() as pin:
    Cylinder(radius=(diam_mm * HALF_FACTOR) - DOWEL_RADIAL_CLEAR_MM, height=length_mm)
  return pin.part


def generate_external_circlip_16mm() -> Part:
  """Generate a DIN 471 external wrist-pin retaining ring (98541A143) centered at origin along Y."""
  with BuildSketch(Plane.XZ) as sk:
    pts = []
    for deg in np.linspace(CIRCLIP_START_DEG, CIRCLIP_END_DEG, CIRCLIP_ARC_SAMPLES):
      rad = np.radians(deg)
      pts.append((CIRCLIP_OUTER_R_MM * float(np.cos(rad)), CIRCLIP_OUTER_R_MM * float(np.sin(rad))))
    for deg in np.linspace(CIRCLIP_END_DEG, CIRCLIP_START_DEG, CIRCLIP_ARC_SAMPLES):
      rad = np.radians(deg)
      pts.append((CIRCLIP_INNER_R_MM * float(np.cos(rad)), CIRCLIP_INNER_R_MM * float(np.sin(rad))))
    Polygon(pts)
  with BuildPart() as clip:
    extrude(sk.sketch, amount=CIRCLIP_HALF_THICK_MM, both=True)
  return clip.part


def generate_viton_oring(id_mm: float = DEFAULT_ORING_ID_MM, cs_mm: float = DEFAULT_ORING_CS_MM, sku: str = "") -> Part:
  """Generate a metric Viton O-ring centered in XY plane with minor clearance."""
  _ = sku
  r_in = (id_mm * HALF_FACTOR) + ORING_RADIAL_CLEAR_MM
  r_out = r_in + max(ORING_MIN_THICK_MM, cs_mm - ORING_RADIAL_TOL_MM)
  with BuildPart() as oring:
    Cylinder(radius=r_out, height=max(ORING_MIN_THICK_MM, cs_mm - ORING_AXIAL_TOL_MM))
    Cylinder(radius=r_in, height=cs_mm + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
  return oring.part


def generate_sump_perimeter_oring() -> Part:
  """Generate the 2.5 mm Viton perimeter O-ring seated inside the sump top groove."""
  cx = SUMP_ORING_CENTER_X_MM
  cy = SUMP_ORING_CENTER_Y_MM
  thick_xy = SUMP_ORING_THICK_MM
  h_z = SUMP_ORING_HEIGHT_MM
  z_mid = SUMP_ORING_Z_MID_MM
  with BuildPart() as seal:
    with Locations((0.0, cy, z_mid), (0.0, -cy, z_mid)):
      Box((cx * 2.0) + thick_xy, thick_xy, h_z)
    with Locations((cx, 0.0, z_mid), (-cx, 0.0, z_mid)):
      Box(thick_xy, (cy * 2.0) + thick_xy, h_z)
  return seal.part


def generate_annular_bearing(
    bore_mm: float,
    od_mm: float,
    width_mm: float,
    axis_along_x: bool = False,
) -> Part:
  """Generate an annular roller/ball bearing or rotary shaft seal solid along Y or X."""
  rot = (0.0, ROT_Y_90_DEG, 0.0) if axis_along_x else (ROT_X_90_DEG, 0.0, 0.0)
  with BuildPart() as brg:
    Cylinder(radius=od_mm * HALF_FACTOR, height=width_mm, rotation=rot)
    Cylinder(radius=bore_mm * HALF_FACTOR, height=width_mm + CUT_MARGIN_MM, rotation=rot, mode=Mode.SUBTRACT)
  return brg.part


def generate_ball_bearing_6206() -> Part:
  """Generate the McMaster 5972K342 6206-2Z shielded main ball bearing along Y."""
  return generate_annular_bearing(bore_mm=BRG_6206_BORE_MM, od_mm=BRG_6206_OD_MM, width_mm=BRG_6206_WIDTH_MM)


def generate_viton_shaft_seal_30x47x7() -> Part:
  """Generate the McMaster 5154T114 30x47x7 mm Viton rotary shaft seal along Y."""
  return generate_annular_bearing(bore_mm=SEAL_30X47X7_BORE_MM, od_mm=SEAL_30X47X7_OD_MM, width_mm=SEAL_30X47X7_WIDTH_MM)


def generate_needle_bearing_nks28() -> Part:
  """Generate the Schaeffler INA NKS28-XL (28x42x20 mm) big-end needle bearing along X."""
  return generate_annular_bearing(
      bore_mm=BRG_NKS28_BORE_MM,
      od_mm=BRG_NKS28_OD_MM,
      width_mm=BRG_NKS28_WIDTH_MM,
      axis_along_x=True,
  )


def generate_sight_glass_m16() -> Part:
  """Generate the McMaster 1079K18 M16x1.5 Viton glass sight-level plug along +X."""
  with BuildPart() as glass:
    with Locations((SIGHT_SHANK_X_MM, 0.0, 0.0)):
      Cylinder(radius=SIGHT_SHANK_R_MM, height=SIGHT_SHANK_LEN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
    with Locations((SIGHT_HEAD_X_MM, 0.0, 0.0)):
      Cylinder(radius=SIGHT_HEAD_R_MM, height=SIGHT_HEAD_LEN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
  return glass.part


def generate_oil_fill_plug_m20() -> Part:
  """Generate the M20x1.5 threaded oil fill plug along +Z."""
  with BuildPart() as plug:
    with Locations((0.0, 0.0, FILL_SHANK_Z_MM)):
      Cylinder(radius=FILL_SHANK_R_MM, height=FILL_SHANK_LEN_MM)
    with Locations((0.0, 0.0, FILL_HEAD_Z_MM)):
      Cylinder(radius=FILL_HEAD_R_MM, height=FILL_HEAD_LEN_MM)
  return plug.part


def generate_magnetic_drain_plug_m12() -> Part:
  """Generate the M12x1.5 neodymium magnetic sump drain plug pointing along +Z into port."""
  with BuildPart() as plug:
    with Locations((0.0, 0.0, DRAIN_SHANK_Z_MM)):
      Cylinder(radius=DRAIN_SHANK_R_MM, height=DRAIN_SHANK_LEN_MM)
    with Locations((0.0, 0.0, DRAIN_HEAD_Z_MM)):
      Cylinder(radius=DRAIN_HEAD_R_MM, height=DRAIN_HEAD_LEN_MM)
  return plug.part


def generate_breather_vent_npt() -> Part:
  """Generate the McMaster 9833K21 1/8-27 NPT sintered bronze breather along Z."""
  with BuildPart() as vent:
    with Locations((0.0, 0.0, BREATHER_SHANK_Z_MM)):
      Cylinder(radius=BREATHER_SHANK_R_MM, height=BREATHER_SHANK_LEN_MM)
    with Locations((0.0, 0.0, BREATHER_CAP_Z_MM)):
      Cylinder(radius=BREATHER_CAP_R_MM, height=BREATHER_CAP_LEN_MM)
  return vent.part


def generate_sprocket_25(teeth: int = DEFAULT_SPROCKET_TEETH, bore_mm: float = DEFAULT_SPROCKET_BORE_MM) -> Part:
  """Generate an ANSI #25 (1/4 in pitch) steel sprocket centered on Y axis."""
  r_pitch = SPROCKET_25_PITCH_MM / (2.0 * float(np.sin(np.pi / teeth)))
  r_outer = r_pitch + SPROCKET_ADDENDUM_MM
  r_hub = max(
      (bore_mm * HALF_FACTOR) + SPROCKET_HUB_MIN_PAD_MM,
      min(r_pitch - SPROCKET_HUB_MAX_INSET_MM, (bore_mm * HALF_FACTOR) + SPROCKET_HUB_PREF_PAD_MM),
  )
  with BuildPart() as sp:
    Cylinder(radius=r_outer, height=SPROCKET_25_PLATE_W_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))
    Cylinder(radius=r_hub, height=SPROCKET_25_HUB_W_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))
    Cylinder(radius=bore_mm * HALF_FACTOR, height=SPROCKET_25_HUB_W_MM + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)
  return sp.part


def generate_roller_chain_25_loop(
    z_crank_mm: float = DEFAULT_CRANK_AXIS_Z_MM,
    z_cam_mm: float = DEFAULT_CAM_AXIS_Z_MM,
    y_center_mm: float = CHAIN_25_CENTER_Y_MM,
) -> Part:
  """Generate the closed ANSI #25 timing chain envelope wrapping 18T crank and 36T cam sprockets."""
  r_pitch_18 = SPROCKET_25_PITCH_MM / (2.0 * float(np.sin(np.pi / CRANK_SPROCKET_TEETH)))
  r_pitch_36 = SPROCKET_25_PITCH_MM / (2.0 * float(np.sin(np.pi / CAM_SPROCKET_TEETH)))
  r_in_18 = r_pitch_18 + SPROCKET_ADDENDUM_MM + CHAIN_25_RADIAL_CLEAR_MM
  r_in_36 = r_pitch_36 + SPROCKET_ADDENDUM_MM + CHAIN_25_RADIAL_CLEAR_MM
  r_out_18 = r_in_18 + CHAIN_25_LINK_THICK_MM
  r_out_36 = r_in_36 + CHAIN_25_LINK_THICK_MM

  def _hull(r_bot: float, r_top: float) -> Part:
    y_back = y_center_mm + (CHAIN_25_WIDTH_Y_MM * HALF_FACTOR)
    pl = Plane(origin=(0.0, y_back, 0.0), x_dir=(1.0, 0.0, 0.0), z_dir=(0.0, -1.0, 0.0))
    with BuildPart() as h:
      with BuildSketch(pl):
        with Locations((0.0, z_crank_mm)):
          Circle(radius=r_bot)
        with Locations((0.0, z_cam_mm)):
          Circle(radius=r_top)
        make_hull()
      extrude(amount=CHAIN_25_WIDTH_Y_MM)
    return h.part

  return _hull(r_out_18, r_out_36) - _hull(r_in_18, r_in_36)


def generate_valve_spring_9573k35(compressed_height_mm: float = DEFAULT_SPRING_HEIGHT_MM) -> Part:
  """Generate the chrome-silicon valve spring (McMaster 9573K35) on Z in [0, height]."""
  with BuildPart() as spring:
    with Locations((0.0, 0.0, compressed_height_mm * HALF_FACTOR)):
      Cylinder(radius=SPRING_OUTER_R_MM, height=compressed_height_mm)
      Cylinder(radius=SPRING_INNER_R_MM, height=compressed_height_mm + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
  return spring.part


def generate_piston_ring_pack_68mm() -> Dict[str, Part]:
  """Generate top, second, and oil 68.0 mm GX200 piston rings centered on piston local Z."""
  rings: Dict[str, Part] = {}
  r_inner = RING_OD_RADIUS_MM - RING_RADIAL_THICK_MM
  for name, z_c, width_z in PISTON_RING_SPECS_MM:
    with BuildPart() as ring:
      with Locations((0.0, 0.0, z_c)):
        Cylinder(radius=RING_OD_RADIUS_MM, height=width_z - (2.0 * RING_AXIAL_CLEAR_MM))
        Cylinder(radius=r_inner, height=width_z + CUT_MARGIN_MM, mode=Mode.SUBTRACT)
    rings[name] = ring.part
  return rings


def generate_wrist_pin_16x45() -> Part:
  """Generate the hollow ground 4140 16x45 mm floating wrist pin along Y."""
  with BuildPart() as pin:
    Cylinder(radius=WRIST_PIN_OUTER_R_MM, height=WRIST_PIN_LEN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0))
    Cylinder(radius=WRIST_PIN_INNER_R_MM, height=WRIST_PIN_LEN_MM + CUT_MARGIN_MM, rotation=(ROT_X_90_DEG, 0.0, 0.0), mode=Mode.SUBTRACT)
  return pin.part


def generate_spark_plug_cmr7h() -> Part:
  """Generate the NGK CMR7H compact M10x1.0 spark plug along +Z from tip s = 0."""
  s1 = PLUG_REACH_LEN_MM
  s2 = s1 + PLUG_HEX_LEN_MM
  s3 = s2 + PLUG_CERAMIC_LEN_MM
  s4 = s3 + PLUG_TERMINAL_LEN_MM
  with BuildPart() as plug:
    with Locations((0.0, 0.0, s1 * HALF_FACTOR)):
      Cylinder(radius=PLUG_THREAD_MAJOR_R_MM, height=s1)
    with Locations((0.0, 0.0, s1 + (PLUG_HEX_LEN_MM * HALF_FACTOR))):
      Cylinder(radius=PLUG_HEX_RADIUS_MM, height=PLUG_HEX_LEN_MM)
    with Locations((0.0, 0.0, s2 + (PLUG_CERAMIC_LEN_MM * HALF_FACTOR))):
      Cylinder(radius=PLUG_CERAMIC_RADIUS_MM, height=PLUG_CERAMIC_LEN_MM)
    with Locations((0.0, 0.0, s3 + (PLUG_TERMINAL_LEN_MM * HALF_FACTOR))):
      Cylinder(radius=PLUG_TERMINAL_RADIUS_MM, height=s4 - s3)
  return plug.part


def generate_carburetor_vm26() -> Part:
  """Generate the Mikuni VM26 26 mm mechanical round-slide carburetor in world coords."""
  mid_x = CARB_MOUNT_X_MM + (CARB_BODY_LEN_X_MM * HALF_FACTOR)
  tower_h = CARB_SLIDE_TOP_Z_MM - CARB_MOUNT_Z_MM
  bowl_h = CARB_MOUNT_Z_MM - CARB_BOWL_BOT_Z_MM
  with BuildPart() as carb:
    with Locations((mid_x, 0.0, CARB_MOUNT_Z_MM)):
      Cylinder(radius=CARB_FLANGE_OUTER_R_MM, height=CARB_BODY_LEN_X_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
    with Locations((mid_x, 0.0, CARB_MOUNT_Z_MM + (tower_h * HALF_FACTOR))):
      Cylinder(radius=CARB_SLIDE_TOWER_R_MM, height=tower_h)
    with Locations((mid_x, 0.0, CARB_MOUNT_Z_MM - (bowl_h * HALF_FACTOR))):
      Box(CARB_BOWL_WIDTH_XY_MM, CARB_BOWL_WIDTH_XY_MM, bowl_h)
    with Locations((mid_x, 0.0, CARB_MOUNT_Z_MM)):
      Cylinder(radius=CARB_VENTURI_INNER_R_MM, height=CARB_BODY_LEN_X_MM + CUT_MARGIN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0), mode=Mode.SUBTRACT)
  return carb.part


def generate_hall_pickup_and_cdi() -> Dict[str, Part]:
  """Generate the Hall-effect trigger pickup and CDI module envelope in world coords."""
  with BuildPart() as pickup:
    with Locations((HALL_SENSOR_X_MM + (HALL_SENSOR_LEN_X_MM * HALF_FACTOR), HALL_SENSOR_Y_MM, HALL_SENSOR_Z_MM)):
      Cylinder(radius=HALL_SENSOR_RADIUS_MM, height=HALL_SENSOR_LEN_X_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
  with BuildPart() as cdi:
    with Locations(CDI_BOX_CENTER_XYZ_MM):
      Box(*CDI_BOX_SIZE_MM)
  return {"hall_pickup": pickup.part, "cdi_module": cdi.part}


def generate_wrench_clearance_zones() -> Dict[str, Part]:
  """Generate tool-access keep-out cylinders for spark plug, fasteners, and oil ports."""
  zones: Dict[str, Part] = {}

  tilt_rad = np.radians(ZONE_PLUG_TILT_DEG)
  plug_origin = np.array(list(ZONE_PLUG_ORIGIN_MM))
  plug_dir = np.array([0.0, -float(np.cos(tilt_rad)), float(np.sin(tilt_rad))])
  sock_center = plug_origin + (plug_dir * ZONE_PLUG_SOCK_DIST_MM)
  with BuildPart() as plug_zone:
    with Locations(tuple(sock_center.tolist())):
      Cylinder(radius=ZONE_PLUG_SOCK_R_MM, height=ZONE_PLUG_SOCK_LEN_MM, rotation=(ZONE_PLUG_ROT_X_DEG, 0.0, 0.0))
  zones["spark_plug_socket"] = plug_zone.part

  with BuildPart() as head_zone:
    for hx in (-ZONE_HEAD_BOLT_XY_OFF_MM, ZONE_HEAD_BOLT_XY_OFF_MM):
      for hy in (-ZONE_HEAD_BOLT_XY_OFF_MM, ZONE_HEAD_BOLT_XY_OFF_MM):
        with Locations((hx, hy, ZONE_HEAD_SOCKET_Z_MM)):
          Cylinder(radius=ZONE_HEAD_SOCKET_R_MM, height=ZONE_HEAD_SOCKET_LEN_MM)
  zones["head_tie_bolt_sockets"] = head_zone.part

  with BuildPart() as base_zone:
    for bx in (-ZONE_BASE_BOLT_X_MM, ZONE_BASE_BOLT_X_MM):
      for by in (-ZONE_BASE_BOLT_Y_MM, ZONE_BASE_BOLT_Y_MM):
        with Locations((bx, by, ZONE_BASE_HEX_Z_MM)):
          Cylinder(radius=ZONE_BASE_HEX_R_MM, height=ZONE_BASE_HEX_LEN_MM)
  zones["barrel_base_hex_keys"] = base_zone.part

  with BuildPart() as drain_zone:
    with Locations(ZONE_DRAIN_POS_MM):
      Cylinder(radius=ZONE_DRAIN_R_MM, height=ZONE_DRAIN_LEN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
  zones["oil_drain_socket"] = drain_zone.part

  with BuildPart() as sight_zone:
    with Locations(ZONE_SIGHT_POS_MM):
      Cylinder(radius=ZONE_SIGHT_R_MM, height=ZONE_SIGHT_LEN_MM, rotation=(0.0, ROT_Y_90_DEG, 0.0))
  zones["sight_glass_socket"] = sight_zone.part

  fill_dir = np.array([float(np.sin(np.radians(ZONE_FILL_TILT_DEG))), 0.0, float(np.cos(np.radians(ZONE_FILL_TILT_DEG)))])
  fill_center = np.array(list(ZONE_FILL_BASE_MM)) + (fill_dir * ZONE_FILL_OFFSET_MM)
  with BuildPart() as fill_zone:
    with Locations(tuple(fill_center.tolist())):
      Cylinder(radius=ZONE_FILL_R_MM, height=ZONE_FILL_LEN_MM, rotation=(0.0, ZONE_FILL_TILT_DEG, 0.0))
  zones["oil_fill_clearance"] = fill_zone.part

  return zones
