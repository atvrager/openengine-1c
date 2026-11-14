"""2D Orthographic CNC Setup Sheet Exporter (SVG and DXF) for NestWorks C500.

Generates multi-view 2D drawings (Top XY, Front XZ, Isometric) via build123d
and annotates SVG setup sheets with datum pins, thread-mill specs, H7/p6 press
and shrink fits, and hand-ream instructions for offline workshop use.
"""

from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import re
from typing import Callable, Dict, List, Tuple, Union

from build123d import Axis, Compound, ExportDXF, ExportSVG, LineType, Part, Shape, Unit

from openengine.cad.connecting_rod import generate_connecting_rod_parts
from openengine.cad.covers_and_aux import generate_flywheel
from openengine.cad.crankcase import generate_crankcase_left, generate_sump_pan
from openengine.cad.crankshaft import generate_crankshaft_parts
from openengine.cad.cylinder_block import generate_cylinder_block
from openengine.cad.cylinder_head import generate_cylinder_head
from openengine.cad.piston import generate_piston
from openengine.cad.valvetrain_hardware import (
    generate_cam_tower,
    generate_sohc_camshaft,
    generate_sohc_camshaft_parts,
)
from openengine.dfm.c500_spec import (
    MODE_FALLBACK_COTS_OR_3AXIS,
    MODE_ROTARY_4_AXIS,
    MODE_THREE_AXIS_FOUR_SIDE_FLIP,
    MODE_THREE_AXIS_HOUSING,
)

ROT_90_DEG: float = 90.0
HALF_FACTOR: float = 0.5
VIEW_DIST_MM: float = 800.0
ISO_COORD_MM: float = 600.0
VIEW_GAP_MM: float = 25.0
ROUND_DIM_DECIMALS: int = 2

MIN_SHEET_WIDTH_MM: float = 160.0
TITLE_BLOCK_HEIGHT_MM: float = 48.0
TITLE_BLOCK_MARGIN_MM: float = 4.0
TITLE_TEXT_INSET_MM: float = 3.0
TITLE_FONT_SIZE_MM: float = 3.6
LINE_STEP_MM: float = 5.4
LINE_IDX_TITLE: int = 1
LINE_IDX_MODE: int = 2
LINE_IDX_DIMS: int = 3
LINE_IDX_DATUM: int = 4
LINE_IDX_THREADS: int = 5
LINE_IDX_FITS: int = 6
LINE_IDX_NOTES: int = 7

LAYER_VISIBLE: str = "Visible"
LAYER_HIDDEN: str = "Hidden"
HIDDEN_LINE_WEIGHT: float = 0.25
VISIBLE_LINE_WEIGHT: float = 0.5

TOP_ORIGIN: Tuple[float, float, float] = (0.0, 0.0, VIEW_DIST_MM)
TOP_UP: Tuple[float, float, float] = (0.0, 1.0, 0.0)
FRONT_ORIGIN: Tuple[float, float, float] = (0.0, -VIEW_DIST_MM, 0.0)
FRONT_UP: Tuple[float, float, float] = (0.0, 0.0, 1.0)
ISO_ORIGIN: Tuple[float, float, float] = (ISO_COORD_MM, -ISO_COORD_MM, ISO_COORD_MM)
ISO_UP: Tuple[float, float, float] = (0.0, 0.0, 1.0)

VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"')
SVG_WIDTH_RE = re.compile(r'width="([^"]+)"')
SVG_HEIGHT_RE = re.compile(r'height="([^"]+)"')


@dataclass(frozen=True, slots=True)
class PartSetupSpec:
  """Shop setup sheet callouts for one NestWorks C500 billet component."""

  sheet_id: str
  title: str
  material: str
  base_mode: str
  is_rotary_dependent: bool
  datum_callout: str
  thread_callout: str
  tolerance_callout: str
  ream_and_ops_notes: str


@dataclass(frozen=True, slots=True)
class SetupSheetArtifact:
  """Generated SVG and DXF setup sheet metadata for one component."""

  sheet_id: str
  title: str
  material: str
  active_mode: str
  measured_dims_mm: Tuple[float, float, float]
  svg_path: str
  dxf_path: str
  datum_callout: str
  thread_callout: str
  tolerance_callout: str


@dataclass(frozen=True, slots=True)
class SetupSheetManifest:
  """Complete manifest of 2D orthographic C500 CNC setup sheets."""

  has_fourth_axis: bool
  output_dir: str
  sheets: Tuple[SetupSheetArtifact, ...]

  def to_dict(self) -> Dict[str, object]:
    """Serialize setup sheet manifest to a dictionary."""
    return asdict(self)

  def to_json(self, filepath: Union[str, Path]) -> Path:
    """Write setup sheet manifest JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(self.to_dict(), indent=2))
    return out_path


PART_SETUP_SPECS: Tuple[PartSetupSpec, ...] = (
    PartSetupSpec(
        sheet_id="cylinder_head",
        title="SOHC Pent-Roof Cylinder Head",
        material="6061-T6 Billet",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 bore center, deck down/up + 2x 6.00 H7 deck dowels",
        thread_callout="Thread mill: 4x M8x1.25 through-clear, M6x1.0 flanges/tower, M10x1.0 spark plug",
        tolerance_callout="10.05 H7/p6 bronze guides & 31.4/27.4 H7/p6 seat shrink fit (180 C head)",
        ream_and_ops_notes="Op1 deck/chamber, Op2 valve pockets/ports, hand-ream 5.00 H7 guide ID",
    ),
    PartSetupSpec(
        sheet_id="cylinder_block",
        title="Finned 4-Side Cylinder Barrel & DOM Liner",
        material="6061-T6 Billet + DOM A513-T5 Liner",
        base_mode=MODE_THREE_AXIS_FOUR_SIDE_FLIP,
        is_rotary_dependent=False,
        datum_callout="G54 cylinder bore center + 4-side square index fixture",
        thread_callout="Thread mill: 4x M8x1.25 head tie-bolt blind bores (22 mm deep)",
        tolerance_callout="74.00 H7/s6 shrink fit barrel-to-liner (-0.05 mm cold, heat barrel 170 C)",
        ream_and_ops_notes="Bore & hone liner 68.00 H7 after shrink fit; deck square within 0.02 mm",
    ),
    PartSetupSpec(
        sheet_id="crankcase_left",
        title="Split Crankcase Left Half (PTO / Timing Half)",
        material="6061-T6 Billet",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 main bearing axis + 2x 6.00 H7 case split dowels",
        thread_callout="Thread mill: M8x1.25 base, M6x1.0 perimeter/covers, M20x1.5 fill, 1/8-27 NPT vent",
        tolerance_callout="62.00 H7/j5 6206-2Z main bearing bore; 47.00 H7 Viton 30x47x7 seal bore",
        ream_and_ops_notes="Hand-ream 6.00 H7 dowel bores prior to final split-line kiss pass",
    ),
    PartSetupSpec(
        sheet_id="sump_pan",
        title="Sloped Wet-Sump Oil Pan (0.65 L Pool)",
        material="6061-T6 Billet",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 top gasket flange + sump perimeter pin stops",
        thread_callout="Thread mill: M12x1.5 magnetic drain plug, M16x1.5 sight glass, M6x1.0 flange",
        tolerance_callout="2.50 mm Viton O-ring cord groove depth 1.90 +/-0.05 mm",
        ream_and_ops_notes="Machine internal baffle ribs and 2-deg drain slope in single 3-axis setup",
    ),
    PartSetupSpec(
        sheet_id="connecting_rod",
        title="Split H-Beam Connecting Rod & Cap",
        material="7075-T6 Billet",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 big-end bore center + 2x 5.00 H7 cap alignment dowels",
        thread_callout="Thread mill: 2x M6x1.0 Class 12.9 cap bolt holes (5.0 mm pilot)",
        tolerance_callout="42.00 H7 big-end (INA NKS28-XL slip/light press), 16.00 H7 small-end",
        ream_and_ops_notes="Torque cap bolts to 14 Nm with 5 mm dowels installed, then hand-ream 16.00 H7 & 42.00 H7",
    ),
    PartSetupSpec(
        sheet_id="crankshaft_web",
        title="3-Piece Crankshaft Cheek & Counterweight Web",
        material="1144 Stressproof Steel",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 main shaft bore axis (Z = -159 mm engine frame)",
        thread_callout="Thread mill: M8x1.25 flywheel/sprocket nose retain taps",
        tolerance_callout="28.00 H7/p6 crankpin press fit & 30.00 H7/p6 main journal press fit",
        ream_and_ops_notes="Press pin & journals at 40-ton arbor press; dial-indicate runout <= 0.025 mm",
    ),
    PartSetupSpec(
        sheet_id="piston",
        title="Three-Ring Slipper Piston (68.0 mm GX200 Pack)",
        material="4032-T6 Billet Aluminum",
        base_mode=MODE_ROTARY_4_AXIS,
        is_rotary_dependent=True,
        datum_callout="A-axis rotary chuck on skirt spigot + crown boss center",
        thread_callout="Thread mill: None (circlip DIN 471 external retention)",
        tolerance_callout="16.00 H7/g6 floating wrist pin bore; 67.87 mm barrel skirt profile",
        ream_and_ops_notes="Hand-ream 16.00 H7 pin bore; fallback to GX200 COTS piston if no 4th axis",
    ),
    PartSetupSpec(
        sheet_id="cam_tower",
        title="SOHC Camshaft Bearing Tower & Rocker Pedestal",
        material="6061-T6 Billet",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 cam center + head deck mounting face down",
        thread_callout="Thread mill: 4x M6x1.0 cam-cover perimeter taps (14 mm deep)",
        tolerance_callout="20.00 H7 SOHC cam plain journal bores (18 um radial running clearance)",
        ream_and_ops_notes="Line-ream 20.00 H7 cam journals & 8.00 H7 rocker bores before honing",
    ),
    PartSetupSpec(
        sheet_id="sohc_camshaft",
        title="SOHC 4-5-6-7 Polynomial Camshaft (2:1 Drive)",
        material="4140 Prehard Steel",
        base_mode=MODE_ROTARY_4_AXIS,
        is_rotary_dependent=True,
        datum_callout="A-axis 4th-axis indexer + 5.00 H7 sprocket index pin",
        thread_callout="Thread mill: M6x1.0 front sprocket retaining tap",
        tolerance_callout="20.00 g6 bearing journals; +/-0.015 mm 4-5-6-7 lobe lift contour",
        ream_and_ops_notes="Polish journals to Ra 0.2 um; fallback to keyed core + 2.5D lobe rings on 3-axis",
    ),
    PartSetupSpec(
        sheet_id="flywheel",
        title="Stepped Inertial Flywheel (160 mm C500 OD)",
        material="1144 Stressproof Steel",
        base_mode=MODE_THREE_AXIS_HOUSING,
        is_rotary_dependent=False,
        datum_callout="G54 central crankshaft hub bore + trigger-wheel bolt circle",
        thread_callout="Thread mill: 3x M6x1.0 trigger-wheel mounting taps",
        tolerance_callout="20.20 H7/p6 hub fit on crankshaft PTO taper/shoulder",
        ream_and_ops_notes="Single 3-axis pocket step on web; dynamically check balance before firing",
    ),
)


def _center_shape_at_origin(shape: Shape) -> Tuple[Shape, Tuple[float, float, float]]:
  """Translate any build123d solid or compound so its bounding box centers on (0,0,0)."""
  bb = shape.bounding_box()
  dx = float(bb.max.X - bb.min.X)
  dy = float(bb.max.Y - bb.min.Y)
  dz = float(bb.max.Z - bb.min.Z)
  cx = (float(bb.min.X) + float(bb.max.X)) * HALF_FACTOR
  cy = (float(bb.min.Y) + float(bb.max.Y)) * HALF_FACTOR
  cz = (float(bb.min.Z) + float(bb.max.Z)) * HALF_FACTOR
  dims = (round(dx, ROUND_DIM_DECIMALS), round(dy, ROUND_DIM_DECIMALS), round(dz, ROUND_DIM_DECIMALS))
  return shape.translate((-cx, -cy, -cz)), dims


def _build_drawing_shape(sheet_id: str, has_fourth_axis: bool) -> Shape:
  """Instantiate the B-Rep solid posed cleanly for 2D orthographic setup projection."""
  factories: Dict[str, Callable[[], Shape]] = {
      "cylinder_head": generate_cylinder_head,
      "cylinder_block": generate_cylinder_block,
      "crankcase_left": lambda: generate_crankcase_left().rotate(Axis.X, ROT_90_DEG),
      "sump_pan": generate_sump_pan,
      "connecting_rod": lambda: Compound(list(generate_connecting_rod_parts())).rotate(Axis.Y, ROT_90_DEG).rotate(Axis.Z, ROT_90_DEG),
      "crankshaft_web": lambda: generate_crankshaft_parts().left_web.rotate(Axis.X, ROT_90_DEG),
      "piston": generate_piston,
      "cam_tower": generate_cam_tower,
      "sohc_camshaft": lambda: (
          generate_sohc_camshaft(True).rotate(Axis.X, ROT_90_DEG)
          if has_fourth_axis
          else generate_sohc_camshaft_parts(False)["intake_lobe_2_5d"].rotate(Axis.X, ROT_90_DEG)
      ),
      "flywheel": lambda: generate_flywheel().rotate(Axis.X, ROT_90_DEG),
  }
  return factories[sheet_id]()


def _project_arranged_views(shape: Shape) -> Tuple[List[Shape], List[Shape]]:
  """Project Top, Front, and Isometric views arranged horizontally without overlap."""
  centered, _ = _center_shape_at_origin(shape)
  views = (
      (TOP_ORIGIN, TOP_UP),
      (FRONT_ORIGIN, FRONT_UP),
      (ISO_ORIGIN, ISO_UP),
  )
  vis_shapes: List[Shape] = []
  hid_shapes: List[Shape] = []
  cursor_x = 0.0

  for origin, up_vec in views:
    vis_edges, hid_edges = centered.project_to_viewport(origin, viewport_up=up_vec)
    all_edges = list(vis_edges) + list(hid_edges)
    if not all_edges:
      continue
    group = Compound(all_edges)
    bb = group.bounding_box()
    shift_x = cursor_x - float(bb.min.X)
    if list(vis_edges):
      vis_shapes.append(Compound(list(vis_edges)).translate((shift_x, 0.0, 0.0)))
    if list(hid_edges):
      hid_shapes.append(Compound(list(hid_edges)).translate((shift_x, 0.0, 0.0)))
    cursor_x += float(bb.max.X - bb.min.X) + VIEW_GAP_MM

  return vis_shapes, hid_shapes


def _export_dxf_sheet(vis_shapes: List[Shape], hid_shapes: List[Shape], dxf_path: Path) -> None:
  """Write visible and hidden projected 2D views to a millimeter DXF file."""
  dxf = ExportDXF(unit=Unit.MM)
  dxf.add_layer(LAYER_VISIBLE, line_type=LineType.CONTINUOUS, line_weight=VISIBLE_LINE_WEIGHT)
  dxf.add_layer(LAYER_HIDDEN, line_type=LineType.DASHED, line_weight=HIDDEN_LINE_WEIGHT)
  for shp in vis_shapes:
    dxf.add_shape(shp, layer=LAYER_VISIBLE)
  for shp in hid_shapes:
    dxf.add_shape(shp, layer=LAYER_HIDDEN)
  dxf.write(dxf_path)


def _export_annotated_svg_sheet(
    vis_shapes: List[Shape],
    hid_shapes: List[Shape],
    spec: PartSetupSpec,
    active_mode: str,
    dims_mm: Tuple[float, float, float],
    svg_path: Path,
) -> None:
  """Export projected views to SVG and inject datum, thread-mill, and fit callouts."""
  svg = ExportSVG(unit=Unit.MM)
  svg.add_layer(LAYER_VISIBLE, line_type=LineType.CONTINUOUS, line_weight=VISIBLE_LINE_WEIGHT)
  svg.add_layer(LAYER_HIDDEN, line_type=LineType.DASHED, line_weight=HIDDEN_LINE_WEIGHT)
  for shp in vis_shapes:
    svg.add_shape(shp, layer=LAYER_VISIBLE)
  for shp in hid_shapes:
    svg.add_shape(shp, layer=LAYER_HIDDEN)
  svg.write(svg_path)

  raw_svg = svg_path.read_text()
  vb_match = VIEWBOX_RE.search(raw_svg)
  if vb_match is None:
    return

  vx, vy, vw, vh = (float(tok) for tok in vb_match.group(1).split())
  panel_w = max(vw, MIN_SHEET_WIDTH_MM)
  panel_x = vx - ((panel_w - vw) * HALF_FACTOR)
  panel_y = vy + vh + TITLE_BLOCK_MARGIN_MM
  new_vh = vh + TITLE_BLOCK_MARGIN_MM + TITLE_BLOCK_HEIGHT_MM

  rows = (
      (LINE_IDX_TITLE, f"{spec.title} ({spec.sheet_id}) | Material: {spec.material}"),
      (LINE_IDX_MODE, f"NestWorks C500 Mode: {active_mode} | Views: Top / Front / Iso"),
      (LINE_IDX_DIMS, f"B-Rep Envelope: {dims_mm[0]:.2f} x {dims_mm[1]:.2f} x {dims_mm[2]:.2f} mm"),
      (LINE_IDX_DATUM, f"Datum: {spec.datum_callout}"),
      (LINE_IDX_THREADS, f"{spec.thread_callout}"),
      (LINE_IDX_FITS, f"Fits: {spec.tolerance_callout}"),
      (LINE_IDX_NOTES, f"Ops: {spec.ream_and_ops_notes}"),
  )
  text_x = panel_x + TITLE_TEXT_INSET_MM
  note_lines = "\n".join(
      f'    <text x="{text_x:.2f}" y="{panel_y + (idx * LINE_STEP_MM):.2f}" '
      f'font-family="monospace" font-size="{TITLE_FONT_SIZE_MM:.2f}" fill="#111111">'
      f"{html.escape(text)}</text>"
      for idx, text in rows
  )
  annotation = (
      f'\n  <g id="c500-setup-notes">\n'
      f'    <rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_w:.2f}" '
      f'height="{TITLE_BLOCK_HEIGHT_MM:.2f}" fill="#f8fafc" stroke="#1e293b" stroke-width="0.4" />\n'
      f"{note_lines}\n"
      f"  </g>\n</svg>"
  )

  updated = VIEWBOX_RE.sub(
      f'viewBox="{panel_x:.3f} {vy:.3f} {panel_w:.3f} {new_vh:.3f}"',
      raw_svg,
      count=1,
  )
  updated = SVG_WIDTH_RE.sub(f'width="{panel_w:.3f}mm"', updated, count=1)
  updated = SVG_HEIGHT_RE.sub(f'height="{new_vh:.3f}mm"', updated, count=1)
  updated = updated.replace("</svg>", annotation)
  svg_path.write_text(updated)


def export_c500_setup_sheets(
    output_dir: Union[str, Path] = "outputs/dfm/drawings",
    has_fourth_axis: bool = True,
) -> SetupSheetManifest:
  """Export 2D orthographic SVG and DXF C500 setup sheets for all billet parts."""
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  sheets: List[SetupSheetArtifact] = []

  for spec in PART_SETUP_SPECS:
    active_mode = (
        MODE_FALLBACK_COTS_OR_3AXIS
        if (spec.is_rotary_dependent and not has_fourth_axis)
        else spec.base_mode
    )
    shape = _build_drawing_shape(spec.sheet_id, has_fourth_axis=has_fourth_axis)
    _, dims_mm = _center_shape_at_origin(shape)
    vis_shapes, hid_shapes = _project_arranged_views(shape)

    svg_file = out_path / f"{spec.sheet_id}_setup.svg"
    dxf_file = out_path / f"{spec.sheet_id}_setup.dxf"
    _export_dxf_sheet(vis_shapes, hid_shapes, dxf_file)
    _export_annotated_svg_sheet(vis_shapes, hid_shapes, spec, active_mode, dims_mm, svg_file)

    sheets.append(
        SetupSheetArtifact(
            sheet_id=spec.sheet_id,
            title=spec.title,
            material=spec.material,
            active_mode=active_mode,
            measured_dims_mm=dims_mm,
            svg_path=str(svg_file),
            dxf_path=str(dxf_file),
            datum_callout=spec.datum_callout,
            thread_callout=spec.thread_callout,
            tolerance_callout=spec.tolerance_callout,
        )
    )

  manifest = SetupSheetManifest(
      has_fourth_axis=has_fourth_axis,
      output_dir=str(out_path),
      sheets=tuple(sheets),
  )
  manifest.to_json(out_path / "setup_sheets_manifest.json")
  return manifest
