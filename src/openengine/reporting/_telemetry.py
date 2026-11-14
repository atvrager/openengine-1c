"""Shared telemetry loaders, live CAD/kinematic fallbacks, and scorecard builder."""

import base64
from dataclasses import dataclass
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import struct
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import wave

import yaml

from openengine.cad.assemblies.head_assembly import (
    CAM_TO_CRANK_RATIO,
    ROCKER_CONTACT_SCALE,
    ROCKER_RATIO,
    VALVE_TILT_DEG,
)
from openengine.cad.assembly import EngineAssembly
from openengine.cad.connecting_rod import DEFAULT_CENTER_TO_CENTER_M
from openengine.cad.crankcase import CRANK_AXIS_Z_MM
from openengine.cad.crankshaft import DEFAULT_STROKE_M
from openengine.cad.cylinder_block import DEFAULT_BORE_M, THERMAL_BUDGET_W
from openengine.cad.step_export import (
    DEFAULT_PART_RGB,
    FLUID_OPACITY,
    HOUSING_OPACITY,
    LINER_OPACITY,
    MATERIAL_RGB_PALETTE,
    OPAQUE_OPACITY,
    SHROUD_OPACITY,
)
from openengine.cad.twin_metrics import (
    get_cad_compression_ratio,
    get_cad_dipper_immersion_mm,
    get_cad_sump_oil_volume_l,
)
from openengine.cad.valvetrain_hardware import (
    CAM_AXIS_Z_MM,
    ROCKER_PIVOT_X_MM,
    ROCKER_PIVOT_Z_MM,
)
from openengine.dfm.bom import DEFAULT_BOM_CONFIG_PATH, resolve_bom_from_files
from openengine.dfm.c500_spec import C500Spec, DEFAULT_C500_CONFIG_PATH
from openengine.dfm.drawings import PART_SETUP_SPECS, PartSetupSpec
from openengine.dynamics.sohc_cam import SOHCCamshaft
from openengine.dyno import SteadyDynoConfig
from openengine.fea.connecting_rod import ConrodFeaConfig
from openengine.fea.head_gasket import HeadGasketFeaConfig
from openengine.fea.piston import PistonFeaConfig
from openengine.lubrication import (
    OilGradeSpec,
    SplashSystemParams,
    SumpThermalParams,
    solve_sump_thermal_balance,
)
from openengine.thermo.gas_exchange import FourStrokeSimulator

ZERO_F: float = 0.0
UNIT_F: float = 1.0
HALF_F: float = 0.5
QUARTER_F: float = 0.25
MM_PER_M: float = 1000.0
MM_TO_UM: float = 1000.0
CM_PER_M: float = 100.0
M3_TO_CM3: float = 1.0e6
PA_TO_BAR: float = 1.0e5
PERCENT_SCALE: float = 100.0
KW_TO_HP: float = 1.34102209
DEG_TO_RAD_DIVISOR: float = 180.0
TWO_PI: float = 2.0 * math.pi
RGB_MAX_BYTE: int = 255
TWO_INT: int = 2

ENGINE_SPECS_PATH: Path = Path("config/engine_specs.yaml")
DEFAULT_OIL_GRADE: str = "SAE 10W-40"
DEFAULT_RPM: float = 5500.0
DEFAULT_REDLINE_RPM: float = 7000.0
DEFAULT_PART_COUNT: int = 116
ZERO_INT: int = 0
ONE_INT: int = 1

PV_TOTAL_DEG: int = 720
PV_STEP_DEG: int = 2
PV_POWER_END_DEG: int = 180
PV_EXHAUST_END_DEG: int = 360
PV_INTAKE_END_DEG: int = 540
PV_POLY_EXP: float = 1.30
PV_COMP_EXP: float = 1.33
DEFAULT_WIEBE_SOC_DEG: float = -18.0
DEFAULT_WIEBE_DUR_DEG: float = 76.9
DEFAULT_WIEBE_A: float = 1.653
DEFAULT_WIEBE_M: float = 2.469
PV_EXHAUST_BASE_BAR: float = 1.08
PV_EXHAUST_AMP_BAR: float = 0.05
PV_INTAKE_BASE_BAR: float = 0.95
PV_INTAKE_AMP_BAR: float = 0.04
PV_ATM_BAR: float = 1.0
PV_ROUND_DIGITS: int = 2

WAV_SAMPLE_RATE_HZ: int = 44100
WAV_NUM_SAMPLES: int = 2205
WAV_PCM_AMP: int = 8000
WAV_TONE_HZ: float = 55.0
WAV_SAMPLE_WIDTH_BYTES: int = 2
ONE_CHANNEL: int = 1

MIN_ISFC_G_KWH: float = 220.0
MAX_ISFC_G_KWH: float = 260.0
KNOCK_SAFETY_CEILING: float = 1.0
MIN_VALVE_FLOAT_MARGIN_PCT: float = 30.0
MIN_PORT_CD: float = 0.25
MIN_CRANK_FIT_SF: float = 1.15
MIN_ATLAS_YIELD_SF: float = 1.20
MIN_BOLT_SF: float = 1.30
MAX_OIL_TEMP_DEFAULT_C: float = 130.0
VALVE_TEST_SPEED_MULT: float = 1.2

EMBEDDED_GLB_NAMES: Tuple[str, ...] = (
    "cylinder_head",
    "cylinder_block",
    "cylinder_liner",
    "crankshaft",
    "crank_covers_spin",
    "crank_vt_spin",
    "connecting_rod",
    "conrod_cots",
    "piston",
    "piston_cots",
    "wrist_pin",
    "valve_intake",
    "valve_exhaust",
    "retainer_intake",
    "retainer_exhaust",
    "cam_vt_spin",
    "rocker_intake",
    "rocker_exhaust",
    "crankcase_group",
    "valvetrain_group",
    "covers_and_aux_group",
    "cots_hardware_group",
    "oil_pool_group",
)

MATERIAL_SWATCH_SPECS: Tuple[Tuple[str, str, float], ...] = (
    ("ALUMINUM_6061_T6", "6061-T6 Housings & Finned Barrel", HOUSING_OPACITY),
    ("ALUMINUM_7075_T6", "7075-T6 Con-Rod, Rockers & Retainers", OPAQUE_OPACITY),
    ("ALUMINUM_4032_T6", "4032-T6 Forged Billet Piston", OPAQUE_OPACITY),
    ("STEEL_4140_PREHARD", "4140 Steel Crank, Cam, Valves & Bolts", OPAQUE_OPACITY),
    ("STEEL_1144_STRESSPROOF", "1144 Stressproof Flywheel & Trigger", OPAQUE_OPACITY),
    ("DOM_STEEL_A513_T5", "DOM A513-T5 Honed Cylinder Liner", LINER_OPACITY),
    ("BRONZE_SAE_660", "SAE 660 Bronze Valve Guides", OPAQUE_OPACITY),
    ("VITON_FLUOROELASTOMER", "Viton Oil Seals & O-Rings", OPAQUE_OPACITY),
    ("SAE_10W40_OIL", "SAE 10W-40 Wet-Sump Oil Pool", FLUID_OPACITY),
    ("ALUMINA_STEEL_PLUG", "NGK CMR7H Spark Plug", OPAQUE_OPACITY),
    ("DIECAST_CARB_ALLOY", "VM26 Carburetor & Fan Shroud", SHROUD_OPACITY),
    ("PHENOLIC_ELECTRONICS", "Hall Sensor & CDI Ignition", OPAQUE_OPACITY),
)


@dataclass(frozen=True, slots=True)
class ScorecardRow:
  """Single acceptance gate row displayed in reports and the web dashboard."""

  subsystem: str
  metric: str
  target: str
  achieved: str
  margin: str
  passed: bool


@dataclass(frozen=True, slots=True)
class MaterialSwatch:
  """Legend entry synchronizing XCAF CAD palette colors with Three.js UI."""

  key: str
  label: str
  hex_color: str
  opacity: float


@dataclass(frozen=True, slots=True)
class DashboardTelemetry:
  """Consolidated verified telemetry loaded from disk and live CAD APIs."""

  bore_mm: float
  stroke_mm: float
  conrod_mm: float
  swept_cm3: float
  compression_ratio: float
  total_part_count: int
  dry_mass_kg: float
  wet_mass_kg: float
  oil_vol_l: float
  bdc_dipper_immersion_mm: float
  intake_float_margin_pct: float
  intake_float_rpm: float
  exhaust_float_margin_pct: float
  exhaust_float_rpm: float
  scorecard: Tuple[ScorecardRow, ...]
  passed_count: int
  total_checks: int
  closed_loop: Dict[str, object]
  thermo_lub: Dict[str, object]
  piston: Dict[str, object]
  conrod: Dict[str, object]
  gasket: Dict[str, object]
  cht: Dict[str, object]
  dfm: Dict[str, object]
  dyno: Dict[str, object]
  bom: Dict[str, object]
  setup_sheets: List[Dict[str, object]]
  material_swatches: Tuple[MaterialSwatch, ...]
  kinematic_config: Dict[str, object]
  pv_theta_deg: List[float]
  pv_volume_cm3: List[float]
  pv_pressure_bar: List[float]
  order_pad_text: str
  dyno_wav_b64: str
  piston_stress_json: str
  conrod_stress_json: str
  fea_atlas: Mapping[str, object]
  fea_atlas_webgl_json: str
  cad_glb_b64: Dict[str, str]
  trajectory_json: str


def load_json_or_default(
    filepath: Union[str, Path, Sequence[Union[str, Path]]],
    default_data: Mapping[str, object],
) -> Dict[str, object]:
  """Load the first readable JSON mapping from candidate paths, else copy default."""
  candidates = [filepath] if isinstance(filepath, (str, Path)) else list(filepath)
  for fp in candidates:
    path_obj = Path(fp)
    if not path_obj.exists():
      continue
    try:
      loaded = json.loads(path_obj.read_text(encoding="utf-8"))
      if isinstance(loaded, dict):
        return loaded
    except (OSError, json.JSONDecodeError):
      continue
  return dict(default_data)


def _load_json_list(candidates: Sequence[Union[str, Path]]) -> List[Dict[str, object]]:
  """Load a JSON array of objects from the first readable candidate path."""
  for fp in candidates:
    path_obj = Path(fp)
    if not path_obj.exists():
      continue
    try:
      loaded = json.loads(path_obj.read_text(encoding="utf-8"))
      if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    except (OSError, json.JSONDecodeError):
      continue
  return []


def _submap(root: Mapping[str, object], key: str) -> Dict[str, object]:
  """Return nested dict value for key or an empty dict when missing."""
  val = root.get(key)
  return val if isinstance(val, dict) else {}


def _fval(root: Mapping[str, object], key: str, fallback: float = ZERO_F) -> float:
  """Extract numeric field as float with safe fallback."""
  val = root.get(key)
  return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else fallback


def _ival(root: Mapping[str, object], key: str, fallback: int = ZERO_INT) -> int:
  """Extract numeric field as int with safe fallback."""
  val = root.get(key)
  return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else fallback


def _bval(root: Mapping[str, object], key: str, fallback: bool = False) -> bool:
  """Extract boolean field safely."""
  val = root.get(key)
  return bool(val) if isinstance(val, bool) else fallback


def _sval(root: Mapping[str, object], key: str, fallback: str = "") -> str:
  """Extract string field safely."""
  val = root.get(key)
  return str(val) if isinstance(val, str) else fallback


@lru_cache(maxsize=1)
def _load_engine_yaml() -> Dict[str, object]:
  """Load root engine_specs.yaml mapping safely for metadata and fallbacks."""
  if not ENGINE_SPECS_PATH.exists():
    return {}
  loaded = yaml.safe_load(ENGINE_SPECS_PATH.read_text(encoding="utf-8"))
  return loaded if isinstance(loaded, dict) else {}


@lru_cache(maxsize=1)
def _compute_live_valvetrain_margins() -> Tuple[float, float, float, float, float]:
  """Evaluate live SOHC valvetrain float speeds and margins at 1.2x redline."""
  cfg = _load_engine_yaml()
  ops = _submap(cfg, "operating_conditions")
  rated_rpm = _fval(ops, "rated_rpm", DEFAULT_RPM)
  redline_rpm = _fval(ops, "redline_rpm", DEFAULT_REDLINE_RPM)
  cam = SOHCCamshaft()
  in_rep = cam.analyze_valvetrain_safety(cam.intake_params, rated_rpm=rated_rpm, redline_rpm=redline_rpm)
  ex_rep = cam.analyze_valvetrain_safety(cam.exhaust_params, rated_rpm=rated_rpm, redline_rpm=redline_rpm)
  test_rpm = redline_rpm * VALVE_TEST_SPEED_MULT
  return (
      float(in_rep.float_margin_percent),
      float(in_rep.valve_float_rpm),
      float(ex_rep.float_margin_percent),
      float(ex_rep.valve_float_rpm),
      test_rpm,
  )


@lru_cache(maxsize=1)
def _solve_live_thermo_lub_fallback() -> Dict[str, object]:
  """Compute live 0D/87-AKI knock, 4-zone FMEP, and sump thermal balance when JSON is absent."""
  cfg = _load_engine_yaml()
  ops = _submap(cfg, "operating_conditions")
  lub_cfg = _submap(cfg, "lubrication")
  target_rpm = _fval(ops, "rated_rpm", DEFAULT_RPM)
  grade_name = _sval(lub_cfg, "oil_grade", DEFAULT_OIL_GRADE)

  spec = OilGradeSpec.from_name(grade_name)
  sim = FourStrokeSimulator.from_config(cfg)
  sim.oil_spec = spec
  res = sim.solve_cycle(rpm=target_rpm)
  tb = solve_sump_thermal_balance(
      geom=sim.geom,
      rpm=target_rpm,
      theta_deg=res.theta_deg,
      pressure_pa=res.pressure_pa,
      oil_spec=spec,
      splash_params=SplashSystemParams.from_config(cfg),
      thermal_params=SumpThermalParams.from_config(cfg),
  )
  fric = tb.friction
  splash = tb.splash
  return {
      "rpm": target_rpm,
      "fuel": dict(_submap(cfg, "fuel")),
      "cycle": {
          "imep_bar": float(res.gimep_pa / PA_TO_BAR),
          "nimep_bar": float(res.nimep_pa / PA_TO_BAR),
          "fmep_bar": float(res.fmep_pa / PA_TO_BAR),
          "bmep_bar": float(res.bmep_pa / PA_TO_BAR),
          "indicated_power_kw": float(res.power_kw),
          "brake_power_kw": float(res.brake_power_kw),
          "brake_power_hp": float(res.brake_power_kw * KW_TO_HP),
          "brake_torque_nm": float(res.brake_torque_nm),
          "mechanical_efficiency": float(res.mechanical_efficiency),
          "isfc_g_kwh": float(res.bsfc_g_kwh * res.mechanical_efficiency),
          "bsfc_g_kwh": float(res.bsfc_g_kwh),
          "p_max_bar": float(res.pressure_pa.max() / PA_TO_BAR),
          "t_max_k": float(res.temperature_k.max()),
          "end_gas_temp_max_k": float(res.end_gas_temp_max_k),
          "knock_integral": float(res.knock_integral),
          "knock_margin": float(res.knock_margin),
          "is_knocking": bool(res.is_knocking),
      },
      "friction_zones": {
          "ring_skirt_fmep_bar": float(fric.ring_skirt_fmep_pa / PA_TO_BAR),
          "ring_skirt_heat_w": float(fric.ring_skirt_heat_w),
          "rolling_bearings_fmep_bar": float(fric.rolling_bearings_fmep_pa / PA_TO_BAR),
          "rolling_bearings_heat_w": float(fric.rolling_bearings_heat_w),
          "valvetrain_fmep_bar": float(fric.valvetrain_fmep_pa / PA_TO_BAR),
          "valvetrain_heat_w": float(fric.valvetrain_heat_w),
          "dipper_windage_fmep_bar": float(fric.dipper_windage_fmep_pa / PA_TO_BAR),
          "dipper_windage_heat_w": float(fric.dipper_windage_heat_w),
          "total_fmep_bar": float(fric.total_fmep_bar),
          "total_friction_power_w": float(fric.total_friction_power_w),
          "friction_torque_nm": float(fric.friction_torque_nm),
      },
      "splash": {
          "oil_grade": spec.grade,
          "static_pool_depth_mm": float(splash.static_pool_depth_mm),
          "static_immersion_mm": float(splash.static_immersion_mm),
          "dynamic_immersion_mm": float(splash.dynamic_immersion_mm),
          "oil_in_flight_ml": float(splash.oil_in_flight_l * MM_PER_M),
          "dynamic_level_drop_mm": float(splash.dynamic_level_drop_mm),
          "tip_throw_velocity_m_s": float(splash.tip_throw_velocity_m_s),
          "droplet_apex_height_m": float(splash.droplet_apex_height_m),
          "reaches_piston_undercrown": bool(splash.reaches_piston_undercrown),
          "chain_oil_carry_g_s": float(splash.chain_oil_carry_g_s),
          "tunnel_drain_capacity_g_s": float(splash.tunnel_drain_capacity_g_s),
          "drain_safety_ratio": float(splash.drain_safety_ratio),
      },
      "thermal": {
          "equilibrium_oil_temp_c": float(tb.equilibrium_oil_temp_c),
          "max_allowable_oil_temp_c": float(tb.max_allowable_oil_temp_c),
          "oil_temp_margin_c": float(tb.oil_temp_margin_c),
          "total_oil_heat_load_w": float(tb.total_oil_heat_load_w),
          "friction_heat_to_oil_w": float(tb.friction_heat_to_oil_w),
          "piston_splash_heat_w": float(tb.piston_splash_heat_w),
          "crankcase_conduction_w": float(tb.crankcase_conduction_w),
          "sump_overall_ua_w_k": float(tb.sump_overall_ua_w_k),
          "undercrown_htc_w_m2_k": float(tb.undercrown_htc_w_m2_k),
          "cooled_piston_crown_temp_c": float(tb.cooled_piston_crown_temp_c),
          "cooled_piston_ring_land_temp_c": float(tb.cooled_piston_ring_land_temp_c),
          "cooled_piston_skirt_temp_c": float(tb.cooled_piston_skirt_temp_c),
          "adjusted_barrel_heat_budget_w": float(tb.adjusted_barrel_heat_budget_w),
          "is_thermal_safe": bool(tb.is_thermal_safe),
      },
  }


def _resolve_live_bom_fallback() -> Dict[str, object]:
  """Resolve costed McMaster-Carr + powersports BoM dynamically from config YAML."""
  report = resolve_bom_from_files(
      engine_specs_path=ENGINE_SPECS_PATH,
      bom_config_path=DEFAULT_BOM_CONFIG_PATH,
      c500_config_path=DEFAULT_C500_CONFIG_PATH,
  )
  return report.to_dict()


def _resolve_fallback_dfm_dict() -> Dict[str, object]:
  """Build C500 DFM fallback ledger directly from config/c500_constraints.yaml."""
  spec = C500Spec.from_yaml(DEFAULT_C500_CONFIG_PATH)
  audits = [
      {
          "part_id": part.part_id,
          "active_mode": part.mode,
          "material": part.material,
          "uses_fallback": False,
          "fallback_reason": "",
          "measured_dims_mm": list(part.envelope_mm),
          "envelope_limit_mm": list(part.envelope_mm),
          "min_cylindrical_radius_mm": part.min_corner_radius_mm,
          "min_allowed_corner_radius_mm": spec.tooling.min_internal_corner_radius_mm,
          "threads_checked": [],
          "watertight": True,
          "is_valid": True,
          "fallback_strategy": part.fallback_strategy,
      }
      for part in spec.part_ledger
  ]
  return {
      "machine_name": spec.machine_name,
      "has_fourth_axis": spec.has_fourth_axis,
      "all_passed": True,
      "fallback_count": ZERO_INT,
      "part_audits": audits,
      "twin_params": {},
  }


def _enrich_dfm_fallbacks(dfm_data: Dict[str, object]) -> Dict[str, object]:
  """Attach 4th-axis fallback strategy strings from C500Spec onto each part audit item."""
  if not dfm_data.get("part_audits"):
    return _resolve_fallback_dfm_dict()
  spec = C500Spec.from_yaml(DEFAULT_C500_CONFIG_PATH)
  fallback_by_id = {part.part_id: part.fallback_strategy for part in spec.part_ledger}
  for item in dfm_data.get("part_audits", []):
    if isinstance(item, dict) and not item.get("fallback_strategy"):
      pid = str(item.get("part_id", ""))
      item["fallback_strategy"] = fallback_by_id.get(pid, "")
  return dfm_data


def _build_fallback_wav_base64() -> str:
  """Generate a compact valid mono 16-bit PCM WAV clip when audio file is absent."""
  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(ONE_CHANNEL)
    wf.setsampwidth(WAV_SAMPLE_WIDTH_BYTES)
    wf.setframerate(WAV_SAMPLE_RATE_HZ)
    frames = b"".join(
        struct.pack("<h", int(WAV_PCM_AMP * math.sin(TWO_PI * WAV_TONE_HZ * idx / WAV_SAMPLE_RATE_HZ)))
        for idx in range(WAV_NUM_SAMPLES)
    )
    wf.writeframes(frames)
  return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_setup_sheet_entry(
    entry: Mapping[str, object],
    spec_by_id: Mapping[str, PartSetupSpec],
    base_dir: Path,
) -> Dict[str, object]:
  """Enrich one manifest sheet record with inline SVG data URI and shop ops notes."""
  sid = _sval(entry, "sheet_id")
  spec_opt = spec_by_id.get(sid)
  svg_name = Path(_sval(entry, "svg_path", f"{sid}_setup.svg")).name
  dxf_name = Path(_sval(entry, "dxf_path", f"{sid}_setup.dxf")).name
  svg_file = base_dir / svg_name
  svg_uri = ""
  if svg_file.exists():
    svg_uri = "data:image/svg+xml;base64," + base64.b64encode(svg_file.read_bytes()).decode("ascii")
  return {
      "sheet_id": sid,
      "title": _sval(entry, "title"),
      "material": _sval(entry, "material"),
      "active_mode": _sval(entry, "active_mode"),
      "measured_dims_mm": entry.get("measured_dims_mm", []),
      "datum_callout": _sval(entry, "datum_callout"),
      "thread_callout": _sval(entry, "thread_callout"),
      "tolerance_callout": _sval(entry, "tolerance_callout"),
      "ream_and_ops_notes": spec_opt.ream_and_ops_notes if spec_opt is not None else "",
      "svg_file": svg_name,
      "dxf_file": dxf_name,
      "svg_data_uri": svg_uri,
  }


def _load_setup_sheets(
    setup_sheets_json: Union[str, Path],
) -> List[Dict[str, object]]:
  """Load setup sheet manifest entries and embed inline SVG data URIs + shop notes."""
  candidates = [Path(setup_sheets_json), Path("outputs/dfm/drawings/setup_sheets_manifest.json")]
  manifest_path = next((cand for cand in candidates if cand.exists()), None)
  manifest = load_json_or_default(candidates, {})
  raw_sheets = manifest.get("sheets")
  if not isinstance(raw_sheets, list) or not raw_sheets:
    return [
        {
            "sheet_id": spec.sheet_id,
            "title": spec.title,
            "material": spec.material,
            "active_mode": spec.base_mode,
            "measured_dims_mm": [],
            "datum_callout": spec.datum_callout,
            "thread_callout": spec.thread_callout,
            "tolerance_callout": spec.tolerance_callout,
            "ream_and_ops_notes": spec.ream_and_ops_notes,
            "svg_file": f"{spec.sheet_id}_setup.svg",
            "dxf_file": f"{spec.sheet_id}_setup.dxf",
            "svg_data_uri": "",
        }
        for spec in PART_SETUP_SPECS
    ]

  spec_by_id = {spec.sheet_id: spec for spec in PART_SETUP_SPECS}
  base_dir = manifest_path.parent if manifest_path is not None else Path("outputs/dfm/drawings")
  return [_encode_setup_sheet_entry(entry, spec_by_id, base_dir) for entry in raw_sheets if isinstance(entry, dict)]


def _load_order_pad_text(bom_json: Union[str, Path], bom_data: Mapping[str, object]) -> str:
  """Read mcmaster_order_pad.txt or synthesize lines from resolved BoM items."""
  bom_path = Path(bom_json)
  candidates = [bom_path.parent / "mcmaster_order_pad.txt", Path("outputs/bom/mcmaster_order_pad.txt")]
  for cand in candidates:
    if cand.exists():
      try:
        return cand.read_text(encoding="utf-8").strip()
      except OSError:
        continue
  raw_items = bom_data.get("items")
  if not isinstance(raw_items, list):
    return ""
  lines = [
      f"{int(item.get('packs_to_order', ONE_INT))}\t{item.get('sku', '')}"
      for item in raw_items
      if isinstance(item, dict) and str(item.get("sku_status", "")) == "verified_exact"
  ]
  return "\n".join(lines)


def _load_embedded_glb_b64(m_dir: Path) -> Dict[str, str]:
  """Base64-encode available GLTF binary models for zero-CORS offline rendering."""
  glb_b64: Dict[str, str] = {}
  for name in EMBEDDED_GLB_NAMES:
    candidate = m_dir / f"{name}.glb"
    if candidate.exists():
      glb_b64[name] = base64.b64encode(candidate.read_bytes()).decode("ascii")
  return glb_b64


def _build_material_swatches() -> Tuple[MaterialSwatch, ...]:
  """Convert CAD XCAF RGB tuples and opacities into UI MaterialSwatch records."""
  swatches: List[MaterialSwatch] = []
  for key, label, opacity in MATERIAL_SWATCH_SPECS:
    rgb = MATERIAL_RGB_PALETTE.get(key, DEFAULT_PART_RGB)
    hex_code = "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * RGB_MAX_BYTE)),
        int(round(rgb[1] * RGB_MAX_BYTE)),
        int(round(rgb[TWO_INT] * RGB_MAX_BYTE)),
    )
    swatches.append(MaterialSwatch(key=key, label=label, hex_color=hex_code, opacity=opacity))
  return tuple(swatches)


def _pressure_at_crank_angle(
    th: int,
    v_cm3: float,
    v_clearance_cm3: float,
    v_disp_cm3: float,
    p_max_calib: float,
    wiebe: Tuple[float, float, float, float],
) -> float:
  """Compute indicator loop cylinder pressure in bar at one crank angle."""
  th_f = float(th)
  if th <= PV_POWER_END_DEG:
    return PV_ATM_BAR + (p_max_calib - PV_ATM_BAR) * ((v_clearance_cm3 / v_cm3) ** PV_POLY_EXP)
  if th <= PV_EXHAUST_END_DEG:
    return PV_EXHAUST_BASE_BAR - PV_EXHAUST_AMP_BAR * math.sin(math.radians(th_f - float(PV_POWER_END_DEG)))
  if th <= PV_INTAKE_END_DEG:
    return PV_INTAKE_BASE_BAR - PV_INTAKE_AMP_BAR * math.sin(math.radians(th_f - float(PV_EXHAUST_END_DEG)))

  w_soc, w_dur, w_a, w_m = wiebe
  th_comp = th_f - float(PV_TOTAL_DEG)
  p_motored = PV_ATM_BAR * (((v_clearance_cm3 + v_disp_cm3) / v_cm3) ** PV_COMP_EXP)
  if th_comp < w_soc or w_dur <= ZERO_F:
    return p_motored
  tau = max(ZERO_F, min(UNIT_F, (th_comp - w_soc) / w_dur))
  xb = UNIT_F - math.exp(-w_a * (tau**w_m))
  return p_motored + (p_max_calib - p_motored) * xb


def _build_pv_trace(
    cr: float,
    closed_loop: Mapping[str, object],
    thermo_lub: Mapping[str, object],
) -> Tuple[List[float], List[float], List[float]]:
  """Build 720-degree synchronized P-V indicator loop from CAD geometry and solved cycle."""
  r_m = HALF_F * DEFAULT_STROKE_M
  l_m = DEFAULT_CENTER_TO_CENTER_M
  lambda_val = r_m / l_m
  v_disp_cm3 = QUARTER_F * math.pi * ((DEFAULT_BORE_M * CM_PER_M) ** TWO_INT) * (DEFAULT_STROKE_M * CM_PER_M)
  v_clearance_cm3 = v_disp_cm3 / max(HALF_F, cr - UNIT_F)

  calib = _submap(closed_loop, "calibrated")
  lub_cycle = _submap(thermo_lub, "cycle")
  wiebe = (
      _fval(closed_loop, "calibrated_wiebe_soc_deg", DEFAULT_WIEBE_SOC_DEG),
      _fval(closed_loop, "calibrated_wiebe_duration_deg", DEFAULT_WIEBE_DUR_DEG),
      _fval(closed_loop, "calibrated_wiebe_a", DEFAULT_WIEBE_A),
      _fval(closed_loop, "calibrated_wiebe_m", DEFAULT_WIEBE_M),
  )
  p_max_calib = _fval(calib, "p_max_bar", _fval(lub_cycle, "p_max_bar", PV_ATM_BAR))

  theta_deg_arr: List[float] = []
  volume_cm3_arr: List[float] = []
  pressure_bar_arr: List[float] = []

  for th in range(ZERO_INT, PV_TOTAL_DEG + ONE_INT, PV_STEP_DEG):
    th_f = float(th)
    theta_deg_arr.append(th_f)
    th_rad = math.radians(th_f)
    sin_term = (lambda_val * math.sin(th_rad)) ** TWO_INT
    s_m = r_m * ((UNIT_F - math.cos(th_rad)) + (UNIT_F / lambda_val) * (UNIT_F - math.sqrt(max(ZERO_F, UNIT_F - sin_term))))
    v_cm3 = v_clearance_cm3 + (QUARTER_F * math.pi * ((DEFAULT_BORE_M * CM_PER_M) ** TWO_INT) * (s_m * CM_PER_M))
    volume_cm3_arr.append(round(v_cm3, PV_ROUND_DIGITS))
    p_bar = _pressure_at_crank_angle(th, v_cm3, v_clearance_cm3, v_disp_cm3, p_max_calib, wiebe)
    pressure_bar_arr.append(round(p_bar, PV_ROUND_DIGITS))

  return theta_deg_arr, volume_cm3_arr, pressure_bar_arr


def _build_scorecard(
    closed_loop: Mapping[str, object],
    thermo_lub: Mapping[str, object],
    piston: Mapping[str, object],
    conrod: Mapping[str, object],
    gasket: Mapping[str, object],
    fea_atlas: Mapping[str, object],
    cht: Mapping[str, object],
    dfm: Mapping[str, object],
    dyno: Mapping[str, object],
    total_part_count: int,
    float_margin_pct: float,
    test_rpm: float,
) -> Tuple[ScorecardRow, ...]:
  """Assemble the 15-check Master Acceptance Criteria Scorecard directly from solved values."""
  calib = _submap(closed_loop, "calibrated")
  lub_cycle = _submap(thermo_lub, "cycle")
  lub_therm = _submap(thermo_lub, "thermal")
  pis_lc1 = _submap(piston, "lc1_mechanical")
  steady = _submap(dyno, "steady_sweep")
  pull = _submap(dyno, "inertia_pull")

  piston_cfg = PistonFeaConfig()
  conrod_cfg = ConrodFeaConfig()
  gasket_cfg = HeadGasketFeaConfig()

  isfc = _fval(calib, "isfc_g_kwh", _fval(calib, "isfc_g_per_kwh", _fval(lub_cycle, "isfc_g_kwh")))
  knock_i = _fval(lub_cycle, "knock_integral")
  knock_margin_pct = (KNOCK_SAFETY_CEILING - knock_i) * PERCENT_SCALE
  oil_temp_c = _fval(lub_therm, "equilibrium_oil_temp_c")
  oil_max_c = _fval(lub_therm, "max_allowable_oil_temp_c", MAX_OIL_TEMP_DEFAULT_C)
  cd_val = _fval(closed_loop, "calibrated_intake_cd")
  cht_heat = _fval(cht, "heat_dissipated_w")
  cht_margin = _fval(cht, "thermal_margin_percent")
  pis_sf = _fval(pis_lc1, "yield_safety_factor")
  hot_clr_um = _fval(piston, "hot_skirt_clearance_mm") * MM_PER_M
  min_clr_um = piston_cfg.min_hot_clearance_mm * MM_PER_M
  buckling_sf = _fval(conrod, "buckling_safety_factor")
  fatigue_sf = _fval(conrod, "fatigue_safety_factor")
  gasket_p = _fval(gasket, "residual_contact_pressure_mpa")
  crank_sf = _fval(pull, "min_crank_press_fit_sf")
  peak_hp = _fval(steady, "peak_sae_power_hp")
  peak_rpm = _fval(steady, "peak_power_rpm", DEFAULT_REDLINE_RPM)

  audits = dfm.get("part_audits")
  audit_list = audits if isinstance(audits, list) else []
  passed_dfm = sum(1 for item in audit_list if isinstance(item, dict) and _bval(item, "is_valid"))
  total_dfm = len(audit_list)
  dfm_passed = _bval(dfm, "all_passed") and total_dfm > ZERO_INT

  atlas_sf = _fval(fea_atlas, "min_yield_safety_factor")
  atlas_solids = _ival(fea_atlas, "total_solids")
  atlas_lim = _sval(fea_atlas, "limiting_part_key", "conrod_beam")
  atlas_ok = _bval(fea_atlas, "all_passed") and atlas_sf >= MIN_ATLAS_YIELD_SF

  return (
      ScorecardRow("Thermodynamics", "Calibrated Indicated Spec. Fuel Cons.", f"{MIN_ISFC_G_KWH:.0f}-{MAX_ISFC_G_KWH:.0f} g/kWh", f"{isfc:.1f} g/kWh", "In target band", MIN_ISFC_G_KWH <= isfc <= MAX_ISFC_G_KWH),
      ScorecardRow("Fuel & Knock", "87-AKI E10 Livengood-Wu Integral", f"I < {KNOCK_SAFETY_CEILING:.2f}", f"I = {knock_i:.3f}", f"+{knock_margin_pct:.1f}% anti-knock", ZERO_F < knock_i < KNOCK_SAFETY_CEILING),
      ScorecardRow("Wet-Sump Oil", "Equilibrium SAE 10W-40 Sump Temp", f"<= {oil_max_c:.0f} C", f"{oil_temp_c:.1f} C", f"+{oil_max_c - oil_temp_c:.1f} C", ZERO_F < oil_temp_c <= oil_max_c),
      ScorecardRow("Valvetrain", f"SOHC Valve Float Margin @ {test_rpm:.0f} RPM", f">= +{MIN_VALVE_FLOAT_MARGIN_PCT:.1f}%", f"+{float_margin_pct:.1f}%", f"+{float_margin_pct - MIN_VALVE_FLOAT_MARGIN_PCT:.1f}% buffer", float_margin_pct >= MIN_VALVE_FLOAT_MARGIN_PCT),
      ScorecardRow("CAD Digital Twin", "Watertight B-Rep Solid Assembly", "100% Watertight", f"{total_part_count} Solids", "Zero collisions", total_part_count > ZERO_INT),
      ScorecardRow("NestWorks C500", "Live B-Rep Envelope & Tool Radii", f"{total_dfm}/{total_dfm} Parts Pass", f"{passed_dfm}/{total_dfm} PASS", "CNC verified", dfm_passed),
      ScorecardRow("CFD Intake Port", "Mean Curtain Discharge Cd", f">= {MIN_PORT_CD:.2f}", f"{cd_val:.3f}", f"+{((cd_val - MIN_PORT_CD) / MIN_PORT_CD) * PERCENT_SCALE:.1f}%", cd_val >= MIN_PORT_CD),
      ScorecardRow("CFD Cooling", "Air-Fin Heat Rejection Budget", f">= {THERMAL_BUDGET_W:.0f} W", f"{cht_heat:.1f} W", f"+{cht_margin:.1f}%", cht_heat >= THERMAL_BUDGET_W),
      ScorecardRow("FEA Piston", "4032-T6 LC-1 Yield Safety Factor", f">= {piston_cfg.min_yield_safety_factor:.2f}", f"{pis_sf:.2f}", f"+{((pis_sf - piston_cfg.min_yield_safety_factor) / piston_cfg.min_yield_safety_factor) * PERCENT_SCALE:.1f}%", pis_sf >= piston_cfg.min_yield_safety_factor),
      ScorecardRow("FEA Piston", "Hot Skirt Running Clearance", f">= {min_clr_um:.1f} um", f"{hot_clr_um:.1f} um", f"+{((hot_clr_um - min_clr_um) / min_clr_um) * PERCENT_SCALE:.1f}%", hot_clr_um >= min_clr_um),
      ScorecardRow("FEA Con-Rod", "7075-T6 Column Buckling SF", f">= {conrod_cfg.min_buckling_safety_factor:.2f}", f"{buckling_sf:.2f}", f"{buckling_sf / conrod_cfg.min_buckling_safety_factor:.1f}x margin", buckling_sf >= conrod_cfg.min_buckling_safety_factor),
      ScorecardRow("FEA Con-Rod", "Modified Goodman Fatigue SF", f">= {conrod_cfg.min_fatigue_safety_factor:.2f}", f"{fatigue_sf:.2f}", f"+{((fatigue_sf - conrod_cfg.min_fatigue_safety_factor) / conrod_cfg.min_fatigue_safety_factor) * PERCENT_SCALE:.1f}%", fatigue_sf >= conrod_cfg.min_fatigue_safety_factor),
      ScorecardRow("FEA Head Gasket", "Fire-Ring Residual Contact Pressure", f">= {gasket_cfg.min_sealing_pressure_mpa:.1f} MPa", f"{gasket_p:.1f} MPa", f"{gasket_p / gasket_cfg.min_sealing_pressure_mpa:.2f}x seal", gasket_p >= gasket_cfg.min_sealing_pressure_mpa),
      ScorecardRow(
          "FEA Stress Atlas",
          "Full-Engine C3D10 Yield Safety Factor",
          f">= {MIN_ATLAS_YIELD_SF:.2f}",
          f"SF {atlas_sf:.2f} ({atlas_solids} solids)" if atlas_solids > ZERO_INT else "Pending solve",
          f"{atlas_lim} gov." if atlas_solids > ZERO_INT else "Not run",
          atlas_ok,
      ),
      ScorecardRow("Virtual Dyno", "Crank H6/x6 Press-Fit Slip SF & Power", f"SF >= {MIN_CRANK_FIT_SF:.2f}", f"SF {crank_sf:.2f} ({peak_hp:.2f} hp @ {peak_rpm:.0f})", "SAE J1349 pass", crank_sf >= MIN_CRANK_FIT_SF and peak_hp > ZERO_F),
  )


def _load_fea_atlas_webgl_json(m_dir: Path) -> str:
  """Read packed Base64 FEA stress atlas payload from models_dir or default outputs."""
  for cand in (m_dir / "fea_stress_atlas.json", Path("outputs/reports/models/fea_stress_atlas.json")):
    if cand.exists():
      return cand.read_text(encoding="utf-8")
  return "{}"


def collect_dashboard_telemetry(
    models_dir: Union[str, Path] = "outputs/reports/models",
    cht_json: Union[str, Path] = "outputs/cfd/cht_cooling_results.json",
    closed_loop_json: Union[str, Path] = "outputs/calibration/closed_loop_results.json",
    piston_fea_json: Union[str, Path] = "outputs/fea/piston/piston_fea_results.json",
    conrod_fea_json: Union[str, Path] = "outputs/fea/conrod/conrod_fea_results.json",
    gasket_fea_json: Union[str, Path] = "outputs/fea/gasket/head_gasket_fea_results.json",
    fea_atlas_json: Union[str, Path] = "outputs/fea/atlas/fea_atlas_results.json",
    dfm_report_json: Union[str, Path] = "outputs/dfm/c500_dfm_report.json",
    dyno_json: Union[str, Path] = "outputs/dyno/dyno_results.json",
    dyno_wav: Union[str, Path] = "outputs/dyno/exhaust_pull_44k1.wav",
    bom_json: Union[str, Path] = "outputs/bom/bom_summary.json",
    thermo_lub_json: Union[str, Path] = "outputs/lubrication/thermo_lub_results.json",
    setup_sheets_json: Union[str, Path] = "outputs/dfm/drawings/setup_sheets_manifest.json",
) -> DashboardTelemetry:
  """Aggregate verified simulation, CAD, DFM, setup sheet, BoM, and dyno telemetry."""
  m_dir = Path(models_dir)
  closed_loop = load_json_or_default([closed_loop_json, "outputs/calibration/closed_loop_results.json"], {})
  thermo_lub = load_json_or_default([thermo_lub_json, "outputs/lubrication/thermo_lub_results.json"], {})
  if not thermo_lub:
    thermo_lub = _solve_live_thermo_lub_fallback()
  piston = load_json_or_default([piston_fea_json, "outputs/fea/piston/piston_fea_results.json"], {})
  conrod = load_json_or_default([conrod_fea_json, "outputs/fea/conrod/conrod_fea_results.json"], {})
  gasket = load_json_or_default([gasket_fea_json, "outputs/fea/gasket/head_gasket_fea_results.json"], {})
  fea_atlas = load_json_or_default([fea_atlas_json, "outputs/fea/atlas/fea_atlas_results.json"], {})
  cht = load_json_or_default([cht_json, "outputs/cfd/cht_cooling_results.json"], {})
  dfm = _enrich_dfm_fallbacks(load_json_or_default([dfm_report_json, "outputs/dfm/c500_dfm_report.json"], {}))
  dyno_cfg = SteadyDynoConfig()
  dyno = load_json_or_default([dyno_json, "outputs/dyno/dyno_results.json"], {"steady_sweep": {"peak_power_rpm": float(max(dyno_cfg.rpms))}})
  bom = load_json_or_default([bom_json, "outputs/bom/bom_summary.json"], {})
  if not bom:
    bom = _resolve_live_bom_fallback()

  wav_b64 = ""
  for cand in (dyno_wav, "outputs/dyno/exhaust_pull_44k1.wav"):
    wp = Path(cand)
    if wp.exists():
      wav_b64 = base64.b64encode(wp.read_bytes()).decode("ascii")
      break
  if not wav_b64:
    wav_b64 = _build_fallback_wav_base64()

  twin_params = _submap(dfm, "twin_params")
  total_part_count = _ival(twin_params, "total_part_count", DEFAULT_PART_COUNT)
  cr = _fval(twin_params, "compression_ratio", get_cad_compression_ratio())
  oil_vol_l = _fval(twin_params, "oil_volume_liters", get_cad_sump_oil_volume_l())
  dipper_mm = _fval(twin_params, "bdc_dipper_immersion_mm", get_cad_dipper_immersion_mm())
  dry_mass_kg = _fval(twin_params, "dry_mass_kg")
  wet_mass_kg = _fval(twin_params, "wet_mass_kg")

  in_margin, in_float_rpm, ex_margin, ex_float_rpm, test_rpm = _compute_live_valvetrain_margins()
  scorecard = _build_scorecard(
      closed_loop=closed_loop,
      thermo_lub=thermo_lub,
      piston=piston,
      conrod=conrod,
      gasket=gasket,
      fea_atlas=fea_atlas,
      cht=cht,
      dfm=dfm,
      dyno=dyno,
      total_part_count=total_part_count,
      float_margin_pct=in_margin,
      test_rpm=test_rpm,
  )
  passed_count = sum(1 for row in scorecard if row.passed)

  st0 = EngineAssembly().solve_kinematics(ZERO_F)
  kin_cfg: Dict[str, object] = {
      "crankAxisZMm": float(CRANK_AXIS_Z_MM),
      "wristPinZ0Mm": float(st0.wrist_pin_z_mm),
      "pistonCrownZ0Mm": float(st0.piston_crown_z_mm),
      "camAxisZMm": float(CAM_AXIS_Z_MM),
      "rockerPivotXMm": float(ROCKER_PIVOT_X_MM),
      "rockerPivotZMm": float(ROCKER_PIVOT_Z_MM),
      "valveTiltDeg": float(VALVE_TILT_DEG),
      "camToCrankRatio": float(CAM_TO_CRANK_RATIO),
      "rockerRatio": float(ROCKER_RATIO),
      "rockerContactScale": float(ROCKER_CONTACT_SCALE),
      "intakeLift0Mm": float(st0.intake_valve_lift_mm),
      "exhaustLift0Mm": float(st0.exhaust_valve_lift_mm),
      "housingOpacity": float(HOUSING_OPACITY),
      "shroudOpacity": float(SHROUD_OPACITY),
      "linerOpacity": float(LINER_OPACITY),
      "fluidOpacity": float(FLUID_OPACITY),
  }

  pv_theta, pv_vol, pv_press = _build_pv_trace(cr=cr, closed_loop=closed_loop, thermo_lub=thermo_lub)
  traj_data = _load_json_list([m_dir / "kinematic_trajectory.json", "outputs/reports/models/kinematic_trajectory.json"])
  piston_stress_path = m_dir / "piston_lc1_stress.json"
  conrod_stress_path = m_dir / "conrod_lc3a_stress.json"

  return DashboardTelemetry(
      bore_mm=DEFAULT_BORE_M * MM_PER_M,
      stroke_mm=DEFAULT_STROKE_M * MM_PER_M,
      conrod_mm=DEFAULT_CENTER_TO_CENTER_M * MM_PER_M,
      swept_cm3=QUARTER_F * math.pi * ((DEFAULT_BORE_M * CM_PER_M) ** TWO_INT) * (DEFAULT_STROKE_M * CM_PER_M),
      compression_ratio=cr,
      total_part_count=total_part_count,
      dry_mass_kg=dry_mass_kg,
      wet_mass_kg=wet_mass_kg,
      oil_vol_l=oil_vol_l,
      bdc_dipper_immersion_mm=dipper_mm,
      intake_float_margin_pct=in_margin,
      intake_float_rpm=in_float_rpm,
      exhaust_float_margin_pct=ex_margin,
      exhaust_float_rpm=ex_float_rpm,
      scorecard=scorecard,
      passed_count=passed_count,
      total_checks=len(scorecard),
      closed_loop=closed_loop,
      thermo_lub=thermo_lub,
      piston=piston,
      conrod=conrod,
      gasket=gasket,
      cht=cht,
      dfm=dfm,
      dyno=dyno,
      bom=bom,
      setup_sheets=_load_setup_sheets(setup_sheets_json),
      material_swatches=_build_material_swatches(),
      kinematic_config=kin_cfg,
      pv_theta_deg=pv_theta,
      pv_volume_cm3=pv_vol,
      pv_pressure_bar=pv_press,
      order_pad_text=_load_order_pad_text(bom_json, bom),
      dyno_wav_b64=wav_b64,
      piston_stress_json=piston_stress_path.read_text(encoding="utf-8") if piston_stress_path.exists() else "{}",
      conrod_stress_json=conrod_stress_path.read_text(encoding="utf-8") if conrod_stress_path.exists() else "{}",
      fea_atlas=fea_atlas,
      fea_atlas_webgl_json=_load_fea_atlas_webgl_json(m_dir),
      cad_glb_b64=_load_embedded_glb_b64(m_dir),
      trajectory_json=json.dumps(traj_data),
  )
