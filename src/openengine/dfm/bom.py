"""Offline parametric McMaster-Carr + powersports Bill of Materials resolver.

Maps parametric engine geometry and lubrication specs from
config/engine_specs.yaml into verified McMaster-Carr SKUs, direct product URLs,
one-paste Order-Pad files, and C500 raw-stock interface validations without
performing any network requests.
"""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
import yaml

from openengine.dfm.c500_spec import (
    C500Spec,
    DEFAULT_C500_CONFIG_PATH,
    PartEnvelopeCheck,
    PartMachiningSpec,
)
from openengine.fea.calculix import MATERIAL_CATALOG

DEFAULT_ENGINE_SPECS_PATH: Path = Path("config/engine_specs.yaml")
DEFAULT_BOM_CONFIG_PATH: Path = Path("config/mcmaster_bom.yaml")
DEFAULT_BOM_OUTPUT_DIR: Path = Path("outputs/bom")

MCMASTER_BASE_URL: str = "https://www.mcmaster.com"
ORDER_PAD_FILENAME: str = "mcmaster_order_pad.txt"
COSTED_CSV_FILENAME: str = "bom_costed.csv"
SUMMARY_JSON_FILENAME: str = "bom_summary.json"

STATUS_VERIFIED_EXACT: str = "verified_exact"
STATUS_ANCHOR_TABLE: str = "anchor_table"
STATUS_COTS_SPEC: str = "cots_spec"

RULE_HEAD_TIE_BOLT: str = "head_tie_bolt"
RULE_WRIST_PIN_DIAM: str = "wrist_pin_diam"
RULE_MAIN_JOURNAL_BEARING: str = "main_journal_bearing"
RULE_MAIN_JOURNAL_SEAL: str = "main_journal_seal"
RULE_CONROD_BILLET_STOCK: str = "conrod_billet_stock"
RULE_PISTON_BAR_STOCK: str = "piston_bar_stock"

NON_FEA_LINER_MATERIAL: str = "DOM_STEEL_A513_T5"
CRANK_SPROCKET_ITEM_ID: str = "crank_timing_sprocket_18t"
CAM_SPROCKET_ITEM_ID: str = "cam_timing_sprocket_36t"

MM_PER_M: float = 1000.0
HEAD_BOLT_STROKE_FACTOR: float = 0.5
HEAD_BOLT_BASE_GRIP_MM: float = 32.0
CONROD_BILLET_ALLOWANCE_MM: float = 52.0
PISTON_STOCK_ALLOWANCE_MM: float = 4.0
LINER_WALL_ALLOWANCE_MM: float = 4.0
VALVE_COIL_BIND_MARGIN_MM: float = 1.5
REQUIRED_TIMING_RATIO: float = 2.0
TIMING_RATIO_TOL: float = 0.01
DIM_MATCH_TOL_MM: float = 0.05
COST_DECIMAL_PLACES: int = 2
MIN_REQUIRED_QTY: int = 1
DEFAULT_FALLBACK_PRICE_USD: float = 15.0
DEFAULT_INTAKE_VALVE_M: float = 0.030
DEFAULT_EXHAUST_VALVE_M: float = 0.026
DEFAULT_VALVE_LIFT_M: float = 0.0085
DEFAULT_SUMP_VOLUME_L: float = 0.65
DEFAULT_OIL_GRADE: str = "SAE 10W-40"

SERIES_SUBKEY_ORDER: Tuple[str, ...] = (
    "threads",
    "diameters",
    "sizes",
    "shaft_diameters",
    "bores",
)


class EngineGeometryConfig(TypedDict):
  """Geometry section slice of config/engine_specs.yaml."""

  bore: float
  stroke: float
  con_rod_length: float
  wrist_pin_diameter: float
  crankpin_diameter: float
  main_journal_diameter: float


class EngineValvesConfig(TypedDict, total=False):
  """Valves section slice of config/engine_specs.yaml."""

  intake_diameter: float
  exhaust_diameter: float
  max_lift: float


class EngineLubricationConfig(TypedDict, total=False):
  """Lubrication section slice of config/engine_specs.yaml."""

  oil_grade: str
  sump_volume_l: float


class EngineSpecsConfigRoot(TypedDict):
  """Root schema slice of config/engine_specs.yaml needed for BoM checks."""

  geometry: EngineGeometryConfig
  valves: EngineValvesConfig
  lubrication: EngineLubricationConfig


class BomMetadataConfig(TypedDict):
  """Metadata section of config/mcmaster_bom.yaml."""

  currency: str
  mcmaster_base_url: str
  sourcing_channels: List[str]


class SkuVariantConfig(TypedDict, total=False):
  """Single catalog row inside a parametric SKU series table."""

  sku: str
  pack_qty: int
  pack_price_usd: float
  length_mm: float
  diameter_mm: float
  bore_mm: float
  od_mm: float
  width_mm: float


class SkuSeriesConfig(TypedDict, total=False):
  """Parametric McMaster SKU family lookup block."""

  family: str
  anchor_sku: str
  anchor_url: str
  threads: Dict[str, List[SkuVariantConfig] | SkuVariantConfig]
  diameters: Dict[str, List[SkuVariantConfig] | SkuVariantConfig]
  sizes: Dict[str, SkuVariantConfig]
  shaft_diameters: Dict[str, SkuVariantConfig]
  bores: Dict[str, SkuVariantConfig]


class BomItemConfig(TypedDict, total=False):
  """Single BoM line item specification in config/mcmaster_bom.yaml."""

  item_id: str
  channel: str
  description: str
  vendor: str
  series: str
  series_key: str
  sku: str
  sku_status: str
  url: str
  thread: str
  parametric_rule: str
  target_length_mm: float
  required_qty: int
  pack_qty: int
  pack_price_usd: float
  selection_hint: str
  teeth: int
  max_travel_mm: float
  material: str
  stock_dims_mm: List[float]
  c500_mode: str
  requires_fourth_axis_active: bool
  fallback_when_no_fourth_axis: bool
  raw_id_mm: float
  raw_od_mm: float
  mating_bore_mm: float
  mating_crankpin_mm: float
  oil_grade: str
  volume_l: float
  intake_head_mm: float
  exhaust_head_mm: float


class BomConfigRoot(TypedDict):
  """Root schema of config/mcmaster_bom.yaml."""

  metadata: BomMetadataConfig
  sku_series: Dict[str, SkuSeriesConfig]
  items: List[BomItemConfig]


@dataclass(frozen=True, slots=True)
class EngineInterfaceTargets:
  """Target engine dimensions [mm] and fluid parameters for COTS fit checks."""

  bore_mm: float
  stroke_mm: float
  con_rod_mm: float
  wrist_pin_mm: float
  crankpin_mm: float
  main_journal_mm: float
  intake_valve_mm: float
  exhaust_valve_mm: float
  max_valve_lift_mm: float
  oil_grade: str
  sump_volume_l: float


@dataclass(frozen=True, slots=True)
class ResolvedBomItem:
  """Resolved purchase line item with verified SKU/URL and interface check."""

  item_id: str
  channel: str
  description: str
  vendor: str
  sku: str
  sku_status: str
  url: str
  selection_hint: str
  required_qty: int
  pack_qty: int
  packs_to_order: int
  pack_price_usd: float
  extended_cost_usd: float
  interface_check: str
  is_interface_valid: bool


@dataclass(frozen=True, slots=True)
class BomValidationSummary:
  """Aggregate validation and sourcing cost summary for the full BoM."""

  is_valid: bool
  total_items: int
  verified_exact_mcmaster_count: int
  anchor_table_mcmaster_count: int
  powersports_cots_count: int
  channel_costs_usd: Dict[str, float]
  total_cost_usd: float
  violations: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedBomReport:
  """Complete offline BoM resolution and C500 ledger verification report."""

  has_fourth_axis: bool
  items: Tuple[ResolvedBomItem, ...]
  c500_checks: Tuple[PartEnvelopeCheck, ...]
  validation: BomValidationSummary


@dataclass(frozen=True, slots=True)
class BomExportPaths:
  """Paths of exported Order-Pad, costed CSV, and JSON summary files."""

  order_pad_txt: Path
  costed_csv: Path
  summary_json: Path


def load_engine_specs(
    path: Path = DEFAULT_ENGINE_SPECS_PATH,
) -> EngineSpecsConfigRoot:
  """Load config/engine_specs.yaml into an EngineSpecsConfigRoot dict."""
  resolved = Path(path)
  if not resolved.exists():
    raise FileNotFoundError(f"Engine config file not found: {resolved}")
  with open(resolved, "r") as handle:
    return yaml.safe_load(handle)


def load_bom_config(path: Path = DEFAULT_BOM_CONFIG_PATH) -> BomConfigRoot:
  """Load config/mcmaster_bom.yaml into a BomConfigRoot dict."""
  resolved = Path(path)
  if not resolved.exists():
    raise FileNotFoundError(f"BoM config file not found: {resolved}")
  with open(resolved, "r") as handle:
    return yaml.safe_load(handle)


def resolve_bom_from_files(
    engine_path: Path = DEFAULT_ENGINE_SPECS_PATH,
    c500_path: Path = DEFAULT_C500_CONFIG_PATH,
    bom_path: Path = DEFAULT_BOM_CONFIG_PATH,
    has_fourth_axis_override: Optional[bool] = None,
) -> ResolvedBomReport:
  """Load YAML configs from disk and resolve the full offline BoM report."""
  engine_cfg = load_engine_specs(engine_path)
  c500_spec = C500Spec.from_yaml(c500_path)
  bom_cfg = load_bom_config(bom_path)
  return resolve_bom(
      engine_cfg=engine_cfg,
      c500_spec=c500_spec,
      bom_cfg=bom_cfg,
      has_fourth_axis_override=has_fourth_axis_override,
  )


def resolve_bom(
    engine_cfg: Optional[EngineSpecsConfigRoot] = None,
    c500_spec: Optional[C500Spec] = None,
    bom_cfg: Optional[BomConfigRoot] = None,
    has_fourth_axis_override: Optional[bool] = None,
) -> ResolvedBomReport:
  """Resolve parametric SKUs, direct URLs, and C500/engine interface checks."""
  eff_engine = engine_cfg if engine_cfg is not None else load_engine_specs()
  eff_c500 = c500_spec if c500_spec is not None else C500Spec.from_yaml()
  eff_bom = bom_cfg if bom_cfg is not None else load_bom_config()

  effective_fourth_axis = (
      eff_c500.has_fourth_axis
      if has_fourth_axis_override is None
      else has_fourth_axis_override
  )
  targets = _extract_engine_targets(eff_engine)
  c500_checks = eff_c500.validate_all_ledger_parts(
      has_fourth_axis_override=effective_fourth_axis
  )
  violations: List[str] = [
      f"C500 ledger '{chk.part_id}': {err}"
      for chk in c500_checks
      if not chk.is_valid
      for err in chk.violations
  ]

  base_url = str(
      eff_bom["metadata"].get("mcmaster_base_url", MCMASTER_BASE_URL)
  ).rstrip("/")
  sku_series = eff_bom["sku_series"]
  active_raw_items = [
      item
      for item in eff_bom["items"]
      if _is_item_active(item, effective_fourth_axis)
  ]

  resolved_items: List[ResolvedBomItem] = []
  sprocket_teeth: Dict[str, int] = {}
  for raw_item in active_raw_items:
    item_resolved, item_errors = _resolve_and_check_item(
        raw_item=raw_item,
        targets=targets,
        c500_spec=eff_c500,
        sku_series=sku_series,
        base_url=base_url,
        effective_fourth_axis=effective_fourth_axis,
    )
    resolved_items.append(item_resolved)
    violations.extend(item_errors)
    if "teeth" in raw_item:
      sprocket_teeth[str(raw_item["item_id"])] = int(raw_item["teeth"])

  _check_sprocket_ratio(sprocket_teeth, violations)
  validation = _summarize_validation(
      items=resolved_items,
      channels=eff_bom["metadata"]["sourcing_channels"],
      violations=violations,
  )
  return ResolvedBomReport(
      has_fourth_axis=effective_fourth_axis,
      items=tuple(resolved_items),
      c500_checks=c500_checks,
      validation=validation,
  )


def export_bom_artifacts(
    report: ResolvedBomReport,
    output_dir: Path = DEFAULT_BOM_OUTPUT_DIR,
) -> BomExportPaths:
  """Write McMaster Order-Pad paste list, costed CSV, and JSON summary."""
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  order_pad_path = out_path / ORDER_PAD_FILENAME
  csv_path = out_path / COSTED_CSV_FILENAME
  json_path = out_path / SUMMARY_JSON_FILENAME

  packs_by_sku: Dict[str, int] = {}
  for item in report.items:
    if item.sku_status == STATUS_VERIFIED_EXACT:
      packs_by_sku[item.sku] = packs_by_sku.get(item.sku, 0) + item.packs_to_order
  order_lines = [f"{packs}\t{sku}\n" for sku, packs in packs_by_sku.items()]
  order_pad_path.write_text("".join(order_lines))

  fieldnames = (
      "item_id",
      "channel",
      "description",
      "vendor",
      "sku",
      "sku_status",
      "url",
      "selection_hint",
      "required_qty",
      "pack_qty",
      "packs_to_order",
      "pack_price_usd",
      "extended_cost_usd",
      "interface_check",
      "is_interface_valid",
  )
  item_rows = [
      {
          "item_id": item.item_id,
          "channel": item.channel,
          "description": item.description,
          "vendor": item.vendor,
          "sku": item.sku,
          "sku_status": item.sku_status,
          "url": item.url,
          "selection_hint": item.selection_hint,
          "required_qty": item.required_qty,
          "pack_qty": item.pack_qty,
          "packs_to_order": item.packs_to_order,
          "pack_price_usd": item.pack_price_usd,
          "extended_cost_usd": item.extended_cost_usd,
          "interface_check": item.interface_check,
          "is_interface_valid": item.is_interface_valid,
      }
      for item in report.items
  ]
  with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(item_rows)

  check_rows = [
      {
          "part_id": chk.part_id,
          "is_valid": chk.is_valid,
          "active_mode": chk.active_mode,
          "uses_fallback": chk.uses_fallback,
          "fallback_reason": chk.fallback_reason,
          "violations": list(chk.violations),
      }
      for chk in report.c500_checks
  ]
  payload = {
      "has_fourth_axis": report.has_fourth_axis,
      "validation": {
          "is_valid": report.validation.is_valid,
          "total_items": report.validation.total_items,
          "verified_exact_mcmaster_count": (
              report.validation.verified_exact_mcmaster_count
          ),
          "anchor_table_mcmaster_count": (
              report.validation.anchor_table_mcmaster_count
          ),
          "powersports_cots_count": report.validation.powersports_cots_count,
          "channel_costs_usd": report.validation.channel_costs_usd,
          "total_cost_usd": report.validation.total_cost_usd,
          "violations": list(report.validation.violations),
      },
      "c500_checks": check_rows,
      "items": item_rows,
  }
  json_path.write_text(json.dumps(payload, indent=2))
  return BomExportPaths(
      order_pad_txt=order_pad_path,
      costed_csv=csv_path,
      summary_json=json_path,
  )


def _extract_engine_targets(
    engine_cfg: EngineSpecsConfigRoot,
) -> EngineInterfaceTargets:
  """Extract normalized metric interface dimensions from engine_specs.yaml."""
  g = engine_cfg["geometry"]
  v = engine_cfg.get("valves", {})
  lub = engine_cfg.get("lubrication", {})
  return EngineInterfaceTargets(
      bore_mm=float(g["bore"]) * MM_PER_M,
      stroke_mm=float(g["stroke"]) * MM_PER_M,
      con_rod_mm=float(g["con_rod_length"]) * MM_PER_M,
      wrist_pin_mm=float(g["wrist_pin_diameter"]) * MM_PER_M,
      crankpin_mm=float(g["crankpin_diameter"]) * MM_PER_M,
      main_journal_mm=float(g["main_journal_diameter"]) * MM_PER_M,
      intake_valve_mm=float(v.get("intake_diameter", DEFAULT_INTAKE_VALVE_M)) * MM_PER_M,
      exhaust_valve_mm=float(v.get("exhaust_diameter", DEFAULT_EXHAUST_VALVE_M)) * MM_PER_M,
      max_valve_lift_mm=float(v.get("max_lift", DEFAULT_VALVE_LIFT_M)) * MM_PER_M,
      oil_grade=str(lub.get("oil_grade", DEFAULT_OIL_GRADE)),
      sump_volume_l=float(lub.get("sump_volume_l", DEFAULT_SUMP_VOLUME_L)),
  )


def _is_item_active(item: BomItemConfig, has_fourth_axis: bool) -> bool:
  """Determine whether an item applies for the active 4th-axis setting."""
  if item.get("requires_fourth_axis_active", False) and not has_fourth_axis:
    return False
  if item.get("fallback_when_no_fourth_axis", False) and has_fourth_axis:
    return False
  return True


def _resolve_and_check_item(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
    c500_spec: C500Spec,
    sku_series: Dict[str, SkuSeriesConfig],
    base_url: str,
    effective_fourth_axis: bool,
) -> Tuple[ResolvedBomItem, List[str]]:
  """Resolve SKU/pricing and validate all C500 & engine mating constraints."""
  sku, sku_status, url, pack_qty, pack_price, hint = _resolve_item_sku(
      raw_item=raw_item,
      targets=targets,
      sku_series=sku_series,
      base_url=base_url,
  )
  req_qty = max(MIN_REQUIRED_QTY, int(raw_item.get("required_qty", MIN_REQUIRED_QTY)))
  safe_pack_qty = max(MIN_REQUIRED_QTY, pack_qty)
  packs_to_order = math.ceil(req_qty / safe_pack_qty)
  ext_cost = round(packs_to_order * pack_price, COST_DECIMAL_PLACES)

  item_errors: List[str] = []
  item_id = str(raw_item["item_id"])
  if "thread" in raw_item:
    thr_err = c500_spec.check_thread(str(raw_item["thread"]))
    if thr_err is not None:
      item_errors.append(f"{item_id}: {thr_err}")

  _validate_stock_and_material(
      raw_item=raw_item,
      c500_spec=c500_spec,
      effective_fourth_axis=effective_fourth_axis,
      errors=item_errors,
  )
  _validate_geometry_mating(
      raw_item=raw_item, targets=targets, c500_spec=c500_spec, errors=item_errors
  )
  _validate_liner_tube_mating(raw_item=raw_item, targets=targets, errors=item_errors)
  _validate_cots_mating(raw_item=raw_item, targets=targets, errors=item_errors)

  check_summary = "PASS" if not item_errors else "; ".join(item_errors)
  item_obj = ResolvedBomItem(
      item_id=item_id,
      channel=str(raw_item["channel"]),
      description=str(raw_item["description"]),
      vendor=str(raw_item["vendor"]),
      sku=sku,
      sku_status=sku_status,
      url=url,
      selection_hint=hint,
      required_qty=req_qty,
      pack_qty=safe_pack_qty,
      packs_to_order=packs_to_order,
      pack_price_usd=round(pack_price, COST_DECIMAL_PLACES),
      extended_cost_usd=ext_cost,
      interface_check=check_summary,
      is_interface_valid=len(item_errors) == 0,
  )
  return item_obj, item_errors


def _resolve_item_sku(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
    sku_series: Dict[str, SkuSeriesConfig],
    base_url: str,
) -> Tuple[str, str, str, int, float, str]:
  """Resolve exact SKU or anchor-table fallback from offline catalog series."""
  hint = str(raw_item.get("selection_hint", ""))
  rule = str(raw_item.get("parametric_rule", ""))
  if rule == RULE_MAIN_JOURNAL_SEAL:
    hint = (
        f"Seals > Metric Rotary Shaft Seals > Shaft Dia: "
        f"{targets.main_journal_mm:g} mm | Bore OD: 47 mm | Width: 7 mm | Lip: Viton FKM"
    )

  series_name = str(raw_item.get("series", ""))
  if not series_name:
    return (
        str(raw_item.get("sku", "")),
        str(raw_item.get("sku_status", STATUS_COTS_SPEC)),
        str(raw_item.get("url", "")),
        int(raw_item.get("pack_qty", MIN_REQUIRED_QTY)),
        float(raw_item.get("pack_price_usd", 0.0)),
        hint,
    )

  series_cfg = sku_series[series_name]
  series_key, req_dim = _compute_series_query(raw_item, targets)
  variant_table = _extract_series_table(series_cfg)
  entry = _select_series_variant(variant_table.get(series_key), req_dim)
  if entry is None:
    anchor_sku = str(series_cfg["anchor_sku"])
    anchor_url = str(series_cfg["anchor_url"])
    target_note = f"{series_key} length >= {req_dim:.1f} mm" if req_dim > 0.0 else series_key
    fallback_hint = f"{hint} | Target {target_note} (exceeds offline SKU table)"
    return (
        anchor_sku,
        STATUS_ANCHOR_TABLE,
        anchor_url,
        MIN_REQUIRED_QTY,
        DEFAULT_FALLBACK_PRICE_USD,
        fallback_hint,
    )

  sku = str(entry["sku"])
  resolved_url = f"{base_url}/{sku}/"
  if req_dim > 0.0:
    hint = f"{hint} | Selected {entry['length_mm']:.0f} mm (min {req_dim:.1f} mm)"
  return (
      sku,
      STATUS_VERIFIED_EXACT,
      resolved_url,
      int(entry.get("pack_qty", MIN_REQUIRED_QTY)),
      float(entry.get("pack_price_usd", 0.0)),
      hint,
  )


def _compute_series_query(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
) -> Tuple[str, float]:
  """Compute parametric series key and required minimum dimension [mm]."""
  rule = str(raw_item.get("parametric_rule", ""))
  base_key = str(raw_item.get("series_key", ""))
  if rule == RULE_HEAD_TIE_BOLT:
    req_len = targets.stroke_mm * HEAD_BOLT_STROKE_FACTOR + HEAD_BOLT_BASE_GRIP_MM
    return base_key, req_len
  if rule == RULE_WRIST_PIN_DIAM:
    return f"{targets.wrist_pin_mm:g}mm", 0.0
  if rule == RULE_MAIN_JOURNAL_BEARING:
    return f"{targets.main_journal_mm:g}mm", 0.0
  return base_key, float(raw_item.get("target_length_mm", 0.0))


def _extract_series_table(
    series_cfg: SkuSeriesConfig,
) -> Dict[str, List[SkuVariantConfig] | SkuVariantConfig]:
  """Extract the size-keyed variant mapping inside a SKU series block."""
  for subkey in SERIES_SUBKEY_ORDER:
    if subkey in series_cfg:
      return dict(series_cfg[subkey])
  return {}


def _select_series_variant(
    table_entry: Optional[List[SkuVariantConfig] | SkuVariantConfig],
    required_dim_mm: float,
) -> Optional[SkuVariantConfig]:
  """Select the smallest catalog entry meeting required_dim_mm."""
  if table_entry is None:
    return None
  if isinstance(table_entry, dict):
    return table_entry
  sorted_entries = sorted(table_entry, key=lambda row: float(row.get("length_mm", 0.0)))
  for row in sorted_entries:
    if float(row.get("length_mm", 0.0)) >= required_dim_mm - DIM_MATCH_TOL_MM:
      return row
  return None


def _validate_stock_and_material(
    raw_item: BomItemConfig,
    c500_spec: C500Spec,
    effective_fourth_axis: bool,
    errors: List[str],
) -> None:
  """Check C500 material policy, FEA catalog coverage, and stock envelopes."""
  item_id = str(raw_item["item_id"])
  material = str(raw_item.get("material", ""))
  if not material:
    return

  mat_err = c500_spec.check_material(material)
  if mat_err is not None:
    errors.append(f"{item_id}: {mat_err}")
  if material != NON_FEA_LINER_MATERIAL and material.upper() not in MATERIAL_CATALOG:
    errors.append(f"{item_id}: Material '{material}' missing from FEA MATERIAL_CATALOG")

  dims = raw_item.get("stock_dims_mm")
  mode = str(raw_item.get("c500_mode", ""))
  if dims and mode:
    stock_part = PartMachiningSpec(
        part_id=item_id,
        module="bom_raw_stock",
        mode=mode,
        fourth_axis_requirement="none",
        envelope_mm=(float(dims[0]), float(dims[1]), float(dims[2])),
        material=material,
        pocket_depth_mm=0.0,
        min_corner_radius_mm=c500_spec.tooling.min_internal_corner_radius_mm,
        fallback_strategy=str(raw_item.get("selection_hint", "")),
    )
    chk = c500_spec.check_part(
        stock_part, has_fourth_axis_override=effective_fourth_axis
    )
    for err in chk.violations:
      errors.append(f"{item_id}: {err}")


def _validate_geometry_mating(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
    c500_spec: C500Spec,
    errors: List[str],
) -> None:
  """Validate raw billet stock dimensions against engine con-rod and piston sizes."""
  item_id = str(raw_item["item_id"])
  rule = str(raw_item.get("parametric_rule", ""))
  dims = raw_item.get("stock_dims_mm", [])
  if not dims:
    return

  if rule == RULE_CONROD_BILLET_STOCK:
    req_len = targets.con_rod_mm + CONROD_BILLET_ALLOWANCE_MM
    if req_len > max(float(d) for d in dims):
      errors.append(
          f"{item_id}: Con-rod billet needs {req_len:.1f} mm, exceeds stock"
      )
    if req_len > max(c500_spec.envelope.max_housing_xy_mm):
      errors.append(
          f"{item_id}: Con-rod billet {req_len:.1f} mm exceeds C500 XY"
      )
    return

  if rule == RULE_PISTON_BAR_STOCK:
    bar_od = float(dims[0])
    min_bar_od = targets.bore_mm + PISTON_STOCK_ALLOWANCE_MM
    if bar_od < min_bar_od:
      errors.append(
          f"{item_id}: Piston stock {bar_od:.1f} mm < required"
          f" {min_bar_od:.1f} mm"
      )


def _validate_liner_tube_mating(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
    errors: List[str],
) -> None:
  """Validate DOM cylinder liner tube raw ID and wall allowance against bore."""
  if "raw_id_mm" not in raw_item or "raw_od_mm" not in raw_item:
    return
  item_id = str(raw_item["item_id"])
  raw_id = float(raw_item["raw_id_mm"])
  raw_od = float(raw_item["raw_od_mm"])
  if raw_id > targets.bore_mm + DIM_MATCH_TOL_MM:
    errors.append(
        f"{item_id}: Liner raw ID {raw_id:.2f} mm exceeds bore"
        f" {targets.bore_mm:.2f} mm"
    )
  if targets.bore_mm + LINER_WALL_ALLOWANCE_MM > raw_od + DIM_MATCH_TOL_MM:
    errors.append(
        f"{item_id}: Liner OD {raw_od:.2f} mm leaves insufficient wall for bore"
        f" {targets.bore_mm:.2f} mm"
    )


def _validate_cots_mating(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
    errors: List[str],
) -> None:
  """Validate COTS component bores, lifts, oil grade/volume, and valves."""
  item_id = str(raw_item["item_id"])
  dim_checks = (
      ("Ring bore", raw_item.get("mating_bore_mm"), targets.bore_mm),
      (
          "Crankpin bearing bore",
          raw_item.get("mating_crankpin_mm"),
          targets.crankpin_mm,
      ),
      ("COTS intake valve", raw_item.get("intake_head_mm"), targets.intake_valve_mm),
      (
          "COTS exhaust valve",
          raw_item.get("exhaust_head_mm"),
          targets.exhaust_valve_mm,
      ),
  )
  for label, raw_val, target_val in dim_checks:
    if raw_val is None:
      continue
    actual_val = float(raw_val)
    if abs(actual_val - target_val) > DIM_MATCH_TOL_MM:
      errors.append(
          f"{item_id}: {label} {actual_val:.1f} mm != target"
          f" {target_val:.1f} mm"
      )

  if "max_travel_mm" in raw_item:
    req_travel = targets.max_valve_lift_mm + VALVE_COIL_BIND_MARGIN_MM
    max_travel = float(raw_item["max_travel_mm"])
    if max_travel < req_travel:
      errors.append(
          f"{item_id}: Spring travel {max_travel:.1f} mm < required"
          f" {req_travel:.1f} mm"
      )

  _validate_oil_pack_mating(raw_item=raw_item, targets=targets, errors=errors)


def _validate_oil_pack_mating(
    raw_item: BomItemConfig,
    targets: EngineInterfaceTargets,
    errors: List[str],
) -> None:
  """Validate COTS lubricant grade and pack volume against sump capacity."""
  item_id = str(raw_item["item_id"])
  if "oil_grade" in raw_item:
    item_grade = str(raw_item["oil_grade"]).strip().upper()
    if item_grade != targets.oil_grade.strip().upper():
      errors.append(
          f"{item_id}: Oil '{raw_item['oil_grade']}' != configured"
          f" '{targets.oil_grade}'"
      )
  if "volume_l" in raw_item:
    vol_l = float(raw_item["volume_l"])
    if vol_l < targets.sump_volume_l:
      errors.append(
          f"{item_id}: Oil pack {vol_l:.2f} L < sump capacity"
          f" {targets.sump_volume_l:.2f} L"
      )


def _check_sprocket_ratio(
    sprocket_teeth: Dict[str, int],
    violations: List[str],
) -> None:
  """Verify the SOHC #25 roller-chain timing drive is an exact 2:1 ratio."""
  crank_t = sprocket_teeth.get(CRANK_SPROCKET_ITEM_ID, 0)
  cam_t = sprocket_teeth.get(CAM_SPROCKET_ITEM_ID, 0)
  if crank_t <= 0 or cam_t <= 0:
    violations.append("Missing crank or cam timing sprocket tooth counts")
    return
  ratio = cam_t / crank_t
  if abs(ratio - REQUIRED_TIMING_RATIO) > TIMING_RATIO_TOL:
    violations.append(
        f"SOHC timing sprocket ratio {cam_t}/{crank_t} = {ratio:.2f} != 2.00"
    )


def _summarize_validation(
    items: List[ResolvedBomItem],
    channels: List[str],
    violations: List[str],
) -> BomValidationSummary:
  """Aggregate per-channel costs, SKU status counts, and overall BoM validity."""
  channel_costs: Dict[str, float] = {ch: 0.0 for ch in channels}
  exact_count = 0
  anchor_count = 0
  cots_count = 0

  for item in items:
    channel_costs[item.channel] = round(
        channel_costs.get(item.channel, 0.0) + item.extended_cost_usd,
        COST_DECIMAL_PLACES,
    )
    if item.sku_status == STATUS_VERIFIED_EXACT:
      exact_count += 1
    elif item.sku_status == STATUS_ANCHOR_TABLE:
      anchor_count += 1
    else:
      cots_count += 1

  total_cost = round(sum(item.extended_cost_usd for item in items), COST_DECIMAL_PLACES)
  return BomValidationSummary(
      is_valid=len(violations) == 0,
      total_items=len(items),
      verified_exact_mcmaster_count=exact_count,
      anchor_table_mcmaster_count=anchor_count,
      powersports_cots_count=cots_count,
      channel_costs_usd=channel_costs,
      total_cost_usd=total_cost,
      violations=tuple(violations),
  )
